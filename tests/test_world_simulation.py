from vision_line_follower.control.base import Controller
from vision_line_follower.control.pid import PIDController
from vision_line_follower.sim.camera import CameraConfig, CameraModel
from vision_line_follower.sim.robot import RobotConfig
from vision_line_follower.sim.world import SimulationConfig, SpeedSchedule, run_simulation
from vision_line_follower.track.generator import TrackSpec, generate_track
from vision_line_follower.track.rendering import TrackRenderConfig, render_track
from vision_line_follower.vision.pipeline import VisionPipeline


class _ConstantOmegaController(Controller):
    """Test-only controller that always commands a fixed turn rate."""

    name = "constant_omega"

    def __init__(self, omega: float) -> None:
        self._omega = omega

    def reset(self) -> None:
        return None

    def compute(self, lateral_error_m, heading_error_rad, curvature, speed_mps, dt) -> float:
        return self._omega


def _make_scene(seed: int = 1):
    track = generate_track(TrackSpec(name="oval", seed=seed))
    world = render_track(track, TrackRenderConfig(seed=0, lighting_strength=0.1))
    camera = CameraModel(CameraConfig(pixel_noise_std=1.0))
    return track, world, camera


def test_short_run_with_pid_stays_on_track() -> None:
    track, world, camera = _make_scene()
    pipeline = VisionPipeline(camera)
    controller = PIDController()
    sim_config = SimulationConfig(max_steps=150, camera_seed=0, robot_seed=0)

    result = run_simulation(
        track, world, camera, pipeline, controller, RobotConfig(), SpeedSchedule(), sim_config
    )

    assert result.steps_run == 150
    # A short run won't complete a full lap, so it legitimately ends at
    # max_steps_reached; what matters is that it did NOT go off-track or
    # lose the line.
    assert result.failure_reason == "max_steps_reached"
    assert result.mean_abs_cross_track_error_m < 0.05
    assert result.poses.shape == (150, 3)
    assert result.cross_track_errors.shape == (150,)
    assert result.vision_found_rate > 0.9


def test_driving_straight_off_a_curved_track_is_detected() -> None:
    track, world, camera = _make_scene()
    pipeline = VisionPipeline(camera)
    controller = _ConstantOmegaController(omega=0.0)
    sim_config = SimulationConfig(max_steps=1000, off_track_max_m=0.09)

    result = run_simulation(
        track, world, camera, pipeline, controller, RobotConfig(), SpeedSchedule(), sim_config
    )

    assert not result.success
    assert result.failure_reason in ("off_track", "vision_lost")


def test_result_metrics_are_finite_and_nonnegative() -> None:
    track, world, camera = _make_scene(seed=3)
    pipeline = VisionPipeline(camera)
    controller = PIDController()
    sim_config = SimulationConfig(max_steps=120)

    result = run_simulation(
        track, world, camera, pipeline, controller, RobotConfig(), SpeedSchedule(), sim_config
    )

    assert result.mean_abs_cross_track_error_m >= 0
    assert result.max_abs_cross_track_error_m >= result.mean_abs_cross_track_error_m
    assert result.rms_cross_track_error_m >= 0
    assert 0.0 <= result.vision_found_rate <= 1.0
