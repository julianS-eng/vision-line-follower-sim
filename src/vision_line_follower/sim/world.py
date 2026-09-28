"""Closed-loop simulation: ties the track, camera, vision pipeline, robot
model and a controller together and runs a fixed-timestep control loop.

At every step the loop:

1. Renders the camera frame for the robot's current true pose.
2. Runs it through the :class:`VisionPipeline` to get lateral/heading error
   and curvature estimates.
3. Computes a curvature-scheduled target speed and asks the controller for
   an angular velocity.
4. Steps the robot's motor/kinematic model.
5. Records ground-truth cross-track error against the track's own
   centerline (independent of the vision estimate) for unbiased benchmarking.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vision_line_follower.control.base import Controller
from vision_line_follower.geometry import Pose
from vision_line_follower.sim.camera import CameraModel
from vision_line_follower.sim.robot import DifferentialDriveRobot, RobotConfig, RobotState
from vision_line_follower.track.generator import Track
from vision_line_follower.track.rendering import WorldImage
from vision_line_follower.vision.pipeline import VisionPipeline, VisionResult
from vision_line_follower.vision.temporal_filter import TemporalFilter, TemporalFilterConfig


@dataclass(frozen=True, slots=True)
class SpeedSchedule:
    """Curvature-aware target-speed scheduler, shared across all controllers.

    ``v_target = clip(base_speed - curvature_gain * |curvature|, min_speed, base_speed)``

    Keeping the speed profile identical across controllers isolates the
    steering-law comparison in the benchmark from confounding speed effects.
    """

    base_speed_mps: float = 0.35
    min_speed_mps: float = 0.14
    curvature_gain: float = 0.22

    def target_speed(self, curvature: float) -> float:
        v = self.base_speed_mps - self.curvature_gain * abs(curvature)
        return max(self.min_speed_mps, min(self.base_speed_mps, v))


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Parameters controlling a single closed-loop run.

    Attributes:
        dt: Control-loop timestep, seconds.
        max_steps: Hard cap on simulation steps (safety bound).
        off_track_max_m: Ground-truth lateral distance from the centerline
            beyond which the run is declared a failure.
        vision_lost_grace_steps: Consecutive frames with no line detected
            that are tolerated (dead-reckoning on the last known estimate)
            before the run is declared a failure.
        nearest_point_window: Half-width (in track points) of the local
            window searched for the ground-truth nearest point each step;
            keeps the search cheap and avoids jumping across self-crossings.
        camera_seed: Seed for camera pixel noise.
        robot_seed: Seed for robot process noise.
        vision_filter_alpha: EMA smoothing factor applied to the vision
            estimate before it reaches the controller (see
            :class:`~vision_line_follower.vision.temporal_filter.TemporalFilter`);
            ``1.0`` disables smoothing.
    """

    dt: float = 0.02
    max_steps: int = 4000
    off_track_max_m: float = 0.09
    vision_lost_grace_steps: int = 15
    nearest_point_window: int = 180
    camera_seed: int = 0
    robot_seed: int = 0
    vision_filter_alpha: float = 0.45


@dataclass(slots=True)
class StepTelemetry:
    t: float
    pose: Pose
    cross_track_error_m: float
    vision_found: bool
    lateral_error_est_m: float
    heading_error_est_rad: float
    speed_mps: float
    omega_cmd: float
    camera_image: np.ndarray | None = None


@dataclass(slots=True)
class SimulationResult:
    """Outcome and aggregate metrics of a closed-loop run."""

    controller_name: str
    track_name: str
    success: bool
    failure_reason: str | None
    steps_run: int
    sim_time_s: float
    laps_completed: float
    mean_abs_cross_track_error_m: float
    max_abs_cross_track_error_m: float
    rms_cross_track_error_m: float
    mean_speed_mps: float
    vision_found_rate: float
    poses: np.ndarray  # (steps, 3) x, y, theta
    cross_track_errors: np.ndarray  # (steps,)
    timestamps: np.ndarray  # (steps,)


def _nearest_point_local(
    query: np.ndarray, centerline: np.ndarray, center_idx: int, window: int
) -> tuple[np.ndarray, int, float]:
    n = centerline.shape[0]
    half_window = min(window, n // 2 - 1)
    offsets = np.arange(-half_window, half_window + 2)
    idxs = (center_idx + offsets) % n
    pts = centerline[idxs]

    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    ab_len_sq = np.sum(ab * ab, axis=1)
    ab_len_sq[ab_len_sq == 0.0] = 1e-12
    t = np.sum((query - a) * ab, axis=1) / ab_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    projections = a + t_clamped[:, None] * ab
    dists = np.linalg.norm(projections - query, axis=1)
    local_best = int(np.argmin(dists))
    closest = projections[local_best]
    seg_dir = ab[local_best] / (np.linalg.norm(ab[local_best]) + 1e-12)
    normal = np.array([-seg_dir[1], seg_dir[0]])
    signed = float(np.dot(query - closest, normal))
    global_idx = int(idxs[local_best])
    return closest, global_idx, signed


def run_simulation(
    track: Track,
    world: WorldImage,
    camera: CameraModel,
    vision_pipeline: VisionPipeline,
    controller: Controller,
    robot_config: RobotConfig,
    speed_schedule: SpeedSchedule,
    sim_config: SimulationConfig,
    on_step: Callable[[StepTelemetry], None] | None = None,
    record_camera_every: int | None = None,
) -> SimulationResult:
    """Run a single closed-loop simulation and return aggregate metrics."""
    camera_rng = np.random.default_rng(sim_config.camera_seed)
    robot_rng = np.random.default_rng(sim_config.robot_seed)

    start = track.centerline[0]
    start_heading = track.heading[0]
    robot = DifferentialDriveRobot(
        robot_config,
        RobotState(x=float(start[0]), y=float(start[1]), theta=float(start_heading)),
        rng=robot_rng,
    )
    controller.reset()
    vision_pipeline.reset()

    dt = sim_config.dt
    last_idx = 0
    progress_unwrapped = 0.0
    n_points = track.n_points

    lost_streak = 0
    last_vision: VisionResult | None = None
    vision_filter = TemporalFilter(TemporalFilterConfig(alpha=sim_config.vision_filter_alpha))

    poses = np.zeros((sim_config.max_steps, 3), dtype=np.float64)
    cross_track_errors = np.zeros(sim_config.max_steps, dtype=np.float64)
    timestamps = np.zeros(sim_config.max_steps, dtype=np.float64)
    speeds = np.zeros(sim_config.max_steps, dtype=np.float64)
    found_flags = np.zeros(sim_config.max_steps, dtype=bool)

    success = False
    failure_reason: str | None = None
    steps_run = 0

    for step in range(sim_config.max_steps):
        pose = robot.pose
        camera_image = camera.render(world, pose, rng=camera_rng)
        vision_result = vision_pipeline.process(camera_image)

        if vision_result.found:
            lost_streak = 0
            last_vision = vision_result
        else:
            lost_streak += 1

        effective = vision_result if vision_result.found else last_vision
        if effective is None:
            lateral_est, heading_est, curvature_est = 0.0, 0.0, 0.0
        elif vision_result.found:
            lateral_est, heading_est, curvature_est = vision_filter.update(
                effective.lateral_error_m, effective.heading_error_rad, effective.curvature
            )
        else:
            # `effective is not None` here implies the filter was primed by
            # an earlier found frame (see `last_vision` above).
            assert vision_filter.last is not None
            lateral_est, heading_est, curvature_est = vision_filter.last

        v_target = speed_schedule.target_speed(curvature_est)
        omega_cmd = controller.compute(lateral_est, heading_est, curvature_est, v_target, dt)
        v_left, v_right = DifferentialDriveRobot.unicycle_to_wheel_speeds(
            v_target, omega_cmd, robot_config.wheel_base_m
        )

        query = np.array([pose.x, pose.y])
        _closest, global_idx, signed_dist = _nearest_point_local(
            query, track.centerline, last_idx, sim_config.nearest_point_window
        )
        delta_idx = global_idx - last_idx
        half_n = n_points / 2.0
        if delta_idx > half_n:
            delta_idx -= n_points
        elif delta_idx < -half_n:
            delta_idx += n_points
        progress_unwrapped += delta_idx
        last_idx = global_idx

        poses[step] = pose.as_array()
        cross_track_errors[step] = signed_dist
        timestamps[step] = step * dt
        speeds[step] = v_target
        found_flags[step] = vision_result.found
        steps_run = step + 1

        if on_step is not None:
            record_img = (
                camera_image if (record_camera_every and step % record_camera_every == 0) else None
            )
            on_step(
                StepTelemetry(
                    t=step * dt,
                    pose=pose,
                    cross_track_error_m=signed_dist,
                    vision_found=vision_result.found,
                    lateral_error_est_m=lateral_est,
                    heading_error_est_rad=heading_est,
                    speed_mps=v_target,
                    omega_cmd=omega_cmd,
                    camera_image=record_img,
                )
            )

        if abs(signed_dist) > sim_config.off_track_max_m:
            failure_reason = "off_track"
            break
        if lost_streak > sim_config.vision_lost_grace_steps:
            failure_reason = "vision_lost"
            break

        robot.step(v_left, v_right, dt)

        if progress_unwrapped >= n_points:
            success = True
            failure_reason = None
            break

    poses = poses[:steps_run]
    cross_track_errors = cross_track_errors[:steps_run]
    timestamps = timestamps[:steps_run]
    speeds = speeds[:steps_run]
    found_flags = found_flags[:steps_run]

    if not success and failure_reason is None:
        failure_reason = "max_steps_reached"

    abs_err = np.abs(cross_track_errors)
    return SimulationResult(
        controller_name=controller.name,
        track_name=track.name,
        success=success,
        failure_reason=failure_reason,
        steps_run=steps_run,
        sim_time_s=steps_run * dt,
        laps_completed=float(progress_unwrapped / n_points),
        mean_abs_cross_track_error_m=float(abs_err.mean()) if steps_run else float("nan"),
        max_abs_cross_track_error_m=float(abs_err.max()) if steps_run else float("nan"),
        rms_cross_track_error_m=float(math.sqrt(np.mean(cross_track_errors**2)))
        if steps_run
        else float("nan"),
        mean_speed_mps=float(speeds.mean()) if steps_run else float("nan"),
        vision_found_rate=float(found_flags.mean()) if steps_run else float("nan"),
        poses=poses,
        cross_track_errors=cross_track_errors,
        timestamps=timestamps,
    )
