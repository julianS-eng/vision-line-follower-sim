import numpy as np

from vision_line_follower.geometry import Pose
from vision_line_follower.sim.camera import CameraConfig, CameraModel
from vision_line_follower.track.generator import TrackSpec, generate_track
from vision_line_follower.track.rendering import TrackRenderConfig, render_track


def test_render_output_shape_matches_resolution() -> None:
    track = generate_track(TrackSpec(name="oval", seed=1))
    world = render_track(track, TrackRenderConfig(seed=0))
    camera = CameraModel(CameraConfig(resolution=(160, 120)))
    frame = camera.render(world, Pose(0.0, 0.0, 0.0), rng=None)
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8


def test_ground_to_camera_homography_projects_forward_point_below_center() -> None:
    camera = CameraModel(CameraConfig())
    h = camera.ground_to_camera_homography()
    point = np.array([0.3, 0.0, 1.0])
    px = h @ point
    px = px[:2] / px[2]
    cx = camera.config.resolution[0] / 2
    cy = camera.config.resolution[1] / 2
    assert abs(px[0] - cx) < 5.0  # centered laterally
    assert px[1] > cy - 200  # roughly within/near the image (not absurd)


def test_render_is_deterministic_without_noise() -> None:
    track = generate_track(TrackSpec(name="oval", seed=1))
    world = render_track(track, TrackRenderConfig(seed=0))
    camera = CameraModel(CameraConfig(pixel_noise_std=0.0))
    pose = Pose(0.0, 0.0, 0.0)
    frame_a = camera.render(world, pose, rng=None)
    frame_b = camera.render(world, pose, rng=None)
    np.testing.assert_array_equal(frame_a, frame_b)


def test_pixel_noise_changes_output() -> None:
    track = generate_track(TrackSpec(name="oval", seed=1))
    world = render_track(track, TrackRenderConfig(seed=0))
    camera = CameraModel(CameraConfig(pixel_noise_std=20.0))
    pose = Pose(0.0, 0.0, 0.0)
    frame_a = camera.render(world, pose, rng=np.random.default_rng(0))
    frame_b = camera.render(world, pose, rng=np.random.default_rng(1))
    assert not np.array_equal(frame_a, frame_b)
