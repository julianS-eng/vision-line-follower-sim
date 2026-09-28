import numpy as np

from vision_line_follower.track.generator import TrackSpec, generate_track
from vision_line_follower.track.rendering import TrackRenderConfig, render_track


def test_render_track_produces_uint8_bgr_image() -> None:
    track = generate_track(TrackSpec(name="oval", seed=1))
    world = render_track(track, TrackRenderConfig(seed=0))
    assert world.image.dtype == np.uint8
    assert world.image.ndim == 3
    assert world.image.shape[2] == 3
    assert world.image.shape[0] > 0
    assert world.image.shape[1] > 0


def test_render_track_world_pixel_affine_roundtrip() -> None:
    track = generate_track(TrackSpec(name="oval", seed=1))
    world = render_track(track, TrackRenderConfig(seed=0))
    to_px = world.world_to_pixel_affine()
    to_world = world.pixel_to_world_affine()
    point_world = np.array([track.centerline[10, 0], track.centerline[10, 1], 1.0])
    px = to_px @ point_world
    back = to_world @ px
    np.testing.assert_allclose(back, point_world, atol=1e-6)


def test_render_track_line_pixels_are_darker_than_floor() -> None:
    track = generate_track(TrackSpec(name="oval", seed=2, width_m=0.03))
    world = render_track(
        track, TrackRenderConfig(seed=0, lighting_strength=0.0, texture_noise_std=0.0)
    )
    to_px = world.world_to_pixel_affine()
    point_on_line = np.array([track.centerline[0, 0], track.centerline[0, 1], 1.0])
    u, v, _ = to_px @ point_on_line
    pixel = world.image[int(v), int(u)]
    corner_pixel = world.image[2, 2]
    assert pixel.astype(int).sum() < corner_pixel.astype(int).sum()


def test_render_track_deterministic_given_seed() -> None:
    track = generate_track(TrackSpec(name="oval", seed=1))
    w1 = render_track(track, TrackRenderConfig(seed=5, occlusion_level=0.5))
    w2 = render_track(track, TrackRenderConfig(seed=5, occlusion_level=0.5))
    np.testing.assert_array_equal(w1.image, w2.image)
