import cv2
import numpy as np
import pytest

from vision_line_follower.geometry import Pose
from vision_line_follower.sim.camera import CameraConfig, CameraModel
from vision_line_follower.track.generator import TrackSpec, generate_track
from vision_line_follower.track.rendering import TrackRenderConfig, render_track
from vision_line_follower.vision.fit import fit_line_polynomial, heading_error_from_slope
from vision_line_follower.vision.pipeline import VisionPipeline
from vision_line_follower.vision.sliding_window import SlidingWindowConfig, sliding_window_search
from vision_line_follower.vision.threshold import ThresholdConfig, binarize_line_mask


def _synthetic_bev(
    line_center_col: int, slope_px_per_row: float, width=200, height=320
) -> np.ndarray:
    """A synthetic bird's-eye BGR image: light floor, dark straight/angled line."""
    img = np.full((height, width, 3), 230, dtype=np.uint8)
    for row in range(height):
        col = int(line_center_col + slope_px_per_row * (height - row))
        cv2.line(img, (col, row), (col, row), (20, 20, 20), thickness=1)
    img = cv2.dilate(255 - img, np.ones((1, 14), np.uint8))
    img = 255 - img
    return img


class TestThreshold:
    def test_binarize_isolates_dark_line(self) -> None:
        img = _synthetic_bev(line_center_col=100, slope_px_per_row=0.0)
        mask = binarize_line_mask(img, ThresholdConfig())
        assert mask.dtype == np.uint8
        # Middle column (where the line is) should be mostly active pixels.
        center_col_activity = mask[:, 95:106].mean()
        edge_col_activity = mask[:, 0:10].mean()
        assert center_col_activity > edge_col_activity


class TestSlidingWindow:
    def test_finds_vertical_line(self) -> None:
        img = _synthetic_bev(line_center_col=100, slope_px_per_row=0.0)
        mask = binarize_line_mask(img, ThresholdConfig())
        result = sliding_window_search(mask, SlidingWindowConfig())
        assert result.found
        assert result.xs_px.size > 0
        assert abs(result.xs_px.mean() - 100) < 10

    def test_no_pixels_reports_not_found(self) -> None:
        mask = np.zeros((320, 200), dtype=np.uint8)
        result = sliding_window_search(mask, SlidingWindowConfig())
        assert not result.found

    def test_prior_hint_restricts_search_to_local_branch(self) -> None:
        # Two separated vertical lines; with a prior hint near one of them,
        # the search should lock onto that branch rather than the (larger)
        # combined histogram peak.
        img = np.full((320, 200, 3), 230, dtype=np.uint8)
        cv2.line(img, (40, 0), (40, 319), (20, 20, 20), thickness=6)
        cv2.line(img, (160, 0), (160, 319), (20, 20, 20), thickness=6)
        mask = binarize_line_mask(img, ThresholdConfig())

        result = sliding_window_search(mask, SlidingWindowConfig(), prior_x_px=40.0)
        assert result.found
        assert abs(result.xs_px.mean() - 40) < 15


class TestFit:
    def test_straight_line_zero_slope(self) -> None:
        forward = np.linspace(0.1, 0.4, 30)
        lateral = np.zeros_like(forward)
        fit = fit_line_polynomial(forward, lateral)
        assert fit.lateral_at(0.1) == pytest.approx(0.0, abs=1e-9)
        assert fit.slope_at(0.1) == pytest.approx(0.0, abs=1e-9)

    def test_linear_slope_recovered(self) -> None:
        forward = np.linspace(0.1, 0.4, 30)
        lateral = 0.5 * forward + 0.02
        fit = fit_line_polynomial(forward, lateral)
        assert fit.slope_at(0.25) == pytest.approx(0.5, abs=1e-3)

    def test_heading_error_from_slope_matches_arctan(self) -> None:
        assert heading_error_from_slope(1.0) == pytest.approx(np.pi / 4)

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_line_polynomial(np.array([0.1]), np.array([0.0]), order=2)


class TestVisionPipelineIntegration:
    def test_robot_on_centerline_reports_near_zero_error(self) -> None:
        track = generate_track(TrackSpec(name="oval", seed=1))
        world = render_track(
            track, TrackRenderConfig(seed=0, lighting_strength=0.0, occlusion_level=0.0)
        )
        camera = CameraModel(CameraConfig(pixel_noise_std=0.0))
        pipeline = VisionPipeline(camera)

        idx = 300
        p = track.centerline[idx]
        pose = Pose(float(p[0]), float(p[1]), float(track.heading[idx]))
        frame = camera.render(world, pose, rng=None)

        result = pipeline.process(frame)
        assert result.found
        assert abs(result.lateral_error_m) < 0.01

    def test_no_line_in_view_reports_not_found(self) -> None:
        track = generate_track(TrackSpec(name="oval", seed=1))
        world = render_track(track, TrackRenderConfig(seed=0))
        camera = CameraModel(CameraConfig(pixel_noise_std=0.0))
        pipeline = VisionPipeline(camera)

        far_away_pose = Pose(1000.0, 1000.0, 0.0)
        frame = camera.render(world, far_away_pose, rng=None)
        result = pipeline.process(frame)
        assert not result.found

    def test_reset_clears_temporal_anchor(self) -> None:
        track = generate_track(TrackSpec(name="oval", seed=1))
        world = render_track(track, TrackRenderConfig(seed=0))
        camera = CameraModel(CameraConfig(pixel_noise_std=0.0))
        pipeline = VisionPipeline(camera)

        p = track.centerline[0]
        pose = Pose(float(p[0]), float(p[1]), float(track.heading[0]))
        frame = camera.render(world, pose, rng=None)
        pipeline.process(frame)
        assert pipeline._prior_x_px is not None
        pipeline.reset()
        assert pipeline._prior_x_px is None
