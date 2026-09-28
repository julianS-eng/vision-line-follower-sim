import numpy as np
import pytest

from vision_line_follower.track.generator import TrackSpec, generate_track


@pytest.mark.parametrize("name", ["oval", "figure_eight", "curvy_loop"])
def test_generate_track_basic_properties(name: str) -> None:
    track = generate_track(TrackSpec(name=name, seed=1))
    assert track.n_points > 100
    assert track.centerline.shape == (track.n_points, 2)
    assert track.heading.shape == (track.n_points,)
    assert track.curvature.shape == (track.n_points,)
    assert track.total_length > 0
    # Curvature should be finite and bounded for a smooth spline track.
    assert np.all(np.isfinite(track.curvature))
    assert np.abs(track.curvature).max() < 50.0


def test_generate_track_is_deterministic_given_seed() -> None:
    a = generate_track(TrackSpec(name="oval", seed=42))
    b = generate_track(TrackSpec(name="oval", seed=42))
    np.testing.assert_array_equal(a.centerline, b.centerline)


def test_generate_track_seed_changes_geometry() -> None:
    a = generate_track(TrackSpec(name="curvy_loop", seed=1))
    b = generate_track(TrackSpec(name="curvy_loop", seed=2))
    assert not np.array_equal(a.centerline, b.centerline)


def test_generate_track_uniform_arc_length_spacing() -> None:
    track = generate_track(TrackSpec(name="oval", seed=3, point_spacing_m=0.01))
    seg_lengths = np.hypot(*np.diff(np.vstack([track.centerline, track.centerline[0]]), axis=0).T)
    assert np.allclose(seg_lengths, seg_lengths[0], rtol=2e-3)


def test_generate_track_closed_loop_heading_continuity() -> None:
    # Heading should wrap continuously across the seam (no ~2*pi jump).
    track = generate_track(TrackSpec(name="curvy_loop", seed=7))
    seam_diff = abs(track.heading[0] - track.heading[-1])
    seam_diff = min(seam_diff, 2 * np.pi - seam_diff)
    assert seam_diff < 0.5


def test_crossing_indices_within_bounds() -> None:
    track = generate_track(TrackSpec(name="oval", seed=1, num_crossings=3))
    assert len(track.crossing_indices) == 3
    for idx in track.crossing_indices:
        assert 0 <= idx < track.n_points


def test_bounding_box_contains_centerline() -> None:
    track = generate_track(TrackSpec(name="figure_eight", seed=9))
    min_x, _min_y, _max_x, max_y = track.bounding_box
    assert track.centerline[:, 0].min() == pytest.approx(min_x)
    assert track.centerline[:, 1].max() == pytest.approx(max_y)
