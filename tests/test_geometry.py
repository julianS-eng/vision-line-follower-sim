import math

import numpy as np
import pytest

from vision_line_follower.geometry import Pose, nearest_point_on_polyline, wrap_angle


@pytest.mark.parametrize(
    ("angle", "expected_cos", "expected_sin"),
    [
        (0.0, 1.0, 0.0),
        (math.pi, -1.0, 0.0),
        (-math.pi, -1.0, 0.0),
        (3 * math.pi, -1.0, 0.0),
        (-3 * math.pi, -1.0, 0.0),
        (2.5 * math.pi, 0.0, 1.0),
    ],
)
def test_wrap_angle(angle: float, expected_cos: float, expected_sin: float) -> None:
    # Compare via cos/sin: +/-pi are the same angle, and which one wrap_angle
    # picks at that exact boundary is a floating-point coin flip we don't
    # want to pin down.
    wrapped = wrap_angle(angle)
    assert -math.pi - 1e-9 <= wrapped <= math.pi + 1e-9
    assert math.cos(wrapped) == pytest.approx(expected_cos, abs=1e-9)
    assert math.sin(wrapped) == pytest.approx(expected_sin, abs=1e-9)


def test_pose_world_to_local_roundtrip() -> None:
    pose = Pose(x=1.5, y=-2.0, theta=math.radians(37.0))
    points = np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.5]])
    local = pose.world_to_local(points)
    back = pose.local_to_world(local)
    np.testing.assert_allclose(back, points, atol=1e-9)


def test_pose_world_to_local_identity_at_origin() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    points = np.array([[2.0, 3.0]])
    local = pose.world_to_local(points)
    np.testing.assert_allclose(local, points)


def test_pose_forward_axis_is_local_x() -> None:
    # A point directly ahead of the robot (along its heading) should map to
    # a purely positive local x with zero local y.
    pose = Pose(x=0.0, y=0.0, theta=math.radians(90.0))
    point_ahead = np.array([[0.0, 5.0]])  # world +y, which is "forward" at theta=90
    local = pose.world_to_local(point_ahead)
    assert local[0, 0] == pytest.approx(5.0, abs=1e-9)
    assert local[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_affine_world_to_local_matches_method() -> None:
    pose = Pose(x=0.4, y=-1.1, theta=1.2)
    points = np.array([[1.0, 2.0], [-3.0, 0.5]])
    via_method = pose.world_to_local(points)

    matrix = pose.affine_world_to_local_matrix()
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    via_matrix = (matrix @ homog.T).T[:, :2]
    np.testing.assert_allclose(via_method, via_matrix, atol=1e-9)


def test_nearest_point_on_polyline_straight_line() -> None:
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
    closest, seg_idx, signed = nearest_point_on_polyline(np.array([5.0, 2.0]), polyline)
    np.testing.assert_allclose(closest, [5.0, 0.0], atol=1e-9)
    assert seg_idx == 0
    assert signed == pytest.approx(2.0)


def test_nearest_point_on_polyline_sign_convention() -> None:
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
    _closest, _idx, signed_left = nearest_point_on_polyline(np.array([5.0, 1.0]), polyline)
    _closest, _idx, signed_right = nearest_point_on_polyline(np.array([5.0, -1.0]), polyline)
    assert signed_left > 0
    assert signed_right < 0
