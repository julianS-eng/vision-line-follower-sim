"""Shared geometric primitives and helper functions.

Coordinate conventions used throughout the project:

- **World frame**: right-handed, ``x`` to the right, ``y`` upward (standard
  Cartesian plane), units in metres. Angles are measured counter-clockwise
  from the positive ``x`` axis, in radians.
- **Robot/vehicle frame**: ``x`` forward, ``y`` to the left, origin at the
  robot's rear-axle midpoint (the point the differential-drive kinematics
  integrate).
- **Image/pixel frame**: ``u`` (column) right, ``v`` (row) down, origin at
  the top-left corner, matching OpenCV/NumPy array indexing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle (radians) to the ``(-pi, pi]`` interval."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True, slots=True)
class Pose:
    """A 2D rigid-body pose in the world frame."""

    x: float
    y: float
    theta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=np.float64)

    def world_to_local(self, points_world: np.ndarray) -> np.ndarray:
        """Transform an ``(N, 2)`` array of world points into this pose's frame.

        Returns an ``(N, 2)`` array of ``(x_forward, y_left)`` coordinates.
        """
        c, s = math.cos(self.theta), math.sin(self.theta)
        delta = points_world - np.array([self.x, self.y])
        x_forward = c * delta[:, 0] + s * delta[:, 1]
        y_left = -s * delta[:, 0] + c * delta[:, 1]
        return np.stack([x_forward, y_left], axis=1)

    def local_to_world(self, points_local: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`world_to_local`."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        x_f, y_l = points_local[:, 0], points_local[:, 1]
        x_world = c * x_f - s * y_l + self.x
        y_world = s * x_f + c * y_l + self.y
        return np.stack([x_world, y_world], axis=1)

    def affine_world_to_local_matrix(self) -> np.ndarray:
        """3x3 homogeneous matrix mapping world ``(x, y, 1)`` to local frame."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array(
            [
                [c, s, -c * self.x - s * self.y],
                [-s, c, s * self.x - c * self.y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def rotation_matrix_2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def nearest_point_on_polyline(
    query: np.ndarray, polyline: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """Find the closest point on a polyline to ``query``.

    Args:
        query: ``(2,)`` point.
        polyline: ``(N, 2)`` array of consecutive vertices.

    Returns:
        Tuple of ``(closest_point, segment_index, signed_lateral_distance)``.
        The signed distance is positive when ``query`` is to the left of the
        segment's direction of travel.
    """
    a = polyline[:-1]
    b = polyline[1:]
    ab = b - a
    ab_len_sq = np.sum(ab * ab, axis=1)
    ab_len_sq[ab_len_sq == 0.0] = 1e-12
    t = np.sum((query - a) * ab, axis=1) / ab_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    projections = a + t_clamped[:, None] * ab
    dists = np.linalg.norm(projections - query, axis=1)
    idx = int(np.argmin(dists))
    closest = projections[idx]
    seg_dir = ab[idx] / (np.linalg.norm(ab[idx]) + 1e-12)
    normal = np.array([-seg_dir[1], seg_dir[0]])
    signed = float(np.dot(query - closest, normal))
    return closest, idx, signed
