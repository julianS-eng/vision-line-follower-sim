"""Procedural generation of closed 2D track centerlines.

Tracks are built from a small set of hand-placed control points that are
randomly perturbed (seeded) and then interpolated with a periodic cubic
B-spline (:func:`scipy.interpolate.splprep`). The resulting curve is
resampled to uniform arc-length spacing so that heading and curvature can be
computed with simple finite differences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import splev, splprep

TrackName = Literal["oval", "figure_eight", "curvy_loop"]


@dataclass(frozen=True, slots=True)
class TrackSpec:
    """Parameters describing how to build a track.

    Attributes:
        name: Which base shape/generator to use.
        seed: Random seed controlling control-point perturbation, applied
            noise texture and occlusion placement (set downstream).
        scale: Overall size multiplier in metres (base shapes are laid out
            on the order of 1-2 m before scaling).
        num_crossings: Number of perpendicular line crossings ("intersections")
            painted across the main line, evenly spaced along the track.
        width_m: Physical width of the painted line, in metres.
        point_spacing_m: Target arc-length spacing between resampled
            centerline points, in metres. Smaller values give a smoother
            curvature estimate at the cost of more points.
    """

    name: TrackName
    seed: int = 0
    scale: float = 1.0
    num_crossings: int = 1
    width_m: float = 0.025
    point_spacing_m: float = 0.004


@dataclass(slots=True)
class Track:
    """A closed 2D track with derived differential-geometry quantities."""

    name: str
    centerline: np.ndarray  # (N, 2) metres, world frame
    heading: np.ndarray  # (N,) radians
    curvature: np.ndarray  # (N,) 1/metres
    arc_length: np.ndarray  # (N,) metres, cumulative from point 0
    total_length: float
    width_m: float
    crossing_indices: list[int] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return int(self.centerline.shape[0])

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        xs, ys = self.centerline[:, 0], self.centerline[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def point_at_arclength(self, s: float) -> np.ndarray:
        s_mod = s % self.total_length
        idx = int(np.searchsorted(self.arc_length, s_mod)) % self.n_points
        return self.centerline[idx]


def _base_control_points(name: TrackName, rng: np.random.Generator) -> np.ndarray:
    """Return a hand-designed set of control points (metres) for each shape."""
    if name == "oval":
        n = 10
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rx, ry = 0.9, 0.6
        pts = np.stack([rx * np.cos(angles), ry * np.sin(angles)], axis=1)
        pts += rng.normal(scale=0.05, size=pts.shape)
        return pts
    if name == "figure_eight":
        n = 16
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        scale_x, scale_y = 0.95, 0.55
        pts = np.stack(
            [scale_x * np.sin(t), scale_y * np.sin(t) * np.cos(t)],
            axis=1,
        )
        pts += rng.normal(scale=0.03, size=pts.shape)
        return pts
    if name == "curvy_loop":
        n = 12
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        base_r = 0.85
        radii = base_r + rng.uniform(-0.35, 0.35, size=n)
        radii = np.convolve(np.r_[radii, radii, radii], np.ones(3) / 3, mode="same")[n : 2 * n]
        pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        return pts
    raise ValueError(f"Unknown track name: {name}")


def _resample_uniform_arclength(
    x: np.ndarray, y: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a closed dense polyline to uniform arc-length spacing."""
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total_length = float(s[-1])
    n_out = max(round(total_length / spacing), 16)
    s_uniform = np.linspace(0.0, total_length, n_out, endpoint=False)
    x_closed = np.r_[x, x[0]]
    y_closed = np.r_[y, y[0]]
    x_out = np.interp(s_uniform, s, x_closed)
    y_out = np.interp(s_uniform, s, y_closed)
    return x_out, y_out, s_uniform


def generate_track(spec: TrackSpec) -> Track:
    """Generate a closed track from a :class:`TrackSpec`.

    The pipeline is: perturbed control points -> periodic cubic B-spline ->
    dense sampling -> uniform arc-length resampling -> finite-difference
    heading/curvature.
    """
    rng = np.random.default_rng(spec.seed)
    control_points = _base_control_points(spec.name, rng) * spec.scale

    # Periodic B-spline through the (closed) control polygon.
    cp = np.vstack([control_points, control_points[0]])
    tck, _u = splprep([cp[:, 0], cp[:, 1]], s=0.0, per=True, k=3)
    u_dense = np.linspace(0.0, 1.0, 4000, endpoint=False)
    x_dense, y_dense = splev(u_dense, tck)
    x_dense, y_dense = np.asarray(x_dense), np.asarray(y_dense)

    spacing = spec.point_spacing_m
    x, y, s = _resample_uniform_arclength(x_dense, y_dense, spacing)
    n = x.shape[0]
    ds = s[1] - s[0] if n > 1 else spacing
    total_length = float(n * ds)

    # Central differences via rolled arrays give a wrap-safe gradient on the
    # closed curve (np.gradient does not wrap at the array boundary).
    dx = (np.roll(x, -1) - np.roll(x, 1)) / (2 * ds)
    dy = (np.roll(y, -1) - np.roll(y, 1)) / (2 * ds)
    heading = np.arctan2(dy, dx)

    # Angle-wrapped central difference: safe across the seam of the closed
    # curve, where a naive np.unwrap over the whole array would pick up a
    # spurious ~2*pi offset (total heading change around a loop is 2*pi).
    raw_diff = np.roll(heading, -1) - np.roll(heading, 1)
    dtheta = (np.mod(raw_diff + math.pi, 2 * math.pi) - math.pi) / (2 * ds)
    curvature = dtheta

    n_crossings = max(spec.num_crossings, 0)
    crossing_indices = [int(i * n / max(n_crossings, 1)) for i in range(n_crossings)]

    return Track(
        name=spec.name,
        centerline=np.stack([x, y], axis=1),
        heading=heading,
        curvature=curvature,
        arc_length=s,
        total_length=total_length,
        width_m=spec.width_m,
        crossing_indices=crossing_indices,
    )
