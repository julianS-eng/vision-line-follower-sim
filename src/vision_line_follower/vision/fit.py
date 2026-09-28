"""Polynomial fitting of the detected line and derived path quantities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LineFit:
    """A quadratic fit ``lateral = c2 * forward^2 + c1 * forward + c0`` (metres)."""

    coeffs: np.ndarray  # [c2, c1, c0]

    def lateral_at(self, forward_m: float) -> float:
        c2, c1, c0 = self.coeffs
        return float(c2 * forward_m**2 + c1 * forward_m + c0)

    def slope_at(self, forward_m: float) -> float:
        c2, c1, _c0 = self.coeffs
        return float(2.0 * c2 * forward_m + c1)

    def curvature_at(self, forward_m: float) -> float:
        c2, c1, _c0 = self.coeffs
        slope = 2.0 * c2 * forward_m + c1
        return float((2.0 * c2) / (1.0 + slope**2) ** 1.5)


def fit_line_polynomial(forward_m: np.ndarray, lateral_m: np.ndarray, order: int = 2) -> LineFit:
    """Least-squares fit of ``lateral`` as a polynomial function of ``forward``.

    Args:
        forward_m: Sample forward distances (independent variable), metres.
        lateral_m: Sample lateral offsets (dependent variable), metres.
        order: Polynomial order (2 = quadratic, matches a constant-curvature
            local approximation of the path).

    Raises:
        ValueError: If there are fewer points than ``order + 1``.
    """
    if forward_m.size < order + 1:
        raise ValueError(f"Need at least {order + 1} points to fit order-{order} polynomial")
    coeffs = np.polyfit(forward_m, lateral_m, order)
    if order == 2:
        full = coeffs
    else:
        full = np.zeros(3)
        full[-len(coeffs) :] = coeffs[-3:] if len(coeffs) >= 3 else coeffs
    return LineFit(coeffs=full)


def heading_error_from_slope(slope: float) -> float:
    """Convert a ``d(lateral)/d(forward)`` slope into a heading-error angle."""
    return math.atan(slope)
