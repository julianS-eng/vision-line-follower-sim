"""Geometric Pure Pursuit steering controller.

Classical Pure Pursuit fits a circular arc from the vehicle's rear axle to a
"lookahead point" a fixed distance ahead on the reference path. Here the
vision pipeline only reports the line's lateral error, heading error and
curvature at a single near-field evaluation point (see
:class:`~vision_line_follower.vision.pipeline.VisionPipeline`), not a full
path. We reconstruct a local quadratic model of the path in the robot frame
via a Taylor expansion around that evaluation point --

``lateral(x) ~= e_y + tan(e_psi) * (x - x0) + 0.5 * kappa_eff * (x - x0)^2``

with ``kappa_eff`` derived from the reported curvature so that the
second-order term matches -- and evaluate it at the lookahead distance. This
keeps every controller in this project consuming the exact same three
scalars from the pipeline, at the cost of accuracy for lookahead distances
that extrapolate far beyond the vision ROI (documented as a known
limitation in ``docs/LEARNING.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision_line_follower.control.base import Controller


@dataclass(frozen=True, slots=True)
class PurePursuitConfig:
    """Parameters for the Pure Pursuit controller.

    Attributes:
        lookahead_base_m: Minimum/base lookahead distance.
        lookahead_speed_gain: Additional lookahead distance per m/s of speed
            (speed-adaptive lookahead improves high-speed stability).
        lookahead_curvature_gain: Lookahead distance *reduction* per unit of
            estimated curvature (1/m). Pure Pursuit's maximum deliverable
            path curvature is ``2 / lookahead``, so shortening the
            lookahead in tight corners is what lets the controller actually
            reach the curvature the path demands instead of being
            geometrically capped by a lookahead sized for straights.
        lookahead_min_m: Hard lower bound on the lookahead distance.
        lookahead_max_m: Hard upper bound on the lookahead distance.
        reference_forward_m: Forward distance (from the robot origin) at
            which the pipeline evaluates ``lateral_error``/``heading_error``
            -- must match ``BirdsEyeConfig.forward_range_m[0]`` plus the
            camera mount offset used by the pipeline that feeds this
            controller.
        max_heading_for_extrapolation_rad: Clamp applied to the heading
            error before it is used inside the local quadratic
            (Taylor-expansion) model of the path -- see the module
            docstring. Bounds the tangent-blow-up that would otherwise make
            the reconstructed lookahead target diverge for near-90-degree
            heading readings.
        omega_limit: Saturation applied to the output angular velocity.
    """

    lookahead_base_m: float = 0.12
    lookahead_speed_gain: float = 0.10
    lookahead_curvature_gain: float = 0.0
    lookahead_min_m: float = 0.07
    lookahead_max_m: float = 0.4
    reference_forward_m: float = 0.155
    max_heading_for_extrapolation_rad: float = 0.9
    omega_limit: float = 6.0


class PurePursuitController(Controller):
    """Pure Pursuit adapted to consume scalar vision-pipeline outputs."""

    name = "pure_pursuit"

    def __init__(self, config: PurePursuitConfig | None = None) -> None:
        self.config = config or PurePursuitConfig()

    def reset(self) -> None:
        return None

    def compute(
        self,
        lateral_error_m: float,
        heading_error_rad: float,
        curvature: float,
        speed_mps: float,
        dt: float,
    ) -> float:
        cfg = self.config
        raw_lookahead = (
            cfg.lookahead_base_m
            + cfg.lookahead_speed_gain * speed_mps
            - cfg.lookahead_curvature_gain * abs(curvature)
        )
        lookahead = max(cfg.lookahead_min_m, min(cfg.lookahead_max_m, raw_lookahead))

        # The Taylor expansion below is only trustworthy for modest slopes;
        # tan() blows up near +/-90 deg and would otherwise let a single
        # large heading-error reading (e.g. while entering a sharp corner)
        # extrapolate to an absurd target point a few centimetres later.
        clamped_heading = max(
            -cfg.max_heading_for_extrapolation_rad,
            min(cfg.max_heading_for_extrapolation_rad, heading_error_rad),
        )
        slope0 = math.tan(clamped_heading)
        second_deriv = curvature * (1.0 + slope0**2) ** 1.5
        dx = lookahead - cfg.reference_forward_m
        y_target = lateral_error_m + slope0 * dx + 0.5 * second_deriv * dx**2

        alpha = math.atan2(y_target, lookahead)
        path_curvature = 2.0 * math.sin(alpha) / lookahead
        omega = speed_mps * path_curvature
        return max(-cfg.omega_limit, min(cfg.omega_limit, omega))
