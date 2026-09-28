"""Stanley steering controller, adapted from bicycle to differential drive.

The original Stanley controller (Thrun et al., 2006, Stanford Racing Team)
computes a front-wheel steering angle

``delta = heading_error + atan2(k * cross_track_error, v + k_soft)``

for a bicycle-model vehicle, then applies it directly since the steered
wheel angle *is* the control input. A differential-drive robot has no
steered wheel: its control input is an angular velocity. We map the Stanley
steering angle to an angular velocity through the same kinematic relation a
bicycle model would use (``omega = v * tan(delta) / L``), with an
"effective wheelbase" ``L_eff`` exposed as a tuning parameter rather than
tied to the physical wheel base -- this is an explicit engineering
adaptation, not the textbook Stanley controller, and is documented as such
in ``docs/LEARNING.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision_line_follower.control.base import Controller


@dataclass(frozen=True, slots=True)
class StanleyConfig:
    """Parameters for the (differential-drive-adapted) Stanley controller.

    Attributes:
        k_cross_track: Gain converting cross-track error into a steering
            contribution; larger values correct lateral offset more
            aggressively at the cost of overshoot.
        k_soft: Softening constant preventing division blow-up at low speed.
        effective_wheelbase_m: Wheelbase used in the ``omega = v * tan(delta)
            / L_eff`` mapping from steering angle to angular velocity.
        heading_gain: Extra gain applied to the heading-error term
            (1.0 recovers the textbook formulation).
        omega_limit: Saturation applied to the output angular velocity.
    """

    k_cross_track: float = 2.5
    k_soft: float = 0.08
    effective_wheelbase_m: float = 0.12
    heading_gain: float = 1.0
    omega_limit: float = 6.0


class StanleyController(Controller):
    """Stanley controller adapted for a differential-drive robot."""

    name = "stanley"

    def __init__(self, config: StanleyConfig | None = None) -> None:
        self.config = config or StanleyConfig()

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
        cross_track_term = math.atan2(cfg.k_cross_track * lateral_error_m, speed_mps + cfg.k_soft)
        delta = cfg.heading_gain * heading_error_rad + cross_track_term
        delta = max(-math.pi / 2.5, min(math.pi / 2.5, delta))
        omega = speed_mps * math.tan(delta) / cfg.effective_wheelbase_m
        return max(-cfg.omega_limit, min(cfg.omega_limit, omega))
