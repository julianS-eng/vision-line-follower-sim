"""PID controller acting on a blended cross-track + heading error."""

from __future__ import annotations

from dataclasses import dataclass

from vision_line_follower.control.base import Controller


@dataclass(frozen=True, slots=True)
class PIDConfig:
    """Gains for the PID steering controller.

    The controller drives ``e = lateral_error_m + heading_weight *
    heading_error_rad`` to zero -- a common practical simplification that
    avoids running two coupled PID loops.

    Attributes:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain (applied to the finite-difference of ``e``).
        heading_weight: Metres of equivalent cross-track error per radian of
            heading error, used to blend the two error signals.
        integral_limit: Anti-windup clamp on the accumulated integral term.
        omega_limit: Saturation applied to the output angular velocity.
    """

    kp: float = 7.0
    ki: float = 0.6
    kd: float = 0.35
    heading_weight: float = 0.12
    integral_limit: float = 0.5
    omega_limit: float = 6.0


class PIDController(Controller):
    """Classic PID controller on the blended line-following error."""

    name = "pid"

    def __init__(self, config: PIDConfig | None = None) -> None:
        self.config = config or PIDConfig()
        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def compute(
        self,
        lateral_error_m: float,
        heading_error_rad: float,
        curvature: float,
        speed_mps: float,
        dt: float,
    ) -> float:
        cfg = self.config
        error = lateral_error_m + cfg.heading_weight * heading_error_rad

        self._integral = max(
            -cfg.integral_limit, min(cfg.integral_limit, self._integral + error * dt)
        )
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error

        omega = cfg.kp * error + cfg.ki * self._integral + cfg.kd * derivative
        return max(-cfg.omega_limit, min(cfg.omega_limit, omega))
