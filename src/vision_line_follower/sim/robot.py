"""Differential-drive robot: kinematics, first-order motor dynamics,
actuator saturation and process noise.

The robot is integrated with a classic unicycle/differential-drive model.
Commanded wheel speeds do not apply instantaneously -- each wheel's actual
linear speed relaxes toward its commanded value with time constant
``motor_time_constant_s`` (a first-order lag approximating motor + gearbox +
wheel-slip response), is clamped to ``max_wheel_speed_mps`` and perturbed by
Gaussian process noise before being integrated into the pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vision_line_follower.geometry import Pose, wrap_angle


@dataclass(frozen=True, slots=True)
class RobotConfig:
    """Physical parameters of the simulated differential-drive robot.

    Attributes:
        wheel_base_m: Distance between the left and right wheel contact
            points.
        max_wheel_speed_mps: Saturation limit for each wheel's linear speed.
        motor_time_constant_s: First-order lag time constant relating
            commanded to actual wheel speed.
        wheel_speed_noise_std: Standard deviation of additive Gaussian noise
            (m/s) applied to each wheel's actual speed at every step,
            modelling wheel slip and encoder/PWM jitter.
    """

    wheel_base_m: float = 0.14
    max_wheel_speed_mps: float = 0.7
    motor_time_constant_s: float = 0.07
    wheel_speed_noise_std: float = 0.0


@dataclass(slots=True)
class RobotState:
    """Full internal state of the robot (pose + actual wheel speeds)."""

    x: float
    y: float
    theta: float
    v_left: float = 0.0
    v_right: float = 0.0

    @property
    def pose(self) -> Pose:
        return Pose(self.x, self.y, self.theta)


class DifferentialDriveRobot:
    """Simulates a differential-drive robot with simple motor dynamics."""

    def __init__(
        self,
        config: RobotConfig,
        initial_state: RobotState,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.state = initial_state
        self._rng = rng if rng is not None else np.random.default_rng()

    @property
    def pose(self) -> Pose:
        return self.state.pose

    def linear_angular_velocity(self) -> tuple[float, float]:
        """Return the current ``(v, omega)`` implied by actual wheel speeds."""
        cfg = self.config
        v = (self.state.v_left + self.state.v_right) / 2.0
        omega = (self.state.v_right - self.state.v_left) / cfg.wheel_base_m
        return v, omega

    def step(self, cmd_v_left: float, cmd_v_right: float, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds given commanded wheel speeds."""
        cfg = self.config
        max_v = cfg.max_wheel_speed_mps
        cmd_v_left = float(np.clip(cmd_v_left, -max_v, max_v))
        cmd_v_right = float(np.clip(cmd_v_right, -max_v, max_v))

        alpha = 1.0 - math.exp(-dt / max(cfg.motor_time_constant_s, 1e-6))
        v_left = self.state.v_left + alpha * (cmd_v_left - self.state.v_left)
        v_right = self.state.v_right + alpha * (cmd_v_right - self.state.v_right)

        if cfg.wheel_speed_noise_std > 0:
            v_left += float(self._rng.normal(scale=cfg.wheel_speed_noise_std))
            v_right += float(self._rng.normal(scale=cfg.wheel_speed_noise_std))
        v_left = float(np.clip(v_left, -max_v, max_v))
        v_right = float(np.clip(v_right, -max_v, max_v))

        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / cfg.wheel_base_m

        # Exact arc integration (better than Euler for large omega*dt).
        theta0 = self.state.theta
        if abs(omega) < 1e-9:
            x1 = self.state.x + v * dt * math.cos(theta0)
            y1 = self.state.y + v * dt * math.sin(theta0)
        else:
            theta1 = theta0 + omega * dt
            x1 = self.state.x + (v / omega) * (math.sin(theta1) - math.sin(theta0))
            y1 = self.state.y - (v / omega) * (math.cos(theta1) - math.cos(theta0))
        theta1 = wrap_angle(theta0 + omega * dt)

        self.state = RobotState(x=x1, y=y1, theta=theta1, v_left=v_left, v_right=v_right)

    @staticmethod
    def unicycle_to_wheel_speeds(
        v: float, omega: float, wheel_base_m: float
    ) -> tuple[float, float]:
        """Convert a desired ``(v, omega)`` command into left/right wheel speeds."""
        v_left = v - omega * wheel_base_m / 2.0
        v_right = v + omega * wheel_base_m / 2.0
        return v_left, v_right
