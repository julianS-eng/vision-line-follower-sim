"""Shared controller interface.

Every controller consumes the vision pipeline's estimate of the line's
lateral offset, heading offset and curvature (plus the current commanded
speed and timestep) and produces a single steering output: the desired
angular velocity ``omega`` (rad/s). Linear speed is decided independently by
a curvature-aware speed scheduler (see
:func:`vision_line_follower.sim.world.speed_schedule`) so that all three
controllers are compared under an identical speed profile.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Controller(ABC):
    """Base class for lateral (steering) controllers."""

    name: str = "controller"

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (integrators, previous error, ...)."""

    @abstractmethod
    def compute(
        self,
        lateral_error_m: float,
        heading_error_rad: float,
        curvature: float,
        speed_mps: float,
        dt: float,
    ) -> float:
        """Return the desired angular velocity ``omega`` in rad/s."""
