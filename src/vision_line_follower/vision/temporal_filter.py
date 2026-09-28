"""Exponential-moving-average smoothing of frame-to-frame vision estimates.

A single ambiguous frame -- most notably near a painted line crossing, where
the sliding-window search can briefly latch onto the wrong branch -- can
otherwise inject a large, one-frame spike into the lateral/heading error
fed to the controller. Real embedded vision stacks routinely smooth their
estimates for exactly this reason; this is a light EMA filter applied
identically for every controller in the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemporalFilterConfig:
    """EMA smoothing factor.

    Attributes:
        alpha: Weight given to the newest measurement, in ``(0, 1]``.
            ``1.0`` disables smoothing (the filter passes measurements
            through unchanged); smaller values smooth more aggressively at
            the cost of added lag.
    """

    alpha: float = 0.5


class TemporalFilter:
    """Stateful EMA filter over ``(lateral, heading, curvature)`` triples."""

    def __init__(self, config: TemporalFilterConfig | None = None) -> None:
        self.config = config or TemporalFilterConfig()
        self._state: tuple[float, float, float] | None = None

    def reset(self) -> None:
        self._state = None

    def update(
        self, lateral: float, heading: float, curvature: float
    ) -> tuple[float, float, float]:
        if self._state is None:
            self._state = (lateral, heading, curvature)
            return self._state
        a = self.config.alpha
        lp, hp, cp = self._state
        self._state = (
            a * lateral + (1 - a) * lp,
            a * heading + (1 - a) * hp,
            a * curvature + (1 - a) * cp,
        )
        return self._state

    @property
    def last(self) -> tuple[float, float, float] | None:
        return self._state
