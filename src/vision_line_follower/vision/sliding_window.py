"""Sliding-window line-pixel search over a binary bird's-eye mask.

Starts from a histogram peak in the bottom rows (nearest to the robot) and
walks a fixed-height window upward (further ahead), recentring on the mean
column of active pixels within each window -- the same technique used in
classical advanced lane-finding pipelines, adapted to a single line instead
of two lane boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SlidingWindowConfig:
    """Parameters controlling the sliding-window search.

    Attributes:
        n_windows: Number of vertical windows stacked from bottom to top.
        margin_px: Half-width of each window, in pixels.
        min_pixels_to_recenter: Minimum active pixels in a window required
            to recentre the next window on their mean column.
        min_total_pixels: Minimum total active pixels across all windows for
            the search to be considered successful.
        prior_search_radius_px: When a ``prior_x_px`` hint is supplied to
            :func:`sliding_window_search` (temporal tracking from the
            previous frame), the base histogram is restricted to this
            radius around the hint instead of a full-width search. This
            keeps the tracker anchored to the previously-followed branch of
            the line when two branches briefly overlap in the ROI (e.g. at
            a painted line crossing), rather than jumping to whichever
            branch happens to have more pixels.
    """

    n_windows: int = 12
    margin_px: int = 28
    min_pixels_to_recenter: int = 25
    min_total_pixels: int = 60
    prior_search_radius_px: int = 45


@dataclass(slots=True)
class SlidingWindowResult:
    found: bool
    xs_px: np.ndarray
    ys_px: np.ndarray
    window_centers: list[tuple[int, int]]


def sliding_window_search(
    mask: np.ndarray,
    config: SlidingWindowConfig | None = None,
    prior_x_px: float | None = None,
) -> SlidingWindowResult:
    """Search a binary mask (255 = line) for line-pixel coordinates.

    Args:
        mask: Binary bird's-eye mask (255 = line pixel).
        config: Search parameters.
        prior_x_px: Optional column hint from the previous frame's base
            window, used to anchor the histogram search locally (see
            :attr:`SlidingWindowConfig.prior_search_radius_px`).
    """
    cfg = config or SlidingWindowConfig()
    h, w = mask.shape[:2]
    window_height = max(h // cfg.n_windows, 1)

    bottom_half = mask[h // 2 :, :]
    histogram = bottom_half.sum(axis=0).astype(np.float64)

    if prior_x_px is not None:
        radius = cfg.prior_search_radius_px
        lo = max(int(prior_x_px - radius), 0)
        hi = min(int(prior_x_px + radius), w)
        local_hist = histogram[lo:hi]
        if local_hist.size > 0 and local_hist.max() > 0:
            histogram = np.zeros_like(histogram)
            histogram[lo:hi] = local_hist

    if histogram.max() <= 0:
        return SlidingWindowResult(False, np.array([]), np.array([]), [])
    current_x = int(np.argmax(histogram))

    nonzero_y, nonzero_x = mask.nonzero()

    collected_x: list[np.ndarray] = []
    collected_y: list[np.ndarray] = []
    centers: list[tuple[int, int]] = []

    for window in range(cfg.n_windows):
        y_hi = h - window * window_height
        y_lo = h - (window + 1) * window_height
        x_lo = max(current_x - cfg.margin_px, 0)
        x_hi = min(current_x + cfg.margin_px, w)

        in_window = (
            (nonzero_y >= y_lo) & (nonzero_y < y_hi) & (nonzero_x >= x_lo) & (nonzero_x < x_hi)
        )
        idx = np.flatnonzero(in_window)
        centers.append((current_x, (y_lo + y_hi) // 2))
        if idx.size > 0:
            collected_x.append(nonzero_x[idx])
            collected_y.append(nonzero_y[idx])
        if idx.size >= cfg.min_pixels_to_recenter:
            current_x = int(np.mean(nonzero_x[idx]))

    if not collected_x:
        return SlidingWindowResult(False, np.array([]), np.array([]), centers)

    xs = np.concatenate(collected_x)
    ys = np.concatenate(collected_y)
    found = xs.size >= cfg.min_total_pixels
    return SlidingWindowResult(found, xs.astype(np.float64), ys.astype(np.float64), centers)
