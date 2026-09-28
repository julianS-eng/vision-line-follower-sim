"""Adaptive thresholding of the bird's-eye image to isolate line pixels.

Adaptive (locally-normalized) thresholding is used instead of a single
global threshold so the pipeline stays robust to the lighting gradients and
vignetting injected by :mod:`vision_line_follower.track.rendering` and
:mod:`vision_line_follower.sim.camera`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Parameters for the adaptive-threshold + morphology stage.

    Attributes:
        block_size: Odd pixel size of the local neighbourhood used by
            ``cv2.adaptiveThreshold``.
        c: Constant subtracted from the local mean (higher = stricter,
            fewer pixels classified as "line").
        morph_kernel: Odd kernel size for the open/close cleanup pass.
        dark_line_on_light: Whether the line is darker than the floor (the
            common case for a painted line on light flooring).
    """

    block_size: int = 31
    c: float = 7.0
    morph_kernel: int = 3
    dark_line_on_light: bool = True


def binarize_line_mask(bev_image: np.ndarray, config: ThresholdConfig | None = None) -> np.ndarray:
    """Return a uint8 binary mask (255 = line pixel) from a BEV BGR image."""
    cfg = config or ThresholdConfig()
    gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    block_size = cfg.block_size if cfg.block_size % 2 == 1 else cfg.block_size + 1
    thresh_type = cv2.THRESH_BINARY_INV if cfg.dark_line_on_light else cv2.THRESH_BINARY
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        block_size,
        cfg.c,
    )

    if cfg.morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
