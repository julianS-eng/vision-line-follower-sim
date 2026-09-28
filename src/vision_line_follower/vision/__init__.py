"""Classical computer-vision pipeline: camera frame -> lateral/heading error."""

from vision_line_follower.vision.birdseye import (
    BirdsEyeCalibration,
    BirdsEyeConfig,
    compute_birdseye_calibration,
    warp_to_birdseye,
)
from vision_line_follower.vision.fit import LineFit, fit_line_polynomial
from vision_line_follower.vision.pipeline import VisionPipeline, VisionResult
from vision_line_follower.vision.sliding_window import SlidingWindowConfig, sliding_window_search
from vision_line_follower.vision.threshold import ThresholdConfig, binarize_line_mask

__all__ = [
    "BirdsEyeCalibration",
    "BirdsEyeConfig",
    "LineFit",
    "SlidingWindowConfig",
    "ThresholdConfig",
    "VisionPipeline",
    "VisionResult",
    "binarize_line_mask",
    "compute_birdseye_calibration",
    "fit_line_polynomial",
    "sliding_window_search",
    "warp_to_birdseye",
]
