"""End-to-end perception pipeline: camera frame -> lateral/heading error.

Stages, mirroring a classical (non-learned) lane-finding pipeline:

1. **Bird's-eye warp** (:mod:`.birdseye`) -- perspective-correct the forward
   camera view into a top-down patch of the ground ahead of the robot.
2. **Adaptive threshold** (:mod:`.threshold`) -- binarize line pixels,
   robust to lighting gradients.
3. **Sliding window search** (:mod:`.sliding_window`) -- collect candidate
   line-pixel coordinates from bottom (near) to top (far).
4. **Polynomial fit** (:mod:`.fit`) -- fit a local quadratic model of the
   line and read off lateral error, heading error and curvature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vision_line_follower.sim.camera import CameraModel
from vision_line_follower.vision.birdseye import (
    BirdsEyeCalibration,
    BirdsEyeConfig,
    compute_birdseye_calibration,
    warp_to_birdseye,
)
from vision_line_follower.vision.fit import LineFit, fit_line_polynomial, heading_error_from_slope
from vision_line_follower.vision.sliding_window import (
    SlidingWindowConfig,
    sliding_window_search,
)
from vision_line_follower.vision.threshold import ThresholdConfig, binarize_line_mask


@dataclass(slots=True)
class VisionResult:
    """Output of :meth:`VisionPipeline.process` for a single frame.

    Attributes:
        found: Whether enough line pixels were detected to trust the fit.
        lateral_error_m: Signed lateral offset of the line from the robot's
            forward axis, evaluated at the near edge of the ROI (positive =
            line is to the left, so the robot should turn left/CCW).
        heading_error_rad: Signed angle between the robot's heading and the
            line's local tangent (positive = line points further left).
        curvature: Local curvature of the fitted line, ``1/m``.
        fit: The underlying :class:`LineFit`, if ``found`` is ``True``.
        n_pixels: Number of line pixels used in the fit.
        debug: Optional dict of intermediate images (populated only when
            ``VisionPipeline(..., debug=True)``), keyed by stage name.
    """

    found: bool
    lateral_error_m: float
    heading_error_rad: float
    curvature: float
    fit: LineFit | None = None
    n_pixels: int = 0
    debug: dict[str, np.ndarray] = field(default_factory=dict)


class VisionPipeline:
    """Wires the bird's-eye/threshold/sliding-window/fit stages together."""

    def __init__(
        self,
        camera: CameraModel,
        birdseye_config: BirdsEyeConfig | None = None,
        threshold_config: ThresholdConfig | None = None,
        window_config: SlidingWindowConfig | None = None,
        debug: bool = False,
    ) -> None:
        self.calibration: BirdsEyeCalibration = compute_birdseye_calibration(
            camera, birdseye_config
        )
        self.threshold_config = threshold_config or ThresholdConfig()
        self.window_config = window_config or SlidingWindowConfig()
        self.debug = debug
        self._prior_x_px: float | None = None

    def reset(self) -> None:
        """Clear temporal tracking state (call between independent runs)."""
        self._prior_x_px = None

    def process(self, camera_image: np.ndarray) -> VisionResult:
        bev = warp_to_birdseye(camera_image, self.calibration)
        mask = binarize_line_mask(bev, self.threshold_config)
        window_result = sliding_window_search(mask, self.window_config, self._prior_x_px)

        debug: dict[str, np.ndarray] = {}
        if self.debug:
            debug["birdseye"] = bev
            debug["mask"] = mask

        if not window_result.found:
            return VisionResult(
                found=False, lateral_error_m=0.0, heading_error_rad=0.0, curvature=0.0, debug=debug
            )

        forward_m, lateral_m = self.calibration.pixel_to_local(
            window_result.xs_px, window_result.ys_px
        )
        try:
            fit = fit_line_polynomial(forward_m, lateral_m, order=2)
        except ValueError:
            return VisionResult(
                found=False, lateral_error_m=0.0, heading_error_rad=0.0, curvature=0.0, debug=debug
            )

        eval_forward = float(self.calibration.config.forward_range_m[0])
        lateral_error = fit.lateral_at(eval_forward)
        heading_error = heading_error_from_slope(fit.slope_at(eval_forward))
        curvature = fit.curvature_at(eval_forward)

        if window_result.window_centers:
            self._prior_x_px = float(window_result.window_centers[0][0])

        return VisionResult(
            found=True,
            lateral_error_m=lateral_error,
            heading_error_rad=heading_error,
            curvature=curvature,
            fit=fit,
            n_pixels=int(window_result.xs_px.size),
            debug=debug,
        )
