"""Inverse perspective mapping (IPM): warp the forward camera view into a
top-down "bird's-eye" patch of the ground immediately ahead of the robot.

The homography is derived analytically from the camera's own calibration
(:meth:`CameraModel.ground_to_camera_homography`) rather than fitted from
point correspondences, mirroring how a real robot would use its known
extrinsic/intrinsic calibration to compute a fixed IPM matrix once at
start-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_line_follower.sim.camera import CameraModel


@dataclass(frozen=True, slots=True)
class BirdsEyeConfig:
    """Defines the ground-plane ROI (robot-local frame) mapped to the BEV image.

    Attributes:
        forward_range_m: ``(near, far)`` distance ahead of the camera mount
            covered by the warp.
        lateral_range_m: ``(right, left)`` extent covered by the warp
            (signed, positive = left of the robot's forward axis).
        output_size: ``(width, height)`` of the bird's-eye image in pixels.
    """

    forward_range_m: tuple[float, float] = (0.10, 0.45)
    lateral_range_m: tuple[float, float] = (-0.22, 0.22)
    output_size: tuple[int, int] = (200, 320)


@dataclass(slots=True)
class BirdsEyeCalibration:
    """Precomputed IPM homography and the pixel<->metre mapping it implies."""

    matrix: np.ndarray  # 3x3, camera pixels -> BEV pixels
    config: BirdsEyeConfig
    meters_per_pixel_x: float
    meters_per_pixel_y: float

    def pixel_to_local(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert BEV pixel coordinates to robot-local ``(forward, lateral)`` metres."""
        far = self.config.forward_range_m[1]
        left = self.config.lateral_range_m[1]
        forward = far - v * self.meters_per_pixel_y
        lateral = left - u * self.meters_per_pixel_x
        return forward, lateral


def compute_birdseye_calibration(
    camera: CameraModel, config: BirdsEyeConfig | None = None
) -> BirdsEyeCalibration:
    """Derive the camera-image -> bird's-eye homography from camera calibration."""
    cfg = config or BirdsEyeConfig()
    near, far = cfg.forward_range_m
    right, left = cfg.lateral_range_m
    out_w, out_h = cfg.output_size

    ground_corners = np.array(
        [
            [near, left],  # near-left
            [near, right],  # near-right
            [far, left],  # far-left
            [far, right],  # far-right
        ],
        dtype=np.float64,
    )
    h_ground2cam = camera.ground_to_camera_homography()
    homog = np.hstack([ground_corners, np.ones((4, 1))])
    cam_homog = (h_ground2cam @ homog.T).T
    cam_px = cam_homog[:, :2] / cam_homog[:, 2:3]

    dst_px = np.array(
        [
            [0, out_h - 1],
            [out_w - 1, out_h - 1],
            [0, 0],
            [out_w - 1, 0],
        ],
        dtype=np.float64,
    )

    matrix = cv2.getPerspectiveTransform(cam_px.astype(np.float32), dst_px.astype(np.float32))
    meters_per_pixel_x = (left - right) / max(out_w - 1, 1)
    meters_per_pixel_y = (far - near) / max(out_h - 1, 1)
    return BirdsEyeCalibration(
        matrix=matrix,
        config=cfg,
        meters_per_pixel_x=meters_per_pixel_x,
        meters_per_pixel_y=meters_per_pixel_y,
    )


def warp_to_birdseye(camera_image: np.ndarray, calibration: BirdsEyeCalibration) -> np.ndarray:
    """Apply the precomputed IPM homography to a camera frame."""
    out_w, out_h = calibration.config.output_size
    return cv2.warpPerspective(
        camera_image,
        calibration.matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
