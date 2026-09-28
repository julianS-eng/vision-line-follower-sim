"""Simulated forward-looking, downward-tilted camera mounted on the robot.

The camera is modelled as an ideal pinhole. Because every point of interest
lies on the flat ground plane (``z = 0`` in the world/robot frames), the
mapping from ground-plane metres to camera pixels is a single 3x3
homography. We exploit this to render the camera view directly with
``cv2.warpPerspective`` on the pre-rendered world raster: the world-pixel ->
world-metre, world-metre -> robot-local-metre and robot-local-metre ->
camera-pixel transforms are each represented as 3x3 homogeneous matrices and
composed into one homography per frame, so the cost of rendering a frame does
not depend on the size of the world image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision_line_follower.geometry import Pose
from vision_line_follower.track.rendering import WorldImage


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Intrinsic and extrinsic parameters of the simulated camera.

    Attributes:
        resolution: ``(width, height)`` in pixels.
        horizontal_fov_deg: Horizontal field of view.
        mount_height_m: Height of the camera optical center above the ground.
        mount_forward_offset_m: Forward offset of the camera from the
            robot's rear-axle origin.
        pitch_deg: Downward tilt of the optical axis from the horizontal
            plane (0 = looking at the horizon, 90 = looking straight down).
        pixel_noise_std: Standard deviation (0-255 scale) of additive
            Gaussian sensor noise.
        vignette_strength: 0-1 strength of a radial darkening at the image
            corners, mimicking cheap wide-angle lenses.
        motion_blur_kernel: Size (odd, pixels) of a directional blur kernel
            applied along the image's vertical axis to approximate rolling
            shutter/motion blur at speed. ``1`` disables it.
    """

    resolution: tuple[int, int] = (320, 240)
    horizontal_fov_deg: float = 62.0
    mount_height_m: float = 0.09
    mount_forward_offset_m: float = 0.055
    pitch_deg: float = 42.0
    pixel_noise_std: float = 4.0
    vignette_strength: float = 0.25
    motion_blur_kernel: int = 1


class CameraModel:
    """Pinhole camera rigidly mounted on the robot, looking forward and down."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._K = self._build_intrinsics()
        self._R = self._build_rotation()

    def _build_intrinsics(self) -> np.ndarray:
        w, h = self.config.resolution
        fov = math.radians(self.config.horizontal_fov_deg)
        fx = (w / 2.0) / math.tan(fov / 2.0)
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _build_rotation(self) -> np.ndarray:
        pitch = math.radians(self.config.pitch_deg)
        # Robot frame axes: e_x forward, e_y left, e_z up.
        z_cam = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])  # optical axis
        x_cam = np.array([0.0, -1.0, 0.0])  # image-right = robot's right
        y_cam = np.cross(z_cam, x_cam)  # image-down
        return np.stack([x_cam, y_cam, z_cam], axis=0)

    def ground_to_camera_homography(self) -> np.ndarray:
        """Homography mapping robot-local ground metres ``(x_f, y_l, 1)`` to
        homogeneous camera pixels, for a target lying exactly on ``z = 0`` in
        the robot frame (i.e. evaluated with the robot at its own origin).
        """
        cfg = self.config
        t = np.array([cfg.mount_forward_offset_m, 0.0, cfg.mount_height_m])
        r0, r1, r2 = self._R[:, 0], self._R[:, 1], self._R[:, 2]
        m_col2 = -r0 * t[0] - r2 * t[2]
        m = np.stack([r0, r1, m_col2], axis=1)
        return self._K @ m

    def world_to_camera_homography(self, world: WorldImage, robot_pose: Pose) -> np.ndarray:
        """Homography mapping world-image pixels ``(u, v, 1)`` to camera pixels."""
        h_ground2cam = self.ground_to_camera_homography()
        t_world2local = robot_pose.affine_world_to_local_matrix()
        s_px2world = world.pixel_to_world_affine()
        return h_ground2cam @ t_world2local @ s_px2world

    def render(
        self,
        world: WorldImage,
        robot_pose: Pose,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Render the camera view for a given robot pose.

        Args:
            world: Pre-rendered world raster and its pixel<->metre mapping.
            robot_pose: Current robot pose in the world frame.
            rng: Optional random generator for sensor noise; omit for a
                noise-free render (useful for calibration/testing).

        Returns:
            ``(H, W, 3)`` uint8 BGR image.
        """
        cfg = self.config
        h_total = self.world_to_camera_homography(world, robot_pose)
        out = cv2.warpPerspective(
            world.image,
            h_total,
            cfg.resolution,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=world.image[0, 0].tolist(),
        )
        if cfg.vignette_strength > 0:
            out = self._apply_vignette(out)
        if cfg.motion_blur_kernel > 1:
            k = cfg.motion_blur_kernel
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[:, k // 2] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)
        if rng is not None and cfg.pixel_noise_std > 0:
            noise = rng.normal(scale=cfg.pixel_noise_std, size=out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        r = np.hypot(xx - cx, yy - cy) / math.hypot(cx, cy)
        factor = np.clip(1.0 - self.config.vignette_strength * (r**2), 0.2, 1.0)[..., None]
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
