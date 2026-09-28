"""Compose an animated GIF of a closed-loop run: top-down track view with the
robot's trail, plus a picture-in-picture inset of what its camera sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vision_line_follower.geometry import Pose
from vision_line_follower.track.generator import Track
from vision_line_follower.track.rendering import WorldImage


@dataclass(frozen=True, slots=True)
class GifExportConfig:
    """Parameters controlling the exported GIF's size/quality trade-off.

    Attributes:
        fps: Playback frame rate.
        max_frames: Hard cap on the number of frames written (frames beyond
            this are subsampled evenly), the main lever for keeping file
            size under control.
        map_size: ``(width, height)`` of the top-down map panel.
        inset_scale: Scale factor applied to the camera frame before it is
            pasted into the map panel as a picture-in-picture inset.
        trail_length: Number of recent robot positions drawn as a trail.
        palette_colors: Number of colors in the shared adaptive palette used
            to encode the GIF; this (plus ``max_frames`` and ``map_size``) is
            the main lever for keeping the output file under the size budget.
    """

    fps: int = 12
    max_frames: int = 110
    map_size: tuple[int, int] = (420, 340)
    inset_scale: float = 0.85
    trail_length: int = 200
    palette_colors: int = 64


def _world_panel(world: WorldImage, map_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize the world image to the map panel and return it with the
    corresponding world->panel-pixel affine matrix."""
    h0, w0 = world.image.shape[:2]
    w1, h1 = map_size
    scale_x, scale_y = w1 / w0, h1 / h0
    resized = cv2.resize(world.image, (w1, h1), interpolation=cv2.INTER_AREA)
    world_to_world_px = world.world_to_pixel_affine()
    scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
    return resized, scale_matrix @ world_to_world_px


def export_run_gif(
    track: Track,
    world: WorldImage,
    poses: list[Pose],
    camera_frames: list[np.ndarray],
    cross_track_errors: list[float],
    controller_name: str,
    output_path: Path,
    config: GifExportConfig | None = None,
) -> None:
    """Write an animated GIF of a recorded run to ``output_path``."""
    cfg = config or GifExportConfig()
    if not poses:
        raise ValueError("Cannot export a GIF from an empty run")

    n_frames = len(poses)
    if n_frames > cfg.max_frames:
        idx = np.linspace(0, n_frames - 1, cfg.max_frames).astype(int)
    else:
        idx = np.arange(n_frames)

    base_panel, world_to_panel = _world_panel(world, cfg.map_size)
    track_homog = np.hstack([track.centerline, np.ones((track.n_points, 1))])
    track_px = (world_to_panel @ track_homog.T).T[:, :2]
    track_px_closed = np.vstack([track_px, track_px[0]]).astype(np.int32)

    frames_out: list[np.ndarray] = []
    trail: list[tuple[int, int]] = []

    for i in idx:
        panel = base_panel.copy()
        cv2.polylines(panel, [track_px_closed], True, (60, 60, 60), 2, cv2.LINE_AA)

        pose = poses[i]
        robot_homog = np.array([pose.x, pose.y, 1.0])
        rx, ry, _ = world_to_panel @ robot_homog
        trail.append((int(rx), int(ry)))
        if len(trail) > cfg.trail_length:
            trail.pop(0)

        if len(trail) >= 2:
            cv2.polylines(
                panel, [np.array(trail, dtype=np.int32)], False, (0, 140, 255), 2, cv2.LINE_AA
            )

        heading_tip = (
            int(rx + 14 * np.cos(-pose.theta)),
            int(ry + 14 * np.sin(-pose.theta)),
        )
        cv2.circle(panel, (int(rx), int(ry)), 6, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.line(panel, (int(rx), int(ry)), heading_tip, (0, 0, 220), 2, cv2.LINE_AA)

        cam_frame = camera_frames[i]
        cam_h, cam_w = cam_frame.shape[:2]
        inset_w = int(cam_w * cfg.inset_scale)
        inset_h = int(cam_h * cfg.inset_scale)
        inset = cv2.resize(cam_frame, (inset_w, inset_h))
        pad = 6
        x0, y0 = panel.shape[1] - inset_w - pad, pad
        cv2.rectangle(
            panel, (x0 - 3, y0 - 3), (x0 + inset_w + 3, y0 + inset_h + 3), (255, 255, 255), -1
        )
        panel[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset
        cv2.rectangle(
            panel, (x0 - 3, y0 - 3), (x0 + inset_w + 3, y0 + inset_h + 3), (40, 40, 40), 2
        )

        cte_cm = cross_track_errors[i] * 100.0
        label = f"{controller_name}  |  cross-track: {cte_cm:+.1f} cm"
        cv2.putText(
            panel,
            label,
            (10, panel.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            label,
            (10, panel.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        frames_out.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gif(frames_out, output_path, cfg)


def _write_gif(frames: list[np.ndarray], output_path: Path, cfg: GifExportConfig) -> None:
    """Encode frames as a paletted, optimized GIF (keeps file size well under
    what an unquantized per-frame-palette encoder would produce)."""
    pil_frames = [Image.fromarray(f) for f in frames]
    shared_palette = pil_frames[0].quantize(
        colors=cfg.palette_colors, method=Image.Quantize.MEDIANCUT
    )
    quantized = [
        frame.quantize(colors=cfg.palette_colors, palette=shared_palette, dither=Image.Dither.NONE)
        for frame in pil_frames
    ]
    duration_ms = round(1000 / cfg.fps)
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
