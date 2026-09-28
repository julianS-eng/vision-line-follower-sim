"""Render a :class:`~vision_line_follower.track.generator.Track` into a
top-down "world" raster image.

The world image is the ground-truth floor texture that the simulated camera
samples from (via a perspective warp, see
:mod:`vision_line_follower.sim.camera`). It bakes in the painted line, a
procedural floor texture, an optional lighting gradient, occlusion blobs and
intersection crossings, all controlled by :class:`TrackRenderConfig` so that
benchmark "difficulty levels" are reproducible from a seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision_line_follower.track.generator import Track


@dataclass(frozen=True, slots=True)
class TrackRenderConfig:
    """Rendering / difficulty parameters for the world image.

    Attributes:
        pixels_per_meter: Raster resolution of the world image.
        margin_m: Empty border kept around the track's bounding box.
        floor_color: BGR base color of the floor.
        line_color: BGR color of the painted line.
        line_width_scale: Multiplier applied to ``track.width_m`` when
            rasterizing the line stroke (>1 makes the line thicker than its
            "official" width, useful for stress-testing the vision pipeline).
        lighting_strength: 0 = perfectly uniform lighting, 1 = strong linear
            gradient plus a soft vignette across the floor.
        texture_noise_std: Standard deviation (0-255 scale) of per-pixel
            floor texture noise.
        occlusion_level: 0 = no occlusions, 1 = frequent large occlusion
            blobs painted over the line (dirt, glare, obstacles).
        seed: Random seed for texture/occlusion generation (independent from
            the track's own generation seed, so the same track can be
            rendered under different conditions).
    """

    pixels_per_meter: float = 250.0
    margin_m: float = 0.25
    floor_color: tuple[int, int, int] = (225, 222, 214)
    line_color: tuple[int, int, int] = (25, 25, 25)
    line_width_scale: float = 1.0
    lighting_strength: float = 0.3
    texture_noise_std: float = 6.0
    occlusion_level: float = 0.0
    seed: int = 0


@dataclass(slots=True)
class WorldImage:
    """A rendered world raster plus the metadata needed to map pixels<->metres."""

    image: np.ndarray  # (H, W, 3) uint8 BGR
    origin_m: tuple[float, float]  # world (x, y) of pixel (0, 0)
    pixels_per_meter: float

    def world_to_pixel_affine(self) -> np.ndarray:
        """3x3 homogeneous matrix mapping world (x, y, 1) -> pixel (u, v, 1)."""
        ppm = self.pixels_per_meter
        ox, oy = self.origin_m
        return np.array(
            [
                [ppm, 0.0, -ox * ppm],
                [0.0, -ppm, oy * ppm],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def pixel_to_world_affine(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_pixel_affine())


def _draw_lighting_gradient(
    image: np.ndarray, strength: float, rng: np.random.Generator
) -> np.ndarray:
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    angle = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    proj = (xx * math.cos(angle) + yy * math.sin(angle)) / max(w, h)
    gradient = 1.0 + strength * (proj - proj.mean()) * 1.4
    cy, cx = h / 2.0, w / 2.0
    r = np.hypot(xx - cx, yy - cy) / math.hypot(cx, cy)
    vignette = 1.0 - strength * 0.5 * (r**2)
    factor = np.clip(gradient * vignette, 0.35, 1.6)[..., None]
    out = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def _draw_occlusions(
    image: np.ndarray,
    centerline_px: np.ndarray,
    level: float,
    background_color: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    if level <= 0:
        return image
    n_points = centerline_px.shape[0]
    n_blobs = int(level * 25)
    out = image.copy()
    for _ in range(n_blobs):
        idx = int(rng.integers(0, n_points))
        cx, cy = centerline_px[idx]
        offset = rng.normal(scale=6.0 + 10.0 * level, size=2)
        center = (int(cx + offset[0]), int(cy + offset[1]))
        radius = int(rng.uniform(4, 6 + 14 * level))
        color = tuple(int(np.clip(c + rng.normal(scale=15), 0, 255)) for c in background_color)
        cv2.circle(out, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def render_track(track: Track, config: TrackRenderConfig) -> WorldImage:
    """Rasterize a track into a :class:`WorldImage`."""
    rng = np.random.default_rng(config.seed)
    min_x, min_y, max_x, max_y = track.bounding_box
    min_x -= config.margin_m
    min_y -= config.margin_m
    max_x += config.margin_m
    max_y += config.margin_m

    width_px = max(int((max_x - min_x) * config.pixels_per_meter), 8)
    height_px = max(int((max_y - min_y) * config.pixels_per_meter), 8)

    world_img = WorldImage(
        image=np.zeros((height_px, width_px, 3), dtype=np.uint8),
        origin_m=(min_x, max_y),
        pixels_per_meter=config.pixels_per_meter,
    )

    base = np.empty((height_px, width_px, 3), dtype=np.uint8)
    base[:] = config.floor_color
    noise = rng.normal(scale=config.texture_noise_std, size=(height_px, width_px, 1))
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=1.0)
    base = _draw_lighting_gradient(base, config.lighting_strength, rng)

    affine = world_img.world_to_pixel_affine()
    homog = np.hstack([track.centerline, np.ones((track.n_points, 1))])
    px = (affine @ homog.T).T[:, :2]
    px_closed = np.vstack([px, px[0]]).astype(np.int32)

    line_thickness_px = max(
        round(track.width_m * config.line_width_scale * config.pixels_per_meter), 1
    )
    cv2.polylines(
        base,
        [px_closed],
        isClosed=True,
        color=config.line_color,
        thickness=line_thickness_px,
        lineType=cv2.LINE_AA,
    )

    for idx in track.crossing_indices:
        center = px[idx]
        heading = track.heading[idx]
        perp = np.array([-math.sin(heading), math.cos(heading)])
        half_len_px = 4.0 * line_thickness_px
        p1 = (center + perp * half_len_px).astype(np.int32)
        p2 = (center - perp * half_len_px).astype(np.int32)
        cv2.line(
            base,
            tuple(p1),
            tuple(p2),
            config.line_color,
            thickness=line_thickness_px,
            lineType=cv2.LINE_AA,
        )

    base = _draw_occlusions(base, px, config.occlusion_level, config.floor_color, rng)

    world_img.image = base
    return world_img
