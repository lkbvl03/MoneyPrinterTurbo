# app/services/utils/video_overlay_effects/_fire.py
import math
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from . import OverlayEffectSpec
from . import _primitives as p


def _fire_glow_layer_fn(t, size):
    width, height = size
    return p._radial_glow_layer(
        t, size, center=(width / 2, height * 1.15), radius=height * 0.55,
        color=(255, 120, 40), alpha_min=0.10, alpha_max=0.30,
        pulse_hz=0.9, pulse_phase=0.0,
    )


def _campfire_flicker_layer_fn(t, size):
    width, height = size
    return p._radial_glow_layer(
        t, size, center=(width / 2, height * 1.05), radius=height * 0.32,
        color=(255, 150, 60), alpha_min=0.15, alpha_max=0.45,
        pulse_hz=2.4, pulse_phase=0.6,
    )


def _plasma_pulse_layer_fn(t, size):
    width, height = size
    return p._radial_glow_layer(
        t, size, center=(width / 2, height / 2), radius=min(width, height) * 0.35,
        color=(120, 180, 255), alpha_min=0.05, alpha_max=0.22,
        pulse_hz=0.6,
    )


def _lightning_flash_layer_fn(t, size):
    width, height = size
    cycle = 3.7
    local_t = t % cycle
    if local_t < 0.06:
        strength = 1.0 - local_t / 0.06
    elif 0.09 <= local_t < 0.16:
        strength = (local_t - 0.09) / 0.07
    else:
        strength = 0.0

    overlay_rgb = np.empty((height, width, 3), dtype=np.uint8)
    overlay_rgb[:, :] = (235, 240, 255)
    overlay_alpha = np.full((height, width), strength * 0.85, dtype=np.float32)
    return overlay_rgb, overlay_alpha


def _midpoint_displacement(
    rng: np.random.Generator,
    start: Tuple[float, float],
    end: Tuple[float, float],
    roughness: float,
    depth: int,
) -> List[Tuple[float, float]]:
    points = [start, end]
    for _ in range(depth):
        new_points = [points[0]]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            length = math.hypot(x2 - x1, y2 - y1)
            offset = float(rng.uniform(-1, 1)) * length * roughness
            dx, dy = x2 - x1, y2 - y1
            norm = math.hypot(dx, dy) or 1.0
            perp = (-dy / norm, dx / norm)
            mx += perp[0] * offset
            my += perp[1] * offset
            new_points.append((mx, my))
            new_points.append((x2, y2))
        points = new_points
        roughness *= 0.6
    return points


def _lightning_branch_layer_fn(t, size):
    width, height = size
    cycle = 5.3
    local_t = t % cycle

    overlay_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    overlay_rgb[:, :] = (225, 235, 255)
    if local_t >= 0.25:
        return overlay_rgb, np.zeros((height, width), dtype=np.float32)

    # 用 cycle 序号做种子，保证同一次"闪电出现"期间的折线形状固定不变，
    # 不会在同一次闪光内逐帧随机跳动。
    strand_seed = int(t // cycle)
    rng = np.random.default_rng(strand_seed)
    start = (float(rng.uniform(0.2, 0.8)) * width, 0.0)
    end = (float(rng.uniform(0.2, 0.8)) * width, height * float(rng.uniform(0.5, 0.85)))
    points = _midpoint_displacement(rng, start, end, roughness=0.4, depth=6)

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.line(points, fill=255, width=3)

    strength = 1.0 - local_t / 0.25
    overlay_alpha = (np.asarray(image).astype(np.float32) / 255.0) * strength
    return overlay_rgb, overlay_alpha


CATEGORY_EFFECTS: Dict[str, OverlayEffectSpec] = {
    "fire_glow": OverlayEffectSpec(
        name="fire_glow", category="fire", kind="overlay",
        generator=p._make_overlay_effect(_fire_glow_layer_fn, "screen"),
    ),
    "lightning_flash": OverlayEffectSpec(
        name="lightning_flash", category="fire", kind="overlay",
        generator=p._make_overlay_effect(_lightning_flash_layer_fn, "screen"),
    ),
    "lightning_branch": OverlayEffectSpec(
        name="lightning_branch", category="fire", kind="overlay",
        generator=p._make_overlay_effect(_lightning_branch_layer_fn, "screen"),
    ),
    "campfire_flicker": OverlayEffectSpec(
        name="campfire_flicker", category="fire", kind="overlay",
        generator=p._make_overlay_effect(_campfire_flicker_layer_fn, "screen"),
    ),
    "plasma_pulse": OverlayEffectSpec(
        name="plasma_pulse", category="fire", kind="overlay",
        generator=p._make_overlay_effect(_plasma_pulse_layer_fn, "screen"),
    ),
}
