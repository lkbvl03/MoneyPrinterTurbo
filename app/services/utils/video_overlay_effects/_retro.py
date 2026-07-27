# app/services/utils/video_overlay_effects/_retro.py
import math
from functools import lru_cache
from typing import Dict

import numpy as np

from . import OverlayEffectSpec
from . import _primitives as p


def _film_grain_frame(frame: np.ndarray, t: float) -> np.ndarray:
    height, width = frame.shape[:2]
    # 用 t 派生种子保证同一帧多次取值时噪声图案一致（纯函数要求）。
    seed = int(round(t * 1000)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 14.0, (height, width, 1)).astype(np.float32)
    grainy = frame.astype(np.float32) + noise
    return np.clip(grainy, 0, 255).astype(np.uint8)


_FILM_SCRATCHES = p._ParticleFieldConfig(
    seed=501, count=4, color=(225, 225, 225),
    radius_range=(0.6, 1.2), fall_speed_range=(0.0, 0.0),
    horizontal_drift=0.0, vertical_drift=0.0, sway_freq=0.0,
    alpha_range=(0.0, 0.35), shape="streak", streak_length=2000.0, twinkle=True,
)


def _film_scratches_layer_fn(t, size):
    return p._particle_field_layer(t, size, _FILM_SCRATCHES)


@lru_cache(maxsize=8)
def _vignette_mask(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    return np.clip(dist - 0.6, 0.0, 1.0) / 0.4


def _vignette_pulse_frame(frame: np.ndarray, t: float) -> np.ndarray:
    height, width = frame.shape[:2]
    mask = _vignette_mask(width, height)
    darkness = 0.35 + 0.15 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.25 * t))
    factor = 1.0 - mask * darkness
    darkened = frame.astype(np.float32) * factor[:, :, None]
    return np.clip(darkened, 0, 255).astype(np.uint8)


def _vhs_glitch_frame(frame: np.ndarray, t: float) -> np.ndarray:
    height, width = frame.shape[:2]
    result = frame.copy()
    seed = int(t * 12) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    band_count = int(rng.integers(1, 3))
    for _ in range(band_count):
        band_top = int(rng.uniform(0, max(height - 1, 1)))
        band_height = max(int(rng.uniform(4, max(height * 0.05, 5))), 1)
        band_bottom = min(band_top + band_height, height)
        if band_bottom <= band_top:
            continue
        shift = int(rng.uniform(-25, 25))
        result[band_top:band_bottom] = np.roll(result[band_top:band_bottom], shift, axis=1)
        color_noise = rng.uniform(-20, 20, (band_bottom - band_top, width, 3))
        result[band_top:band_bottom] = np.clip(
            result[band_top:band_bottom].astype(np.float32) + color_noise, 0, 255
        ).astype(np.uint8)
    return result


def _chromatic_aberration_frame(frame: np.ndarray, t: float) -> np.ndarray:
    shift = 3
    result = frame.copy()
    result[:, :, 0] = np.roll(frame[:, :, 0], shift, axis=1)
    result[:, :, 2] = np.roll(frame[:, :, 2], -shift, axis=1)
    return result


def _film_burn_edges_layer_fn(t, size):
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    edge_dist = np.minimum.reduce([xx, width - 1 - xx, yy, height - 1 - yy])
    band = max(width, height) * 0.08
    alpha = np.clip(1.0 - edge_dist / band, 0.0, 1.0) ** 1.5
    flicker = 0.6 + 0.4 * math.sin(2 * math.pi * 0.3 * t)

    overlay_rgb = np.empty((height, width, 3), dtype=np.uint8)
    overlay_rgb[:, :] = (255, 100, 30)
    return overlay_rgb, (alpha * flicker * 0.5).astype(np.float32)


def _double_exposure_ghost_frame(frame: np.ndarray, t: float) -> np.ndarray:
    offset_x = int(10 + 6 * math.sin(2 * math.pi * 0.2 * t))
    ghost = np.roll(frame, offset_x, axis=1)
    blended = frame.astype(np.float32) * 0.75 + ghost.astype(np.float32) * 0.25
    return np.clip(blended, 0, 255).astype(np.uint8)


def _static_noise_layer_fn(t, size):
    width, height = size
    seed = int(round(t * 1000)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 255, (height, width), dtype=np.uint8)
    overlay_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    overlay_alpha = np.full((height, width), 0.08, dtype=np.float32)
    return overlay_rgb, overlay_alpha


def _color_drift_flicker_frame(frame: np.ndarray, t: float) -> np.ndarray:
    brightness = 1.0 + 0.04 * math.sin(2 * math.pi * 0.12 * t)
    warm_shift = 0.03 * math.sin(2 * math.pi * 0.05 * t + 1.3)
    result = frame.astype(np.float32) * brightness
    result[:, :, 0] *= 1.0 + warm_shift
    result[:, :, 2] *= 1.0 - warm_shift
    return np.clip(result, 0, 255).astype(np.uint8)


def _motion_streak_blur_frame(frame: np.ndarray, t: float) -> np.ndarray:
    offsets = (-6, -3, 0, 3, 6)
    accum = np.zeros_like(frame, dtype=np.float32)
    for offset in offsets:
        accum += np.roll(frame, offset, axis=1).astype(np.float32)
    return np.clip(accum / len(offsets), 0, 255).astype(np.uint8)


CATEGORY_EFFECTS: Dict[str, OverlayEffectSpec] = {
    "film_grain": OverlayEffectSpec(
        name="film_grain", category="retro", kind="transform",
        generator=p._make_transform_effect(_film_grain_frame),
    ),
    "film_scratches": OverlayEffectSpec(
        name="film_scratches", category="retro", kind="overlay",
        generator=p._make_overlay_effect(_film_scratches_layer_fn, "screen"),
    ),
    "vignette_pulse": OverlayEffectSpec(
        name="vignette_pulse", category="retro", kind="transform",
        generator=p._make_transform_effect(_vignette_pulse_frame),
    ),
    "vhs_glitch": OverlayEffectSpec(
        name="vhs_glitch", category="retro", kind="transform",
        generator=p._make_transform_effect(_vhs_glitch_frame),
    ),
    "chromatic_aberration": OverlayEffectSpec(
        name="chromatic_aberration", category="retro", kind="transform",
        generator=p._make_transform_effect(_chromatic_aberration_frame),
    ),
    "film_burn_edges": OverlayEffectSpec(
        name="film_burn_edges", category="retro", kind="overlay",
        generator=p._make_overlay_effect(_film_burn_edges_layer_fn, "screen"),
    ),
    "double_exposure_ghost": OverlayEffectSpec(
        name="double_exposure_ghost", category="retro", kind="transform",
        generator=p._make_transform_effect(_double_exposure_ghost_frame),
    ),
    "static_noise_overlay": OverlayEffectSpec(
        name="static_noise_overlay", category="retro", kind="overlay",
        generator=p._make_overlay_effect(_static_noise_layer_fn, "screen"),
    ),
    "color_drift_flicker": OverlayEffectSpec(
        name="color_drift_flicker", category="retro", kind="transform",
        generator=p._make_transform_effect(_color_drift_flicker_frame),
    ),
    "motion_streak_blur": OverlayEffectSpec(
        name="motion_streak_blur", category="retro", kind="transform",
        generator=p._make_transform_effect(_motion_streak_blur_frame),
    ),
}
