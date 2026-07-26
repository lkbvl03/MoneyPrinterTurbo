# app/services/utils/video_overlay_effects/_primitives.py
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Tuple

import numpy as np
from moviepy import Clip
from PIL import Image


def _blend_frame(
    base: np.ndarray,
    overlay_rgb: np.ndarray,
    overlay_alpha: np.ndarray,
    mode: str,
) -> np.ndarray:
    base_f = base.astype(np.float32)
    overlay_f = overlay_rgb.astype(np.float32)
    alpha = overlay_alpha.astype(np.float32)
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]
    alpha = np.clip(alpha, 0.0, 1.0)

    if mode == "normal_alpha":
        composite = overlay_f
    elif mode == "screen":
        composite = 255.0 - (255.0 - base_f) * (255.0 - overlay_f) / 255.0
    elif mode == "lighten":
        composite = np.maximum(base_f, overlay_f)
    else:
        raise ValueError(f"unknown blend mode: {mode}")

    blended = base_f * (1.0 - alpha) + composite * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _make_overlay_effect(
    layer_fn: Callable[[float, Tuple[int, int]], Tuple[np.ndarray, np.ndarray]],
    blend_mode: str,
) -> Callable[[Clip], Clip]:
    def generator(clip: Clip) -> Clip:
        def transform(get_frame, t: float):
            frame = get_frame(t)
            height, width = frame.shape[:2]
            overlay_rgb, overlay_alpha = layer_fn(t, (width, height))
            return _blend_frame(frame, overlay_rgb, overlay_alpha, blend_mode)

        return clip.transform(transform)

    return generator


def _make_transform_effect(
    frame_fn: Callable[[np.ndarray, float], np.ndarray],
) -> Callable[[Clip], Clip]:
    def generator(clip: Clip) -> Clip:
        def transform(get_frame, t: float):
            return frame_fn(get_frame(t), t)

        return clip.transform(transform)

    return generator


# ---------------------------------------------------------------------------
# 粒子场共用：雨/雪/灰尘/火星/灰烬/萤火虫/虚焦/胶片划痕。
# 逐个 Blit 小 'stamp'（不扫描整帧）让性能不依赖分辨率——这很重要，因为素材可能达到 1080x1920。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParticleFieldConfig:
    seed: int
    count: int
    color: Tuple[int, int, int]
    radius_range: Tuple[float, float]
    fall_speed_range: Tuple[float, float]
    horizontal_drift: float
    vertical_drift: float
    sway_freq: float
    alpha_range: Tuple[float, float]
    shape: str  # "dot" | "streak"
    streak_length: float
    twinkle: bool


@lru_cache(maxsize=64)
def _circular_stamp(radius_px: int) -> np.ndarray:
    radius_px = max(radius_px, 1)
    size = radius_px * 2 + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) - radius_px
    dist = np.sqrt(xx**2 + yy**2)
    return np.clip(1.0 - dist / radius_px, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=64)
def _streak_stamp(radius_px: int, length_px: int) -> np.ndarray:
    radius_px = max(radius_px, 1)
    length_px = max(length_px, radius_px * 2)
    width = radius_px * 2 + 1
    yy, xx = np.mgrid[0:length_px, 0:width].astype(np.float32)
    dx = np.abs(xx - radius_px)
    dy = np.abs(yy - length_px / 2.0)
    horizontal = np.clip(1.0 - dx / radius_px, 0.0, 1.0)
    vertical = np.clip(1.0 - dy / (length_px / 2.0), 0.0, 1.0)
    return (horizontal * vertical).astype(np.float32)


def _blit_stamp(
    alpha_canvas: np.ndarray, cx: int, cy: int, stamp: np.ndarray, weight: float
) -> None:
    height, width = alpha_canvas.shape
    stamp_h, stamp_w = stamp.shape
    top, left = cy - stamp_h // 2, cx - stamp_w // 2
    bottom, right = top + stamp_h, left + stamp_w

    canvas_top, canvas_left = max(top, 0), max(left, 0)
    canvas_bottom, canvas_right = min(bottom, height), min(right, width)
    if canvas_top >= canvas_bottom or canvas_left >= canvas_right:
        return

    stamp_top = canvas_top - top
    stamp_left = canvas_left - left
    stamp_bottom = stamp_top + (canvas_bottom - canvas_top)
    stamp_right = stamp_left + (canvas_right - canvas_left)

    region = alpha_canvas[canvas_top:canvas_bottom, canvas_left:canvas_right]
    np.maximum(
        region,
        stamp[stamp_top:stamp_bottom, stamp_left:stamp_right] * weight,
        out=region,
    )


def _particle_field_layer(
    t: float, size: Tuple[int, int], config: _ParticleFieldConfig
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = size
    overlay_alpha = np.zeros((height, width), dtype=np.float32)
    overlay_rgb = np.empty((height, width, 3), dtype=np.uint8)
    overlay_rgb[:, :] = config.color

    if config.count <= 0:
        return overlay_rgb, overlay_alpha

    rng = np.random.default_rng(config.seed)
    x0 = rng.uniform(0, width, config.count)
    y0 = rng.uniform(0, height, config.count)
    radii = rng.uniform(config.radius_range[0], config.radius_range[1], config.count)
    speeds = rng.uniform(
        config.fall_speed_range[0], config.fall_speed_range[1], config.count
    )
    base_alphas = rng.uniform(
        config.alpha_range[0], config.alpha_range[1], config.count
    )
    phase = rng.uniform(0, 2 * np.pi, config.count)

    margin = float(radii.max())
    sway_x = config.horizontal_drift * np.sin(2 * np.pi * config.sway_freq * t + phase)
    sway_y = config.vertical_drift * np.cos(2 * np.pi * config.sway_freq * t + phase)
    y = (y0 + speeds * t + sway_y) % (height + 2 * margin) - margin
    x = (x0 + sway_x) % width

    alphas = base_alphas
    if config.twinkle:
        alphas = alphas * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t + phase))

    for i in range(config.count):
        radius_px = max(int(round(radii[i])), 1)
        if config.shape == "streak":
            stamp = _streak_stamp(radius_px, int(round(config.streak_length)))
        else:
            stamp = _circular_stamp(radius_px)
        _blit_stamp(overlay_alpha, int(round(x[i])), int(round(y[i])), stamp, float(alphas[i]))

    return overlay_rgb, overlay_alpha


# ---------------------------------------------------------------------------
# 滚动值噪声：用于雾/烟。不增加外部依赖，自生成低分辨率随机网格，用 PIL 双线性插值放大。
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _value_noise_grid(seed: int, grid_w: int, grid_h: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (grid_h, grid_w)).astype(np.float32)


def _sample_scrolling_noise(
    size: Tuple[int, int], *, seed: int, scale: float, t: float, speed: float
) -> np.ndarray:
    width, height = size
    grid_w = max(int(width / scale), 2)
    grid_h = max(int(height / scale), 2)
    grid = _value_noise_grid(seed, grid_w, grid_h)
    image = Image.fromarray((grid * 255).astype(np.uint8), "L")

    offset_x = int((speed * t) % grid_w)
    if offset_x:
        shifted = Image.new("L", (grid_w, grid_h))
        shifted.paste(image.crop((offset_x, 0, grid_w, grid_h)), (0, 0))
        shifted.paste(image.crop((0, 0, offset_x, grid_h)), (grid_w - offset_x, 0))
        image = shifted

    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def _noise_field_layer(
    t: float,
    size: Tuple[int, int],
    *,
    seed: int,
    scale: float,
    speed: float,
    color: Tuple[int, int, int],
    alpha_min: float,
    alpha_max: float,
    vertical_bias: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = size
    noise = _sample_scrolling_noise(size, seed=seed, scale=scale, t=t, speed=speed)
    alpha = alpha_min + (alpha_max - alpha_min) * noise

    if vertical_bias > 0:
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        bias_mask = np.clip(yy - (1.0 - vertical_bias), 0.0, vertical_bias) / vertical_bias
        alpha = alpha * bias_mask

    overlay_rgb = np.empty((height, width, 3), dtype=np.uint8)
    overlay_rgb[:, :] = color
    return overlay_rgb, alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# 径向辉光：柔和的圆形辉光，用于火焰/光线/能量。
# ---------------------------------------------------------------------------


def _radial_glow_layer(
    t: float,
    size: Tuple[int, int],
    *,
    center: Tuple[float, float],
    radius: float,
    color: Tuple[int, int, int],
    alpha_min: float,
    alpha_max: float,
    pulse_hz: float = 0.0,
    pulse_phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = center
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1.0)
    falloff = np.clip(1.0 - dist, 0.0, 1.0) ** 2

    if pulse_hz > 0:
        intensity = alpha_min + (alpha_max - alpha_min) * (
            0.5 + 0.5 * math.sin(2 * math.pi * pulse_hz * t + pulse_phase)
        )
    else:
        intensity = alpha_max

    overlay_rgb = np.empty((height, width, 3), dtype=np.uint8)
    overlay_rgb[:, :] = color
    return overlay_rgb, (falloff * intensity).astype(np.float32)
