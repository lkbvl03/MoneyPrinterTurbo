# app/services/utils/video_overlay_effects/_light.py
import math
from typing import Dict, Tuple

import numpy as np

from . import OverlayEffectSpec
from . import _primitives as p


def _lens_flare_sweep_layer_fn(t, size):
    width, height = size
    period = 5.0
    progress = (t % period) / period
    cx = width * progress
    cy = height * 0.3
    return p._radial_glow_layer(
        t, size, center=(cx, cy), radius=width * 0.18,
        color=(255, 244, 220), alpha_min=0.0, alpha_max=0.5,
    )


def _light_leak_layer_fn(t, size):
    width, height = size
    return p._radial_glow_layer(
        t, size, center=(0, height), radius=max(width, height) * 0.6,
        color=(255, 90, 40), alpha_min=0.05, alpha_max=0.28,
        pulse_hz=0.15,
    )


def _god_rays_layer_fn(t, size):
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = math.radians(20)
    projected = xx * math.cos(angle) + yy * math.sin(angle)
    bands = 0.5 + 0.5 * np.sin(projected / 40.0 + 0.4 * t)
    vertical_mask = np.clip(1.0 - yy / (height * 0.75), 0.0, 1.0)
    alpha = bands * vertical_mask * 0.25

    overlay_rgb = np.empty((height, width, 3), dtype=np.uint8)
    overlay_rgb[:, :] = (255, 244, 200)
    return overlay_rgb, alpha.astype(np.float32)


_BOKEH_LIGHTS = p._ParticleFieldConfig(
    seed=401, count=18, color=(255, 235, 190),
    radius_range=(6.0, 14.0), fall_speed_range=(0, 0),
    horizontal_drift=50, vertical_drift=35, sway_freq=0.08,
    alpha_range=(0.2, 0.45), shape="dot", streak_length=0, twinkle=False,
)


def _bokeh_lights_layer_fn(t, size):
    return p._particle_field_layer(t, size, _BOKEH_LIGHTS)


def _sun_pulse_layer_fn(t, size):
    width, height = size
    return p._radial_glow_layer(
        t, size, center=(width * 0.85, height * 0.12), radius=width * 0.25,
        color=(255, 250, 230), alpha_min=0.1, alpha_max=0.35,
        pulse_hz=0.35,
    )


def _golden_hour_glow_layer_fn(t, size) -> Tuple[np.ndarray, np.ndarray]:
    width, height = size
    intensity = 0.12 + 0.06 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.08 * t))
    overlay_rgb = np.empty((height, width, 3), dtype=np.uint8)
    overlay_rgb[:, :] = (255, 175, 90)
    overlay_alpha = np.full((height, width), intensity, dtype=np.float32)
    return overlay_rgb, overlay_alpha


CATEGORY_EFFECTS: Dict[str, OverlayEffectSpec] = {
    "lens_flare_sweep": OverlayEffectSpec(
        name="lens_flare_sweep", category="light", kind="overlay",
        generator=p._make_overlay_effect(_lens_flare_sweep_layer_fn, "screen"),
    ),
    "light_leak": OverlayEffectSpec(
        name="light_leak", category="light", kind="overlay",
        generator=p._make_overlay_effect(_light_leak_layer_fn, "screen"),
    ),
    "god_rays": OverlayEffectSpec(
        name="god_rays", category="light", kind="overlay",
        generator=p._make_overlay_effect(_god_rays_layer_fn, "screen"),
    ),
    "bokeh_lights": OverlayEffectSpec(
        name="bokeh_lights", category="light", kind="overlay",
        generator=p._make_overlay_effect(_bokeh_lights_layer_fn, "screen"),
    ),
    "sun_pulse": OverlayEffectSpec(
        name="sun_pulse", category="light", kind="overlay",
        generator=p._make_overlay_effect(_sun_pulse_layer_fn, "screen"),
    ),
    "golden_hour_glow": OverlayEffectSpec(
        name="golden_hour_glow", category="light", kind="overlay",
        generator=p._make_overlay_effect(_golden_hour_glow_layer_fn, "screen"),
    ),
}
