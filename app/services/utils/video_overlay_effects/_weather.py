# app/services/utils/video_overlay_effects/_weather.py
from typing import Dict

from . import OverlayEffectSpec
from . import _primitives as p


def _rain_layer_fn(config: p._ParticleFieldConfig):
    return lambda t, size: p._particle_field_layer(t, size, config)


_RAIN_LIGHT = p._ParticleFieldConfig(
    seed=101, count=90, color=(200, 210, 230),
    radius_range=(1.0, 2.0), fall_speed_range=(900, 1300),
    horizontal_drift=6, vertical_drift=0, sway_freq=0.3,
    alpha_range=(0.25, 0.45), shape="streak", streak_length=18, twinkle=False,
)
_RAIN_HEAVY = p._ParticleFieldConfig(
    seed=102, count=220, color=(190, 200, 220),
    radius_range=(1.5, 3.0), fall_speed_range=(1200, 1700),
    horizontal_drift=10, vertical_drift=0, sway_freq=0.3,
    alpha_range=(0.35, 0.6), shape="streak", streak_length=28, twinkle=False,
)
_SNOW_LIGHT = p._ParticleFieldConfig(
    seed=103, count=60, color=(255, 255, 255),
    radius_range=(2.0, 4.0), fall_speed_range=(40, 90),
    horizontal_drift=40, vertical_drift=0, sway_freq=0.5,
    alpha_range=(0.5, 0.85), shape="dot", streak_length=0, twinkle=False,
)
_SNOW_HEAVY = p._ParticleFieldConfig(
    seed=104, count=160, color=(255, 255, 255),
    radius_range=(2.0, 6.0), fall_speed_range=(60, 140),
    horizontal_drift=70, vertical_drift=0, sway_freq=0.4,
    alpha_range=(0.55, 0.9), shape="dot", streak_length=0, twinkle=False,
)


def _fog_drift_layer_fn(t, size):
    return p._noise_field_layer(
        t, size, seed=201, scale=10.0, speed=25.0,
        color=(225, 225, 230), alpha_min=0.06, alpha_max=0.22,
    )


def _mist_ground_layer_fn(t, size):
    return p._noise_field_layer(
        t, size, seed=202, scale=14.0, speed=15.0,
        color=(215, 218, 222), alpha_min=0.05, alpha_max=0.35,
        vertical_bias=0.45,
    )


CATEGORY_EFFECTS: Dict[str, OverlayEffectSpec] = {
    "rain_light": OverlayEffectSpec(
        name="rain_light", category="weather", kind="overlay",
        generator=p._make_overlay_effect(_rain_layer_fn(_RAIN_LIGHT), "normal_alpha"),
    ),
    "rain_heavy": OverlayEffectSpec(
        name="rain_heavy", category="weather", kind="overlay",
        generator=p._make_overlay_effect(_rain_layer_fn(_RAIN_HEAVY), "normal_alpha"),
    ),
    "snow_light": OverlayEffectSpec(
        name="snow_light", category="weather", kind="overlay",
        generator=p._make_overlay_effect(_rain_layer_fn(_SNOW_LIGHT), "normal_alpha"),
    ),
    "snow_heavy": OverlayEffectSpec(
        name="snow_heavy", category="weather", kind="overlay",
        generator=p._make_overlay_effect(_rain_layer_fn(_SNOW_HEAVY), "normal_alpha"),
    ),
    "fog_drift": OverlayEffectSpec(
        name="fog_drift", category="weather", kind="overlay",
        generator=p._make_overlay_effect(_fog_drift_layer_fn, "normal_alpha"),
    ),
    "mist_ground": OverlayEffectSpec(
        name="mist_ground", category="weather", kind="overlay",
        generator=p._make_overlay_effect(_mist_ground_layer_fn, "normal_alpha"),
    ),
}
