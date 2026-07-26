# app/services/utils/video_overlay_effects/_atmosphere.py
from typing import Dict

from . import OverlayEffectSpec
from . import _primitives as p


def _particle_layer_fn(config: p._ParticleFieldConfig):
    return lambda t, size: p._particle_field_layer(t, size, config)


def _smoke_drift_layer_fn(t, size):
    return p._noise_field_layer(
        t, size, seed=301, scale=9.0, speed=18.0,
        color=(115, 115, 125), alpha_min=0.05, alpha_max=0.28,
    )


_DUST_MOTES = p._ParticleFieldConfig(
    seed=302, count=70, color=(255, 244, 214),
    radius_range=(1.0, 2.5), fall_speed_range=(-15, 15),
    horizontal_drift=25, vertical_drift=20, sway_freq=0.2,
    alpha_range=(0.15, 0.35), shape="dot", streak_length=0, twinkle=True,
)
_EMBERS_RISING = p._ParticleFieldConfig(
    seed=303, count=50, color=(255, 140, 60),
    radius_range=(1.5, 3.0), fall_speed_range=(-160, -90),
    horizontal_drift=15, vertical_drift=0, sway_freq=0.35,
    alpha_range=(0.4, 0.75), shape="dot", streak_length=0, twinkle=True,
)
_ASH_FALLING = p._ParticleFieldConfig(
    seed=304, count=80, color=(180, 175, 168),
    radius_range=(1.5, 3.5), fall_speed_range=(35, 70),
    horizontal_drift=20, vertical_drift=0, sway_freq=0.25,
    alpha_range=(0.3, 0.5), shape="dot", streak_length=0, twinkle=False,
)
_FIREFLIES = p._ParticleFieldConfig(
    seed=305, count=14, color=(210, 255, 120),
    radius_range=(2.0, 3.0), fall_speed_range=(0, 0),
    horizontal_drift=60, vertical_drift=45, sway_freq=0.15,
    alpha_range=(0.5, 0.9), shape="dot", streak_length=0, twinkle=True,
)
_SPARKS = p._ParticleFieldConfig(
    seed=306, count=40, color=(255, 200, 90),
    radius_range=(1.0, 1.8), fall_speed_range=(-260, -150),
    horizontal_drift=25, vertical_drift=0, sway_freq=0.5,
    alpha_range=(0.5, 0.9), shape="streak", streak_length=10, twinkle=True,
)


CATEGORY_EFFECTS: Dict[str, OverlayEffectSpec] = {
    "smoke_drift": OverlayEffectSpec(
        name="smoke_drift", category="atmosphere", kind="overlay",
        generator=p._make_overlay_effect(_smoke_drift_layer_fn, "screen"),
    ),
    "dust_motes": OverlayEffectSpec(
        name="dust_motes", category="atmosphere", kind="overlay",
        generator=p._make_overlay_effect(_particle_layer_fn(_DUST_MOTES), "screen"),
    ),
    "embers_rising": OverlayEffectSpec(
        name="embers_rising", category="atmosphere", kind="overlay",
        generator=p._make_overlay_effect(_particle_layer_fn(_EMBERS_RISING), "screen"),
    ),
    "ash_falling": OverlayEffectSpec(
        name="ash_falling", category="atmosphere", kind="overlay",
        generator=p._make_overlay_effect(_particle_layer_fn(_ASH_FALLING), "normal_alpha"),
    ),
    "fireflies": OverlayEffectSpec(
        name="fireflies", category="atmosphere", kind="overlay",
        generator=p._make_overlay_effect(_particle_layer_fn(_FIREFLIES), "screen"),
    ),
    "sparks": OverlayEffectSpec(
        name="sparks", category="atmosphere", kind="overlay",
        generator=p._make_overlay_effect(_particle_layer_fn(_SPARKS), "screen"),
    ),
}
