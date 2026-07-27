# app/services/utils/video_overlay_effects/__init__.py
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from moviepy import Clip

OVERLAY_EFFECTS: Dict[str, "OverlayEffectSpec"] = {}


@dataclass(frozen=True)
class OverlayEffectSpec:
    name: str
    category: str
    kind: str  # "overlay" | "transform"
    generator: Callable[[Clip], Clip]


def resolve_overlay_effect_name(
    style: Optional[str], *, random_choice=None
) -> Optional[str]:
    if random_choice is None:
        random_choice = random.choice
    if style is None or style == "none":
        return None
    if style == "random":
        return random_choice(list(OVERLAY_EFFECTS))
    return style


def apply_overlay_effect(clip: Clip, effect_name: str) -> Clip:
    spec = OVERLAY_EFFECTS[effect_name]
    return spec.generator(clip)


from . import _weather

OVERLAY_EFFECTS.update(_weather.CATEGORY_EFFECTS)

from . import _atmosphere

OVERLAY_EFFECTS.update(_atmosphere.CATEGORY_EFFECTS)

from . import _fire

OVERLAY_EFFECTS.update(_fire.CATEGORY_EFFECTS)

from . import _light

OVERLAY_EFFECTS.update(_light.CATEGORY_EFFECTS)

from . import _retro

OVERLAY_EFFECTS.update(_retro.CATEGORY_EFFECTS)

# === register category modules below ===
