# app/services/utils/card_text/_effects_light.py
import math
from typing import Callable, Dict

import numpy as np
from moviepy import Clip

from ._primitives import _ease_in_cubic, _ease_out_cubic, _make_rigid_effect, _phase_progress

GROUP_NAME = "light"

_PULSE_HZ = 1.5
_PULSE_SCALE_AMPLITUDE = 0.04


def _glow_pulse_transform_fn(t, enter_duration, hold_duration, exit_duration):
    phase, progress = _phase_progress(t, enter_duration, hold_duration, exit_duration)
    if phase == "enter":
        opacity = _ease_out_cubic(progress)
        scale = 0.9 + 0.1 * _ease_out_cubic(progress)
    elif phase == "hold":
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * _PULSE_HZ * t)
        opacity = 1.0
        scale = 1.0 + _PULSE_SCALE_AMPLITUDE * pulse
    else:
        opacity = 1.0 - _ease_in_cubic(progress)
        scale = 1.0 - 0.1 * _ease_in_cubic(progress)
    return 0.0, 0.0, max(scale, 0.01), opacity


glow_pulse = _make_rigid_effect(_glow_pulse_transform_fn)


CATEGORY_EFFECTS: Dict[str, Callable[[np.ndarray, float, float, float], Clip]] = {
    "glow_pulse": glow_pulse,
}
