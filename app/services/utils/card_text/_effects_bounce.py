# app/services/utils/card_text/_effects_bounce.py
from typing import Callable, Dict

import numpy as np
from moviepy import Clip

from ._primitives import _ease_in_cubic, _ease_out_back, _make_rigid_effect, _phase_progress

GROUP_NAME = "bounce"


def _bounce_transform_fn(t, enter_duration, hold_duration, exit_duration):
    phase, progress = _phase_progress(t, enter_duration, hold_duration, exit_duration)
    if phase == "enter":
        scale = _ease_out_back(progress)
        opacity = min(progress * 3, 1.0)
    elif phase == "hold":
        scale, opacity = 1.0, 1.0
    else:
        scale = 1.0 - _ease_in_cubic(progress)
        opacity = max(1.0 - progress * 3, 0.0)
    return 0.0, 0.0, max(scale, 0.01), opacity


bounce = _make_rigid_effect(_bounce_transform_fn)


CATEGORY_EFFECTS: Dict[str, Callable[[np.ndarray, float, float, float], Clip]] = {
    "bounce": bounce,
}
