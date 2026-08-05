# app/services/utils/card_text/_effects_motion.py
from typing import Callable, Dict

import numpy as np
from moviepy import Clip

from ._primitives import _ease_in_cubic, _ease_out_cubic, _make_rigid_effect, _phase_progress

GROUP_NAME = "motion"

_SLIDE_AMOUNT = 0.45


def _slide_transform_fn(dx_sign: float = 0.0, dy_sign: float = 0.0):
    def transform_fn(t, enter_duration, hold_duration, exit_duration):
        phase, progress = _phase_progress(t, enter_duration, hold_duration, exit_duration)
        if phase == "enter":
            amount = 1.0 - _ease_out_cubic(progress)
        elif phase == "hold":
            amount = 0.0
        else:
            amount = _ease_in_cubic(progress)
        return dx_sign * amount, dy_sign * amount, 1.0, 1.0

    return transform_fn


slide_left = _make_rigid_effect(_slide_transform_fn(dx_sign=-_SLIDE_AMOUNT))


CATEGORY_EFFECTS: Dict[str, Callable[[np.ndarray, float, float, float], Clip]] = {
    "slide_left": slide_left,
}
