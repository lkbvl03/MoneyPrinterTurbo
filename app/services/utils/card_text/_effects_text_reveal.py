# app/services/utils/card_text/_effects_text_reveal.py
from typing import Callable, Dict

import numpy as np
from moviepy import Clip

from ._primitives import _make_custom_frame_effect, _phase_progress

GROUP_NAME = "text_reveal"


def _typewriter_frame_fn(card_rgba, t, enter_duration, hold_duration, exit_duration):
    phase, progress = _phase_progress(t, enter_duration, hold_duration, exit_duration)
    card_w = card_rgba.shape[1]

    if phase == "enter":
        reveal_ratio = progress
    elif phase == "hold":
        reveal_ratio = 1.0
    else:
        # “反向擦除”：从右边缘把显示区域收窄回去，和 enter 阶段从左到右
        # 显现的方向对称（不是按打字顺序把已经打出的字逐个擦掉）——这是
        # 刻意简化，详见需求文档。
        reveal_ratio = 1.0 - progress

    reveal_px = int(card_w * reveal_ratio)
    frame = card_rgba.copy()
    frame[:, reveal_px:, 3] = 0
    return frame


typewriter = _make_custom_frame_effect(_typewriter_frame_fn)


CATEGORY_EFFECTS: Dict[str, Callable[[np.ndarray, float, float, float], Clip]] = {
    "typewriter": typewriter,
}
