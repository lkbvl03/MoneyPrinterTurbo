# app/services/utils/card_text/_effects_rotation.py
from typing import Callable, Dict

import numpy as np
from moviepy import Clip
from PIL import Image

from ._primitives import _ease_in_cubic, _ease_out_cubic, _make_custom_frame_effect, _phase_progress

GROUP_NAME = "rotation"

_MAX_ANGLE_DEGREES = 25.0


def _rotate_in_frame_fn(card_rgba, t, enter_duration, hold_duration, exit_duration):
    # 旋转角度不能用 (dx, dy, scale, opacity) 四元组表示，所以这里用
    # _make_custom_frame_effect：每一帧都用 PIL 对卡片原图重新旋转。
    phase, progress = _phase_progress(t, enter_duration, hold_duration, exit_duration)
    if phase == "enter":
        angle = (1.0 - _ease_out_cubic(progress)) * _MAX_ANGLE_DEGREES
        opacity = min(progress * 3, 1.0)
    elif phase == "hold":
        angle, opacity = 0.0, 1.0
    else:
        angle = _ease_in_cubic(progress) * _MAX_ANGLE_DEGREES
        opacity = max(1.0 - progress * 3, 0.0)

    # expand=False 保持画布尺寸和卡片原图一致（与 _make_custom_frame_effect
    # 的约定一致：frame_fn 返回的数组尺寸必须和 card_rgba 相同）。
    image = Image.fromarray(card_rgba).rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=False
    )
    if opacity < 1.0:
        alpha = image.getchannel("A").point(lambda a: int(a * opacity))
        image.putalpha(alpha)
    return np.asarray(image)


rotate_in = _make_custom_frame_effect(_rotate_in_frame_fn)


CATEGORY_EFFECTS: Dict[str, Callable[[np.ndarray, float, float, float], Clip]] = {
    "rotate_in": rotate_in,
}
