# app/services/utils/video_overlay_effects/_camera.py
from typing import Dict

import numpy as np

from app.services.utils import video_effects

from . import OverlayEffectSpec
from . import _primitives as p


def _heat_haze_frame(frame: np.ndarray, t: float) -> np.ndarray:
    """热浪扭曲效果：通过正弦波周期位移每行像素，创建热浪般的扭曲视觉。

    幅度根据帧宽度缩放，波长根据帧高度计算，时间参数控制动画速度。"""
    height, width = frame.shape[:2]
    amplitude = max(width * 0.01, 2.0)
    wavelength = height / 6.0
    row_idx = np.arange(height)
    shift = (amplitude * np.sin(2 * np.pi * (row_idx / wavelength - 0.6 * t))).astype(np.int32)
    col_idx = (np.arange(width)[None, :] - shift[:, None]) % width
    return np.take_along_axis(frame, col_idx[:, :, None].repeat(3, axis=2), axis=1)


CATEGORY_EFFECTS: Dict[str, OverlayEffectSpec] = {
    "zoom_in": OverlayEffectSpec(
        name="zoom_in", category="camera", kind="transform",
        generator=lambda clip: video_effects.zoomin_transition(clip, 1),
    ),
    "zoom_out": OverlayEffectSpec(
        name="zoom_out", category="camera", kind="transform",
        generator=lambda clip: video_effects.zoomout_transition(clip, 1),
    ),
    "pan_left": OverlayEffectSpec(
        name="pan_left", category="camera", kind="transform",
        generator=lambda clip: video_effects.pan_left_transition(clip, 1),
    ),
    "pan_right": OverlayEffectSpec(
        name="pan_right", category="camera", kind="transform",
        generator=lambda clip: video_effects.pan_right_transition(clip, 1),
    ),
    "heat_haze": OverlayEffectSpec(
        name="heat_haze", category="camera", kind="transform",
        generator=p._make_transform_effect(_heat_haze_frame),
    ),
}
