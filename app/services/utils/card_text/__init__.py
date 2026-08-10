import random
from typing import Callable, Dict, Optional

import numpy as np
from moviepy import Clip

from . import (
    _effects_bounce,
    _effects_light,
    _effects_motion,
    _effects_rotation,
    _effects_text_reveal,
    _styles,
)

CARD_STYLES: Dict[str, Callable[[str], np.ndarray]] = dict(_styles.CATEGORY_STYLES)

CARD_EFFECTS: Dict[str, Callable[[np.ndarray, float, float, float], Clip]] = {}
EFFECT_GROUPS: Dict[str, str] = {}
for _module in (
    _effects_motion,
    _effects_bounce,
    _effects_rotation,
    _effects_text_reveal,
    _effects_light,
):
    CARD_EFFECTS.update(_module.CATEGORY_EFFECTS)
    for _name in _module.CATEGORY_EFFECTS:
        EFFECT_GROUPS[_name] = _module.GROUP_NAME


# enter/exit 固定各占 0.4 秒，总时长中剩下的部分作为 hold。
# 参见 spec:"默认显示时长：3 秒"。
_DEFAULT_ENTER_DURATION = 0.4
_DEFAULT_EXIT_DURATION = 0.4
_DEFAULT_TOTAL_DURATION = 3.0

# 公开（不带 `_` 前缀），因为 video.py（Task 11）需要知道这个默认值，
# 以便在下一张卡片提前于 3 秒内开始时自动缩短当前卡片的时长，避免两张
# 卡片重叠——这与 spec 中的决策一致。
DEFAULT_CARD_DURATION_SECONDS = _DEFAULT_TOTAL_DURATION


def resolve_style_name(style: Optional[str], *, random_choice=None) -> str:
    if random_choice is None:
        random_choice = random.choice
    if style is None or style == "random":
        return random_choice(list(CARD_STYLES))
    return style


def resolve_effect_name(effect: Optional[str], *, random_choice=None) -> str:
    if random_choice is None:
        random_choice = random.choice
    if effect is None or effect == "random":
        return random_choice(list(CARD_EFFECTS))
    return effect


def render_card_clip(
    text: str,
    style_name: str,
    effect_name: str,
    duration: Optional[float] = None,
    font_name: Optional[str] = None,
    font_size: Optional[int] = None,
) -> Clip:
    """根据 catalog 里的模板名 + 效果名，渲染一张带出现/消失动画的完整卡片
    Clip。duration 是期望的总显示时长（默认 3 秒）；enter/exit 固定各占
    0.4 秒，剩下的时间作为 hold（若总时长很短，会按比例压缩 enter/exit，
    保证三段加起来正好等于 duration，不会出现负数 hold）。font_name/
    font_size 为 None 时各 style 使用自己的默认字体/字号（调用方
    video.py 已确认 font_name 对应的字体文件确实存在，这里不再重复校验）。"""
    if style_name not in CARD_STYLES:
        raise KeyError(f"unknown card style: {style_name}")
    if effect_name not in CARD_EFFECTS:
        raise KeyError(f"unknown card effect: {effect_name}")

    total_duration = duration if duration is not None else _DEFAULT_TOTAL_DURATION
    enter_duration = min(_DEFAULT_ENTER_DURATION, total_duration / 3)
    exit_duration = min(_DEFAULT_EXIT_DURATION, total_duration / 3)
    hold_duration = max(total_duration - enter_duration - exit_duration, 0.0)

    card_rgba = CARD_STYLES[style_name](text, font_name, font_size)
    build = CARD_EFFECTS[effect_name]
    return build(card_rgba, enter_duration, hold_duration, exit_duration)
