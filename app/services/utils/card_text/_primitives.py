# app/services/utils/card_text/_primitives.py
import math
from typing import Callable, Optional, Tuple

import numpy as np
from moviepy import Clip, VideoClip
from PIL import Image, ImageDraw, ImageFont

# --- Easing ------------------------------------------------------------------


def _ease_out_cubic(x: float) -> float:
    return 1 - (1 - x) ** 3


def _ease_in_cubic(x: float) -> float:
    return x**3


def _ease_in_out_sine(x: float) -> float:
    return -(math.cos(math.pi * x) - 1) / 2


def _ease_out_back(x: float, overshoot: float = 1.70158) -> float:
    c = overshoot + 1
    return 1 + c * (x - 1) ** 3 + overshoot * (x - 1) ** 2


def _phase_progress(
    t: float, enter_duration: float, hold_duration: float, exit_duration: float
) -> Tuple[str, float]:
    """在卡片自己的时间轴上（t=0 是卡片刚开始出现的时刻），判断当前处于
    enter/hold/exit 哪个阶段，以及该阶段内的 0..1 进度。"""
    if t < enter_duration:
        return "enter", t / max(enter_duration, 1e-6)
    if t < enter_duration + hold_duration:
        return "hold", 1.0
    exit_t = t - enter_duration - hold_duration
    return "exit", min(exit_t / max(exit_duration, 1e-6), 1.0)


# --- Canvas placement ----------------------------------------------------------


def _place_card_on_canvas(
    card_rgba: np.ndarray,
    dx_pct: float,
    dy_pct: float,
    scale: float,
    opacity: float,
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """把已经渲染好的静态卡片图像 (RGBA) 按给定的水平/垂直偏移比例（相对画
    布宽高，正值向右/下）、缩放倍数（围绕卡片中心）、不透明度，贴到一张透
    明画布上，返回画布的 RGBA 数组。"""
    card_h, card_w = card_rgba.shape[:2]
    scale = max(scale, 0.01)
    opacity = min(max(opacity, 0.0), 1.0)

    new_w, new_h = max(int(card_w * scale), 1), max(int(card_h * scale), 1)
    image = Image.fromarray(card_rgba)
    if (new_w, new_h) != (card_w, card_h):
        image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    if opacity < 1.0:
        alpha = image.getchannel("A").point(lambda a: int(a * opacity))
        image.putalpha(alpha)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    offset_x = int((canvas_w - new_w) / 2 + dx_pct * canvas_w)
    offset_y = int((canvas_h - new_h) / 2 + dy_pct * canvas_h)
    canvas.alpha_composite(image, (offset_x, offset_y))
    return np.asarray(canvas)


# --- Effect wrappers -------------------------------------------------------------

RigidTransformFn = Callable[
    [float, float, float, float], Tuple[float, float, float, float]
]
CustomFrameFn = Callable[[np.ndarray, float, float, float, float], np.ndarray]

# 画布比卡片本身宽/高出这么多倍，给滑入/滑出等超出卡片本身边界的位移动画
# 留出空间；数值是经验值，具体观感由各效果自己的位移幅度决定。
_CANVAS_MARGIN_RATIO = 2.0


def _make_rigid_effect(transform_fn: RigidTransformFn):
    """把 transform_fn(t, enter_duration, hold_duration, exit_duration) ->
    (dx_pct, dy_pct, scale, opacity) 包装成完整的动画 Clip 构造函数。用于只
    需要整体移动/缩放/淡入淡出的效果，不需要重绘卡片内部内容。"""

    def build(
        card_rgba: np.ndarray,
        enter_duration: float,
        hold_duration: float,
        exit_duration: float,
    ) -> Clip:
        card_h, card_w = card_rgba.shape[:2]
        canvas_w = int(card_w * _CANVAS_MARGIN_RATIO)
        canvas_h = int(card_h * _CANVAS_MARGIN_RATIO)
        total_duration = enter_duration + hold_duration + exit_duration

        def make_frame(t):
            dx, dy, scale, _opacity = transform_fn(
                t, enter_duration, hold_duration, exit_duration
            )
            frame = _place_card_on_canvas(
                card_rgba, dx, dy, scale, 1.0, canvas_w, canvas_h
            )
            return frame[:, :, :3]

        def make_mask(t):
            dx, dy, scale, opacity = transform_fn(
                t, enter_duration, hold_duration, exit_duration
            )
            frame = _place_card_on_canvas(
                card_rgba, dx, dy, scale, opacity, canvas_w, canvas_h
            )
            return frame[:, :, 3].astype(np.float64) / 255.0

        clip = VideoClip(make_frame, duration=total_duration)
        mask_clip = VideoClip(make_mask, duration=total_duration, is_mask=True)
        return clip.with_mask(mask_clip)

    return build


def _make_custom_frame_effect(frame_fn: CustomFrameFn):
    """把 frame_fn(card_rgba, t, enter_duration, hold_duration, exit_duration)
    -> RGBA ndarray 包装成完整的动画 Clip 构造函数。用于需要按时间重绘/遮罩
    卡片内部内容的效果（例如打字机、故障闪烁）——无法只用整体位置/缩放/
    不透明度描述。frame_fn 返回的数组尺寸必须和 card_rgba 一致。"""

    def build(
        card_rgba: np.ndarray,
        enter_duration: float,
        hold_duration: float,
        exit_duration: float,
    ) -> Clip:
        total_duration = enter_duration + hold_duration + exit_duration

        def make_frame(t):
            frame = frame_fn(card_rgba, t, enter_duration, hold_duration, exit_duration)
            return frame[:, :, :3]

        def make_mask(t):
            frame = frame_fn(card_rgba, t, enter_duration, hold_duration, exit_duration)
            return frame[:, :, 3].astype(np.float64) / 255.0

        clip = VideoClip(make_frame, duration=total_duration)
        mask_clip = VideoClip(make_mask, duration=total_duration, is_mask=True)
        return clip.with_mask(mask_clip)

    return build


# --- Card box rendering (dùng chung cho mọi mẫu thẻ trong _styles.py) -----------

# Spec yêu cầu chiều rộng thẻ tối đa = 80% chiều rộng khung hình thật (thay
# đổi theo video_width, ví dụ 864px cho portrait 1080, 1536px cho landscape
# 1920). Ở Plan A dùng cố định 640px làm giới hạn wrap nội bộ, đơn giản hóa
# có chủ đích: 640px luôn nhỏ hơn 80% mọi độ phân giải MPT hỗ trợ (portrait/
# square/landscape), nên thẻ KHÔNG BAO GIỜ vượt quá giới hạn 80% (an toàn,
# không tràn khung hình) — chỉ khác là thẻ không tự phóng to hết cỡ trên
# khung hình rộng (16:9). Nếu cần thẻ to hơn tương ứng theo từng độ phân giải,
# đó là việc của Plan B (thêm tham số max_text_width cho render_card_clip),
# không phải bug cần chặn Plan A.
_CARD_PADDING = 24
_CARD_MAX_TEXT_WIDTH = 640
_CARD_LINE_SPACING = 8


def _wrap_text_to_width(
    text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> list:
    """按像素宽度换行（用实际要渲染的字体测量，不是猜字符数）——和
    video.py 给字幕换行用的思路一致。"""
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = font.getbbox(candidate)
        width = bbox[2] - bbox[0]
        if width > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return lines


def render_card_box(
    text: str,
    *,
    font_path: str,
    font_size: int,
    text_color: Tuple[int, int, int, int],
    background_color: Tuple[int, int, int, int],
    border_color: Optional[Tuple[int, int, int, int]],
    border_width: int,
    corner_radius: int,
) -> np.ndarray:
    """画一张静态卡片（背景+边框+已自动换行并居中的文字），返回 RGBA
    数组。_styles.py 里的 15 个模板都只需要传入不同的颜色/字体/边框参数
    调用这一个函数，不用各自重复实现换行、居中、画框的逻辑。"""
    font = ImageFont.truetype(font_path, font_size)
    lines = _wrap_text_to_width(text, font, _CARD_MAX_TEXT_WIDTH - 2 * _CARD_PADDING)

    line_widths = []
    line_heights = []
    for line in lines:
        bbox = font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    text_block_width = max(line_widths) if line_widths else 0
    text_block_height = sum(line_heights) + _CARD_LINE_SPACING * (len(lines) - 1)

    card_width = int(text_block_width + 2 * _CARD_PADDING)
    card_height = int(text_block_height + 2 * _CARD_PADDING)

    image = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, card_width - 1, card_height - 1),
        radius=corner_radius,
        fill=background_color,
        outline=border_color,
        width=border_width if border_color else 0,
    )

    y = _CARD_PADDING
    for line, line_w, line_h in zip(lines, line_widths, line_heights):
        x = (card_width - line_w) / 2
        draw.text((x, y), line, font=font, fill=text_color)
        y += line_h + _CARD_LINE_SPACING

    return np.asarray(image)
