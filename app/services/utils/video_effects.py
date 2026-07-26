import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy 内置 SlideIn 在当前这条处理链里对全屏素材不稳定，
    # 会出现“逻辑上应用了转场，但画面几乎看不出变化”的情况。
    # 这里改成显式黑底 + 位移动画，保证转场效果可见且行为可控。
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_start = max(clip.duration - t, 0)

    # SlideOut 同样改成显式位移，保证片段末尾能稳定滑出画面。
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(t, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# 保留原始设计的 20% 缩放幅度，让三秒左右的短片也有清晰可见的 Ken Burns 运动感。
# 缩放稳定性由下方的亚像素中心采样保证，不通过削弱效果幅度来掩盖源视频编码闪烁。
_ZOOM_MAX_SCALE = 1.2


def _crop_frame(
    frame: np.ndarray, left: float, top: float, right: float, bottom: float
) -> np.ndarray:
    """裁剪 (left, top, right, bottom) 区域并放大回原始画布尺寸。共享给
    zoom（居中裁剪）和 pan（水平偏移裁剪）复用，避免重复实现同一套
    PIL EXTENT 采样逻辑。

    裁剪边界必须保持浮点精度，不能舍入到整数再使用——连续的缩放或平移
    变化会导致跨帧在整数边界处跳变，造成边界闪烁。

    采样使用 BILINEAR 而非 BICUBIC/LANCZOS 等更锐利的滤波器——视频需要
    逐帧连续性，锐利滤波器在高频纹理采样栅格连续变化时会产生光晕和亮度
    闪烁。"""
    height, width = frame.shape[:2]
    if (
        abs(left) < 1e-9
        and abs(top) < 1e-9
        and abs(right - width) < 1e-9
        and abs(bottom - height) < 1e-9
    ):
        return frame

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """使用亚像素中心裁剪实现无黑边且稳定的缩放效果，参见 `_crop_frame`。"""
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    return _crop_frame(frame, left, top, left + crop_width, top + crop_height)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """在整个片段内从原始画面平滑放大到 1.2 倍。"""
    # t 暂时保留，用于与其它转场函数保持统一调用签名；缩放需要覆盖完整片段，
    # 否则短暂缩放结束后画面会突然静止，不适合静态或低运动量素材。
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect)


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """在整个片段内从 1.2 倍平滑缩小到原始画面。"""
    # 与 zoomin_transition 一致，t 仅用于兼容统一的转场调用接口。
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect)
