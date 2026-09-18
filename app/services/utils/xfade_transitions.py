import random
from typing import List, Tuple

# Validated ffmpeg xfade transition catalog (ffmpeg >= 4.3). Do not add, remove,
# or rename entries without re-verifying against a real ffmpeg build -- an
# invalid name fails the whole render with "No such transition".
XFADE_TRANSITIONS: List[str] = [
    "fade", "fadeblack", "fadewhite", "fadegrays",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circlecrop", "circleopen", "circleclose",
    "rectcrop",
    "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "zoomin", "squeezeh", "squeezev",
    "dissolve", "pixelize", "radial", "hblur", "distance",
    "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
    "horzopen", "horzclose", "vertopen", "vertclose",
]

_MINIMUM_TRANSITION_DURATION = 0.1
_TRANSITION_DURATION_SAFETY_RATIO = 0.4


def resolve_transition_name(style: str, *, random_choice=None) -> str:
    if random_choice is None:
        random_choice = random.choice
    if style == "random":
        return random_choice(XFADE_TRANSITIONS)
    return style


def compute_transition_durations(
    clip_durations: List[float], requested_duration: float
) -> List[float]:
    durations = []
    for left, right in zip(clip_durations, clip_durations[1:]):
        shorter = min(left, right)
        clamped = min(requested_duration, shorter * _TRANSITION_DURATION_SAFETY_RATIO)
        durations.append(max(clamped, _MINIMUM_TRANSITION_DURATION))
    return durations


def compute_xfade_offsets(
    clip_durations: List[float], transition_durations: List[float]
) -> List[float]:
    offsets = []
    running = clip_durations[0] - transition_durations[0]
    offsets.append(running)
    for index in range(1, len(transition_durations)):
        running = running + clip_durations[index] - transition_durations[index]
        offsets.append(running)
    return offsets


def build_xfade_filter_complex(
    clip_durations: List[float],
    transition_names: List[str],
    transition_durations: List[float],
) -> Tuple[str, str]:
    if len(clip_durations) < 2:
        raise ValueError("build_xfade_filter_complex requires at least 2 clips")

    offsets = compute_xfade_offsets(clip_durations, transition_durations)
    filters = []
    previous_label = "[0:v]"
    last_index = len(transition_names) - 1
    for index, (name, duration, offset) in enumerate(
        zip(transition_names, transition_durations, offsets)
    ):
        next_input = f"[{index + 1}:v]"
        output_label = "[outv]" if index == last_index else f"[v{index + 1}]"
        filters.append(
            f"{previous_label}{next_input}xfade=transition={name}:"
            f"duration={duration:.3f}:offset={offset:.3f}{output_label}"
        )
        previous_label = output_label

    return ";".join(filters), "[outv]"


def build_xfade_audio_filter_complex(
    transition_durations: List[float],
) -> Tuple[str, str]:
    """构造与 build_xfade_filter_complex 对应的音频交叉淡化链。

    ffmpeg 的 acrossfade 语义上和 xfade 一致（在两段衔接处按给定时长互相
    淡入淡出），只是不需要单独的 offset 参数——它总是用前一段的结尾和
    后一段的开头做交叉。只要用同一组 transition_durations，就能让声音的
    过渡节奏和画面转场精确对上。
    """
    if not transition_durations:
        raise ValueError(
            "build_xfade_audio_filter_complex requires at least 1 transition"
        )

    filters = []
    previous_label = "[0:a]"
    last_index = len(transition_durations) - 1
    for index, duration in enumerate(transition_durations):
        next_input = f"[{index + 1}:a]"
        output_label = "[outa]" if index == last_index else f"[a{index + 1}]"
        filters.append(
            f"{previous_label}{next_input}acrossfade=d={duration:.3f}{output_label}"
        )
        previous_label = output_label

    return ";".join(filters), "[outa]"
