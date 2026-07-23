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
