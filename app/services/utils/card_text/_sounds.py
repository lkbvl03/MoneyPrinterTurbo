# app/services/utils/card_text/_sounds.py
from typing import Callable, Dict, Optional

import numpy as np

SAMPLE_RATE = 44100


def _envelope_linear(
    num_samples: int, attack_ratio: float = 0.1, release_ratio: float = 0.6
) -> np.ndarray:
    """线性 attack-sustain-release 包络，避免声音突然开始/结束产生咔哒声。"""
    attack_len = max(int(num_samples * attack_ratio), 1)
    release_len = max(int(num_samples * release_ratio), 1)
    sustain_len = max(num_samples - attack_len - release_len, 0)
    envelope = np.concatenate(
        [
            np.linspace(0.0, 1.0, attack_len),
            np.ones(sustain_len),
            np.linspace(1.0, 0.0, release_len),
        ]
    )
    if len(envelope) < num_samples:
        envelope = np.pad(envelope, (0, num_samples - len(envelope)))
    return envelope[:num_samples]


def whoosh(duration: float) -> np.ndarray:
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    freq_sweep = np.linspace(2000, 200, num_samples)
    phase = np.cumsum(2 * np.pi * freq_sweep / SAMPLE_RATE)
    tone = np.sin(phase)
    # 固定种子保证纯函数（相同 duration 输入必须产出相同波形）。
    noise = np.random.default_rng(0).uniform(-1, 1, num_samples)
    signal = 0.6 * tone + 0.4 * noise
    envelope = _envelope_linear(num_samples, attack_ratio=0.15, release_ratio=0.55)
    return (signal * envelope * 0.5).astype(np.float32)


def boing(duration: float) -> np.ndarray:
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    pitch_curve = 300 + 250 * np.exp(-6 * t / max(duration, 1e-6))
    phase = np.cumsum(2 * np.pi * pitch_curve / SAMPLE_RATE)
    tone = np.sin(phase)
    envelope = _envelope_linear(num_samples, attack_ratio=0.05, release_ratio=0.7)
    return (tone * envelope * 0.6).astype(np.float32)


def sparkle_chime(duration: float) -> np.ndarray:
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    frequencies = (1200, 1600, 2000, 2600)
    signal = np.zeros(num_samples)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    signal /= len(frequencies)
    envelope = _envelope_linear(num_samples, attack_ratio=0.02, release_ratio=0.8)
    return (signal * envelope * 0.4).astype(np.float32)


CARD_SOUNDS: Dict[str, Callable[[float], np.ndarray]] = {
    "whoosh": whoosh,
    "boing": boing,
    "sparkle_chime": sparkle_chime,
}

# 尚未在此表中拥有默认音效的效果组（如 "rotation"、"text_reveal"）会在
# resolve_sound_name 中回退到 CARD_SOUNDS 里的第一个音效 —— Plan B 补齐全部
# 10 个音效后将覆盖全部 6 个效果组，届时该回退分支在实际使用中不会再被触发，
# 但仍保留作为安全网。
_GROUP_DEFAULT_SOUND: Dict[str, str] = {
    "motion": "whoosh",
    "bounce": "boing",
    "light": "sparkle_chime",
}


def resolve_sound_name(sound: Optional[str], effect_group: str) -> Optional[str]:
    if sound == "none":
        return None
    if sound is None or sound == "auto":
        default_name = _GROUP_DEFAULT_SOUND.get(effect_group)
        if default_name is not None:
            return default_name
        return next(iter(CARD_SOUNDS))
    return sound
