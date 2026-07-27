# test/services/test_video_overlay_effects_retro.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moviepy import ImageClip

from app.services.utils.video_overlay_effects import OVERLAY_EFFECTS, apply_overlay_effect
from app.services.utils.video_overlay_effects._retro import CATEGORY_EFFECTS

_EFFECT_NAMES = [
    "film_grain", "film_scratches", "vignette_pulse", "vhs_glitch",
    "chromatic_aberration", "film_burn_edges", "double_exposure_ghost",
    "static_noise_overlay", "color_drift_flicker", "motion_streak_blur",
]


def _gradient_clip(width=80, height=60, duration=1.0):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    frame = np.stack(np.meshgrid(x, y), axis=-1).sum(axis=-1) % 256
    rgb = np.stack([frame] * 3, axis=-1).astype(np.uint8)
    return ImageClip(rgb).with_duration(duration)


class TestRetroCategoryRegistration(unittest.TestCase):
    def test_all_ten_names_are_registered_globally(self):
        for name in _EFFECT_NAMES:
            self.assertIn(name, OVERLAY_EFFECTS)
            self.assertIn(name, CATEGORY_EFFECTS)
            self.assertEqual(OVERLAY_EFFECTS[name].category, "retro")


class TestRetroEffectsPreserveClipShape(unittest.TestCase):
    def test_each_effect_keeps_size_and_duration(self):
        for name in _EFFECT_NAMES:
            with self.subTest(effect=name):
                clip = _gradient_clip()
                self.addCleanup(clip.close)
                result = apply_overlay_effect(clip, name)
                self.addCleanup(result.close)
                self.assertEqual(result.size, clip.size)
                self.assertEqual(result.duration, clip.duration)
                frame = result.get_frame(0.4)
                self.assertEqual(frame.shape, (60, 80, 3))
                self.assertEqual(frame.dtype, np.uint8)

    def test_each_effect_visibly_changes_pixels(self):
        for name in _EFFECT_NAMES:
            with self.subTest(effect=name):
                clip = _gradient_clip()
                self.addCleanup(clip.close)
                result = apply_overlay_effect(clip, name)
                self.addCleanup(result.close)
                base = clip.get_frame(0.5)
                out = result.get_frame(0.5)
                self.assertTrue((base != out).any())


class TestChromaticAberration(unittest.TestCase):
    def test_red_and_blue_channels_shift_in_opposite_directions(self):
        clip = _gradient_clip()
        self.addCleanup(clip.close)
        result = apply_overlay_effect(clip, "chromatic_aberration")
        self.addCleanup(result.close)
        base = clip.get_frame(0.0)
        out = result.get_frame(0.0)
        np.testing.assert_array_equal(out[:, :, 0], np.roll(base[:, :, 0], 3, axis=1))
        np.testing.assert_array_equal(out[:, :, 2], np.roll(base[:, :, 2], -3, axis=1))


class TestVignettePulse(unittest.TestCase):
    def test_corners_are_darker_than_center(self):
        clip = _gradient_clip(width=100, height=100)
        self.addCleanup(clip.close)
        result = apply_overlay_effect(clip, "vignette_pulse")
        self.addCleanup(result.close)
        frame = result.get_frame(0.0)
        base = clip.get_frame(0.0)
        corner_dim = int(base[0, 0, 0]) - int(frame[0, 0, 0])
        center_dim = int(base[50, 50, 0]) - int(frame[50, 50, 0])
        self.assertGreaterEqual(corner_dim, center_dim)


if __name__ == "__main__":
    unittest.main()
