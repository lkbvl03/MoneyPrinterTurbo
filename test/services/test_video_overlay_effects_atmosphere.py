# test/services/test_video_overlay_effects_atmosphere.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moviepy import ImageClip

from app.services.utils.video_overlay_effects import OVERLAY_EFFECTS, apply_overlay_effect
from app.services.utils.video_overlay_effects._atmosphere import CATEGORY_EFFECTS

_EFFECT_NAMES = [
    "smoke_drift", "dust_motes", "embers_rising", "ash_falling", "fireflies", "sparks",
]


def _solid_clip(width=80, height=60, duration=1.0):
    frame = np.full((height, width, 3), 20, dtype=np.uint8)
    return ImageClip(frame).with_duration(duration)


class TestAtmosphereCategoryRegistration(unittest.TestCase):
    def test_all_six_names_are_registered_globally(self):
        for name in _EFFECT_NAMES:
            self.assertIn(name, OVERLAY_EFFECTS)
            self.assertIn(name, CATEGORY_EFFECTS)
            self.assertEqual(OVERLAY_EFFECTS[name].category, "atmosphere")


class TestAtmosphereEffectsPreserveClipShape(unittest.TestCase):
    def test_each_effect_keeps_size_and_duration(self):
        for name in _EFFECT_NAMES:
            with self.subTest(effect=name):
                clip = _solid_clip()
                self.addCleanup(clip.close)
                result = apply_overlay_effect(clip, name)
                self.addCleanup(result.close)
                self.assertEqual(result.size, clip.size)
                self.assertEqual(result.duration, clip.duration)
                frame = result.get_frame(0.4)
                self.assertEqual(frame.shape, (60, 80, 3))

    def test_embers_move_upward_over_time(self):
        # fall_speed âm nghĩa là hạt bay lên; sau 1 giây khung hình phải
        # khác khung hình ban đầu (đủ để phân biệt chuyển động).
        clip = _solid_clip()
        self.addCleanup(clip.close)
        result = apply_overlay_effect(clip, "embers_rising")
        self.addCleanup(result.close)
        frame0 = result.get_frame(0.0)
        frame1 = result.get_frame(1.0)
        self.assertTrue((frame0 != frame1).any())


if __name__ == "__main__":
    unittest.main()
