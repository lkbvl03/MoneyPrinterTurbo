# test/services/test_video_overlay_effects_light.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moviepy import ImageClip

from app.services.utils.video_overlay_effects import OVERLAY_EFFECTS, apply_overlay_effect
from app.services.utils.video_overlay_effects._light import CATEGORY_EFFECTS

_EFFECT_NAMES = [
    "lens_flare_sweep", "light_leak", "god_rays", "bokeh_lights", "sun_pulse", "golden_hour_glow",
]


def _solid_clip(width=80, height=60, duration=1.0):
    frame = np.full((height, width, 3), 25, dtype=np.uint8)
    return ImageClip(frame).with_duration(duration)


class TestLightCategoryRegistration(unittest.TestCase):
    def test_all_six_names_are_registered_globally(self):
        for name in _EFFECT_NAMES:
            self.assertIn(name, OVERLAY_EFFECTS)
            self.assertIn(name, CATEGORY_EFFECTS)
            self.assertEqual(OVERLAY_EFFECTS[name].category, "light")


class TestLightEffectsPreserveClipShape(unittest.TestCase):
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

    def test_all_effects_brighten_a_dark_frame_since_they_blend_screen(self):
        for name in _EFFECT_NAMES:
            with self.subTest(effect=name):
                clip = _solid_clip()
                self.addCleanup(clip.close)
                result = apply_overlay_effect(clip, name)
                self.addCleanup(result.close)
                base_frame = clip.get_frame(0.5).astype(int)
                out_frame = result.get_frame(0.5).astype(int)
                self.assertTrue((out_frame >= base_frame).all())


class TestLensFlareSweep(unittest.TestCase):
    def test_bright_spot_moves_horizontally_over_time(self):
        clip = _solid_clip()
        self.addCleanup(clip.close)
        result = apply_overlay_effect(clip, "lens_flare_sweep")
        self.addCleanup(result.close)
        frame_a = result.get_frame(0.0)
        frame_b = result.get_frame(2.5)
        self.assertFalse(np.array_equal(frame_a, frame_b))


if __name__ == "__main__":
    unittest.main()
