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

        # Get frames at start and halfway through the period (5.0s)
        # Formula: cx = width * progress where progress = (t % period) / period
        # At t=0.0: progress=0, cx=0 (left edge)
        # At t=2.5: progress=0.5, cx=width/2 (center)
        frame_a = result.get_frame(0.0)
        frame_b = result.get_frame(2.5)

        # Find the brightest column index in each frame by summing brightness
        # across all rows and channels for each column, then finding the argmax.
        brightness_per_column_a = frame_a.sum(axis=(0, 2))
        brightness_per_column_b = frame_b.sum(axis=(0, 2))

        brightest_col_a = np.argmax(brightness_per_column_a)
        brightest_col_b = np.argmax(brightness_per_column_b)

        # The spot should move right (increasing column index) as time increases within the period
        self.assertGreater(brightest_col_b, brightest_col_a,
                          f"Lens flare should move right: brightest column at t=0.0 was {brightest_col_a}, "
                          f"at t=2.5 was {brightest_col_b}")


if __name__ == "__main__":
    unittest.main()
