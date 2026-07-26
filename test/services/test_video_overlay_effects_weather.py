# test/services/test_video_overlay_effects_weather.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moviepy import ImageClip

from app.services.utils.video_overlay_effects import OVERLAY_EFFECTS, apply_overlay_effect
from app.services.utils.video_overlay_effects._weather import CATEGORY_EFFECTS

_EFFECT_NAMES = [
    "rain_light", "rain_heavy", "snow_light", "snow_heavy", "fog_drift", "mist_ground",
]


def _solid_clip(width=80, height=60, duration=1.0):
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    return ImageClip(frame).with_duration(duration)


class TestWeatherCategoryRegistration(unittest.TestCase):
    def test_all_six_names_are_registered_globally(self):
        for name in _EFFECT_NAMES:
            self.assertIn(name, OVERLAY_EFFECTS)
            self.assertIn(name, CATEGORY_EFFECTS)

    def test_all_effects_are_tagged_weather_category(self):
        for name in _EFFECT_NAMES:
            self.assertEqual(OVERLAY_EFFECTS[name].category, "weather")


class TestWeatherEffectsPreserveClipShape(unittest.TestCase):
    def test_each_effect_keeps_size_and_duration(self):
        for name in _EFFECT_NAMES:
            with self.subTest(effect=name):
                clip = _solid_clip()
                self.addCleanup(clip.close)
                result = apply_overlay_effect(clip, name)
                self.addCleanup(result.close)
                self.assertEqual(result.size, clip.size)
                self.assertEqual(result.duration, clip.duration)
                frame = result.get_frame(0.3)
                self.assertEqual(frame.shape, (60, 80, 3))
                self.assertEqual(frame.dtype, np.uint8)

    def test_particle_effects_visibly_change_some_pixels(self):
        for name in ["rain_light", "rain_heavy", "snow_light", "snow_heavy"]:
            with self.subTest(effect=name):
                clip = _solid_clip()
                self.addCleanup(clip.close)
                result = apply_overlay_effect(clip, name)
                self.addCleanup(result.close)
                base_frame = clip.get_frame(0.5)
                out_frame = result.get_frame(0.5)
                self.assertTrue((base_frame != out_frame).any())


if __name__ == "__main__":
    unittest.main()
