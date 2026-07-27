# test/services/test_video_overlay_effects_camera.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moviepy import ImageClip

from app.services.utils.video_overlay_effects import OVERLAY_EFFECTS, apply_overlay_effect
from app.services.utils.video_overlay_effects._camera import CATEGORY_EFFECTS

_EFFECT_NAMES = ["zoom_in", "zoom_out", "pan_left", "pan_right", "heat_haze"]


def _gradient_clip(width=80, height=60, duration=1.0):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    frame = np.stack(np.meshgrid(x, y), axis=-1).sum(axis=-1) % 256
    rgb = np.stack([frame] * 3, axis=-1).astype(np.uint8)
    return ImageClip(rgb).with_duration(duration)


class TestCameraCategoryRegistration(unittest.TestCase):
    def test_all_five_names_are_registered_globally(self):
        for name in _EFFECT_NAMES:
            self.assertIn(name, OVERLAY_EFFECTS)
            self.assertIn(name, CATEGORY_EFFECTS)
            self.assertEqual(OVERLAY_EFFECTS[name].category, "camera")

    def test_full_catalog_has_at_least_38_effects(self):
        self.assertGreaterEqual(len(OVERLAY_EFFECTS), 38)


class TestCameraEffectsPreserveClipShape(unittest.TestCase):
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


class TestHeatHaze(unittest.TestCase):
    def test_visibly_distorts_the_frame(self):
        clip = _gradient_clip()
        self.addCleanup(clip.close)
        result = apply_overlay_effect(clip, "heat_haze")
        self.addCleanup(result.close)
        base = clip.get_frame(0.5)
        out = result.get_frame(0.5)
        self.assertTrue((base != out).any())


if __name__ == "__main__":
    unittest.main()
