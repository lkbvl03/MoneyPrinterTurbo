# test/services/test_video_overlay_effects_fire.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moviepy import ImageClip

from app.services.utils.video_overlay_effects import OVERLAY_EFFECTS, apply_overlay_effect
from app.services.utils.video_overlay_effects._fire import CATEGORY_EFFECTS, _lightning_branch_layer_fn

_EFFECT_NAMES = [
    "fire_glow", "lightning_flash", "lightning_branch", "campfire_flicker", "plasma_pulse",
]


def _solid_clip(width=80, height=60, duration=1.0):
    frame = np.full((height, width, 3), 15, dtype=np.uint8)
    return ImageClip(frame).with_duration(duration)


class TestFireCategoryRegistration(unittest.TestCase):
    def test_all_five_names_are_registered_globally(self):
        for name in _EFFECT_NAMES:
            self.assertIn(name, OVERLAY_EFFECTS)
            self.assertIn(name, CATEGORY_EFFECTS)
            self.assertEqual(OVERLAY_EFFECTS[name].category, "fire")


class TestFireEffectsPreserveClipShape(unittest.TestCase):
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


class TestLightningFlash(unittest.TestCase):
    def test_flash_peak_is_brighter_than_quiet_period(self):
        clip = _solid_clip()
        self.addCleanup(clip.close)
        result = apply_overlay_effect(clip, "lightning_flash")
        self.addCleanup(result.close)
        peak_frame = result.get_frame(0.0)
        quiet_frame = result.get_frame(1.0)
        self.assertGreater(int(peak_frame[0, 0, 0]), int(quiet_frame[0, 0, 0]))


class TestLightningBranch(unittest.TestCase):
    def test_shape_matches_frame_size(self):
        rgb, alpha = _lightning_branch_layer_fn(0.0, (80, 60))
        self.assertEqual(rgb.shape, (60, 80, 3))
        self.assertEqual(alpha.shape, (60, 80))

    def test_same_flash_window_uses_the_same_branch_shape(self):
        _, alpha_a = _lightning_branch_layer_fn(0.01, (80, 60))
        _, alpha_b = _lightning_branch_layer_fn(0.02, (80, 60))
        # Cùng trong 1 cửa sổ chớp (local_t < 0.25) -> vị trí đường sét
        # (khác 0) phải giống hệt nhau, chỉ độ sáng (strength) khác.
        mask_a = alpha_a > 0
        mask_b = alpha_b > 0
        np.testing.assert_array_equal(mask_a, mask_b)

    def test_no_branch_outside_the_flash_window(self):
        _, alpha = _lightning_branch_layer_fn(2.0, (80, 60))
        self.assertEqual(alpha.max(), 0.0)


if __name__ == "__main__":
    unittest.main()
