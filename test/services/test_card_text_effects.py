# test/services/test_card_text_effects.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.utils.card_text._effects_bounce import (
    CATEGORY_EFFECTS as BOUNCE_EFFECTS,
    GROUP_NAME as BOUNCE_GROUP,
)
from app.services.utils.card_text._effects_motion import (
    CATEGORY_EFFECTS as MOTION_EFFECTS,
    GROUP_NAME as MOTION_GROUP,
)
from app.services.utils.card_text._effects_rotation import (
    CATEGORY_EFFECTS as ROTATION_EFFECTS,
    GROUP_NAME as ROTATION_GROUP,
)


def _solid_card(width=100, height=50):
    card = np.zeros((height, width, 4), dtype=np.uint8)
    card[:, :] = (255, 0, 0, 255)
    return card


class TestMotionEffects(unittest.TestCase):
    def test_group_name(self):
        self.assertEqual(MOTION_GROUP, "motion")

    def test_slide_left_registered(self):
        self.assertIn("slide_left", MOTION_EFFECTS)

    def test_slide_left_produces_valid_clip(self):
        card = _solid_card()
        clip = MOTION_EFFECTS["slide_left"](card, 0.3, 1.0, 0.3)
        self.addCleanup(clip.close)
        self.assertAlmostEqual(clip.duration, 1.6)
        frame = clip.get_frame(0.8)
        self.assertEqual(frame.shape[2], 3)

    def test_slide_left_is_more_visible_mid_hold_than_at_the_very_start(self):
        card = _solid_card()
        clip = MOTION_EFFECTS["slide_left"](card, 0.3, 1.0, 0.3)
        self.addCleanup(clip.close)
        start_mask = clip.mask.get_frame(0.0)
        mid_mask = clip.mask.get_frame(0.9)
        self.assertLess(start_mask.sum(), mid_mask.sum())


class TestBounceEffects(unittest.TestCase):
    def test_group_name(self):
        self.assertEqual(BOUNCE_GROUP, "bounce")

    def test_bounce_registered(self):
        self.assertIn("bounce", BOUNCE_EFFECTS)

    def test_bounce_produces_valid_clip(self):
        card = _solid_card()
        clip = BOUNCE_EFFECTS["bounce"](card, 0.3, 1.0, 0.3)
        self.addCleanup(clip.close)
        self.assertAlmostEqual(clip.duration, 1.6)


class TestRotationEffects(unittest.TestCase):
    def test_group_name(self):
        self.assertEqual(ROTATION_GROUP, "rotation")

    def test_rotate_in_registered(self):
        self.assertIn("rotate_in", ROTATION_EFFECTS)

    def test_rotate_in_produces_valid_clip_with_matching_frame_shape(self):
        card = _solid_card(120, 60)
        clip = ROTATION_EFFECTS["rotate_in"](card, 0.3, 1.0, 0.3)
        self.addCleanup(clip.close)
        frame = clip.get_frame(0.9)
        self.assertEqual(frame.shape[:2], (60, 120))


if __name__ == "__main__":
    unittest.main()
