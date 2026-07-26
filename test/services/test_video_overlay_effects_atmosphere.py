# test/services/test_video_overlay_effects_atmosphere.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moviepy import ImageClip

from app.services.utils.video_overlay_effects import OVERLAY_EFFECTS, apply_overlay_effect
from app.services.utils.video_overlay_effects._atmosphere import CATEGORY_EFFECTS, _EMBERS_RISING
from app.services.utils.video_overlay_effects import _primitives as p

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
                self.assertEqual(frame.dtype, np.uint8)

    def test_embers_move_upward_over_time(self):
        # Verify that embers actually move upward (toward row 0) over time
        # by comparing alpha-weighted centroid row position at t=0.0 and t=0.15.
        # Use small time interval (max displacement ~24px) to avoid wrapping with height=60.
        width, height = 80, 60
        size = (width, height)

        # Get overlay layers at early and late times
        # With fall_speed_range=(-160, -90) and t=0.15, displacement is -24 to -13.5 pixels,
        # which is well within the canvas before wrapping occurs.
        overlay_rgb_0, overlay_alpha_0 = p._particle_field_layer(0.0, size, _EMBERS_RISING)
        overlay_rgb_1, overlay_alpha_1 = p._particle_field_layer(0.15, size, _EMBERS_RISING)

        # Compute alpha-weighted mean row index for each time
        def compute_centroid_row(alpha):
            total_alpha = alpha.sum(axis=1)  # sum alpha across width for each row
            if total_alpha.sum() == 0:
                return None
            return np.average(np.arange(height), weights=total_alpha)

        centroid_row_0 = compute_centroid_row(overlay_alpha_0)
        centroid_row_1 = compute_centroid_row(overlay_alpha_1)

        self.assertIsNotNone(centroid_row_0, "Embers should have nonzero alpha at t=0.0")
        self.assertIsNotNone(centroid_row_1, "Embers should have nonzero alpha at t=0.15")

        # Embers move upward => centroid row decreases (lower row index = higher up)
        self.assertLess(
            centroid_row_1, centroid_row_0,
            f"Embers should move upward: centroid_row at t=0.15 ({centroid_row_1:.1f}) "
            f"should be less than at t=0.0 ({centroid_row_0:.1f})"
        )


if __name__ == "__main__":
    unittest.main()
