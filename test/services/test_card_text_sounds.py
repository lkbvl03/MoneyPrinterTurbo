# test/services/test_card_text_sounds.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.utils.card_text._sounds import CARD_SOUNDS, SAMPLE_RATE, resolve_sound_name

_SOUND_NAMES = ["whoosh", "boing", "sparkle_chime"]


class TestCardSounds(unittest.TestCase):
    def test_all_three_mvp_sounds_registered(self):
        for name in _SOUND_NAMES:
            self.assertIn(name, CARD_SOUNDS)

    def test_each_sound_returns_correct_length_array(self):
        for name in _SOUND_NAMES:
            with self.subTest(sound=name):
                result = CARD_SOUNDS[name](0.5)
                expected_len = int(0.5 * SAMPLE_RATE)
                self.assertEqual(len(result), expected_len)

    def test_each_sound_stays_within_valid_amplitude_range(self):
        for name in _SOUND_NAMES:
            with self.subTest(sound=name):
                result = CARD_SOUNDS[name](0.5)
                self.assertTrue(np.all(np.isfinite(result)))
                self.assertLessEqual(np.abs(result).max(), 1.0)

    def test_sound_is_deterministic_across_calls(self):
        for name in _SOUND_NAMES:
            with self.subTest(sound=name):
                first = CARD_SOUNDS[name](0.4)
                second = CARD_SOUNDS[name](0.4)
                np.testing.assert_array_equal(first, second)


class TestResolveSoundName(unittest.TestCase):
    def test_none_string_returns_none(self):
        self.assertIsNone(resolve_sound_name("none", "motion"))

    def test_auto_matches_group_default(self):
        self.assertEqual(resolve_sound_name("auto", "motion"), "whoosh")
        self.assertEqual(resolve_sound_name("auto", "bounce"), "boing")
        self.assertEqual(resolve_sound_name("auto", "light"), "sparkle_chime")

    def test_none_value_treated_same_as_auto(self):
        self.assertEqual(resolve_sound_name(None, "motion"), "whoosh")

    def test_auto_for_group_without_default_falls_back_to_first_sound(self):
        result = resolve_sound_name("auto", "rotation")
        self.assertEqual(result, next(iter(CARD_SOUNDS)))

    def test_specific_name_returned_unchanged_regardless_of_group(self):
        self.assertEqual(resolve_sound_name("boing", "motion"), "boing")


if __name__ == "__main__":
    unittest.main()
