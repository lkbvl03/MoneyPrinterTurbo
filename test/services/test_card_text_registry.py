import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.utils import card_text as ct

_STYLE_NAMES = ["minimal_white", "bold_yellow_box", "gradient_pop", "sticky_note", "comic_burst"]
_EFFECT_NAMES = ["slide_left", "bounce", "rotate_in", "typewriter", "glow_pulse"]


class TestCatalogRegistration(unittest.TestCase):
    def test_all_five_styles_registered(self):
        for name in _STYLE_NAMES:
            self.assertIn(name, ct.CARD_STYLES)

    def test_all_five_effects_registered(self):
        for name in _EFFECT_NAMES:
            self.assertIn(name, ct.CARD_EFFECTS)

    def test_effect_groups_mapped_correctly(self):
        self.assertEqual(ct.EFFECT_GROUPS["slide_left"], "motion")
        self.assertEqual(ct.EFFECT_GROUPS["bounce"], "bounce")
        self.assertEqual(ct.EFFECT_GROUPS["rotate_in"], "rotation")
        self.assertEqual(ct.EFFECT_GROUPS["typewriter"], "text_reveal")
        self.assertEqual(ct.EFFECT_GROUPS["glow_pulse"], "light")


class TestResolveStyleName(unittest.TestCase):
    def test_none_and_random_call_random_choice(self):
        with patch.object(ct, "CARD_STYLES", {"a": lambda t: None}):
            self.assertEqual(
                ct.resolve_style_name("random", random_choice=lambda seq: list(seq)[0]), "a"
            )
            self.assertEqual(
                ct.resolve_style_name(None, random_choice=lambda seq: list(seq)[0]), "a"
            )

    def test_fixed_name_returned_unchanged(self):
        self.assertEqual(ct.resolve_style_name("minimal_white"), "minimal_white")


class TestResolveEffectName(unittest.TestCase):
    def test_none_and_random_call_random_choice(self):
        with patch.object(ct, "CARD_EFFECTS", {"a": lambda *args: None}):
            self.assertEqual(
                ct.resolve_effect_name("random", random_choice=lambda seq: list(seq)[0]), "a"
            )
            self.assertEqual(
                ct.resolve_effect_name(None, random_choice=lambda seq: list(seq)[0]), "a"
            )

    def test_fixed_name_returned_unchanged(self):
        self.assertEqual(ct.resolve_effect_name("bounce"), "bounce")


class TestRenderCardClip(unittest.TestCase):
    def test_builds_a_working_clip_for_every_style_effect_combination(self):
        for style_name in _STYLE_NAMES:
            for effect_name in _EFFECT_NAMES:
                with self.subTest(style=style_name, effect=effect_name):
                    clip = ct.render_card_clip("Kiểm thử", style_name, effect_name)
                    self.addCleanup(clip.close)
                    self.assertGreater(clip.duration, 0)

    def test_unknown_style_raises_key_error(self):
        with self.assertRaises(KeyError):
            ct.render_card_clip("Test", "khong_ton_tai", "bounce")

    def test_unknown_effect_raises_key_error(self):
        with self.assertRaises(KeyError):
            ct.render_card_clip("Test", "minimal_white", "khong_ton_tai")

    def test_custom_duration_is_respected(self):
        clip = ct.render_card_clip("Test", "minimal_white", "bounce", duration=1.0)
        self.addCleanup(clip.close)
        self.assertAlmostEqual(clip.duration, 1.0, places=2)

    def test_font_name_and_font_size_are_forwarded_to_the_style(self):
        captured = {}

        def fake_style(text, font_name=None, font_size=None):
            captured["args"] = (text, font_name, font_size)
            import numpy as np

            return np.zeros((10, 10, 4), dtype=np.uint8)

        with patch.object(ct, "CARD_STYLES", {"fake": fake_style}):
            clip = ct.render_card_clip(
                "Test",
                "fake",
                "bounce",
                font_name="BeVietnamPro-Bold.ttf",
                font_size=64,
            )
        self.addCleanup(clip.close)
        self.assertEqual(captured["args"], ("Test", "BeVietnamPro-Bold.ttf", 64))


if __name__ == "__main__":
    unittest.main()
