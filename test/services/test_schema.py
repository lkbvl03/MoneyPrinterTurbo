import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pydantic

from app.models.schema import VideoAspect, VideoParams


class TestVideoAspect(unittest.TestCase):
    def test_to_resolution_known_aspects(self):
        self.assertEqual(VideoAspect.landscape.to_resolution(), (1920, 1080))
        self.assertEqual(VideoAspect.portrait.to_resolution(), (1080, 1920))
        self.assertEqual(VideoAspect.square.to_resolution(), (1080, 1080))

    def test_to_resolution_rejects_unsupported_value(self):
        with self.assertRaises(ValueError):
            VideoAspect.to_resolution("4:5")


class TestVideoParams(unittest.TestCase):
    def test_video_transition_style_defaults_to_none(self):
        params = VideoParams(video_subject="test")
        self.assertIsNone(params.video_transition_style)

    def test_video_transition_style_accepts_a_transition_name(self):
        params = VideoParams(video_subject="test", video_transition_style="wipeleft")
        self.assertEqual(params.video_transition_style, "wipeleft")

    def test_video_transition_style_accepts_random(self):
        params = VideoParams(video_subject="test", video_transition_style="random")
        self.assertEqual(params.video_transition_style, "random")

    def test_video_transition_style_rejects_unknown_name(self):
        with self.assertRaises(pydantic.ValidationError):
            VideoParams(
                video_subject="test",
                video_transition_style="not-a-real-transition",
            )

    def test_video_overlay_effect_defaults_to_none(self):
        params = VideoParams(video_subject="test")
        self.assertIsNone(params.video_overlay_effect)

    def test_video_overlay_effect_accepts_a_catalog_name(self):
        params = VideoParams(video_subject="test", video_overlay_effect="rain_light")
        self.assertEqual(params.video_overlay_effect, "rain_light")

    def test_video_overlay_effect_accepts_random(self):
        params = VideoParams(video_subject="test", video_overlay_effect="random")
        self.assertEqual(params.video_overlay_effect, "random")

    def test_video_overlay_effect_accepts_none_string(self):
        params = VideoParams(video_subject="test", video_overlay_effect="none")
        self.assertEqual(params.video_overlay_effect, "none")

    def test_video_overlay_effect_rejects_unknown_name(self):
        with self.assertRaises(pydantic.ValidationError):
            VideoParams(video_subject="test", video_overlay_effect="not-a-real-effect")

    def test_image_clip_duration_defaults_to_four_seconds(self):
        params = VideoParams(video_subject="test")
        self.assertEqual(params.image_clip_duration, 4)

    def test_image_clip_duration_accepts_a_value_longer_than_clip_duration_cap(self):
        # video_clip_duration 在 WebUI 里被限制在个位数秒级；image_clip_duration
        # 必须完全独立、没有类似上限，用户才能让静态图配文案停留更久。
        params = VideoParams(video_subject="test", image_clip_duration=120)
        self.assertEqual(params.image_clip_duration, 120)

    def test_image_clip_duration_rejects_non_positive_values(self):
        with self.assertRaises(pydantic.ValidationError):
            VideoParams(video_subject="test", image_clip_duration=0)

    def test_output_dir_and_filename_default_to_none(self):
        params = VideoParams(video_subject="test")
        self.assertIsNone(params.output_dir)
        self.assertIsNone(params.output_filename)

    def test_output_dir_strips_whitespace_and_empty_becomes_none(self):
        params = VideoParams(video_subject="test", output_dir="  D:\\Videos  ")
        self.assertEqual(params.output_dir, "D:\\Videos")
        self.assertIsNone(VideoParams(video_subject="test", output_dir="   ").output_dir)

    def test_output_filename_accepts_a_plain_name(self):
        params = VideoParams(video_subject="test", output_filename="my-video")
        self.assertEqual(params.output_filename, "my-video")

    def test_output_filename_empty_becomes_none(self):
        params = VideoParams(video_subject="test", output_filename="   ")
        self.assertIsNone(params.output_filename)

    def test_output_filename_rejects_path_separators_and_reserved_chars(self):
        for bad_name in ("a/b", "a\\b", "a:b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b"):
            with self.subTest(bad_name=bad_name):
                with self.assertRaises(pydantic.ValidationError):
                    VideoParams(video_subject="test", output_filename=bad_name)


def _make_params(**overrides):
    return VideoParams(video_subject="test", **overrides)


class TestCardTextConfigValidation(unittest.TestCase):
    def test_none_is_accepted_and_left_as_none(self):
        params = _make_params(card_text_config=None)
        self.assertIsNone(params.card_text_config)

    def test_empty_string_is_accepted_and_normalized_to_none(self):
        params = _make_params(card_text_config="")
        self.assertIsNone(params.card_text_config)

    def test_valid_single_slot_config_is_kept_unchanged(self):
        raw = '[{"slot": 1, "style": "minimal_white", "effect": "slide_left", "sound": "auto"}]'
        params = _make_params(card_text_config=raw)
        self.assertEqual(params.card_text_config, raw)

    def test_sound_field_is_optional_per_slot(self):
        raw = '[{"slot": 1, "style": "minimal_white", "effect": "slide_left"}]'
        params = _make_params(card_text_config=raw)
        self.assertEqual(params.card_text_config, raw)

    def test_random_is_accepted_for_style_and_effect(self):
        raw = '[{"slot": 1, "style": "random", "effect": "random"}]'
        params = _make_params(card_text_config=raw)
        self.assertEqual(params.card_text_config, raw)

    def test_multiple_slots_are_accepted(self):
        raw = (
            '[{"slot": 1, "style": "minimal_white", "effect": "slide_left"},'
            ' {"slot": 2, "style": "bold_yellow_box", "effect": "bounce", "sound": "none"}]'
        )
        params = _make_params(card_text_config=raw)
        self.assertEqual(params.card_text_config, raw)

    def test_malformed_json_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            _make_params(card_text_config="not json")

    def test_non_list_json_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            _make_params(card_text_config='{"slot": 1}')

    def test_missing_required_key_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            _make_params(card_text_config='[{"slot": 1, "style": "minimal_white"}]')

    def test_unknown_style_name_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            _make_params(
                card_text_config='[{"slot": 1, "style": "nonexistent", "effect": "slide_left"}]'
            )

    def test_unknown_effect_name_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            _make_params(
                card_text_config='[{"slot": 1, "style": "minimal_white", "effect": "nonexistent"}]'
            )

    def test_unknown_sound_name_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            _make_params(
                card_text_config=(
                    '[{"slot": 1, "style": "minimal_white", "effect": "slide_left",'
                    ' "sound": "nonexistent"}]'
                )
            )

    def test_non_positive_slot_number_is_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            _make_params(
                card_text_config='[{"slot": 0, "style": "minimal_white", "effect": "slide_left"}]'
            )

    def test_duplicate_slot_numbers_are_rejected(self):
        raw = (
            '[{"slot": 1, "style": "minimal_white", "effect": "slide_left"},'
            ' {"slot": 1, "style": "bold_yellow_box", "effect": "bounce"}]'
        )
        with self.assertRaises(pydantic.ValidationError):
            _make_params(card_text_config=raw)


if __name__ == "__main__":
    unittest.main()
