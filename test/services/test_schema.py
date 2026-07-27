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


if __name__ == "__main__":
    unittest.main()
