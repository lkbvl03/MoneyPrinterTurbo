# test/services/test_card_text_primitives.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.utils.card_text import _primitives as p


def _solid_card(width=100, height=50, color=(255, 0, 0, 255)):
    card = np.zeros((height, width, 4), dtype=np.uint8)
    card[:, :] = color
    return card


class TestPhaseProgress(unittest.TestCase):
    def test_enter_phase(self):
        phase, progress = p._phase_progress(0.1, 0.5, 1.0, 0.5)
        self.assertEqual(phase, "enter")
        self.assertAlmostEqual(progress, 0.2)

    def test_hold_phase(self):
        phase, progress = p._phase_progress(0.7, 0.5, 1.0, 0.5)
        self.assertEqual(phase, "hold")
        self.assertEqual(progress, 1.0)

    def test_exit_phase(self):
        phase, progress = p._phase_progress(1.75, 0.5, 1.0, 0.5)
        self.assertEqual(phase, "exit")
        self.assertAlmostEqual(progress, 0.5)

    def test_exit_progress_clamped_to_one(self):
        phase, progress = p._phase_progress(100.0, 0.5, 1.0, 0.5)
        self.assertEqual(phase, "exit")
        self.assertEqual(progress, 1.0)


class TestPlaceCardOnCanvas(unittest.TestCase):
    def test_returns_correct_canvas_shape(self):
        card = _solid_card(100, 50)
        result = p._place_card_on_canvas(card, 0.0, 0.0, 1.0, 1.0, 200, 150)
        self.assertEqual(result.shape, (150, 200, 4))

    def test_zero_opacity_produces_fully_transparent_output(self):
        card = _solid_card(100, 50)
        result = p._place_card_on_canvas(card, 0.0, 0.0, 1.0, 0.0, 200, 150)
        self.assertTrue(np.all(result[:, :, 3] == 0))

    def test_full_opacity_centered_has_visible_pixels(self):
        card = _solid_card(100, 50)
        result = p._place_card_on_canvas(card, 0.0, 0.0, 1.0, 1.0, 200, 150)
        center_alpha = result[75, 100, 3]
        self.assertGreater(center_alpha, 0)

    def test_scale_up_produces_larger_visible_region(self):
        card = _solid_card(60, 40)
        small = p._place_card_on_canvas(card, 0.0, 0.0, 0.5, 1.0, 200, 150)
        large = p._place_card_on_canvas(card, 0.0, 0.0, 1.5, 1.0, 200, 150)
        self.assertGreater(
            np.count_nonzero(large[:, :, 3]), np.count_nonzero(small[:, :, 3])
        )


class TestMakeRigidEffect(unittest.TestCase):
    def test_builds_clip_with_correct_total_duration(self):
        def transform_fn(t, enter, hold, exit_):
            return 0.0, 0.0, 1.0, 1.0

        card = _solid_card()
        build = p._make_rigid_effect(transform_fn)
        clip = build(card, enter_duration=0.3, hold_duration=1.0, exit_duration=0.3)
        self.addCleanup(clip.close)
        self.assertAlmostEqual(clip.duration, 1.6)

    def test_frame_and_mask_reflect_transform_fn_output(self):
        def transform_fn(t, enter, hold, exit_):
            opacity = 0.0 if t < 0.5 else 1.0
            return 0.0, 0.0, 1.0, opacity

        card = _solid_card()
        build = p._make_rigid_effect(transform_fn)
        clip = build(card, enter_duration=0.5, hold_duration=0.5, exit_duration=0.5)
        self.addCleanup(clip.close)
        early_mask = clip.mask.get_frame(0.1)
        late_mask = clip.mask.get_frame(0.6)
        self.assertTrue(np.all(early_mask == 0))
        self.assertGreater(late_mask.max(), 0)


class TestMakeCustomFrameEffect(unittest.TestCase):
    def test_builds_clip_with_correct_total_duration(self):
        def frame_fn(card_rgba, t, enter, hold, exit_):
            return card_rgba

        card = _solid_card()
        build = p._make_custom_frame_effect(frame_fn)
        clip = build(card, enter_duration=0.2, hold_duration=0.4, exit_duration=0.2)
        self.addCleanup(clip.close)
        self.assertAlmostEqual(clip.duration, 0.8)

    def test_frame_fn_receives_correct_arguments(self):
        received = []

        def frame_fn(card_rgba, t, enter, hold, exit_):
            received.append((t, enter, hold, exit_))
            return card_rgba

        card = _solid_card()
        build = p._make_custom_frame_effect(frame_fn)
        clip = build(card, enter_duration=0.2, hold_duration=0.4, exit_duration=0.2)
        self.addCleanup(clip.close)
        clip.get_frame(0.3)
        self.assertEqual(received[-1][1:], (0.2, 0.4, 0.2))


class TestRenderCardBox(unittest.TestCase):
    def test_returns_rgba_array_with_visible_content(self):
        from app.services import video as video_service

        font_path = video_service.resolve_font_path(None)
        result = p.render_card_box(
            "Xin chào",
            font_path=font_path,
            font_size=32,
            text_color=(0, 0, 0, 255),
            background_color=(255, 255, 255, 230),
            border_color=(0, 0, 0, 255),
            border_width=2,
            corner_radius=8,
        )
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[2], 4)
        self.assertGreater(result.shape[0], 0)
        self.assertGreater(result.shape[1], 0)
        self.assertGreater(result[:, :, 3].max(), 0)

    def test_long_text_wraps_into_multiple_lines_and_grows_height(self):
        from app.services import video as video_service

        font_path = video_service.resolve_font_path(None)
        short = p.render_card_box(
            "Ngắn",
            font_path=font_path, font_size=32,
            text_color=(0, 0, 0, 255), background_color=(255, 255, 255, 230),
            border_color=None, border_width=0, corner_radius=0,
        )
        long_text = " ".join(["từ"] * 40)
        long_card = p.render_card_box(
            long_text,
            font_path=font_path, font_size=32,
            text_color=(0, 0, 0, 255), background_color=(255, 255, 255, 230),
            border_color=None, border_width=0, corner_radius=0,
        )
        self.assertGreater(long_card.shape[0], short.shape[0])


if __name__ == "__main__":
    unittest.main()
