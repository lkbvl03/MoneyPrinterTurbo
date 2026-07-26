import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moviepy import ImageClip

from app.services.utils.video_overlay_effects import (
    OVERLAY_EFFECTS,
    OverlayEffectSpec,
    apply_overlay_effect,
    resolve_overlay_effect_name,
)
from app.services.utils.video_overlay_effects import _primitives as p


def _solid_clip(width=64, height=48, duration=1.0, color=(40, 40, 40)):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return ImageClip(frame).with_duration(duration)


class TestBlendFrame(unittest.TestCase):
    def test_normal_alpha_full_opacity_replaces_base(self):
        base = np.zeros((2, 2, 3), dtype=np.uint8)
        overlay = np.full((2, 2, 3), 200, dtype=np.uint8)
        alpha = np.ones((2, 2), dtype=np.float32)
        result = p._blend_frame(base, overlay, alpha, "normal_alpha")
        np.testing.assert_array_equal(result, overlay)

    def test_zero_alpha_keeps_base_unchanged(self):
        base = np.full((2, 2, 3), 77, dtype=np.uint8)
        overlay = np.full((2, 2, 3), 255, dtype=np.uint8)
        alpha = np.zeros((2, 2), dtype=np.float32)
        result = p._blend_frame(base, overlay, alpha, "screen")
        np.testing.assert_array_equal(result, base)

    def test_screen_mode_is_brighter_than_either_input(self):
        base = np.full((1, 1, 3), 100, dtype=np.uint8)
        overlay = np.full((1, 1, 3), 100, dtype=np.uint8)
        alpha = np.ones((1, 1), dtype=np.float32)
        result = p._blend_frame(base, overlay, alpha, "screen")
        # screen(100, 100) = 255 - (155*155)/255 ≈ 155.6 > 100
        self.assertGreater(int(result[0, 0, 0]), 100)

    def test_lighten_mode_keeps_the_brighter_pixel(self):
        base = np.array([[[10, 200, 50]]], dtype=np.uint8)
        overlay = np.array([[[80, 20, 60]]], dtype=np.uint8)
        alpha = np.ones((1, 1), dtype=np.float32)
        result = p._blend_frame(base, overlay, alpha, "lighten")
        np.testing.assert_array_equal(result[0, 0], [80, 200, 60])

    def test_unknown_mode_raises(self):
        base = np.zeros((1, 1, 3), dtype=np.uint8)
        overlay = np.zeros((1, 1, 3), dtype=np.uint8)
        alpha = np.zeros((1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            p._blend_frame(base, overlay, alpha, "bogus")


class TestMakeOverlayEffect(unittest.TestCase):
    def test_wraps_layer_fn_and_preserves_clip_shape(self):
        def layer_fn(t, size):
            width, height = size
            rgb = np.full((height, width, 3), 255, dtype=np.uint8)
            alpha = np.full((height, width), 0.5, dtype=np.float32)
            return rgb, alpha

        clip = _solid_clip(color=(0, 0, 0))
        self.addCleanup(clip.close)
        generator = p._make_overlay_effect(layer_fn, "normal_alpha")
        result = generator(clip)
        self.addCleanup(result.close)

        self.assertEqual(result.size, clip.size)
        self.assertEqual(result.duration, clip.duration)
        frame = result.get_frame(0)
        # alpha=0.5 giữa nền đen (0) và overlay trắng (255) -> ~127
        self.assertTrue(120 <= int(frame[0, 0, 0]) <= 135)


class TestMakeTransformEffect(unittest.TestCase):
    def test_wraps_frame_fn_and_applies_it_per_frame(self):
        def invert(frame, t):
            return 255 - frame

        clip = _solid_clip(color=(10, 20, 30))
        self.addCleanup(clip.close)
        generator = p._make_transform_effect(invert)
        result = generator(clip)
        self.addCleanup(result.close)

        frame = result.get_frame(0)
        np.testing.assert_array_equal(frame[0, 0], [245, 235, 225])


class TestParticleFieldLayer(unittest.TestCase):
    def test_returns_correct_shapes_and_valid_ranges(self):
        config = p._ParticleFieldConfig(
            seed=1, count=20, color=(255, 0, 0),
            radius_range=(1, 3), fall_speed_range=(10, 20),
            horizontal_drift=5, vertical_drift=0, sway_freq=0.2,
            alpha_range=(0.2, 0.6), shape="dot", streak_length=0, twinkle=False,
        )
        rgb, alpha = p._particle_field_layer(0.5, (100, 80), config)
        self.assertEqual(rgb.shape, (80, 100, 3))
        self.assertEqual(alpha.shape, (80, 100))
        self.assertTrue(np.all(alpha >= 0.0) and np.all(alpha <= 1.0))
        self.assertGreater(alpha.max(), 0.0)

    def test_is_deterministic_for_the_same_time(self):
        config = p._ParticleFieldConfig(
            seed=7, count=15, color=(0, 255, 0),
            radius_range=(1, 2), fall_speed_range=(5, 5),
            horizontal_drift=3, vertical_drift=3, sway_freq=0.3,
            alpha_range=(0.3, 0.3), shape="streak", streak_length=10, twinkle=True,
        )
        rgb1, alpha1 = p._particle_field_layer(1.23, (64, 64), config)
        rgb2, alpha2 = p._particle_field_layer(1.23, (64, 64), config)
        np.testing.assert_array_equal(alpha1, alpha2)
        np.testing.assert_array_equal(rgb1, rgb2)

    def test_zero_count_returns_empty_alpha(self):
        config = p._ParticleFieldConfig(
            seed=1, count=0, color=(1, 1, 1),
            radius_range=(1, 1), fall_speed_range=(0, 0),
            horizontal_drift=0, vertical_drift=0, sway_freq=0,
            alpha_range=(0, 0), shape="dot", streak_length=0, twinkle=False,
        )
        _, alpha = p._particle_field_layer(0.0, (32, 32), config)
        self.assertEqual(alpha.max(), 0.0)


class TestNoiseFieldLayer(unittest.TestCase):
    def test_returns_correct_shapes_and_valid_alpha_range(self):
        rgb, alpha = p._noise_field_layer(
            0.0, (100, 60), seed=3, scale=8.0, speed=10.0,
            color=(200, 200, 200), alpha_min=0.1, alpha_max=0.3,
        )
        self.assertEqual(rgb.shape, (60, 100, 3))
        self.assertEqual(alpha.shape, (60, 100))
        self.assertTrue(np.all(alpha >= 0.0) and np.all(alpha <= 1.0))

    def test_vertical_bias_confines_alpha_near_bottom(self):
        _, alpha = p._noise_field_layer(
            0.0, (50, 100), seed=3, scale=8.0, speed=0.0,
            color=(200, 200, 200), alpha_min=0.0, alpha_max=1.0, vertical_bias=0.3,
        )
        top_band = alpha[:30, :].mean()
        bottom_band = alpha[-30:, :].mean()
        self.assertLess(top_band, bottom_band)


class TestRadialGlowLayer(unittest.TestCase):
    def test_center_is_brighter_than_edges(self):
        _, alpha = p._radial_glow_layer(
            0.0, (100, 100), center=(50, 50), radius=40,
            color=(255, 200, 100), alpha_min=0.0, alpha_max=1.0,
        )
        self.assertGreater(alpha[50, 50], alpha[0, 0])

    def test_pulse_varies_alpha_over_time(self):
        _, alpha_a = p._radial_glow_layer(
            0.0, (20, 20), center=(10, 10), radius=15,
            color=(255, 255, 255), alpha_min=0.0, alpha_max=1.0, pulse_hz=0.25,
        )
        _, alpha_b = p._radial_glow_layer(
            1.0, (20, 20), center=(10, 10), radius=15,
            color=(255, 255, 255), alpha_min=0.0, alpha_max=1.0, pulse_hz=0.25,
        )
        self.assertFalse(np.array_equal(alpha_a, alpha_b))


class TestResolveOverlayEffectName(unittest.TestCase):
    def test_none_and_literal_none_return_none(self):
        self.assertIsNone(resolve_overlay_effect_name(None))
        self.assertIsNone(resolve_overlay_effect_name("none"))

    def test_fixed_name_returned_unchanged(self):
        self.assertEqual(resolve_overlay_effect_name("rain_light"), "rain_light")

    def test_random_calls_injected_random_choice_over_catalog_keys(self):
        with patch.dict(
            OVERLAY_EFFECTS,
            {"dummy_a": OverlayEffectSpec("dummy_a", "test", "transform", lambda c: c)},
            clear=True,
        ):
            result = resolve_overlay_effect_name(
                "random", random_choice=lambda seq: list(seq)[0]
            )
        self.assertEqual(result, "dummy_a")


class TestApplyOverlayEffect(unittest.TestCase):
    def test_calls_the_registered_generator(self):
        clip = _solid_clip()
        self.addCleanup(clip.close)
        sentinel = _solid_clip(color=(9, 9, 9))
        self.addCleanup(sentinel.close)

        dummy_spec = OverlayEffectSpec(
            name="dummy", category="test", kind="transform",
            generator=lambda c: sentinel,
        )
        with patch.dict(OVERLAY_EFFECTS, {"dummy": dummy_spec}, clear=True):
            result = apply_overlay_effect(clip, "dummy")
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
