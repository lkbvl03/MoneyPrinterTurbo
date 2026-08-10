# test/services/test_card_text_styles.py
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import video as video_service
from app.services.utils.card_text._styles import (
    CATEGORY_STYLES,
    _resolve_card_font_path,
)

_STYLE_NAMES = [
    "minimal_white", "bold_yellow_box", "gradient_pop", "sticky_note", "comic_burst",
]


class TestCardStyles(unittest.TestCase):
    def test_all_five_mvp_styles_registered(self):
        for name in _STYLE_NAMES:
            self.assertIn(name, CATEGORY_STYLES)

    def test_each_style_returns_valid_rgba_image_with_content(self):
        for name in _STYLE_NAMES:
            with self.subTest(style=name):
                result = CATEGORY_STYLES[name]("Nội dung kiểm thử")
                self.assertEqual(result.ndim, 3)
                self.assertEqual(result.shape[2], 4)
                self.assertGreater(result.shape[0], 0)
                self.assertGreater(result.shape[1], 0)
                self.assertGreater(result[:, :, 3].max(), 0)

    def test_different_styles_produce_visually_different_output(self):
        white = CATEGORY_STYLES["minimal_white"]("Test")
        yellow = CATEGORY_STYLES["bold_yellow_box"]("Test")
        # 尺寸可能不同（字体/边距不同），所以比较平均背景色而不是直接比较数组
        white_avg = white[:, :, :3].astype(float).mean()
        yellow_avg = yellow[:, :, :3].astype(float).mean()
        self.assertNotAlmostEqual(white_avg, yellow_avg, delta=1.0)

    def test_custom_font_size_produces_a_larger_card_than_default(self):
        for name in _STYLE_NAMES:
            with self.subTest(style=name):
                default_size = CATEGORY_STYLES[name]("Test")
                bigger_size = CATEGORY_STYLES[name]("Test", None, 150)
                self.assertGreater(
                    bigger_size.shape[0] * bigger_size.shape[1],
                    default_size.shape[0] * default_size.shape[1],
                )

    def test_custom_font_name_is_used_when_provided(self):
        with patch.object(
            video_service, "resolve_font_path", wraps=video_service.resolve_font_path
        ) as resolve_font_path:
            CATEGORY_STYLES["minimal_white"]("Test", "BeVietnamPro-Bold.ttf", None)
        resolve_font_path.assert_called_once_with("BeVietnamPro-Bold.ttf")


class TestResolveCardFontPath(unittest.TestCase):
    def test_delegates_to_subtitle_font_glyph_fallback(self):
        # card_text dung lai co che fallback font cua phu de (khong tu suy
        # nghi lai) - regression test cho bug de quy vo han da tung xay ra
        # khi noi day nham goi lai chinh no thay vi goi
        # video_service._resolve_subtitle_font_path.
        with patch.object(
            video_service,
            "_resolve_subtitle_font_path",
            return_value="/fake/fallback-font.ttf",
        ) as fallback:
            result = _resolve_card_font_path("Ban van tieng Viet")

        fallback.assert_called_once()
        called_font_path, called_text = fallback.call_args.args
        self.assertTrue(called_font_path.endswith("STHeitiMedium.ttc"))
        self.assertEqual(called_text, "Ban van tieng Viet")
        self.assertEqual(result, "/fake/fallback-font.ttf")

    def test_falls_back_to_vietnamese_font_for_glyphs_missing_from_default(self):
        # "Ạ" (U+1EA0) khong co trong STHeitiMedium.ttc (font mac dinh cua
        # the chu) va tung bi render ra o vuong (tofu) tren video that.
        text_with_dot_below_diacritic = "0.3 GIÂY — NÃO BẠN ĐÃ TIN"
        font_path = _resolve_card_font_path(text_with_dot_below_diacritic)
        self.assertNotIn("STHeitiMedium.ttc", font_path)
        self.assertTrue(
            video_service.subtitle_font_supports_text(
                font_path, text_with_dot_below_diacritic
            )
        )

    def test_each_style_renders_diacritic_heavy_vietnamese_text_without_error(self):
        text_with_dot_below_diacritic = "0.3 GIÂY — NÃO BẠN ĐÃ TIN"
        for name in _STYLE_NAMES:
            with self.subTest(style=name):
                result = CATEGORY_STYLES[name](text_with_dot_below_diacritic)
                self.assertGreater(result[:, :, 3].max(), 0)


if __name__ == "__main__":
    unittest.main()
