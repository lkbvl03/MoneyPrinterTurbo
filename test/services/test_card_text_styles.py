# test/services/test_card_text_styles.py
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.utils.card_text._styles import CATEGORY_STYLES

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


if __name__ == "__main__":
    unittest.main()
