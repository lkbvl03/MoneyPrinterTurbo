import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.utils.card_markers import CardMarker, extract_card_markers


class TestExtractCardMarkers(unittest.TestCase):
    def test_no_markers_returns_script_unchanged(self):
        script = "Đây là một kịch bản bình thường không có thẻ nào."
        cleaned, markers = extract_card_markers(script)
        self.assertEqual(cleaned, script)
        self.assertEqual(markers, [])

    def test_single_unnumbered_marker_defaults_to_slot_1(self):
        script = "Xin chào [card: Nội dung thẻ] các bạn."
        cleaned, markers = extract_card_markers(script)
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].slot, 1)
        self.assertEqual(markers[0].text, "Nội dung thẻ")
        self.assertNotIn("[card", cleaned)
        self.assertIn("Xin chào", cleaned)
        self.assertIn("các bạn", cleaned)

    def test_numbered_marker_captures_slot_number(self):
        script = "Một [card 3: Ba] hai [card 7: Bảy]."
        cleaned, markers = extract_card_markers(script)
        self.assertEqual([m.slot for m in markers], [3, 7])
        self.assertEqual([m.text for m in markers], ["Ba", "Bảy"])

    def test_marker_content_removed_from_cleaned_script(self):
        script = "A [card: X] B [card 2: Y] C"
        cleaned, markers = extract_card_markers(script)
        self.assertNotIn("[card", cleaned)
        self.assertNotIn("X", cleaned)
        self.assertNotIn("Y", cleaned)

    def test_case_insensitive_matching(self):
        script = "Trước [CARD: hoa] sau"
        cleaned, markers = extract_card_markers(script)
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].text, "hoa")

    def test_anchor_word_index_counts_words_before_marker(self):
        script = "Một hai ba [card: X] bốn năm"
        cleaned, markers = extract_card_markers(script)
        # "Một hai ba" = 3 từ trước thẻ -> anchor là từ cuối cùng, index 0-based = 2.
        self.assertEqual(markers[0].anchor_word_index, 2)

    def test_marker_at_very_start_anchors_to_word_zero(self):
        script = "[card: Mở đầu] Nội dung tiếp theo"
        cleaned, markers = extract_card_markers(script)
        self.assertEqual(markers[0].anchor_word_index, 0)

    def test_multiple_markers_anchor_indices_increase(self):
        script = "Một [card: A] hai ba [card: B] bốn"
        cleaned, markers = extract_card_markers(script)
        self.assertEqual(markers[0].anchor_word_index, 0)  # ngay sau "Một"
        self.assertEqual(markers[1].anchor_word_index, 2)  # ngay sau "Một hai ba"

    def test_marker_with_blank_content_is_kept_as_plain_text(self):
        # "[card: ]" khớp regex (nội dung là 1 dấu cách, vẫn thỏa [^\]]+),
        # nhưng sau .strip() thành chuỗi rỗng -> coi như không phải thẻ hợp
        # lệ, giữ nguyên cụm "[card: ]" trong kịch bản thay vì âm thầm xóa.
        script = "Trước [card: ] sau"
        cleaned, markers = extract_card_markers(script)
        self.assertEqual(markers, [])
        self.assertEqual(cleaned, script)

    def test_default_marker_has_no_style_effect_sound(self):
        script = "Xin chào [card: Nội dung] các bạn."
        _, markers = extract_card_markers(script)
        self.assertIsNone(markers[0].style)
        self.assertIsNone(markers[0].effect)
        self.assertIsNone(markers[0].sound)

    def test_inline_attributes_are_parsed_without_slot_number(self):
        script = "[card style=bold_yellow_box effect=typewriter sound=whoosh: Diem nhan]"
        _, markers = extract_card_markers(script)
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].slot, 1)
        self.assertEqual(markers[0].style, "bold_yellow_box")
        self.assertEqual(markers[0].effect, "typewriter")
        self.assertEqual(markers[0].sound, "whoosh")
        self.assertEqual(markers[0].text, "Diem nhan")

    def test_inline_attributes_are_parsed_with_slot_number(self):
        script = "[card 3 style=gradient_pop effect=bounce: Diem nhan thu 3]"
        _, markers = extract_card_markers(script)
        self.assertEqual(markers[0].slot, 3)
        self.assertEqual(markers[0].style, "gradient_pop")
        self.assertEqual(markers[0].effect, "bounce")
        self.assertIsNone(markers[0].sound)

    def test_attributes_are_case_insensitive_keyword_but_case_sensitive_value(self):
        script = "[CARD STYLE=bold_yellow_box: Text]"
        _, markers = extract_card_markers(script)
        self.assertEqual(markers[0].style, "bold_yellow_box")

    def test_partial_attributes_only_sets_provided_ones(self):
        script = "[card effect=rotate_in: Only effect set]"
        _, markers = extract_card_markers(script)
        self.assertIsNone(markers[0].style)
        self.assertEqual(markers[0].effect, "rotate_in")
        self.assertIsNone(markers[0].sound)

    def test_unknown_attribute_keys_are_ignored(self):
        script = "[card style=bold_yellow_box color=red: Text]"
        _, markers = extract_card_markers(script)
        self.assertEqual(markers[0].style, "bold_yellow_box")

    def test_font_and_font_size_attributes_are_parsed(self):
        script = "[card font=BeVietnamPro-Bold.ttf font_size=48: Text]"
        _, markers = extract_card_markers(script)
        self.assertEqual(markers[0].font, "BeVietnamPro-Bold.ttf")
        self.assertEqual(markers[0].font_size, 48)
        self.assertIsInstance(markers[0].font_size, int)

    def test_non_numeric_font_size_is_ignored(self):
        script = "[card font_size=abc: Text]"
        _, markers = extract_card_markers(script)
        self.assertIsNone(markers[0].font_size)

    def test_double_quoted_text_has_quotes_stripped(self):
        script = '[card: "Noi dung co ngoac kep"]'
        _, markers = extract_card_markers(script)
        self.assertEqual(markers[0].text, "Noi dung co ngoac kep")

    def test_single_quoted_text_has_quotes_stripped(self):
        script = "[card: 'Noi dung co ngoac don']"
        _, markers = extract_card_markers(script)
        self.assertEqual(markers[0].text, "Noi dung co ngoac don")

    def test_unmatched_quote_is_left_untouched(self):
        script = '[card: 6" man hinh]'
        _, markers = extract_card_markers(script)
        self.assertEqual(markers[0].text, '6" man hinh')


if __name__ == "__main__":
    unittest.main()
