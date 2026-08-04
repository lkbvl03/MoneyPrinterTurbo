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


if __name__ == "__main__":
    unittest.main()
