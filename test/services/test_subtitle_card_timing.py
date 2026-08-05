import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import subtitle
from app.services.utils.card_markers import CardMarker


class TestResolveCardMarkerTimestamps(unittest.TestCase):
    def test_anchor_within_range_uses_that_words_end_time(self):
        # spec: start = whisper_words[anchor_word_index].end -- thẻ xuất hiện
        # ngay sau khi từ liền trước nó (anchor) vừa đọc XONG, không phải lúc
        # từ đó bắt đầu được đọc.
        markers = [CardMarker(slot=1, text="Xin chào", anchor_word_index=2)]
        whisper_words = [
            ("hello", 0.0, 0.3),
            ("there", 0.3, 0.6),
            ("friend", 0.6, 1.0),
            ("today", 1.0, 1.4),
        ]
        results = subtitle._resolve_card_marker_timestamps(markers, whisper_words)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].slot, 1)
        self.assertEqual(results[0].text, "Xin chào")
        self.assertEqual(results[0].start_time, 1.0)

    def test_anchor_beyond_available_words_extrapolates_forward(self):
        # 4 từ, tổng 1.2s -> tốc độ trung bình 0.3s/từ. anchor_word_index=5
        # tức là 2 từ sau từ cuối cùng (index 3) -> 1.2 + 2 * 0.3 = 1.8
        markers = [CardMarker(slot=2, text="Bùng nổ", anchor_word_index=5)]
        whisper_words = [
            ("a", 0.0, 0.3),
            ("b", 0.3, 0.6),
            ("c", 0.6, 0.9),
            ("d", 0.9, 1.2),
        ]
        results = subtitle._resolve_card_marker_timestamps(markers, whisper_words)
        self.assertAlmostEqual(results[0].start_time, 1.8)

    def test_multiple_markers_preserve_input_order_and_own_slot(self):
        markers = [
            CardMarker(slot=3, text="Một", anchor_word_index=0),
            CardMarker(slot=1, text="Hai", anchor_word_index=1),
        ]
        whisper_words = [("x", 0.0, 0.5), ("y", 0.5, 1.0)]
        results = subtitle._resolve_card_marker_timestamps(markers, whisper_words)
        self.assertEqual([r.slot for r in results], [3, 1])
        self.assertEqual([r.text for r in results], ["Một", "Hai"])

    def test_empty_markers_returns_empty_list(self):
        self.assertEqual(subtitle._resolve_card_marker_timestamps([], []), [])


class TestCreateWithCardMarkers(unittest.TestCase):
    def test_create_returns_resolved_card_timings_when_markers_present(self):
        class _FakeWhisperModel:
            def __init__(self, **kwargs):
                pass

            def transcribe(self, audio_file, **kwargs):
                words = [
                    SimpleNamespace(start=0.0, end=0.3, word="hello"),
                    SimpleNamespace(start=0.3, end=0.6, word=" there"),
                    SimpleNamespace(start=0.6, end=1.0, word=" friend"),
                ]
                segment = SimpleNamespace(start=0.0, end=1.0, words=words)
                info = SimpleNamespace(language="en", language_probability=0.99)
                return [segment], info

        markers = [CardMarker(slot=1, text="Chào!", anchor_word_index=2)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "generated.srt"
            with patch.object(subtitle, "model", None), patch.object(
                subtitle, "WhisperModel", _FakeWhisperModel,
            ):
                card_timings = subtitle.create(
                    "audio.mp3",
                    str(subtitle_file),
                    video_script="hello there friend.",
                    card_markers=markers,
                )

        self.assertEqual(len(card_timings), 1)
        self.assertEqual(card_timings[0].slot, 1)
        self.assertEqual(card_timings[0].start_time, 1.0)

    def test_create_returns_empty_list_when_no_markers_given(self):
        class _FakeWhisperModel:
            def __init__(self, **kwargs):
                pass

            def transcribe(self, audio_file, **kwargs):
                words = [SimpleNamespace(start=0.0, end=0.3, word="hi")]
                segment = SimpleNamespace(start=0.0, end=0.3, words=words)
                info = SimpleNamespace(language="en", language_probability=0.99)
                return [segment], info

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "generated.srt"
            with patch.object(subtitle, "model", None), patch.object(
                subtitle, "WhisperModel", _FakeWhisperModel,
            ):
                card_timings = subtitle.create(
                    "audio.mp3", str(subtitle_file), video_script="hi",
                )

        self.assertEqual(card_timings, [])


if __name__ == "__main__":
    unittest.main()
