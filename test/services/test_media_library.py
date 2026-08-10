import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect
from app.services import media_library


class TestScoreMatch(unittest.TestCase):
    def test_tag_match_scores_higher_than_description_only_match(self):
        entry_tag_hit = {"description": "a quiet street", "tags": ["ocean", "sunset"]}
        entry_description_only = {
            "description": "a boat on the ocean at sunset",
            "tags": ["boat", "sky"],
        }
        tag_score = media_library._score_match("ocean", entry_tag_hit)
        description_score = media_library._score_match("ocean", entry_description_only)
        self.assertGreater(tag_score, description_score)

    def test_no_overlap_scores_zero(self):
        entry = {"description": "a mountain trail", "tags": ["hiking", "forest"]}
        self.assertEqual(media_library._score_match("city traffic", entry), 0)

    def test_multi_token_query_accumulates_score(self):
        entry = {"description": "", "tags": ["ocean", "sunset", "beach"]}
        self.assertEqual(media_library._score_match("ocean sunset", entry), 4)


class TestDeriveTagsFromPath(unittest.TestCase):
    def test_splits_filename_on_common_separators(self):
        derived = media_library._derive_tags_from_path("red_car-photo.jpg")
        self.assertEqual(derived["tags"], ["red", "car", "photo"])
        self.assertEqual(derived["description"], "red car photo")

    def test_includes_parent_folder_tokens(self):
        derived = media_library._derive_tags_from_path("thien_nhien/bien_hoanghon.mp4")
        self.assertEqual(derived["tags"], ["thien", "nhien", "bien", "hoanghon"])

    def test_lowercases_and_dedupes_tokens(self):
        derived = media_library._derive_tags_from_path("Car/Car-Red.jpg")
        self.assertEqual(derived["tags"], ["car", "red"])


class TestScanAndTagLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.directory = self._tmp_dir.name

        patcher = patch.object(
            media_library, "aspect_dir", return_value=self.directory
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _write_image(self, name: str, content: bytes = b"fake-image-bytes"):
        path = os.path.join(self.directory, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_new_image_gets_tagged_from_filename_no_api_call(self):
        self._write_image("red_car.jpg")
        stats = media_library.scan_and_tag_library(VideoAspect.portrait)

        self.assertEqual(stats, {"scanned": 1, "tagged": 1, "skipped": 0, "failed": 0})

        tags = media_library._load_tags(self.directory)
        self.assertEqual(tags["red_car.jpg"]["tags"], ["red", "car"])
        self.assertEqual(tags["red_car.jpg"]["description"], "red car")
        self.assertEqual(tags["red_car.jpg"]["type"], "image")
        self.assertIsNone(tags["red_car.jpg"]["duration"])

    def test_unchanged_file_is_skipped_on_second_scan(self):
        self._write_image("red_car.jpg")
        media_library.scan_and_tag_library(VideoAspect.portrait)

        stats = media_library.scan_and_tag_library(VideoAspect.portrait)
        self.assertEqual(stats, {"scanned": 1, "tagged": 0, "skipped": 1, "failed": 0})

    def test_force_retags_even_when_unchanged(self):
        self._write_image("red_car.jpg")
        media_library.scan_and_tag_library(VideoAspect.portrait)

        stats = media_library.scan_and_tag_library(VideoAspect.portrait, force=True)
        self.assertEqual(stats["tagged"], 1)

    def test_video_duration_is_probed_and_stored(self):
        self._write_image("clip.mp4")
        with patch.object(
            media_library, "_probe_video_duration", return_value=12.5
        ) as probe:
            stats = media_library.scan_and_tag_library(VideoAspect.portrait)

        probe.assert_called_once()
        self.assertEqual(stats["tagged"], 1)
        tags = media_library._load_tags(self.directory)
        self.assertEqual(tags["clip.mp4"]["duration"], 12.5)
        self.assertEqual(tags["clip.mp4"]["type"], "video")

    def test_failed_duration_probe_is_recorded_and_does_not_stop_the_scan(self):
        self._write_image("bad.mp4")
        self._write_image("good.jpg")
        with patch.object(
            media_library,
            "_probe_video_duration",
            side_effect=RuntimeError("cannot open video"),
        ):
            stats = media_library.scan_and_tag_library(VideoAspect.portrait)

        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["tagged"], 1)

        tags = media_library._load_tags(self.directory)
        self.assertNotIn("bad.mp4", tags)
        self.assertIn("good.jpg", tags)

    def test_deleted_file_is_removed_from_tags_on_next_scan(self):
        self._write_image("red_car.jpg")
        media_library.scan_and_tag_library(VideoAspect.portrait)
        os.remove(os.path.join(self.directory, "red_car.jpg"))

        media_library.scan_and_tag_library(VideoAspect.portrait)
        tags = media_library._load_tags(self.directory)
        self.assertNotIn("red_car.jpg", tags)

    def test_files_in_subfolders_are_tagged_with_relative_path_key(self):
        self._write_image("thien_nhien/bien.jpg")
        stats = media_library.scan_and_tag_library(VideoAspect.portrait)

        self.assertEqual(stats, {"scanned": 1, "tagged": 1, "skipped": 0, "failed": 0})
        tags = media_library._load_tags(self.directory)
        self.assertIn("thien_nhien/bien.jpg", tags)
        self.assertEqual(tags["thien_nhien/bien.jpg"]["tags"], ["thien", "nhien", "bien"])

    def test_same_basename_in_different_subfolders_does_not_collide(self):
        self._write_image("thien_nhien/car.jpg")
        self._write_image("thanh_pho/car.jpg")
        stats = media_library.scan_and_tag_library(VideoAspect.portrait)

        self.assertEqual(stats["tagged"], 2)
        tags = media_library._load_tags(self.directory)
        self.assertIn("thien_nhien/car.jpg", tags)
        self.assertIn("thanh_pho/car.jpg", tags)

    def test_hidden_subfolder_is_skipped(self):
        self._write_image(".cache/car.jpg")
        stats = media_library.scan_and_tag_library(VideoAspect.portrait)
        self.assertEqual(stats["scanned"], 0)


class TestSearchLocalLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.directory = self._tmp_dir.name

        patcher = patch.object(
            media_library, "aspect_dir", return_value=self.directory
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _write_tags(self, tags: dict):
        for name in tags:
            path = os.path.join(self.directory, name)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            open(path, "wb").close()
        media_library._save_tags(self.directory, tags)

    def test_short_video_below_minimum_duration_is_excluded(self):
        self._write_tags(
            {
                "short.mp4": {
                    "type": "video",
                    "duration": 2.0,
                    "description": "ocean waves",
                    "tags": ["ocean"],
                },
                "long.mp4": {
                    "type": "video",
                    "duration": 10.0,
                    "description": "ocean waves",
                    "tags": ["ocean"],
                },
            }
        )
        results = media_library.search_local_library(
            "ocean", minimum_duration=5, video_aspect=VideoAspect.portrait
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].url.endswith("long.mp4"))

    def test_results_are_ranked_by_score_descending(self):
        self._write_tags(
            {
                "weak.mp4": {
                    "type": "video",
                    "duration": 10.0,
                    "description": "a car driving near the ocean",
                    "tags": ["car"],
                },
                "strong.mp4": {
                    "type": "video",
                    "duration": 10.0,
                    "description": "ocean waves",
                    "tags": ["ocean", "waves"],
                },
            }
        )
        results = media_library.search_local_library(
            "ocean", minimum_duration=1, video_aspect=VideoAspect.portrait
        )
        self.assertEqual([os.path.basename(r.url) for r in results], ["strong.mp4", "weak.mp4"])

    def test_image_entries_are_not_filtered_by_original_duration(self):
        self._write_tags(
            {
                "photo.jpg": {
                    "type": "image",
                    "duration": None,
                    "description": "a mountain",
                    "tags": ["mountain"],
                }
            }
        )
        results = media_library.search_local_library(
            "mountain", minimum_duration=7, video_aspect=VideoAspect.portrait
        )
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0].duration, 7)

    def test_finds_matching_file_stored_in_a_subfolder(self):
        self._write_tags(
            {
                "thien_nhien/bien.mp4": {
                    "type": "video",
                    "duration": 10.0,
                    "description": "ocean waves at the beach",
                    "tags": ["ocean", "beach"],
                }
            }
        )
        results = media_library.search_local_library(
            "ocean", minimum_duration=1, video_aspect=VideoAspect.portrait
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(
            results[0].url.replace("\\", "/").endswith("thien_nhien/bien.mp4")
        )

    def test_no_tags_file_returns_empty_list(self):
        results = media_library.search_local_library(
            "anything", minimum_duration=1, video_aspect=VideoAspect.portrait
        )
        self.assertEqual(results, [])

    def test_all_items_populate_provider_as_local_library(self):
        self._write_tags(
            {
                "clip.mp4": {
                    "type": "video",
                    "duration": 10.0,
                    "description": "sunset",
                    "tags": ["sunset"],
                }
            }
        )
        results = media_library.search_local_library(
            "sunset", minimum_duration=1, video_aspect=VideoAspect.portrait
        )
        self.assertEqual(results[0].provider, "local_library")


if __name__ == "__main__":
    unittest.main()
