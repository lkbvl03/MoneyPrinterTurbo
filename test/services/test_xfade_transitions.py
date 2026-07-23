import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.utils import xfade_transitions as xt


class TestTransitionCatalog(unittest.TestCase):
    def test_catalog_has_48_unique_names(self):
        self.assertEqual(len(xt.XFADE_TRANSITIONS), 48)
        self.assertEqual(len(set(xt.XFADE_TRANSITIONS)), 48)

    def test_catalog_contains_expected_names(self):
        for name in ("fade", "wipeleft", "circleopen", "dissolve", "zoomin", "vertclose"):
            self.assertIn(name, xt.XFADE_TRANSITIONS)


class TestResolveTransitionName(unittest.TestCase):
    def test_returns_fixed_name_unchanged(self):
        self.assertEqual(xt.resolve_transition_name("wipeleft"), "wipeleft")

    def test_random_calls_random_choice_over_catalog(self):
        with patch.object(xt.random, "choice", return_value="circleopen") as mock_choice:
            result = xt.resolve_transition_name("random")
        mock_choice.assert_called_once_with(xt.XFADE_TRANSITIONS)
        self.assertEqual(result, "circleopen")

    def test_random_choice_injection_is_used_when_provided(self):
        result = xt.resolve_transition_name("random", random_choice=lambda seq: seq[0])
        self.assertEqual(result, xt.XFADE_TRANSITIONS[0])


class TestComputeTransitionDurations(unittest.TestCase):
    def test_keeps_requested_duration_when_clips_are_long_enough(self):
        # shorter clip in each pair is 5s; 40% of 5s = 2s, so a 1s request fits unclamped.
        durations = xt.compute_transition_durations([5.0, 5.0, 5.0], 1.0)
        self.assertEqual(durations, [1.0, 1.0])

    def test_clamps_to_40_percent_of_shorter_clip_in_pair(self):
        # pair (2.0, 5.0): shorter is 2.0, 40% = 0.8, so a 1.0s request clamps to 0.8.
        durations = xt.compute_transition_durations([2.0, 5.0], 1.0)
        self.assertEqual(durations, [0.8])

    def test_never_goes_below_the_minimum_floor(self):
        # pair (0.1, 5.0): 40% of 0.1 is 0.04, below the 0.1s floor -> clamps up to 0.1.
        durations = xt.compute_transition_durations([0.1, 5.0], 1.0)
        self.assertEqual(durations, [0.1])


class TestComputeXfadeOffsets(unittest.TestCase):
    def test_matches_worked_three_clip_example(self):
        # a=5s, b=4s, c=6s; transitions 1s (a->b), 1.5s (b->c).
        offsets = xt.compute_xfade_offsets([5.0, 4.0, 6.0], [1.0, 1.5])
        self.assertEqual(offsets, [4.0, 6.5])

    def test_two_clips_returns_single_offset(self):
        offsets = xt.compute_xfade_offsets([3.0, 4.0], [1.0])
        self.assertEqual(offsets, [2.0])


class TestBuildXfadeFilterComplex(unittest.TestCase):
    def test_two_clips_builds_single_xfade_to_outv(self):
        filter_complex, output_label = xt.build_xfade_filter_complex(
            [5.0, 4.0], ["wipeleft"], [1.0]
        )
        self.assertEqual(output_label, "[outv]")
        self.assertEqual(
            filter_complex,
            "[0:v][1:v]xfade=transition=wipeleft:duration=1.000:offset=4.000[outv]",
        )

    def test_three_clips_chains_through_intermediate_label(self):
        filter_complex, output_label = xt.build_xfade_filter_complex(
            [5.0, 4.0, 6.0], ["wipeleft", "circleopen"], [1.0, 1.5]
        )
        self.assertEqual(output_label, "[outv]")
        self.assertEqual(
            filter_complex,
            "[0:v][1:v]xfade=transition=wipeleft:duration=1.000:offset=4.000[v1];"
            "[v1][2:v]xfade=transition=circleopen:duration=1.500:offset=6.500[outv]",
        )

    def test_raises_value_error_with_fewer_than_two_clips(self):
        with self.assertRaises(ValueError):
            xt.build_xfade_filter_complex([5.0], [], [])


if __name__ == "__main__":
    unittest.main()
