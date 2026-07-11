"""Field-projection contracts for the canary semantic/calibration hash helpers.

``_execution_event_projection`` and ``_calibration_event_projection`` build the
stable, hash-friendly view of a canary event by conditionally copying each
optional field only when it is present on the source mapping. These tests
exercise both the "field absent" (skip) and "field present" (project) sides of
every conditional, including the ``rank_count`` fallback that only applies
when ``ranks`` itself is missing.
"""

from __future__ import annotations

import unittest

from commcanary.artifacts.canary_hashes import (
    _calibration_event_projection,
    _execution_event_projection,
)


class ExecutionEventProjectionTests(unittest.TestCase):
    def test_projection_of_a_field_free_event_is_empty(self):
        # None of phase/op/bytes/ranks/group/concurrent_groups/
        # execution_occurrence_base/rank_count/timing_samples are present, so
        # every optional projection must be skipped rather than raising or
        # inventing a default value.
        self.assertEqual(_execution_event_projection({}), {})

    def test_every_present_optional_field_is_projected(self):
        event = {
            "phase": "decode",
            "op": "all_reduce",
            "bytes": 128,
            "ranks": [0, 1],
            "group": "tp",
            "concurrent_groups": 2,
            "execution_occurrence_base": 3,
        }
        projected = _execution_event_projection(event)
        self.assertEqual(projected["phase"], "decode")
        self.assertEqual(projected["op"], "all_reduce")
        self.assertEqual(projected["bytes"], 128)
        self.assertEqual(projected["ranks"], [0, 1])
        self.assertEqual(projected["group"], "tp")
        self.assertEqual(projected["concurrent_groups"], 2)
        self.assertEqual(projected["execution_occurrence_base"], 3)
        # rank_count is derived from the normalized ranks list when present.
        self.assertEqual(projected["rank_count"], 2)

    def test_rank_count_falls_back_to_explicit_field_only_when_ranks_absent(self):
        # No "ranks" key at all, so the explicit rank_count field is trusted
        # directly instead of being derived from a normalized ranks list.
        projected = _execution_event_projection({"rank_count": 5})
        self.assertEqual(projected, {"rank_count": 5})

    def test_ranks_present_takes_priority_over_explicit_rank_count(self):
        # When both are present, the derived count from ranks wins, proving
        # the fallback branch is only reachable when ranks is truly absent.
        projected = _execution_event_projection({"ranks": [0, 1, 2], "rank_count": 999})
        self.assertEqual(projected["rank_count"], 3)

    def test_non_list_timing_samples_do_not_produce_timing_runs(self):
        projected = _execution_event_projection({"timing_samples": "not-a-list"})
        self.assertNotIn("timing_runs", projected)


class CalibrationEventProjectionTests(unittest.TestCase):
    def test_projection_of_a_field_free_event_is_empty(self):
        # phase/op/group (loop), bytes, ranks, concurrent_groups, and
        # timing_samples are all absent, exercising every skip path.
        self.assertEqual(_calibration_event_projection({}), {})

    def test_every_present_optional_field_is_projected(self):
        event = {
            "phase": "decode",
            "op": "all_reduce",
            "group": "tp",
            "bytes": 256,
            "ranks": [0, 1],
            "concurrent_groups": 4,
        }
        projected = _calibration_event_projection(event)
        self.assertEqual(projected["phase"], "decode")
        self.assertEqual(projected["op"], "all_reduce")
        self.assertEqual(projected["group"], "tp")
        self.assertEqual(projected["bytes"], 256)
        self.assertEqual(projected["ranks"], [0, 1])
        self.assertEqual(projected["concurrent_groups"], 4)

    def test_non_list_timing_samples_do_not_produce_observed_runs(self):
        projected = _calibration_event_projection({"timing_samples": {"not": "a-list"}})
        self.assertNotIn("observed_runs", projected)


if __name__ == "__main__":
    unittest.main()
