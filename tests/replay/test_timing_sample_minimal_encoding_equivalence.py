"""A timing sample may omit its cached ``gap_sum_us``/``source_start``/
``source_end`` fields and rely on ``gap_us``+``weight`` and ``source_index``
instead; ``validate_canary`` explicitly allows this (see
``_record_gap_sum``/``_occurrence_base`` fallbacks and ``_sample_interval``'s
``source_index``-only branch in ``artifacts/canary_validation.py``). This
module checks that replay treats the two encodings of the same single
occurrence (``weight == 1``) as exactly equivalent.
"""

from __future__ import annotations

import copy
import unittest

from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary
from commcanary.schema import TRACE_FORMAT, validate_canary
from tests.artifact_helpers import refresh_canary_hashes


def _two_occurrence_trace():
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "minimal-encoding"},
        "events": [
            {
                "id": "a",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 1024,
                "ranks": [0, 1],
                "group": "tp",
                "gap_us": 3.0,
                "rank_arrival_us": {"0": 0.0, "1": 1.0},
            },
            {
                "id": "b",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 1024,
                "ranks": [0, 1],
                "group": "tp",
                "gap_us": 7.0,
                "rank_arrival_us": {"0": 0.0, "1": 2.0},
            },
        ],
    }


class TimingSampleMinimalEncodingEquivalenceTests(unittest.TestCase):
    def test_dropping_cached_source_and_gap_sum_fields_replays_identically(self):
        canary = compile_trace(_two_occurrence_trace())
        timing_samples = canary["events"][0]["timing_samples"]
        # Sanity check on the fixture: a weight-1 record compiled the "full"
        # way, with both the cache fields and the fields they duplicate.
        self.assertEqual(timing_samples[0]["weight"], 1)
        self.assertEqual(timing_samples[0]["source_start"], timing_samples[0]["source_index"])
        self.assertEqual(timing_samples[0]["gap_sum_us"], timing_samples[0]["gap_us"])

        baseline_samples = replay_canary(canary, include_samples=True, seed=5)["samples"]

        minimal = copy.deepcopy(canary)
        minimal_sample = minimal["events"][0]["timing_samples"][0]
        del minimal_sample["source_start"]
        del minimal_sample["source_end"]
        del minimal_sample["gap_sum_us"]
        refresh_canary_hashes(minimal)

        # The minimal encoding is still a schema-valid canary.
        validate_canary(minimal)

        minimal_samples = replay_canary(minimal, include_samples=True, seed=5)["samples"]
        self.assertEqual(minimal_samples, baseline_samples)


if __name__ == "__main__":
    unittest.main()
