"""Malformed replay-configuration rejections for ``replay_canary``.

Each case sends one invalid keyword argument through the public API and
checks both that a ``SchemaError`` is raised and that the message names the
actual offending field, so a future refactor cannot silently swap two
validation branches and still pass this suite.
"""

from __future__ import annotations

import unittest

from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary
from commcanary.schema import TRACE_FORMAT, SchemaError


def _single_event_trace(**overrides):
    event = {
        "id": "only",
        "phase": "decode",
        "op": "all_reduce",
        "bytes": 1024,
        "ranks": [0, 1],
        "rank_arrival_us": {"0": 0.0, "1": 0.0},
        "compute_overlap_us": 500.0,
    }
    event.update(overrides)
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "replay-config-validation"},
        "events": [event],
    }


class ReplayParameterValidationTests(unittest.TestCase):
    def setUp(self):
        self.canary = compile_trace(_single_event_trace())

    def test_ablations_accept_comma_separated_string_and_apply(self):
        baseline = replay_canary(self.canary, include_samples=True, seed=1)["samples"][0]
        self.assertEqual(baseline["compute_overlap_us"], 500.0)

        report = replay_canary(self.canary, include_samples=True, seed=1, ablations="compute_overlap")
        self.assertEqual(report["replay_protocol"]["ablations"], ["compute_overlap"])
        self.assertEqual(report["backend"]["ablations"], ["compute_overlap"])
        self.assertEqual(report["samples"][0]["compute_overlap_us"], 0.0)

    def test_unsupported_ablation_name_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "unsupported replay ablation"):
            replay_canary(self.canary, ablations=["not_a_real_ablation"])

    def test_empty_backend_label_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "backend_label must be a non-empty string"):
            replay_canary(self.canary, backend_label="")

    def test_non_boolean_include_samples_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "include_samples must be a boolean"):
            replay_canary(self.canary, include_samples=1)

    def test_max_replay_events_below_one_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "max_replay_events must be at least 1"):
            replay_canary(self.canary, max_replay_events=0)

    def test_non_positive_bandwidth_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "bandwidth_gbps must be positive"):
            replay_canary(self.canary, bandwidth_gbps=0.0)

    def test_negative_latency_floor_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "latency_floor_us must be non-negative"):
            replay_canary(self.canary, latency_floor_us=-1.0)


if __name__ == "__main__":
    unittest.main()
