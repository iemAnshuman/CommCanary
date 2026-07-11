"""Cumulative replay clock (arrival-time accumulation) budget rejection.

This is distinct from the completion-time budget check: it fires while the
running ``logical_clock_us`` is being advanced by per-step gaps, strictly
before any collective duration for that step is computed. To isolate it we
keep each individual event's ``gap_us`` comfortably under the duration limit
(so the trace/canary schema accepts it) while making the *sum* of two such
gaps cross the limit.
"""

from __future__ import annotations

import unittest

from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary
from commcanary.schema import TRACE_FORMAT, SchemaError, validate_canary


class ReplayClockBudgetExceededTests(unittest.TestCase):
    def test_cumulative_gap_sum_exceeding_limit_is_rejected_before_completion(self):
        # MAX_TIME_US is 1e12; each gap is 6e11 (comfortably below the per-field
        # limit) but two of them sum to 1.2e12, which exceeds it.
        half_budget_gap_us = 6e11
        event = {
            "phase": "decode",
            "op": "all_reduce",
            "bytes": 1024,
            "ranks": [0, 1],
            "gap_us": half_budget_gap_us,
            "rank_arrival_us": {"0": 0.0, "1": 0.0},
        }
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "clock-budget"},
            "events": [
                {"id": "first", **event},
                {"id": "second", **event},
            ],
        }
        canary = compile_trace(trace)
        # The canary itself is well-formed; only the accumulated replay clock
        # crosses the limit.
        validate_canary(canary)
        with self.assertRaisesRegex(SchemaError, "replay logical clock exceeds maximum supported duration"):
            replay_canary(canary)


if __name__ == "__main__":
    unittest.main()
