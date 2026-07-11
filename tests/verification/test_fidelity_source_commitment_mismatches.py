"""Source-commitment failures when a canary is checked against the wrong trace.

``verify_canary_fidelity`` independently recomputes each canary event's
source correspondence from the trace it is handed. If that trace has fewer
events than the canary's own ``compiler.source_events`` declares (for example
because the wrong trace file was passed in), the reconstructed source-event
pool is shorter than what the canary's event repeats consume. This must be
reported as an explicit, itemized failure rather than raising or silently
under-checking, since assurance tooling depends on those failure reasons to
explain *why* an artifact does not correspond to its claimed source.
"""

from __future__ import annotations

import unittest

from commcanary.compiler import compile_trace, verify_canary_fidelity
from commcanary.schema import TRACE_FORMAT


def _identical_event_trace(count: int) -> dict:
    events = []
    for index in range(count):
        events.append(
            {
                "id": f"event-{index}",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 1024,
                "ranks": [0, 1],
                "group": "tp",
                "gap_us": 1.0,
                "rank_arrival_us": {"0": 0.0, "1": 0.0},
            }
        )
    return {"format": TRACE_FORMAT, "workload": {"name": "identical-events"}, "events": events}


class FidelitySourceCommitmentMismatchTests(unittest.TestCase):
    def test_verifying_against_a_shorter_trace_reports_explicit_source_shortfalls(self):
        # All 20 events share one operation identity, so the compiler
        # collapses them into a single canary event with repeat=20.
        full_trace = _identical_event_trace(20)
        canary = compile_trace(full_trace)
        self.assertEqual(len(canary["events"]), 1)
        self.assertEqual(canary["events"][0]["repeat"], 20)
        self.assertEqual(canary["compiler"]["source_events"], 20)

        # A different (shorter) trace of the same shape as the one that
        # produced the canary: only 5 of the 20 claimed source events exist.
        shorter_trace = _identical_event_trace(5)

        verification = verify_canary_fidelity(shorter_trace, canary)

        self.assertEqual(verification["status"], "failed")
        source_commitments = next(check for check in verification["checks"] if check["name"] == "source_commitments")
        self.assertEqual(source_commitments["status"], "fail")
        reasons = {failure["reason"] for failure in source_commitments["failures"]}

        # The per-event repeat could not be satisfied from the shorter pool.
        shortfall = next(
            failure
            for failure in source_commitments["failures"]
            if failure["reason"] == "source slice shorter than canary repeat"
        )
        self.assertEqual(shortfall["expected"], 20)
        self.assertEqual(shortfall["actual"], 5)

        # Because the single event's repeat could not be consumed, the
        # overall pointer never reaches the end of the (shorter) source pool
        # either, for both the logical-event walk and the stored-block walk.
        self.assertIn("canary events do not consume all selected source events", reasons)
        self.assertIn("stored source blocks do not consume all selected source events", reasons)


if __name__ == "__main__":
    unittest.main()
