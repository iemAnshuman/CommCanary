"""The ``operation_ordering`` ablation resequences logical steps deterministically."""

from __future__ import annotations

import unittest

from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary
from commcanary.schema import TRACE_FORMAT


def _two_phase_trace():
    common = {
        "op": "all_reduce",
        "bytes": 1024,
        "ranks": [0, 1],
        "group": "tp",
        "rank_arrival_us": {"0": 0.0, "1": 0.0},
    }
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "operation-ordering"},
        "events": [
            {"id": "z", "phase": "zzz_last", **common},
            {"id": "a", "phase": "aaa_first", **common},
        ],
    }


class OperationOrderingAblationTests(unittest.TestCase):
    def test_default_order_matches_source_order(self):
        canary = compile_trace(_two_phase_trace())
        samples = replay_canary(canary, include_samples=True, seed=1)["samples"]
        self.assertEqual([sample["phase"] for sample in samples], ["zzz_last", "aaa_first"])

    def test_operation_ordering_ablation_sorts_by_scheduler_key(self):
        canary = compile_trace(_two_phase_trace())
        samples = replay_canary(
            canary,
            include_samples=True,
            seed=1,
            ablations=["operation_ordering"],
        )["samples"]
        self.assertEqual([sample["phase"] for sample in samples], ["aaa_first", "zzz_last"])


if __name__ == "__main__":
    unittest.main()
