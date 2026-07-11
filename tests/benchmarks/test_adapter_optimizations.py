"""Behavioral guards for measured adapter-scale optimizations."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.fixtures import generate_compressed_canary, generate_trace, materialize_capture_shards
from commcanary.adapters import capture_merge as capture_merge_module
from commcanary.adapters import param as param_module
from commcanary.artifacts.canary import iter_canary_logical_events
from commcanary.artifacts.wire import write_json
from commcanary.errors import SchemaError
from commcanary.formats import TRACE_FORMAT


class CaptureMergeOptimizationTests(unittest.TestCase):
    def test_singleton_identities_do_not_allocate_contribution_lists(self) -> None:
        trace = generate_trace(16)
        with tempfile.TemporaryDirectory() as temp_dir:
            materialize_capture_shards(trace, Path(temp_dir))
            with mock.patch.object(
                capture_merge_module,
                "_coalesce_events",
                side_effect=AssertionError("singleton identities must use the direct finalizer"),
            ):
                merged = capture_merge_module.merge_trace_shards(temp_dir, workload_name="singleton-fast-path")

        self.assertEqual(len(merged["events"]), 16)
        self.assertEqual(
            [event["collective_id"] for event in merged["events"]],
            [f"capture-{index:06d}" for index in range(16)],
        )

    def test_cross_rank_collision_promotes_to_a_contribution_list(self) -> None:
        session = "capture-optimization-collision"
        workload = {"name": "collision"}
        with tempfile.TemporaryDirectory() as temp_dir:
            for rank in (0, 1):
                shard = {
                    "format": TRACE_FORMAT,
                    "workload": workload,
                    "system": {
                        "capture_session_id": session,
                        "clock_offset_us": 0.0,
                        "rank": str(rank),
                    },
                    "events": [
                        {
                            "id": f"event-rank-{rank}",
                            "phase": "decode",
                            "op": "all_reduce",
                            "bytes": 1024,
                            "ranks": [0, 1],
                            "group": "tp",
                            "start_us": 10.0 + rank,
                            "rank_arrival_us": {str(rank): 0.0},
                            "partial_rank_arrival": True,
                            "recorder_rank": str(rank),
                            "capture_session_id": session,
                            "collective_id": "collective-0",
                            "collective_seq": 0,
                        }
                    ],
                }
                write_json(str(Path(temp_dir, f"rank-{rank}.trace.json")), shard)

            with mock.patch.object(
                capture_merge_module,
                "_coalesce_events",
                wraps=capture_merge_module._coalesce_events,
            ) as coalesce:
                merged = capture_merge_module.merge_trace_shards(temp_dir, workload_name="collision")

        coalesce.assert_called_once()
        self.assertEqual(merged["events"][0]["recorder_ranks"], ["0", "1"])
        self.assertEqual(merged["events"][0]["rank_arrival_us"], {"0": 0.0, "1": 1.0})


class ParamExportOptimizationTests(unittest.TestCase):
    def test_production_motif_templates_are_normalized_once_and_match_uncached_output(self) -> None:
        canary = generate_compressed_canary(64, motif_pattern_length=2)

        with (
            mock.patch.object(param_module, "normalize_ranks", wraps=param_module.normalize_ranks) as normalize,
            mock.patch.object(param_module, "_expanded_gaps_us", wraps=param_module._expanded_gaps_us) as gaps,
        ):
            cached = param_module.canary_to_param_comms_trace(canary)

        def uncached_iterator(events: object, **kwargs: object):
            yield from iter_canary_logical_events(events, **kwargs)

        uncached = param_module.export_param_comms_trace(
            canary,
            logical_event_iterator=uncached_iterator,
        )

        logical_count = 64
        self.assertLess(normalize.call_count, logical_count)
        self.assertLess(gaps.call_count, logical_count)
        self.assertEqual(cached, uncached)

        body = [entry for entry in cached if entry.get("comms") != "init"]
        self.assertIsNot(body[0]["global_ranks"], body[1]["global_ranks"])
        self.assertIsNot(body[0]["markers"], body[1]["markers"])
        body[0]["global_ranks"].append(99)
        body[0]["markers"].append("mutated")
        self.assertNotIn(99, body[1]["global_ranks"])
        self.assertNotIn("mutated", body[1]["markers"])

    def test_injected_iterator_does_not_cache_mutated_template_objects(self) -> None:
        canary = generate_compressed_canary(16, motif_pattern_length=2)
        shared_ranks = [0, 1]
        event = {
            "op": "all_reduce",
            "bytes": 1024,
            "ranks": shared_ranks,
            "group": "dynamic",
            "timing_samples": [{"gap_us": 0.0, "weight": 1}],
        }

        def mutating_iterator(_events: object, **_kwargs: object):
            yield copy.copy(event)
            shared_ranks.append(2)
            yield copy.copy(event)

        with self.assertRaisesRegex(SchemaError, "two different rank sets"):
            param_module.export_param_comms_trace(
                canary,
                logical_event_iterator=mutating_iterator,
            )


if __name__ == "__main__":
    unittest.main()
