from __future__ import annotations

import copy
import dataclasses
import math
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pytest

import commcanary.adapters.capture as capture_module
from commcanary.adapters.capture import require_readable_shard
from commcanary.artifacts.json_codec import formatted_json_bytes
from commcanary.capture import TraceRecorder, _rank_label, merge_trace_shards
from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary
from commcanary.resources import DEFAULT_RESOURCE_LIMITS
from commcanary.schema import TRACE_FORMAT, SchemaError, load_json, write_json

ROOT = Path(__file__).resolve().parents[2]


class CaptureTests(unittest.TestCase):
    def test_capture_does_not_accept_stale_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "trace.json")
            write_json(output, {"format": TRACE_FORMAT, "workload": {"name": "stale"}, "events": []})
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "commcanary",
                    "capture",
                    "--output",
                    output,
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": "src"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("did not write a trace", completed.stderr)
            self.assertEqual(load_json(output)["workload"]["name"], "stale")

    def test_capture_merges_rank_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "trace.json")
            script = (
                "from commcanary.capture import record_collective\n"
                "record_collective(op='all_reduce', bytes=16, ranks=[0,1], "
                "rank_arrival_us={'0':0,'1':2})\n"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "commcanary",
                    "capture",
                    "--output",
                    output,
                    "--",
                    sys.executable,
                    "-c",
                    script,
                ],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": "src", "RANK": "7"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            trace = load_json(output)
            self.assertEqual(len(trace["events"]), 1)
            self.assertEqual(trace["system"]["shards"], 1)

    def test_capture_coalesces_duplicate_rank_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard0 = {
                "format": TRACE_FORMAT,
                "workload": {"name": "ranked"},
                "system": {"rank": "0", "capture_session_id": "s", "clock_offset_us": 0.0},
                "events": [
                    {
                        "id": "event-000000",
                        "capture_session_id": "s",
                        "collective_id": "c0",
                        "collective_seq": 0,
                        "recorder_rank": "0",
                        "phase": "decode",
                        "op": "all_reduce",
                        "bytes": 16,
                        "ranks": [0, 1],
                        "start_us": 10.0,
                        "rank_arrival_us": {"0": 0.0},
                        "partial_rank_arrival": True,
                    }
                ],
            }
            shard1 = {
                "format": TRACE_FORMAT,
                "workload": {"name": "ranked"},
                "system": {"rank": "1", "capture_session_id": "s", "clock_offset_us": -2.0},
                "events": [
                    {
                        "id": "event-000000",
                        "capture_session_id": "s",
                        "collective_id": "c0",
                        "collective_seq": 0,
                        "recorder_rank": "1",
                        "phase": "decode",
                        "op": "all_reduce",
                        "bytes": 16,
                        "ranks": [0, 1],
                        "start_us": 12.0,
                        "rank_arrival_us": {"1": 3.0},
                        "partial_rank_arrival": True,
                    }
                ],
            }
            write_json(os.path.join(tmp, "rank-0.trace.json"), shard0)
            write_json(os.path.join(tmp, "rank-1.trace.json"), shard1)
            merged = merge_trace_shards(tmp, workload_name="ranked")
            self.assertEqual(len(merged["events"]), 1)
            event = merged["events"][0]
            self.assertEqual(event["rank_arrival_us"], {"0": 0.0, "1": 3.0})
            self.assertEqual(event["merged_shards"], ["rank-0.trace.json", "rank-1.trace.json"])

    def test_capture_accepts_identical_full_maps_and_aggregates_compute_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank, start_us, overlap_us in ((0, 10.0, 0.0), (1, 0.0, 1000.0)):
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "ranked"},
                        "system": {"rank": str(rank), "capture_session_id": "s"},
                        "events": [
                            {
                                "id": "event-000000",
                                "capture_session_id": "s",
                                "collective_id": "c0",
                                "collective_seq": 0,
                                "recorder_rank": str(rank),
                                "phase": "decode",
                                "op": "all_reduce",
                                "bytes": 16,
                                "ranks": [0, 1],
                                "start_us": start_us,
                                "rank_arrival_us": {"0": 0.0, "1": 5.0},
                                "compute_overlap_us": overlap_us,
                            }
                        ],
                    },
                )
            merged = merge_trace_shards(tmp, workload_name="ranked")
            event = merged["events"][0]
            self.assertEqual(event["rank_arrival_us"], {"0": 0.0, "1": 5.0})
            self.assertEqual(event["compute_overlap_us"], 0.0)
            self.assertTrue(event["compute_fields_uncertain"])
            self.assertEqual(event["compute_by_rank"]["0"]["compute_overlap_us"], 0.0)
            self.assertEqual(event["compute_by_rank"]["1"]["compute_overlap_us"], 1000.0)
            canary = compile_trace(merged)
            self.assertTrue(canary["events"][0]["compute_fields_uncertain"])
            self.assertEqual(
                canary["compiler"]["capture_uncertainty"]["compute_fields_uncertain_events"],
                1,
            )
            report = replay_canary(canary, include_samples=True)
            self.assertTrue(report["samples"][0]["compute_fields_uncertain"])
            comparison = compare_reports(report, copy.deepcopy(report))
            self.assertEqual(comparison["verdict"], "warn")
            self.assertTrue(any("uncertain rank-local compute fields" in reason for reason in comparison["reasons"]))

    def test_capture_merge_preserves_unknown_rank_overlap_and_compile_refuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank in (0, 1):
                event = {
                    "id": "event-000000",
                    "capture_session_id": "s",
                    "collective_id": "c0",
                    "collective_seq": 0,
                    "recorder_rank": str(rank),
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "start_us": float(rank),
                    "rank_arrival_us": {"0": 0.0, "1": 1.0},
                }
                if rank == 0:
                    event["compute_overlap_us"] = 3.0
                else:
                    event["compute_overlap_unknown"] = True
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "ranked"},
                        "system": {"rank": str(rank), "capture_session_id": "s"},
                        "events": [event],
                    },
                )

            merged = merge_trace_shards(tmp, workload_name="ranked")
            merged_event = merged["events"][0]
            self.assertNotIn("compute_overlap_us", merged_event)
            self.assertTrue(merged_event["compute_overlap_unknown"])
            self.assertEqual(merged_event["compute_by_rank"]["0"]["compute_overlap_us"], 3.0)
            self.assertTrue(merged_event["compute_by_rank"]["1"]["compute_overlap_unknown"])
            with self.assertRaisesRegex(SchemaError, "unknown compute overlap"):
                compile_trace(merged)

    def test_compute_uncertainty_is_record_scoped_inside_repeated_motifs(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "uncertainty-scope"},
            "events": [
                {
                    "id": "a",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                    "compute_overlap_us": 0.0,
                    "compute_fields_uncertain": True,
                },
                {
                    "id": "b",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                    "compute_overlap_us": 0.0,
                },
            ],
        }
        canary = compile_trace(trace)
        self.assertEqual(canary["events"][0]["repeat"], 2)
        self.assertTrue(canary["events"][0]["compute_fields_uncertain"])
        self.assertEqual(
            canary["compiler"]["capture_uncertainty"]["compute_fields_uncertain_events"],
            1,
        )
        samples = replay_canary(canary, include_samples=True)["samples"]
        self.assertEqual(
            [sample.get("compute_fields_uncertain", False) for sample in samples],
            [True, False],
        )

    def test_capture_rejects_rank_swapped_partial_arrivals_and_mixed_clocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {
                "capture_session_id": "s",
                "collective_id": "c0",
                "collective_seq": 0,
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 0.0,
                "partial_rank_arrival": True,
            }
            write_json(
                os.path.join(tmp, "rank-0.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "0", "clock_offset_us": 0.0},
                    "events": [{**base, "recorder_rank": "0", "rank_arrival_us": {"1": 0.0}}],
                },
            )
            write_json(
                os.path.join(tmp, "rank-1.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "1", "clock_offset_us": 0.0},
                    "events": [{**base, "recorder_rank": "1", "rank_arrival_us": {"0": 0.0}}],
                },
            )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="ranked")

        with tempfile.TemporaryDirectory() as tmp:
            for rank in (0, 1):
                system = {"rank": str(rank)}
                if rank == 0:
                    system["clock_offset_us"] = 0.0
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "ranked"},
                        "system": system,
                        "events": [
                            {
                                **base,
                                "recorder_rank": str(rank),
                                "rank_arrival_us": {str(rank): 0.0},
                            }
                        ],
                    },
                )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="ranked")

    def test_capture_rejects_reused_and_conflicting_collective_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            for index in range(2):
                events.append(
                    {
                        "id": f"event-{index}",
                        "capture_session_id": "s",
                        "collective_id": "c0",
                        "collective_seq": index,
                        "recorder_rank": "0",
                        "phase": "decode",
                        "op": "all_reduce",
                        "bytes": 16,
                        "ranks": [0, 1],
                        "start_us": float(index),
                        "rank_arrival_us": {"0": 0.0},
                        "partial_rank_arrival": True,
                    }
                )
            write_json(
                os.path.join(tmp, "rank-0.trace.json"),
                {"format": TRACE_FORMAT, "workload": {"name": "ranked"}, "system": {"rank": "0"}, "events": events},
            )
            write_json(
                os.path.join(tmp, "rank-1.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "1"},
                    "events": [
                        {
                            **events[0],
                            "id": "event-0-r1",
                            "recorder_rank": "1",
                            "rank_arrival_us": {"1": 0.0},
                        }
                    ],
                },
            )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="ranked")

        with tempfile.TemporaryDirectory() as tmp:
            base = {
                "capture_session_id": "s",
                "collective_id": "c0",
                "collective_seq": 0,
                "phase": "decode",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 0.0,
                "partial_rank_arrival": True,
            }
            write_json(
                os.path.join(tmp, "rank-0.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "0"},
                    "events": [{**base, "op": "all_reduce", "recorder_rank": "0", "rank_arrival_us": {"0": 0.0}}],
                },
            )
            write_json(
                os.path.join(tmp, "rank-1.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "1"},
                    "events": [{**base, "op": "broadcast", "recorder_rank": "1", "rank_arrival_us": {"1": 0.0}}],
                },
            )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="ranked")

    def test_capture_merges_send_recv_as_point_to_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {
                "capture_session_id": "s",
                "collective_id": "msg-0",
                "collective_seq": 0,
                "phase": "decode",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 0.0,
                "partial_rank_arrival": True,
            }
            write_json(
                os.path.join(tmp, "rank-0.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "0", "capture_session_id": "s", "clock_offset_us": 0.0},
                    "events": [{**base, "op": "send", "recorder_rank": "0", "rank_arrival_us": {"0": 0.0}}],
                },
            )
            write_json(
                os.path.join(tmp, "rank-1.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "1", "capture_session_id": "s", "clock_offset_us": 0.0},
                    "events": [{**base, "op": "recv", "recorder_rank": "1", "rank_arrival_us": {"1": 0.0}}],
                },
            )
            merged = merge_trace_shards(tmp, workload_name="ranked")
            event = merged["events"][0]
            self.assertEqual(event["op"], "point_to_point")
            self.assertEqual(event["sender_rank"], 0)
            self.assertEqual(event["receiver_rank"], 1)

    def test_capture_requires_session_and_all_scalar_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = {
                "collective_id": "c0",
                "collective_seq": 0,
                "recorder_rank": "0",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 0.0,
                "arrival_skew_us": 1.0,
            }
            write_json(
                os.path.join(tmp, "rank-0.trace.json"),
                {"format": TRACE_FORMAT, "workload": {"name": "ranked"}, "system": {"rank": "0"}, "events": [event]},
            )
            write_json(
                os.path.join(tmp, "rank-1.trace.json"),
                {"format": TRACE_FORMAT, "workload": {"name": "ranked"}, "system": {"rank": "1"}, "events": []},
            )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="ranked")

        with tempfile.TemporaryDirectory() as tmp:
            event = {
                "capture_session_id": "s",
                "collective_id": "c0",
                "collective_seq": 0,
                "recorder_rank": "0",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 0.0,
                "arrival_skew_us": 1.0,
            }
            write_json(
                os.path.join(tmp, "rank-0.trace.json"),
                {"format": TRACE_FORMAT, "workload": {"name": "ranked"}, "system": {"rank": "0"}, "events": [event]},
            )
            write_json(
                os.path.join(tmp, "rank-1.trace.json"),
                {"format": TRACE_FORMAT, "workload": {"name": "ranked"}, "system": {"rank": "1"}, "events": []},
            )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="ranked")

    def test_capture_rejects_missing_rank_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_event = {
                "capture_session_id": "s",
                "collective_id": "c0",
                "collective_seq": 0,
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0, 1, 2],
                "start_us": 10.0,
                "partial_rank_arrival": True,
            }
            write_json(
                os.path.join(tmp, "rank-0.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "0", "capture_session_id": "s"},
                    "events": [{**base_event, "recorder_rank": "0", "rank_arrival_us": {"0": 0.0}}],
                },
            )
            write_json(
                os.path.join(tmp, "rank-1.trace.json"),
                {
                    "format": TRACE_FORMAT,
                    "workload": {"name": "ranked"},
                    "system": {"rank": "1", "capture_session_id": "s"},
                    "events": [{**base_event, "recorder_rank": "1", "rank_arrival_us": {"1": 0.0}}],
                },
            )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="ranked")

    def test_capture_preserves_scalar_skew_when_rank_maps_are_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank in (0, 1):
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "ranked"},
                        "system": {"rank": str(rank), "capture_session_id": "s"},
                        "events": [
                            {
                                "id": "event-000000",
                                "capture_session_id": "s",
                                "collective_id": "c0",
                                "collective_seq": 0,
                                "recorder_rank": str(rank),
                                "phase": "decode",
                                "op": "all_reduce",
                                "bytes": 16,
                                "ranks": [0, 1],
                                "start_us": 10.0 + rank,
                                "arrival_skew_us": 7.0,
                            }
                        ],
                    },
                )
            merged = merge_trace_shards(tmp, workload_name="ranked")
            event = merged["events"][0]
            self.assertNotIn("rank_arrival_us", event)
            self.assertEqual(event["arrival_skew_us"], 7.0)
            self.assertTrue(event["arrival_skew_unknown"])

    def test_uncalibrated_rank_maps_do_not_invent_cross_rank_skew(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank, start_us in ((0, 0.0), (1, 20.0)):
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "ranked"},
                        "system": {"rank": str(rank), "capture_session_id": "s"},
                        "events": [
                            {
                                "id": "event-000000",
                                "capture_session_id": "s",
                                "collective_id": "c0",
                                "collective_seq": 0,
                                "recorder_rank": str(rank),
                                "phase": "decode",
                                "op": "all_reduce",
                                "bytes": 16,
                                "ranks": [0, 1],
                                "start_us": start_us,
                                "rank_arrival_us": {str(rank): 0.0},
                                "partial_rank_arrival": True,
                            }
                        ],
                    },
                )
            merged = merge_trace_shards(tmp, workload_name="ranked")
            event = merged["events"][0]
            self.assertNotIn("rank_arrival_us", event)
            self.assertTrue(event["arrival_skew_unknown"])
            with self.assertRaises(SchemaError):
                compile_trace(merged)

    def test_uncalibrated_disjoint_rank_domains_are_not_globally_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank, collective_id, ranks, start_us in (
                (0, "a", [0, 1], 100.0),
                (1, "a", [0, 1], 105.0),
                (2, "b", [2, 3], 0.0),
                (3, "b", [2, 3], 5.0),
            ):
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "clock-domains"},
                        "system": {"rank": str(rank), "capture_session_id": "s"},
                        "events": [
                            {
                                "id": f"{collective_id}-{rank}",
                                "capture_session_id": "s",
                                "collective_id": collective_id,
                                "collective_seq": 0,
                                "recorder_rank": str(rank),
                                "phase": "decode",
                                "op": "all_reduce",
                                "bytes": 16,
                                "ranks": ranks,
                                "start_us": start_us,
                                "rank_arrival_us": {str(rank): 0.0},
                                "partial_rank_arrival": True,
                            }
                        ],
                    },
                )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="clock-domains")

    def test_uncalibrated_overlapping_rank_domains_are_not_globally_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_events = {
                0: [("a", [0, 1], 500.0)],
                1: [("a", [0, 1], 0.0), ("b", [1, 2], 10.0)],
                2: [("b", [1, 2], 20.0)],
            }
            for rank, records in shard_events.items():
                events = []
                for collective_id, ranks, start_us in records:
                    events.append(
                        {
                            "id": f"{collective_id}-{rank}",
                            "capture_session_id": "s",
                            "collective_id": collective_id,
                            "collective_seq": 0,
                            "recorder_rank": str(rank),
                            "phase": "decode",
                            "op": "all_reduce",
                            "bytes": 16,
                            "ranks": ranks,
                            "start_us": start_us,
                            "rank_arrival_us": {str(rank): 0.0},
                            "partial_rank_arrival": True,
                        }
                    )
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "clock-domains"},
                        "system": {"rank": str(rank), "capture_session_id": "s"},
                        "events": events,
                    },
                )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="clock-domains")

    def test_uncalibrated_merge_uses_canonical_rank_and_conservative_compute(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank, start_us, overlap, pressure, before in (
                (0, 100.0, 0.0, 0.25, 2.0),
                (1, 0.0, 1000.0, 1.2, 7.0),
            ):
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "ranked"},
                        "system": {"rank": str(rank), "capture_session_id": "s"},
                        "events": [
                            {
                                "id": "event-000000",
                                "capture_session_id": "s",
                                "collective_id": "c0",
                                "collective_seq": 0,
                                "recorder_rank": str(rank),
                                "phase": "decode",
                                "op": "all_reduce",
                                "bytes": 16,
                                "ranks": [0, 1],
                                "start_us": start_us,
                                "rank_arrival_us": {str(rank): 0.0},
                                "partial_rank_arrival": True,
                                "compute_overlap_us": overlap,
                                "compute_pressure": pressure,
                                "compute_before_us": before,
                            }
                        ],
                    },
                )
            merged = merge_trace_shards(tmp, workload_name="ranked")
            event = merged["events"][0]
            self.assertEqual(event["start_us"], 100.0)
            self.assertEqual(event["compute_overlap_us"], 0.0)
            self.assertEqual(event["compute_pressure"], 1.2)
            self.assertEqual(event["compute_before_us"], 7.0)
            self.assertTrue(event["compute_fields_uncertain"])
            self.assertEqual(event["compute_by_rank"]["0"]["compute_overlap_us"], 0.0)
            self.assertEqual(event["compute_by_rank"]["1"]["compute_overlap_us"], 1000.0)

    def test_rank_label_prefers_global_rank_and_sharded_names_are_merged(self):
        original = {name: os.environ.get(name) for name in ("COMMCANARY_RANK", "RANK", "LOCAL_RANK")}
        try:
            os.environ.pop("COMMCANARY_RANK", None)
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "8"
            self.assertEqual(_rank_label(), "8")
            os.environ["COMMCANARY_RANK"] = "42"
            self.assertEqual(_rank_label(), "42")
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        with tempfile.TemporaryDirectory() as tmp:
            shard = {
                "format": TRACE_FORMAT,
                "workload": {"name": "ranked"},
                "system": {"rank": "0", "capture_session_id": "s"},
                "events": [
                    {
                        "id": "event-000000",
                        "capture_session_id": "s",
                        "collective_id": "c0",
                        "collective_seq": 0,
                        "recorder_rank": "0",
                        "phase": "decode",
                        "op": "all_reduce",
                        "bytes": 16,
                        "ranks": [0],
                        "start_us": 0.0,
                        "rank_arrival_us": {"0": 0.0},
                    }
                ],
            }
            write_json(os.path.join(tmp, "capture.trace.rank-0-pid-123.json"), shard)
            merged = merge_trace_shards(tmp, workload_name="ranked")
            self.assertEqual(len(merged["events"]), 1)

    def test_direct_recorder_resets_after_fork_without_global_singleton(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                "from commcanary.capture import TraceRecorder\n"
                f"rec = TraceRecorder({os.path.join(tmp, 'trace.json')!r})\n"
                "pid = __import__('os').fork()\n"
                "if pid == 0:\n"
                "    rec.record_collective(op='all_reduce', bytes=16, ranks=[0], rank_arrival_us={'0': 0.0})\n"
                "    rec.save()\n"
                "    raise SystemExit(0)\n"
                "__import__('os').waitpid(pid, 0)\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": "src", "RANK": "0"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_recorder_rejects_fractional_nonfinite_and_negative_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TraceRecorder(os.path.join(tmp, "trace.json"))
            with self.assertRaises(SchemaError):
                recorder.record_collective(op="all_reduce", bytes=1.5, ranks=[0])
            with self.assertRaises(SchemaError):
                recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], compute_pressure=math.nan)
            with self.assertRaises(SchemaError):
                recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], observed_exposed_us=-1.0)
            with self.assertRaises(SchemaError):
                recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], concurrent_groups=1.5)

    def test_recorder_trace_snapshot_does_not_share_nested_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            workload = {"name": "snapshot", "tags": ["before"]}
            metadata = {"nested": {"value": 1}}
            recorder = TraceRecorder(
                os.path.join(tmp, "trace.json"),
                workload=workload,
            )
            recorder.record_collective(
                op="all_reduce",
                bytes=16,
                ranks=[0],
                rank_arrival_us={"0": 0.0},
                metadata=metadata,
            )
            workload["tags"].append("after")
            metadata["nested"]["value"] = 2
            trace = recorder.to_trace()
            trace["workload"]["tags"].append("after")
            trace["events"][0]["metadata"]["nested"]["value"] = 2

            self.assertEqual(recorder.workload["tags"], ["before"])
            self.assertEqual(recorder.events[0]["metadata"]["nested"]["value"], 1)

    def test_recorder_shards_are_unique_and_sessions_do_not_mix(self):
        original_dir = os.environ.get("COMMCANARY_TRACE_DIR")
        original_rank = os.environ.get("RANK")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["COMMCANARY_TRACE_DIR"] = tmp
                os.environ["RANK"] = "0"
                first = TraceRecorder(os.path.join(tmp, "trace.json"))
                second = TraceRecorder(os.path.join(tmp, "trace.json"))
                self.assertNotEqual(first.output_path, second.output_path)
                first.record_collective(op="all_reduce", bytes=16, ranks=[0], rank_arrival_us={"0": 0.0})
                second.record_collective(op="all_reduce", bytes=16, ranks=[0], rank_arrival_us={"0": 0.0})
                first.save()
                second.save()
                self.assertTrue(os.path.exists(first.output_path))
                self.assertTrue(os.path.exists(second.output_path))
        finally:
            if original_dir is None:
                os.environ.pop("COMMCANARY_TRACE_DIR", None)
            else:
                os.environ["COMMCANARY_TRACE_DIR"] = original_dir
            if original_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = original_rank

        with tempfile.TemporaryDirectory() as tmp:
            for rank, session in ((0, "s0"), (1, "s1")):
                write_json(
                    os.path.join(tmp, f"rank-{rank}.trace.json"),
                    {
                        "format": TRACE_FORMAT,
                        "workload": {"name": "mixed"},
                        "system": {"rank": str(rank), "capture_session_id": session},
                        "events": [
                            {
                                "capture_session_id": session,
                                "collective_id": "c0",
                                "recorder_rank": str(rank),
                                "op": "all_reduce",
                                "bytes": 16,
                                "ranks": [0, 1],
                                "start_us": 0.0,
                                "rank_arrival_us": {str(rank): 0.0},
                                "partial_rank_arrival": True,
                            }
                        ],
                    },
                )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="mixed")

        with tempfile.TemporaryDirectory() as tmp:
            events = []
            for index, session in enumerate(("s0", "s1")):
                events.append(
                    {
                        "id": f"event-{index}",
                        "capture_session_id": session,
                        "op": "all_reduce",
                        "bytes": 16,
                        "ranks": [0],
                        "start_us": float(index),
                        "rank_arrival_us": {"0": 0.0},
                    }
                )
            write_json(
                os.path.join(tmp, "single.trace.json"),
                {"format": TRACE_FORMAT, "workload": {"name": "mixed"}, "system": {}, "events": events},
            )
            with self.assertRaises(SchemaError):
                merge_trace_shards(tmp, workload_name="mixed")


def test_capture_refuses_to_write_a_shard_it_could_not_read_back() -> None:
    """Write and read must agree, or a finished run is lost.

    validate_trace bounds stored events, but merging applies the bounded JSON
    loader, whose item budget binds far earlier -- roughly 21 items per event
    for a coalescing shard against a declared max_stored_events of 1,000,000.
    Without this check capture writes a shard merge_trace_shards then refuses,
    and it is discovered only after the workload has finished.
    """

    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_json_items=2_000)
    events = [
        {
            "op": "all_reduce",
            "bytes": 1024,
            "ranks": [0, 1],
            "dtype": "bfloat16",
            "reduction_op": "sum",
            "start_us": float(index),
            "compute_overlap_us": 1.0,
        }
        for index in range(400)
    ]
    trace = {
        "format": TRACE_FORMAT,
        "workload": {"name": "readability"},
        "system": {"world_size": 2},
        "events": events,
    }

    with pytest.raises(SchemaError, match="could not read back|would exceed the bounded JSON loader"):
        require_readable_shard(trace, limits=limits, path="/tmp/shard.trace.json")

    # The same document is accepted once the budget can actually admit it.
    require_readable_shard(trace, limits=DEFAULT_RESOURCE_LIMITS, path="/tmp/shard.trace.json")


def test_capture_rolls_one_rank_into_readable_shards_and_merges_every_event_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    limits = dataclasses.replace(DEFAULT_RESOURCE_LIMITS, max_json_items=500)
    source_ids = [f"collective-{index:04d}" for index in range(80)]
    direct_output = tmp_path / "capture.json"
    recorder = TraceRecorder(
        str(direct_output),
        workload={"name": "rollover"},
        limits=limits,
    )
    try:
        for index, collective_id in enumerate(source_ids):
            recorder.record_collective(
                op="all_reduce",
                bytes=16,
                ranks=[0],
                start_us=float(index),
                rank_arrival_us={"0": 0.0},
                collective_id=collective_id,
            )
            if index == 4:
                recorder.save()
        recorder.save()
    finally:
        recorder.close()

    shard_paths = sorted(tmp_path.glob("*.json"))
    assert len(shard_paths) >= 2
    assert direct_output not in shard_paths
    headroom_limits = dataclasses.replace(limits, max_json_items=limits.max_json_items - 1)
    for shard_path in shard_paths:
        load_json(str(shard_path), limits=headroom_limits)

    merged = merge_trace_shards(str(tmp_path), workload_name="rollover", limits=limits)
    merged_ids = [event["collective_id"] for event in merged["events"]]
    assert len(merged_ids) == len(source_ids)
    assert sorted(merged_ids) == source_ids


def test_small_capture_keeps_the_single_shard_wire_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_created_at = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)

    class FixedDatetime:
        @classmethod
        def now(cls, tz: timezone) -> datetime:
            assert tz is timezone.utc
            return fixed_created_at

    monkeypatch.setattr(capture_module, "datetime", FixedDatetime)
    monkeypatch.setenv("RANK", "0")
    output_path = tmp_path / "capture.trace.json"
    recorder = TraceRecorder(
        str(output_path),
        workload={"name": "small"},
        system={"world_size": 1},
    )
    try:
        recorder.record_collective(
            op="all_reduce",
            bytes=16,
            ranks=[0],
            start_us=1.0,
            rank_arrival_us={"0": 0.0},
        )
        expected = {
            "format": TRACE_FORMAT,
            "created_at": fixed_created_at.isoformat(),
            "workload": {"name": "small"},
            "system": {
                "world_size": 1,
                "pid": os.getpid(),
                "rank": "0",
                "capture_session_id": recorder._session_id,
                "recorder_id": recorder._recorder_id,
            },
            "events": copy.deepcopy(recorder.events),
        }
        recorder.save()
    finally:
        recorder.close()

    assert sorted(tmp_path.glob("*.json")) == [output_path]
    assert output_path.read_bytes() == formatted_json_bytes(expected, indent=2)
