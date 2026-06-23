from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from commcanary.compare import compare_reports
from commcanary.cli import main as cli_main
from commcanary.compiler import compile_trace
from commcanary.capture import TraceRecorder, _rank_label, merge_trace_shards
from commcanary.html_report import render_compare_html, render_report_html
from commcanary.replay import replay_canary
from commcanary.schema import (
    SchemaError,
    TRACE_FORMAT,
    as_int,
    canary_execution_sha256,
    load_json,
    validate_canary,
    validate_report,
    validate_trace,
    write_json,
)


def small_trace():
    ranks = [0, 1, 2, 3]
    events = []
    for index in range(6):
        events.append(
            {
                "id": f"decode-{index}",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 128 * 1024,
                "ranks": ranks,
                "group": "tp0",
                "start_us": index * 40.0,
                "rank_arrival_us": {"0": 0.0, "1": 2.0, "2": 4.0, "3": 10.0 + (index % 2)},
                "compute_overlap_us": 15.0,
            }
        )
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "unit"},
        "events": events,
    }


class CommCanaryTests(unittest.TestCase):
    def test_compile_compresses_repeated_events(self):
        canary = compile_trace(small_trace())
        self.assertEqual(canary["format"], "commcanary.canary.v2")
        self.assertLess(len(canary["events"]), len(small_trace()["events"]))
        self.assertEqual(canary["compiler"]["source_events"], 6)

    def test_replay_produces_tail_metrics(self):
        canary = compile_trace(small_trace())
        report = replay_canary(canary, seed=3)
        metrics = report["metrics"]
        self.assertEqual(metrics["count"], 6)
        self.assertGreaterEqual(metrics["p99_us"], metrics["median_us"])
        self.assertGreater(metrics["arrival_skew_p95_us"], 0.0)

    def test_compare_detects_candidate_regression(self):
        canary = compile_trace(small_trace())
        baseline = replay_canary(canary, latency_floor_us=7.0, seed=3)
        candidate = replay_canary(canary, latency_floor_us=20.0, seed=3)
        comparison = compare_reports(baseline, candidate, p99_threshold_pct=10.0)
        self.assertEqual(comparison["verdict"], "fail")
        self.assertGreater(comparison["delta"]["p99_pct"], 10.0)

    def test_non_finite_metrics_are_rejected(self):
        canary = compile_trace(small_trace())
        baseline = replay_canary(canary, seed=3)
        candidate = replay_canary(canary, seed=3)
        candidate["metrics"]["p99_us"] = math.nan
        with self.assertRaises(SchemaError):
            compare_reports(baseline, candidate)
        candidate = replay_canary(canary, seed=3)
        candidate["metrics"] = {}
        with self.assertRaises(SchemaError):
            compare_reports(baseline, candidate)
        with tempfile.TemporaryDirectory() as tmp:
            candidate = replay_canary(canary, seed=3)
            candidate["metrics"]["p99_us"] = math.nan
            with self.assertRaises(SchemaError):
                write_json(os.path.join(tmp, "bad.json"), candidate)

    def test_timing_samples_preserve_joint_correlation_and_are_bounded(self):
        trace = small_trace()
        trace["events"] = []
        for index in range(100):
            trace["events"].append(
                {
                    "id": f"decode-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 128 * 1024,
                    "ranks": [0, 1, 2, 3],
                    "start_us": index * 40.0,
                    "rank_arrival_us": {"0": 0.0, "1": 1.0, "2": 2.0, "3": float(index)},
                    "compute_overlap_us": float(index * 10),
                }
            )
        canary = compile_trace(trace, timing_sample_limit=8)
        event = canary["events"][0]
        self.assertEqual(event["repeat"], 100)
        self.assertLessEqual(len(event["timing_samples"]), 8)
        for sample in event["timing_samples"]:
            self.assertEqual(sample["compute_overlap_us"], sample["arrival_offsets_us"][3] * 10)
        self.assertNotIn("source_event_ids", event)
        self.assertEqual(event["source"]["count"], 100)
        source_indices = [sample["source_index"] for sample in event["timing_samples"]]
        self.assertEqual(source_indices, sorted(source_indices))
        self.assertEqual(sum(sample["weight"] for sample in event["timing_samples"]), 100)

    def test_pressure_and_compute_before_fidelity_are_audited(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "pressure"}, "events": []}
        for index in range(80):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "start_us": index * 10.0,
                    "rank_arrival_us": {"0": 0.0, "1": float(index % 7)},
                    "compute_before_us": float((index * 13) % 23),
                    "compute_pressure": ((index * 37) % 101) / 100.0,
                }
            )
        canary = compile_trace(trace, timing_sample_limit=4)
        fidelity = canary["compiler"]["fidelity"]
        self.assertEqual(fidelity["mode"], "bounded_approximate")
        self.assertGreater(fidelity["max_pressure_error"], 0.0)
        self.assertGreater(fidelity["max_compute_before_error_us"], 0.0)

        malformed = copy.deepcopy(canary)
        malformed["compiler"]["fidelity"]["max_pressure_error"] = 0.0
        with self.assertRaises(SchemaError):
            validate_canary(malformed)

        malformed = copy.deepcopy(canary)
        malformed["compiler"]["fidelity"]["max_compute_before_error_us"] = 0.0
        with self.assertRaises(SchemaError):
            validate_canary(malformed)

        malformed = copy.deepcopy(canary)
        interval = next(
            sample
            for sample in malformed["events"][0]["timing_samples"]
            if sample.get("approximation") == "bounded_interval"
        )
        interval.pop("max_pressure_error")
        with self.assertRaises(SchemaError):
            validate_canary(malformed)

    def test_max_events_sorts_before_truncating_and_rejects_negative(self):
        trace = small_trace()
        trace["events"] = [
            {**trace["events"][0], "id": "late", "start_us": 100.0, "bytes": 256},
            {**trace["events"][1], "id": "early", "start_us": 1.0, "bytes": 512},
        ]
        canary = compile_trace(trace, max_events=1)
        self.assertEqual(canary["events"][0]["bytes"], 512)
        with self.assertRaises(SchemaError):
            compile_trace(trace, max_events=-1)

    def test_invalid_ranks_and_arrival_maps_are_rejected(self):
        trace = small_trace()
        trace["events"][0]["ranks"] = [0, 1, 1]
        with self.assertRaises(SchemaError):
            compile_trace(trace)
        trace = small_trace()
        trace["events"][0]["rank_arrival_us"] = {"0": 0.0, "1": 1.0, "2": 2.0}
        with self.assertRaises(SchemaError):
            compile_trace(trace)
        trace = small_trace()
        trace["events"][0]["ranks"] = [0, 1.5, 2, 3]
        with self.assertRaises(SchemaError):
            compile_trace(trace)

    def test_replay_rejects_bad_iterations_and_preserves_latency_invariant(self):
        canary = compile_trace(small_trace())
        with self.assertRaises(SchemaError):
            replay_canary(canary, iterations=0)
        with self.assertRaises(SchemaError):
            replay_canary(canary, overlap_efficiency=-0.1)
        report = replay_canary(canary, include_samples=True, overlap_efficiency=1.0)
        for sample in report["samples"]:
            self.assertAlmostEqual(sample["hidden_us"] + sample["exposed_us"], sample["total_us"], places=3)

    def test_replay_resets_scheduler_state_per_iteration(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "one-event"},
            "events": [
                {
                    "id": "only",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "start_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                }
            ],
        }
        report = replay_canary(compile_trace(trace), iterations=3, include_samples=True)
        self.assertEqual([sample["queue_wait_us"] for sample in report["samples"]], [0.0, 0.0, 0.0])

    def test_replay_readiness_uses_gaps_not_compute_before(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "same-start"},
            "events": [
                {
                    "id": "a",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "group": "a",
                    "start_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                    "compute_before_us": 100.0,
                },
                {
                    "id": "b",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "group": "b",
                    "start_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                    "compute_before_us": 100.0,
                },
            ],
        }
        report = replay_canary(compile_trace(trace), include_samples=True)
        self.assertEqual([sample["ready_us"] for sample in report["samples"]], [0.0, 0.0])

    def test_compute_before_is_gap_fallback_when_timestamps_are_absent(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "compute-before"},
            "events": [
                {
                    "id": "a",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                },
                {
                    "id": "b",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                    "compute_before_us": 1_000_000.0,
                },
            ],
        }
        report = replay_canary(compile_trace(trace), include_samples=True)
        self.assertEqual([sample["ready_us"] for sample in report["samples"]], [0.0, 1_000_000.0])

    def test_scheduler_overlaps_queue_delay_with_arrival_skew(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "scheduler"},
            "events": [
                {
                    "id": "a",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "group": "tp",
                    "start_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 10.0},
                },
                {
                    "id": "b",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "group": "tp",
                    "start_us": 5.0,
                    "rank_arrival_us": {"0": 0.0, "1": 10.0},
                },
            ],
        }
        samples = replay_canary(compile_trace(trace), include_samples=True, seed=1)["samples"]
        first_completion = samples[0]["collective_start_us"] + samples[0]["collective_us"]
        expected_start = max(samples[1]["last_arrival_us"], first_completion)
        self.assertAlmostEqual(samples[1]["collective_start_us"], expected_start, places=3)
        self.assertAlmostEqual(samples[1]["queue_wait_us"], expected_start - samples[1]["last_arrival_us"], places=3)

    def test_default_scheduler_resources_are_rank_scoped(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "independent-default-groups"},
            "events": [
                {
                    "id": "a",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "start_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                },
                {
                    "id": "b",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [2, 3],
                    "start_us": 0.0,
                    "rank_arrival_us": {"2": 0.0, "3": 0.0},
                },
            ],
        }
        samples = replay_canary(compile_trace(trace), include_samples=True, seed=1)["samples"]
        self.assertEqual(samples[0]["group"], "default")
        self.assertEqual(samples[1]["group"], "default")
        self.assertNotEqual(samples[0]["scheduler_resource"], samples[1]["scheduler_resource"])
        self.assertEqual([sample["queue_wait_us"] for sample in samples], [0.0, 0.0])

    def test_report_mismatch_is_rejected_by_default(self):
        baseline = replay_canary(compile_trace(small_trace()), seed=3)
        other = small_trace()
        other["events"][0]["bytes"] = 64 * 1024
        candidate = replay_canary(compile_trace(other), seed=3)
        with self.assertRaises(SchemaError):
            compare_reports(baseline, candidate)
        comparison = compare_reports(baseline, candidate, require_compatible=False)
        self.assertFalse(comparison["compatibility"]["compatible"])

        same_canary_candidate = replay_canary(compile_trace(small_trace()), seed=99)
        with self.assertRaises(SchemaError):
            compare_reports(baseline, same_canary_candidate)
        comparison = compare_reports(baseline, same_canary_candidate, require_compatible=False)
        self.assertIn(comparison["verdict"], {"warn", "fail"})
        self.assertFalse(comparison["compatibility"]["compatible"])

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
                cwd=os.path.dirname(os.path.dirname(__file__)),
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
                cwd=os.path.dirname(os.path.dirname(__file__)),
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
            self.assertTrue(
                any("uncertain rank-local compute fields" in reason for reason in comparison["reasons"])
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
                cwd=os.path.dirname(os.path.dirname(__file__)),
                env={**os.environ, "PYTHONPATH": "src", "RANK": "0"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_model_is_monotonic_and_bandwidth_uses_gbits(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "sizes"},
            "events": [
                {
                    "id": "x",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 262144,
                    "ranks": [0, 1, 2, 3],
                    "rank_arrival_us": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0},
                }
            ],
        }
        small = replay_canary(compile_trace(trace), seed=1)
        trace["events"][0]["bytes"] = 262145
        large = replay_canary(compile_trace(trace), seed=1)
        self.assertGreaterEqual(large["metrics"]["median_us"], small["metrics"]["median_us"])

        slow = replay_canary(compile_trace(trace), seed=1, bandwidth_gbps=0.1)
        fast = replay_canary(compile_trace(trace), seed=1, bandwidth_gbps=1.0)
        self.assertGreater(slow["metrics"]["median_us"], fast["metrics"]["median_us"])

    def test_high_bandwidth_model_is_monotonic_for_small_messages(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "sizes"},
            "events": [
                {
                    "id": "x",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1,
                    "ranks": [0, 1, 2, 3],
                    "rank_arrival_us": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0},
                }
            ],
        }
        previous = None
        for size in (1, 8, 64, 512, 4096, 32768, 65536):
            trace["events"][0]["bytes"] = size
            median_us = replay_canary(compile_trace(trace), seed=1, bandwidth_gbps=500.0)["metrics"]["median_us"]
            if previous is not None:
                self.assertGreaterEqual(median_us, previous)
            previous = median_us

    def test_timing_compression_preserves_periodic_skew_and_rare_gaps(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "patterns"}, "events": []}
        for index in range(100):
            skew = 0.0 if index % 2 == 0 else 100.0
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "start_us": float(index),
                    "rank_arrival_us": {"0": 0.0, "1": skew},
                }
            )
        canary = compile_trace(trace, timing_sample_limit=8)
        validate_canary(canary)
        self.assertEqual(canary["events"][0]["arrival_skew_us"], 50.0)
        report = replay_canary(canary, include_samples=True, seed=1)
        self.assertEqual(report["metrics"]["arrival_skew_median_us"], 50.0)

        trace = {"format": TRACE_FORMAT, "workload": {"name": "gaps"}, "events": []}
        for index in range(20):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": 1000.0 if index == 10 else 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                }
            )
        canary = compile_trace(trace, timing_sample_limit=5)
        event = canary["events"][0]
        self.assertEqual(sum(sample["gap_sum_us"] for sample in event["timing_samples"]), 1000.0)
        report = replay_canary(canary, include_samples=True, seed=1)
        self.assertEqual(max(sample["gap_us"] for sample in report["samples"]), 1000.0)

    def test_irregular_compression_is_bounded(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "irregular"}, "events": []}
        for index in range(1000):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "start_us": float(index * 3),
                    "rank_arrival_us": {"0": 0.0, "1": float(index % 37)},
                }
            )
        canary = compile_trace(trace, timing_sample_limit=8)
        self.assertEqual(len(canary["events"]), 1)
        self.assertLessEqual(canary["compiler"]["recursive_timing_records"], 8)
        for event in canary["events"]:
            self.assertLessEqual(len(event["timing_samples"]), 8)

    def test_bounded_compression_uses_joint_medoid_and_exact_gap_sum(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "outlier"}, "events": []}
        expected_gap = 0.0
        for index in range(100):
            gap_us = 0.04852 if index else 0.0
            expected_gap += gap_us
            skew = 100.0 if index == 50 else 0.0
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": gap_us,
                    "rank_arrival_us": {"0": 0.0, "1": skew},
                }
            )
        canary = compile_trace(trace, timing_sample_limit=8)
        validate_canary(canary)
        samples = replay_canary(canary, include_samples=True, seed=1)["samples"]
        self.assertLessEqual(sum(1 for sample in samples if sample["arrival_skew_us"] > 0.0), 2)
        self.assertAlmostEqual(sum(sample["gap_us"] for sample in samples), expected_gap, places=6)

    def test_timing_sample_limit_rejects_single_record_bound(self):
        with self.assertRaises(SchemaError):
            compile_trace(small_trace(), timing_sample_limit=1)

    def test_replay_protocol_cap_does_not_affect_compatibility(self):
        canary = compile_trace(small_trace())
        baseline = replay_canary(canary, seed=3, max_replay_events=100)
        candidate = replay_canary(canary, seed=3, max_replay_events=200)
        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])

    def test_schema_rejects_inconsistent_canaries_and_reports(self):
        canary = compile_trace(small_trace())
        canary["events"][0]["rank_count"] = 100
        with self.assertRaises(SchemaError):
            validate_canary(canary)

        report = replay_canary(compile_trace(small_trace()), seed=3)
        report["metrics"]["p95_us"] = report["metrics"]["median_us"] - 1.0
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), seed=3)
        report["replay_protocol"]["seed"] = 99
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), seed=3)
        report["canary"]["sha256"] = "not-a-sha"
        with self.assertRaises(SchemaError):
            validate_report(report)

        canary = compile_trace(small_trace())
        del canary["compiler"]["source_events"]
        with self.assertRaises(SchemaError):
            validate_canary(canary)

    def test_schema_rejects_bad_arrival_maps_samples_and_integer_strings(self):
        trace = small_trace()
        trace["events"][0]["rank_arrival_us"] = {"0": -5.0, "1": 0.0, "2": 0.0, "3": 0.0}
        with self.assertRaises(SchemaError):
            compile_trace(trace)

        trace = small_trace()
        trace["events"][0]["rank_arrival_us"] = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}
        with self.assertRaises(SchemaError):
            compile_trace(trace)

        report = replay_canary(compile_trace(small_trace()), include_samples=True)
        report["samples"] = [3]
        with self.assertRaises(SchemaError):
            validate_report(report)

        with self.assertRaises(SchemaError):
            as_int("9" * 5000)

    def test_public_apis_reject_non_mapping_inputs_and_bad_categories(self):
        with self.assertRaises(SchemaError):
            validate_trace([])
        with self.assertRaises(SchemaError):
            compile_trace([])
        with self.assertRaises(SchemaError):
            validate_canary([])
        with self.assertRaises(SchemaError):
            validate_report([])

        trace = small_trace()
        trace["events"][0]["phase"] = []
        with self.assertRaises(SchemaError):
            compile_trace(trace)

        canary = compile_trace(small_trace())
        canary["events"][0]["group"] = {}
        with self.assertRaises(SchemaError):
            validate_canary(canary)

    def test_schema_rejects_interval_skew_and_report_closure_gaps(self):
        canary = compile_trace(small_trace())
        sample = canary["events"][0]["timing_samples"][0]
        sample["arrival_skew_us"] = sample["arrival_skew_us"] + 10.0
        with self.assertRaises(SchemaError):
            validate_canary(canary)

        canary = compile_trace(small_trace())
        event = canary["events"][0]
        event["timing_samples"][0]["source_start"] = 2
        event["timing_samples"][0]["source_end"] = 1
        with self.assertRaises(SchemaError):
            validate_canary(canary)

        report = replay_canary(compile_trace(small_trace()), include_samples=True)
        report["replay_protocol"]["iterations"] = 0
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), include_samples=True)
        report["by_phase"] = []
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), include_samples=True)
        report["samples"][0].pop("queue_wait_us")
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), include_samples=True)
        report["samples"][0]["collective_us"] += 1000000.0
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), include_samples=True)
        report["samples"][0]["hidden_us"] = report["samples"][0]["total_us"]
        report["samples"][0]["exposed_us"] = 0.0
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), include_samples=True)
        report["samples"][0]["exposed_us"] = report["samples"][0]["total_us"]
        report["samples"][0]["hidden_us"] = 0.0
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), include_samples=True)
        report["samples"][0]["index"] = 2
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()), iterations=2, include_samples=True)
        report["samples"][-1]["iteration"] = 0
        with self.assertRaises(SchemaError):
            validate_report(report)

        report = replay_canary(compile_trace(small_trace()))
        report["metrics"]["mean_us"] = report["metrics"]["max_us"] + 1.0
        with self.assertRaises(SchemaError):
            validate_report(report)

    def test_event_local_tail_rng_is_stable_for_later_events(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "rng"},
            "events": [
                {
                    "id": "a",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "group": "a",
                    "start_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                },
                {
                    "id": "b",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "group": "b",
                    "start_us": 10.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                },
            ],
        }
        baseline = replay_canary(compile_trace(trace), include_samples=True, seed=9)["samples"][1]["collective_us"]
        trace["events"][0]["rank_arrival_us"] = {"0": 0.0, "1": 500.0}
        changed = replay_canary(compile_trace(trace), include_samples=True, seed=9)["samples"][1]["collective_us"]
        self.assertEqual(baseline, changed)

    def test_allow_mismatch_includes_reasons_even_with_latency_failure(self):
        canary = compile_trace(small_trace())
        baseline = replay_canary(canary, seed=3)
        candidate = replay_canary(canary, seed=99, latency_floor_us=1000.0)
        comparison = compare_reports(baseline, candidate, require_compatible=False, p99_threshold_pct=1.0)
        self.assertEqual(comparison["verdict"], "fail")
        self.assertTrue(any("replay protocol" in reason for reason in comparison["reasons"]))

    def test_semantic_canary_hash_allows_provenance_only_changes(self):
        baseline_canary = compile_trace(small_trace())
        candidate_canary = compile_trace(small_trace(), max_skew_error_us=1000.0)
        baseline = replay_canary(baseline_canary, seed=3)
        candidate = replay_canary(candidate_canary, seed=3)
        self.assertNotEqual(baseline["canary"]["sha256"], candidate["canary"]["sha256"])
        self.assertEqual(
            baseline["canary"]["execution_semantic_sha256"],
            candidate["canary"]["execution_semantic_sha256"],
        )
        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])

    def test_semantic_canary_hash_ignores_non_executed_event_fields(self):
        baseline_canary = compile_trace(small_trace())
        candidate_canary = copy.deepcopy(baseline_canary)
        candidate_canary["events"][0]["source"]["first_id"] = "renamed-source-event"
        candidate_canary["events"][0]["source"]["digest"] = "0" * 64
        candidate_canary["events"][0]["arrival_skew_us"] = 999.0
        candidate_canary["events"][0]["compute_pressure"] = 1.25
        candidate_canary["compiler"]["execution_semantic_sha256"] = canary_execution_sha256(candidate_canary)

        self.assertEqual(
            baseline_canary["compiler"]["execution_semantic_sha256"],
            candidate_canary["compiler"]["execution_semantic_sha256"],
        )
        baseline = replay_canary(baseline_canary, seed=3)
        candidate = replay_canary(candidate_canary, seed=3)
        self.assertEqual(baseline["metrics"], candidate["metrics"])
        self.assertTrue(compare_reports(baseline, candidate)["compatibility"]["compatible"])

    def test_semantic_canary_hash_canonicalizes_rank_count_but_includes_phase(self):
        baseline_canary = compile_trace(small_trace())
        without_rank_count = copy.deepcopy(baseline_canary)
        without_rank_count["events"][0].pop("rank_count")
        without_rank_count["compiler"]["execution_semantic_sha256"] = canary_execution_sha256(without_rank_count)
        self.assertEqual(
            baseline_canary["compiler"]["execution_semantic_sha256"],
            without_rank_count["compiler"]["execution_semantic_sha256"],
        )
        self.assertTrue(
            compare_reports(
                replay_canary(baseline_canary, seed=3),
                replay_canary(without_rank_count, seed=3),
            )["compatibility"]["compatible"]
        )

        changed_phase = copy.deepcopy(baseline_canary)
        changed_phase["events"][0]["phase"] = "prefill"
        changed_phase["compiler"]["execution_semantic_sha256"] = canary_execution_sha256(changed_phase)
        self.assertNotEqual(
            baseline_canary["compiler"]["execution_semantic_sha256"],
            changed_phase["compiler"]["execution_semantic_sha256"],
        )
        with self.assertRaises(SchemaError):
            compare_reports(replay_canary(baseline_canary, seed=3), replay_canary(changed_phase, seed=3))

    def test_zero_baseline_regression_is_not_clamped_to_100_percent(self):
        baseline = replay_canary(compile_trace(small_trace()), seed=3)
        candidate = copy.deepcopy(baseline)
        for report, p99 in ((baseline, 0.0), (candidate, 15.0)):
            report["metrics"].update(
                {
                    "median_us": 0.0,
                    "p95_us": 0.0,
                    "p99_us": p99,
                    "max_us": p99,
                    "mean_us": p99 / max(1, report["metrics"]["count"]),
                }
            )
            for key in ("by_phase", "by_op"):
                for row in report[key]:
                    row.update(
                        {
                            "median_us": 0.0,
                            "p95_us": 0.0,
                            "p99_us": p99,
                            "max_us": p99,
                            "mean_us": report["metrics"]["mean_us"],
                        }
                    )
        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["verdict"], "fail")
        self.assertIsNone(comparison["delta"]["p99_pct"])
        self.assertEqual(comparison["delta"]["p99_relative_status"], "new_nonzero_regression")

    def test_rare_phase_regression_fails_global_verdict(self):
        baseline = replay_canary(compile_trace(small_trace()), seed=3)
        candidate = copy.deepcopy(baseline)
        for report in (baseline, candidate):
            report["canary"]["source_events"] = 1000
            report["metrics"].update(
                {
                    "count": 1000,
                    "median_us": 10.0,
                    "p95_us": 10.0,
                    "p99_us": 10.0,
                    "max_us": 10.0,
                    "mean_us": 10.0,
                    "arrival_skew_median_us": 0.0,
                    "arrival_skew_p95_us": 0.0,
                    "arrival_skew_max_us": 0.0,
                    "avg_rank_wait_median_us": 0.0,
                    "communication_hidden_pct": 0.0,
                }
            )
            report["by_phase"] = [
                {
                    "name": "common",
                    "count": 999,
                    "median_us": 10.0,
                    "p95_us": 10.0,
                    "p99_us": 10.0,
                    "max_us": 10.0,
                    "mean_us": 10.0,
                },
                {
                    "name": "rare_finalize",
                    "count": 1,
                    "median_us": 10.0,
                    "p95_us": 10.0,
                    "p99_us": 10.0,
                    "max_us": 10.0,
                    "mean_us": 10.0,
                },
            ]
            report["by_op"] = [
                {
                    "name": "all_reduce",
                    "count": 1000,
                    "median_us": 10.0,
                    "p95_us": 10.0,
                    "p99_us": 10.0,
                    "max_us": 10.0,
                    "mean_us": 10.0,
                }
            ]
        candidate["metrics"]["max_us"] = 1000.0
        candidate["metrics"]["mean_us"] = 10.99
        candidate["by_phase"][1].update(
            {
                "median_us": 1000.0,
                "p95_us": 1000.0,
                "p99_us": 1000.0,
                "max_us": 1000.0,
                "mean_us": 1000.0,
            }
        )
        candidate["by_op"][0].update({"max_us": 1000.0, "mean_us": 10.99})

        comparison = compare_reports(baseline, candidate, p99_threshold_pct=15.0)
        self.assertEqual(comparison["verdict"], "fail")
        self.assertEqual(comparison["delta"]["p99_pct"], 0.0)
        self.assertTrue(any("phase 'rare_finalize' p99" in reason for reason in comparison["reasons"]))

    def test_p95_and_hidden_drop_regressions_fail_comparison(self):
        baseline = replay_canary(compile_trace(small_trace()), seed=3)
        candidate = copy.deepcopy(baseline)
        for report, p95, hidden, mean in (
            (baseline, 10.0, 90.0, 20.0),
            (candidate, 90.0, 0.0, 30.0),
        ):
            report["canary"]["source_events"] = 100
            report["metrics"].update(
                {
                    "count": 100,
                    "median_us": 10.0,
                    "p95_us": p95,
                    "p99_us": 100.0,
                    "max_us": 100.0,
                    "mean_us": mean,
                    "communication_hidden_pct": hidden,
                }
            )
            row = {
                "name": "decode",
                "count": 100,
                "median_us": 10.0,
                "p95_us": p95,
                "p99_us": 100.0,
                "max_us": 100.0,
                "mean_us": mean,
            }
            report["by_phase"] = [dict(row)]
            report["by_op"] = [{**row, "name": "all_reduce"}]

        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["verdict"], "fail")
        self.assertEqual(comparison["delta"]["p99_pct"], 0.0)
        self.assertTrue(any("p95 regression" in reason for reason in comparison["reasons"]))
        self.assertTrue(any("hidden percentage dropped" in reason for reason in comparison["reasons"]))

    def test_write_json_wraps_bad_parent_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_file = os.path.join(tmp, "not-a-dir")
            with open(parent_file, "w", encoding="utf-8") as handle:
                handle.write("x")
            with self.assertRaises(SchemaError):
                write_json(os.path.join(parent_file, "out.json"), {"ok": True})

    def test_compile_rejects_non_json_serializable_trace_data(self):
        trace = small_trace()
        trace["workload"]["path"] = Path("not-json")
        with self.assertRaises(SchemaError):
            compile_trace(trace)

    def test_compute_overlap_can_hide_scheduled_latency(self):
        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "queue-overlap"},
            "events": [
                {
                    "id": "a",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "group": "tp",
                    "start_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                },
                {
                    "id": "b",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "group": "tp",
                    "start_us": 1.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                    "compute_overlap_us": 10000.0,
                },
            ],
        }
        samples = replay_canary(compile_trace(trace), include_samples=True, seed=1, overlap_efficiency=1.0)["samples"]
        self.assertGreater(samples[1]["queue_wait_us"], 0.0)
        self.assertAlmostEqual(samples[1]["hidden_us"], samples[1]["total_us"], places=3)

    def test_html_escapes_once_and_allowlists_verdict_class(self):
        canary = compile_trace(small_trace())
        report = replay_canary(canary)
        report["workload"]["name"] = "A&B"
        html = render_report_html(report)
        self.assertIn("<title>CommCanary Report - A&amp;B</title>", html)
        self.assertNotIn("A&amp;amp;B", html)

        comparison = {
            "verdict": 'fail" onclick="bad',
            "created_at": "now",
            "delta": {"median_pct": 0, "p95_pct": 0, "p99_pct": 0},
            "baseline": {"metrics": report["metrics"]},
            "candidate": {"metrics": report["metrics"]},
            "reasons": ["A&B"],
        }
        compare_html = render_compare_html(comparison)
        self.assertIn('class="hero warn"', compare_html)
        self.assertNotIn("onclick", compare_html)
        self.assertIn("A&amp;B", compare_html)

    def test_html_report_rejects_malformed_metadata(self):
        report = replay_canary(compile_trace(small_trace()))
        report["workload"] = []
        with self.assertRaises(SchemaError):
            render_report_html(report)

    def test_cli_compile_replay_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "trace.json")
            canary_path = os.path.join(tmp, "canary.json")
            report_path = os.path.join(tmp, "report.json")
            html_path = os.path.join(tmp, "report.html")
            write_json(trace_path, small_trace())
            self.assertEqual(
                cli_main(
                    [
                        "compile",
                        trace_path,
                        "--output",
                        canary_path,
                        "--max-compute-before-error-us",
                        "0",
                        "--max-pressure-error",
                        "0",
                    ]
                ),
                0,
            )
            self.assertEqual(cli_main(["replay", canary_path, "--output", report_path, "--include-samples"]), 0)
            self.assertEqual(cli_main(["report", report_path, "--output", html_path]), 0)
            self.assertTrue(os.path.exists(html_path))


    def test_tiny_periodic_gaps_preserve_exact_timeline(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "tiny-gaps"}, "events": []}
        expected = 0.0
        for index in range(1000):
            gap = 0.0001 if index % 2 == 0 else 0.0004
            expected += gap
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": gap,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                }
            )
        canary = compile_trace(trace, timing_sample_limit=8)
        samples = replay_canary(canary, include_samples=True, seed=1)["samples"]
        self.assertAlmostEqual(sum(sample["gap_us"] for sample in samples), expected, places=7)
        malformed = copy.deepcopy(canary)
        malformed["events"][0]["timing_samples"][0]["gap_sum_us"] += 1.0
        with self.assertRaises(SchemaError):
            validate_canary(malformed)

    def test_exact_period_detection_exceeds_sixteen_when_budget_allows(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "period-17"}, "events": []}
        for index in range(17 * 10):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": float(index % 17),
                    "rank_arrival_us": {"0": 0.0, "1": float((index % 17) * 2)},
                }
            )
        canary = compile_trace(trace, timing_sample_limit=32)
        timing_samples = canary["events"][0]["timing_samples"]
        self.assertEqual(len(timing_samples), 1)
        self.assertEqual(len(timing_samples[0]["timing_pattern"]), 17)
        self.assertEqual(timing_samples[0]["pattern_repeats"], 10)
        validate_canary(canary)

    def test_mixed_timing_modes_and_fractional_limits_are_rejected(self):
        base = {
            "phase": "decode",
            "op": "all_reduce",
            "ranks": [0, 1],
            "rank_arrival_us": {"0": 0.0, "1": 0.0},
        }
        ambiguous = {
            "format": TRACE_FORMAT,
            "events": [
                {**base, "id": "a", "bytes": 111, "start_us": 100.0},
                {**base, "id": "b", "bytes": 222},
            ],
        }
        with self.assertRaises(SchemaError):
            compile_trace(ambiguous)
        explicit = copy.deepcopy(ambiguous)
        explicit["events"][0]["gap_us"] = 0.0
        explicit["events"][1]["gap_us"] = 2.0
        canary = compile_trace(explicit)
        self.assertEqual([event["bytes"] for event in canary["events"]], [111, 222])
        conflicting = copy.deepcopy(explicit)
        conflicting["events"][1]["start_us"] = 110.0
        conflicting["events"][1]["gap_us"] = 2.0
        with self.assertRaises(SchemaError):
            compile_trace(conflicting)
        with self.assertRaises(SchemaError):
            compile_trace(small_trace(), max_events=1.5)
        with self.assertRaises(SchemaError):
            compile_trace(small_trace(), timing_sample_limit=2.5)

    def test_observed_tail_signal_is_preserved_and_calibrated(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "observed-tail"}, "events": []}
        for index in range(100):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": float(index % 7),
                    "rank_arrival_us": {"0": 0.0, "1": float(index % 13)},
                    "observed_exposed_us": 1000.0 if index == 77 else 10.0,
                }
            )
        canary = compile_trace(trace, timing_sample_limit=8)
        self.assertEqual(canary["compiler"]["tail_signal"], "observed_exposed_us")
        encoded = canary["events"][0]["timing_samples"]
        self.assertTrue(any(sample.get("observed_exposed_us") == 1000.0 for sample in encoded))
        report = replay_canary(canary, include_samples=True)
        self.assertEqual(report["calibration"]["count"], 100)
        validate_report(report)
        partial = copy.deepcopy(trace)
        del partial["events"][0]["observed_exposed_us"]
        with self.assertRaises(SchemaError):
            compile_trace(partial, timing_sample_limit=8)

    def test_fidelity_budgets_fail_closed(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "budget"}, "events": []}
        for index in range(40):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": float((index * 7) % 19),
                    "rank_arrival_us": {"0": 0.0, "1": float((index * 11) % 31)},
                }
            )
        canary = compile_trace(trace, timing_sample_limit=2)
        self.assertEqual(canary["compiler"]["fidelity"]["mode"], "bounded_approximate")
        with self.assertRaises(SchemaError):
            compile_trace(trace, timing_sample_limit=2, max_skew_error_us=0.0)
        with self.assertRaises(SchemaError):
            compile_trace(trace, timing_sample_limit=2, max_prefix_gap_error_us=0.0)
        malformed = copy.deepcopy(canary)
        malformed["compiler"]["fidelity_budget"] = {
            "max_skew_error_us": malformed["compiler"]["fidelity"]["max_skew_error_us"] - 0.001
        }
        with self.assertRaises(SchemaError):
            validate_canary(malformed)

    def test_lossless_timing_is_an_explicit_invariant(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "lossless"}, "events": []}
        for index in range(30):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": 1.0,
                    "rank_arrival_us": {"0": 0.0, "1": 2.0},
                    "compute_before_us": float(index),
                }
            )
        with self.assertRaises(SchemaError):
            compile_trace(trace, timing_sample_limit=2, require_lossless_timing=True)
        with self.assertRaises(SchemaError):
            compile_trace(trace, require_lossless_timing=1)
        exact = compile_trace(small_trace(), require_lossless_timing=True)
        self.assertEqual(exact["compiler"]["fidelity"]["mode"], "lossless_timing")

    def test_report_samples_reconcile_and_comparison_localises_regressions(self):
        trace = small_trace()
        trace["events"][0]["phase"] = "prefill"
        canary = compile_trace(trace)
        baseline = replay_canary(canary, latency_floor_us=7.0, include_samples=True)
        candidate = replay_canary(canary, latency_floor_us=20.0, include_samples=True)
        comparison = compare_reports(baseline, candidate)
        self.assertIn("by_phase", comparison["breakdown_delta"])
        self.assertEqual(comparison["worst_regressions"]["operation"]["name"], "all_reduce")
        malformed = copy.deepcopy(baseline)
        malformed["samples"][0]["exposed_us"] += 10.0
        with self.assertRaises(SchemaError):
            validate_report(malformed)

    def test_report_breakdowns_reconcile_without_samples(self):
        trace = small_trace()
        trace["events"][0]["phase"] = "prefill"
        report = replay_canary(compile_trace(trace))
        malformed = copy.deepcopy(report)
        malformed["by_phase"][0]["mean_us"] += 1.0
        with self.assertRaises(SchemaError):
            validate_report(malformed)
        malformed = copy.deepcopy(report)
        self.assertGreaterEqual(len(malformed["by_phase"]), 2)
        malformed["by_phase"][1]["name"] = malformed["by_phase"][0]["name"]
        with self.assertRaises(SchemaError):
            validate_report(malformed)

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
            recorder = TraceRecorder(
                os.path.join(tmp, "trace.json"),
                workload={"name": "snapshot", "tags": ["before"]},
            )
            recorder.record_collective(
                op="all_reduce",
                bytes=16,
                ranks=[0],
                rank_arrival_us={"0": 0.0},
                metadata={"nested": {"value": 1}},
            )
            trace = recorder.to_trace()
            trace["workload"]["tags"].append("after")
            trace["events"][0]["metadata"]["nested"]["value"] = 2

            self.assertEqual(recorder.workload["tags"], ["before"])
            self.assertEqual(recorder.events[0]["metadata"]["nested"]["value"], 1)

    def test_one_rank_skew_and_malformed_gap_sum_are_rejected(self):
        trace = {
            "format": TRACE_FORMAT,
            "events": [
                {
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0],
                    "arrival_skew_us": 2.0,
                }
            ],
        }
        with self.assertRaises(SchemaError):
            compile_trace(trace)
        canary = compile_trace(small_trace())
        malformed = copy.deepcopy(canary)
        sample = malformed["events"][0]["timing_samples"][0]
        self.assertNotIn("timing_pattern", sample)
        sample["gap_sum_us"] += 1.0
        with self.assertRaises(SchemaError):
            validate_canary(malformed)

        malformed = compile_trace(
            {
                "format": TRACE_FORMAT,
                "events": [
                    {
                        "op": "all_reduce",
                        "bytes": 16,
                        "ranks": [0, 1],
                        "rank_arrival_us": {"0": 0.0, "1": 4.0},
                    }
                ],
            }
        )
        malformed["events"][0]["arrival_skew_us"] = 999.0
        with self.assertRaises(SchemaError):
            validate_canary(malformed)

    def test_empty_trace_requires_explicit_override(self):
        empty = {"format": TRACE_FORMAT, "events": []}
        with self.assertRaises(SchemaError):
            compile_trace(empty)
        canary = compile_trace(empty, allow_empty=True)
        self.assertEqual(canary["compiler"]["source_events"], 0)

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



if __name__ == "__main__":
    unittest.main()
