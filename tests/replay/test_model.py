from __future__ import annotations

import copy
import unittest

from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary
from commcanary.schema import TRACE_FORMAT, SchemaError, validate_canary, validate_report
from tests.builders import small_trace


class ReplayTests(unittest.TestCase):
    def test_replay_produces_tail_metrics(self):
        canary = compile_trace(small_trace())
        report = replay_canary(canary, seed=3)
        metrics = report["metrics"]
        self.assertEqual(metrics["count"], 6)
        self.assertGreaterEqual(metrics["p99_us"], metrics["median_us"])
        self.assertGreater(metrics["arrival_skew_p95_us"], 0.0)

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

    def test_replay_protocol_cap_does_not_affect_compatibility(self):
        canary = compile_trace(small_trace())
        baseline = replay_canary(canary, seed=3, max_replay_events=100)
        candidate = replay_canary(canary, seed=3, max_replay_events=200)
        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])

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

    def test_unrelated_prefix_removal_does_not_rekey_retained_event_noise(self):
        event_a = {
            "id": "a",
            "phase": "decode",
            "op": "all_reduce",
            "bytes": 1024,
            "ranks": [0, 1],
            "group": "a",
            "start_us": 0.0,
            "rank_arrival_us": {"0": 0.0, "1": 0.0},
        }
        event_b = {
            "id": "b",
            "phase": "decode",
            "op": "all_reduce",
            "bytes": 2048,
            "ranks": [2, 3],
            "group": "b",
            "start_us": 0.0,
            "rank_arrival_us": {"2": 0.0, "3": 0.0},
        }
        with_prefix = {
            "format": TRACE_FORMAT,
            "workload": {"name": "rng-prefix"},
            "events": [event_a, event_b],
        }
        without_prefix = {
            "format": TRACE_FORMAT,
            "workload": {"name": "rng-prefix"},
            "events": [event_b],
        }
        prefixed_sample = replay_canary(compile_trace(with_prefix), include_samples=True, seed=9)["samples"][1]
        retained_sample = replay_canary(compile_trace(without_prefix), include_samples=True, seed=9)["samples"][0]
        self.assertEqual(prefixed_sample["collective_us"], retained_sample["collective_us"])

    def test_repeated_identical_occurrences_receive_distinct_noise(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "repeated-noise"}, "events": []}
        for index in range(100):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                }
            )
        samples = replay_canary(compile_trace(trace), include_samples=True, seed=9)["samples"]
        unique_durations = {round(sample["collective_us"], 6) for sample in samples}
        self.assertGreater(len(unique_durations), 1)

    def test_nonconsecutive_identical_events_do_not_share_noise_identity(self):
        events = []
        for index, bytes_ in enumerate((1024, 2048, 1024, 2048)):
            events.append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": bytes_,
                    "ranks": [0, 1],
                    "gap_us": 0.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                }
            )
        samples = replay_canary(
            compile_trace({"format": TRACE_FORMAT, "workload": {"name": "alternating-noise"}, "events": events}),
            include_samples=True,
            seed=9,
        )["samples"]
        self.assertNotEqual(samples[0]["collective_us"], samples[2]["collective_us"])

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

    def test_cumulative_replay_clock_is_bounded(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "cumulative-clock"}, "events": []}
        for index in range(101):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "gap_us": 10_000_000_000.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.1},
                }
            )
        canary = compile_trace(trace)
        validate_canary(canary)
        with self.assertRaises(SchemaError):
            replay_canary(canary)
        with self.assertRaises(SchemaError):
            replay_canary(canary, include_samples=True)

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

    def test_sample_free_report_backend_and_calibration_are_validated(self):
        report = replay_canary(compile_trace(small_trace()))
        for key, value in (
            ("bandwidth_gbps", -5.0),
            ("latency_floor_us", -7.0),
            ("compute_pressure", -1.0),
            ("overlap_efficiency", 3.0),
        ):
            malformed = copy.deepcopy(report)
            malformed["backend"][key] = value
            with self.assertRaises(SchemaError):
                validate_report(malformed)

        trace = small_trace()
        for event in trace["events"]:
            event["observed_exposed_us"] = 20.0
        report = replay_canary(compile_trace(trace))
        malformed = copy.deepcopy(report)
        malformed["calibration"]["count"] -= 1
        with self.assertRaises(SchemaError):
            validate_report(malformed)

    def test_point_to_point_identity_affects_scheduler_hash_and_resource(self):
        base = {
            "id": "msg",
            "phase": "decode",
            "op": "point_to_point",
            "bytes": 4096,
            "ranks": [0, 1],
            "group": "pp",
            "gap_us": 1.0,
            "rank_arrival_us": {"0": 0.0, "1": 0.0},
            "sender_rank": 0,
            "receiver_rank": 1,
            "tag": "kv",
            "channel": "pipe",
            "message_sequence": 7,
        }
        trace_a = {"format": TRACE_FORMAT, "workload": {"name": "p2p-a"}, "events": [base]}
        trace_b = {
            "format": TRACE_FORMAT,
            "workload": {"name": "p2p-b"},
            "events": [{**base, "sender_rank": 1, "receiver_rank": 0}],
        }
        canary_a = compile_trace(trace_a)
        canary_b = compile_trace(trace_b)
        self.assertNotEqual(
            canary_a["compiler"]["scheduler_execution_sha256"],
            canary_b["compiler"]["scheduler_execution_sha256"],
        )
        sample = replay_canary(canary_a, include_samples=True)["samples"][0]
        self.assertIn("p2p:pp:0->1", sample["scheduler_resource"])
        self.assertIn("channel=pipe", sample["scheduler_resource"])
        self.assertIn("tag=kv", sample["scheduler_resource"])
