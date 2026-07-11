from __future__ import annotations

import copy
import os
import tempfile
import unittest

from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary
from commcanary.schema import (
    TRACE_FORMAT,
    SchemaError,
    as_int,
    validate_canary,
    validate_report,
    validate_trace,
    write_json,
)
from tests.artifact_helpers import refresh_canary_hashes
from tests.builders import small_trace


class RuntimeArtifactTests(unittest.TestCase):
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
        refresh_canary_hashes(candidate_canary)

        self.assertEqual(
            baseline_canary["compiler"]["execution_semantic_sha256"],
            candidate_canary["compiler"]["execution_semantic_sha256"],
        )
        baseline = replay_canary(baseline_canary, seed=3)
        candidate = replay_canary(candidate_canary, seed=3)
        self.assertEqual(baseline["metrics"], candidate["metrics"])
        self.assertTrue(compare_reports(baseline, candidate)["compatibility"]["compatible"])

    def test_semantic_identity_is_independent_of_source_event_ids(self):
        renamed = small_trace()
        for index, event in enumerate(renamed["events"]):
            event["id"] = f"renamed-{index}"
        baseline_canary = compile_trace(small_trace())
        renamed_canary = compile_trace(renamed)
        self.assertNotEqual(
            baseline_canary["events"][0]["source"]["digest"],
            renamed_canary["events"][0]["source"]["digest"],
        )
        self.assertEqual(
            baseline_canary["compiler"]["execution_semantic_sha256"],
            renamed_canary["compiler"]["execution_semantic_sha256"],
        )
        self.assertEqual(
            replay_canary(baseline_canary, seed=3)["metrics"],
            replay_canary(renamed_canary, seed=3)["metrics"],
        )

    def test_semantic_canary_hash_canonicalizes_rank_count_but_includes_phase(self):
        baseline_canary = compile_trace(small_trace())
        without_rank_count = copy.deepcopy(baseline_canary)
        without_rank_count["events"][0].pop("rank_count")
        refresh_canary_hashes(without_rank_count)
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
        refresh_canary_hashes(changed_phase)
        self.assertNotEqual(
            baseline_canary["compiler"]["execution_semantic_sha256"],
            changed_phase["compiler"]["execution_semantic_sha256"],
        )
        with self.assertRaises(SchemaError):
            compare_reports(replay_canary(baseline_canary, seed=3), replay_canary(changed_phase, seed=3))

    def test_semantic_hash_and_replay_normalize_numeric_json_forms(self):
        canary = compile_trace(small_trace())
        changed = copy.deepcopy(canary)
        event = changed["events"][0]
        event["bytes"] = str(event["bytes"])
        event["ranks"] = [str(rank) for rank in event["ranks"]]
        event["rank_count"] = str(event["rank_count"])
        event["concurrent_groups"] = str(event["concurrent_groups"])
        for sample in event["timing_samples"]:
            sample["gap_us"] = str(sample["gap_us"])
            sample["gap_sum_us"] = str(sample["gap_sum_us"])
            sample["arrival_offsets_us"] = [str(offset) for offset in sample["arrival_offsets_us"]]
            sample["compute_overlap_us"] = str(sample["compute_overlap_us"])
            sample["compute_pressure"] = str(sample["compute_pressure"])
        refresh_canary_hashes(changed)

        self.assertEqual(
            canary["compiler"]["execution_semantic_sha256"],
            changed["compiler"]["execution_semantic_sha256"],
        )
        self.assertEqual(
            replay_canary(canary, seed=3)["metrics"],
            replay_canary(changed, seed=3)["metrics"],
        )

    def test_scheduler_hash_ignores_calibration_and_noncausal_compute_before(self):
        observed_trace = small_trace()
        for event in observed_trace["events"]:
            event["observed_exposed_us"] = 20.0
        observed_canary = compile_trace(observed_trace)
        changed_observed = copy.deepcopy(observed_canary)
        changed_observed["events"][0]["observed_exposed_us"] = 999.0
        for sample in changed_observed["events"][0]["timing_samples"]:
            sample["observed_exposed_us"] = 999.0
        refresh_canary_hashes(changed_observed)
        self.assertEqual(
            observed_canary["compiler"]["execution_semantic_sha256"],
            changed_observed["compiler"]["execution_semantic_sha256"],
        )
        self.assertTrue(
            compare_reports(
                replay_canary(observed_canary, seed=3),
                replay_canary(changed_observed, seed=3),
            )["compatibility"]["compatible"]
        )

        trace = {
            "format": TRACE_FORMAT,
            "workload": {"name": "compute-before-hash"},
            "events": [
                {
                    "id": "event-0",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": 1.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                    "compute_before_us": 1.0,
                }
            ],
        }
        canary = compile_trace(trace)
        changed_before = copy.deepcopy(canary)
        changed_before["events"][0]["compute_before_us"] = 1_000_000.0
        changed_before["events"][0]["timing_samples"][0]["compute_before_us"] = 1_000_000.0
        refresh_canary_hashes(changed_before)
        self.assertEqual(
            canary["compiler"]["execution_semantic_sha256"],
            changed_before["compiler"]["execution_semantic_sha256"],
        )
        self.assertEqual(
            replay_canary(canary, seed=3)["metrics"],
            replay_canary(changed_before, seed=3)["metrics"],
        )

    def test_scheduler_hash_canonicalizes_equivalent_timing_encodings(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "flat-vs-pattern"}, "events": []}
        for index in range(4):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "gap_us": 1.0,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                }
            )
        flat = compile_trace(trace)
        patterned = copy.deepcopy(flat)
        pattern_parent = copy.deepcopy(patterned["events"][0]["timing_samples"][0])
        child = copy.deepcopy(pattern_parent)
        child.update({"source_index": 0, "source_start": 0, "source_end": 0, "weight": 1, "gap_sum_us": 1.0})
        pattern_parent.update(
            {
                "source_index": 0,
                "source_start": 0,
                "source_end": 3,
                "weight": 4,
                "gap_sum_us": 4.0,
                "timing_pattern": [child],
                "pattern_repeats": 4,
            }
        )
        patterned["events"][0]["timing_samples"] = [pattern_parent]
        patterned["events"][0]["source"]["sampled_timing_records"] = 2
        patterned["compiler"]["recursive_timing_records"] = 2
        patterned["compiler"]["stored_recursive_timing_records"] = 2
        refresh_canary_hashes(patterned)
        self.assertEqual(
            flat["compiler"]["execution_semantic_sha256"],
            patterned["compiler"]["execution_semantic_sha256"],
        )
        self.assertEqual(
            replay_canary(flat, seed=3)["metrics"],
            replay_canary(patterned, seed=3)["metrics"],
        )

    def test_write_json_wraps_bad_parent_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_file = os.path.join(tmp, "not-a-dir")
            with open(parent_file, "w", encoding="utf-8") as handle:
                handle.write("x")
            with self.assertRaises(SchemaError):
                write_json(os.path.join(parent_file, "out.json"), {"ok": True})

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

    def test_canary_rejects_multi_weight_sample_without_source_interval(self):
        canary = compile_trace(small_trace())
        tampered = copy.deepcopy(canary)
        sample = tampered["events"][0]["timing_samples"][0]
        del sample["source_start"]
        del sample["source_end"]
        sample["weight"] = 2
        with self.assertRaisesRegex(SchemaError, "requires source_start and source_end"):
            validate_canary(tampered)

    def test_canary_rejects_source_interval_without_matching_weight(self):
        canary = compile_trace(small_trace())
        tampered = copy.deepcopy(canary)
        sample = tampered["events"][0]["timing_samples"][0]
        del sample["weight"]
        sample["source_end"] = as_int(sample["source_start"]) + 1
        with self.assertRaisesRegex(SchemaError, "weight must match source interval length"):
            validate_canary(tampered)

    def test_canary_rejects_gap_between_timing_sample_intervals(self):
        canary = compile_trace(small_trace())
        tampered = copy.deepcopy(canary)
        samples = tampered["events"][0]["timing_samples"]
        self.assertGreater(len(samples), 2)
        del samples[1]
        with self.assertRaisesRegex(SchemaError, "contiguous"):
            validate_canary(tampered)

    def test_as_int_rejects_non_ascii_digit_strings(self):
        self.assertEqual(as_int("5"), 5)
        self.assertEqual(as_int("-5"), -5)
        with self.assertRaises(SchemaError):
            as_int("٥")
