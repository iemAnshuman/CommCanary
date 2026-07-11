from __future__ import annotations

import copy
import unittest
from pathlib import Path

from commcanary.compiler import (
    compile_trace,
    synthesize_behavioral_canary,
    verify_canary_behavior,
    verify_canary_fidelity,
)
from commcanary.replay import replay_canary
from commcanary.schema import TRACE_FORMAT, SchemaError, validate_canary
from tests.builders import (
    adversarial_ranking_configs,
    adversarial_ranking_trace,
    small_trace,
    two_group_refinement_trace,
)


class CompilationTests(unittest.TestCase):
    def test_compile_compresses_repeated_events(self):
        canary = compile_trace(small_trace())
        self.assertEqual(canary["format"], "commcanary.canary.v2")
        self.assertLess(len(canary["events"]), len(small_trace()["events"]))
        self.assertEqual(canary["compiler"]["source_events"], 6)

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

    def test_behavior_search_finds_smallest_verified_candidate(self):
        trace = adversarial_ranking_trace()
        canary = synthesize_behavioral_canary(
            trace,
            min_timing_sample_limit=2,
            max_timing_sample_limit=32,
            behavior_configurations=adversarial_ranking_configs(),
            ranking_tie_tolerance_us=0.0,
        )
        search = canary["compiler"]["behavior_search"]
        self.assertEqual(canary["compiler"]["behavior_verification_status"], "behaviorally_verified")
        self.assertEqual(search["ranking_status"], "pass")
        self.assertGreater(search["accepted_candidates"], 0)
        self.assertGreaterEqual(search["selected_timing_sample_limit"], 2)
        self.assertLessEqual(search["selected_timing_sample_limit"], 32)
        selected_rows = [
            row
            for row in search["candidates"]
            if row["status"] == "behaviorally_verified"
            and row["canary_bytes"] == search["selected_canary_bytes_without_search_metadata"]
        ]
        self.assertTrue(selected_rows)
        self.assertEqual(
            verify_canary_behavior(
                trace,
                canary,
                configurations=adversarial_ranking_configs(),
                ranking_tie_tolerance_us=0.0,
            )["status"],
            "behaviorally_verified",
        )

    def test_behavior_search_fails_when_budget_cannot_preserve_ranking(self):
        with self.assertRaises(SchemaError):
            synthesize_behavioral_canary(
                adversarial_ranking_trace(),
                min_timing_sample_limit=2,
                max_timing_sample_limit=2,
                behavior_configurations=adversarial_ranking_configs(),
                ranking_tie_tolerance_us=0.0,
            )

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

    def test_exact_pattern_uncertainty_is_not_double_counted(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "uncertain-pattern"}, "events": []}
        for index in range(200):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                    "compute_fields_uncertain": index % 2 == 0,
                }
            )
        canary = compile_trace(trace, timing_sample_limit=8)
        self.assertEqual(canary["events"][0]["repeat"], 200)
        self.assertEqual(len(canary["events"][0]["timing_samples"][0]["timing_pattern"]), 2)
        self.assertEqual(
            canary["compiler"]["capture_uncertainty"]["compute_fields_uncertain_events"],
            100,
        )
        report = replay_canary(canary, include_samples=True)
        self.assertEqual(
            sum(1 for sample in report["samples"] if sample.get("compute_fields_uncertain")),
            100,
        )
        malformed = copy.deepcopy(canary)
        malformed["compiler"]["capture_uncertainty"]["compute_fields_uncertain_events"] = 101
        with self.assertRaises(SchemaError):
            validate_canary(malformed)

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

    def test_compile_rejects_non_json_serializable_trace_data(self):
        trace = small_trace()
        trace["workload"]["path"] = Path("not-json")
        with self.assertRaises(SchemaError):
            compile_trace(trace)

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

    def test_extremely_large_timestamps_are_rejected_early(self):
        trace = {
            "format": TRACE_FORMAT,
            "events": [
                {
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "start_us": 1e15,
                    "rank_arrival_us": {"0": 0.0, "1": 0.0},
                }
            ],
        }
        with self.assertRaises(SchemaError):
            compile_trace(trace)

        canary = compile_trace(small_trace())
        canary["events"][0]["timing_samples"][0]["gap_us"] = 1e15
        with self.assertRaises(SchemaError):
            validate_canary(canary)

    def test_empty_trace_requires_explicit_override(self):
        empty = {"format": TRACE_FORMAT, "events": []}
        with self.assertRaises(SchemaError):
            compile_trace(empty)
        canary = compile_trace(empty, allow_empty=True)
        self.assertEqual(canary["compiler"]["source_events"], 0)

    def test_compile_rejects_non_boolean_feature_switches(self):
        with self.assertRaisesRegex(SchemaError, "allow_empty must be a boolean"):
            compile_trace(small_trace(), allow_empty=1)
        with self.assertRaisesRegex(SchemaError, "enable_sequence_motifs must be a boolean"):
            compile_trace(small_trace(), enable_sequence_motifs=1)

    def test_sequence_motif_compression_preserves_scheduler_execution(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "abab"}, "events": []}
        for index in range(8):
            op = "all_reduce" if index % 2 == 0 else "all_gather"
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": op,
                    "bytes": 1024 + (index % 2) * 512,
                    "ranks": [0, 1],
                    "group": "tp",
                    "gap_us": 1.0,
                    "rank_arrival_us": {"0": 0.0, "1": 1.0},
                    "compute_overlap_us": 2.0,
                }
            )
        flat = compile_trace(trace, enable_sequence_motifs=False)
        motif = compile_trace(trace)
        self.assertEqual(motif["compiler"]["sequence_motif_count"], 1)
        self.assertEqual(motif["compiler"]["canary_events"], 1)
        self.assertLess(
            motif["compiler"]["stored_recursive_timing_records"],
            motif["compiler"]["recursive_timing_records"],
        )
        self.assertEqual(
            flat["compiler"]["scheduler_execution_sha256"],
            motif["compiler"]["scheduler_execution_sha256"],
        )
        self.assertEqual(
            replay_canary(flat, include_samples=True)["samples"],
            replay_canary(motif, include_samples=True)["samples"],
        )
        self.assertEqual(verify_canary_fidelity(trace, motif)["status"], "source_verified")

    def test_behavior_search_refines_per_group_timing_budgets(self):
        trace = two_group_refinement_trace()
        canary = synthesize_behavioral_canary(
            trace,
            min_timing_sample_limit=2,
            max_timing_sample_limit=20,
            behavior_configurations=adversarial_ranking_configs(),
            relative_tolerance_pct=1000.0,
            absolute_tolerance_us=1000.0,
            hidden_tolerance_points=100.0,
            tail_recall_threshold=0.0,
            ranking_tie_tolerance_us=0.0,
        )
        search = canary["compiler"]["behavior_search"]
        refinement = search["per_group_refinement"]
        self.assertEqual(canary["compiler"]["timing_sample_limit_mode"], "per_group")
        self.assertEqual(refinement["status"], "refined")
        self.assertGreaterEqual(refinement["accepted_candidates"], 1)
        self.assertIn("1", canary["compiler"]["timing_sample_limits_by_group"])

        fidelity = verify_canary_fidelity(trace, canary)
        behavior = verify_canary_behavior(
            trace,
            canary,
            configurations=adversarial_ranking_configs(),
            relative_tolerance_pct=1000.0,
            absolute_tolerance_us=1000.0,
            hidden_tolerance_points=100.0,
            tail_recall_threshold=0.0,
            ranking_tie_tolerance_us=0.0,
        )
        self.assertEqual(fidelity["status"], "source_verified")
        self.assertEqual(behavior["status"], "behaviorally_verified")

    def test_compile_rejects_invalid_per_group_timing_budget(self):
        with self.assertRaises(SchemaError):
            compile_trace(small_trace(), timing_sample_limit=8, timing_sample_limits_by_group={99: 2})
        with self.assertRaises(SchemaError):
            compile_trace(small_trace(), timing_sample_limit=8, timing_sample_limits_by_group={0: 1})
