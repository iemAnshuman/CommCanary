from __future__ import annotations

import copy
import unittest

from commcanary.compiler import (
    compile_trace,
    verify_canary_behavior,
    verify_canary_fidelity,
)
from commcanary.replay import replay_canary, verify_report_against_canary
from commcanary.schema import TRACE_FORMAT, SchemaError, validate_canary, validate_report
from tests.artifact_helpers import refresh_canary_hashes
from tests.builders import adversarial_ranking_configs, adversarial_ranking_trace, small_trace


class VerificationTests(unittest.TestCase):
    def test_report_verification_recomputes_declared_model(self):
        canary = compile_trace(small_trace())
        report = replay_canary(canary, seed=3)
        verification = verify_report_against_canary(report, canary)
        self.assertEqual(verification["status"], "model_recomputed")

        forged = copy.deepcopy(report)
        forged["metrics"].update(
            {
                "median_us": 1000.0,
                "p95_us": 1000.0,
                "p99_us": 1000.0,
                "max_us": 1000.0,
                "mean_us": 1000.0,
            }
        )
        for key in ("by_phase", "by_op"):
            for row in forged[key]:
                row.update(
                    {
                        "median_us": 1000.0,
                        "p95_us": 1000.0,
                        "p99_us": 1000.0,
                        "max_us": 1000.0,
                        "mean_us": 1000.0,
                    }
                )
        validate_report(forged)
        verification = verify_report_against_canary(forged, canary)
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(
            any(check["name"] == "metrics" and check["status"] == "fail" for check in verification["checks"])
        )

    def test_report_verification_rejects_forged_identity_and_summary_metadata(self):
        canary = compile_trace(small_trace())
        report = replay_canary(canary, seed=3)

        forged_canary = copy.deepcopy(report)
        forged_canary["canary"]["scheduler_execution_sha256"] = "0" * 64
        validate_report(forged_canary)
        verification = verify_report_against_canary(forged_canary, canary)
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(
            any(check["name"] == "canary" and check["status"] == "fail" for check in verification["checks"])
        )

        forged_summary = copy.deepcopy(report)
        forged_summary["canary_summary"]["source_events"] += 1
        validate_report(forged_summary)
        verification = verify_report_against_canary(forged_summary, canary)
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(
            any(check["name"] == "canary_summary" and check["status"] == "fail" for check in verification["checks"])
        )

    def test_replay_ablations_are_report_verifiable_and_visible_in_samples(self):
        canary = compile_trace(small_trace())
        report = replay_canary(
            canary,
            include_samples=True,
            ablations=["arrival_skew", "compute_overlap", "rare_tail_windows"],
        )
        self.assertEqual(report["backend"]["ablations"], ["arrival_skew", "compute_overlap", "rare_tail_windows"])
        self.assertTrue(all(sample["arrival_skew_us"] == 0.0 for sample in report["samples"]))
        self.assertTrue(all(sample["compute_overlap_us"] == 0.0 for sample in report["samples"]))
        self.assertEqual(verify_report_against_canary(report, canary)["status"], "model_recomputed")

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

    def test_source_assisted_fidelity_verification_rejects_tampered_errors(self):
        trace = {"format": TRACE_FORMAT, "workload": {"name": "verify-fidelity"}, "events": []}
        for index in range(80):
            trace["events"].append(
                {
                    "id": f"event-{index}",
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 1024,
                    "ranks": [0, 1],
                    "start_us": index * 10.0,
                    "rank_arrival_us": {"0": 0.0, "1": float(index % 21)},
                    "compute_overlap_us": float((index * 17) % 29),
                    "compute_pressure": ((index * 31) % 100) / 100.0,
                }
            )
        canary = compile_trace(trace, timing_sample_limit=4)
        self.assertEqual(verify_canary_fidelity(trace, canary)["status"], "source_verified")

        tampered = copy.deepcopy(canary)
        fidelity_fields = (
            "max_gap_error_us",
            "max_skew_error_us",
            "max_arrival_offset_error_us",
            "max_compute_before_error_us",
            "max_overlap_error_us",
            "max_pressure_error",
            "max_observed_exposed_error_us",
            "max_prefix_gap_error_us",
        )

        def zero_record_errors(records):
            for record in records:
                for field in fidelity_fields:
                    if field in record:
                        record[field] = 0.0
                    if isinstance(record.get("error_vector"), dict) and field in record["error_vector"]:
                        record["error_vector"][field] = 0.0
                if isinstance(record.get("timing_pattern"), list):
                    zero_record_errors(record["timing_pattern"])

        zero_record_errors(tampered["events"][0]["timing_samples"])
        for field in fidelity_fields:
            if field in tampered["compiler"]["fidelity"]:
                tampered["compiler"]["fidelity"][field] = 0.0
        refresh_canary_hashes(tampered)
        validate_canary(tampered)
        verification = verify_canary_fidelity(trace, tampered)
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(
            any(check["name"] == "fidelity" and check["status"] == "fail" for check in verification["checks"])
        )

    def test_source_commitments_are_independently_verified(self):
        trace = adversarial_ranking_trace()
        canary = compile_trace(trace, timing_sample_limit=2)
        interval = next(
            sample
            for sample in canary["events"][0]["timing_samples"]
            if sample.get("approximation") == "bounded_interval"
        )
        self.assertIn("source_count", interval)
        self.assertIn("source_segment_sha256", interval)
        self.assertIn("source_gap_sum_us", interval)
        self.assertIn("representative_selection_method", interval)
        self.assertIn("error_vector", interval)
        self.assertEqual(verify_canary_fidelity(trace, canary)["status"], "source_verified")

        tampered = copy.deepcopy(canary)
        interval = next(
            sample
            for sample in tampered["events"][0]["timing_samples"]
            if sample.get("approximation") == "bounded_interval"
        )
        interval["source_segment_sha256"] = "0" * 64
        refresh_canary_hashes(tampered)
        validate_canary(tampered)
        verification = verify_canary_fidelity(trace, tampered)
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(
            any(check["name"] == "source_commitments" and check["status"] == "fail" for check in verification["checks"])
        )

    def test_behavior_verification_compares_source_and_canary_replay(self):
        trace = small_trace()
        canary = compile_trace(trace)
        verification = verify_canary_behavior(trace, canary)
        self.assertEqual(verification["status"], "behaviorally_verified")
        self.assertEqual(verification["ranking"]["status"], "pass")

        changed = copy.deepcopy(canary)
        changed["events"][0]["bytes"] *= 64
        refresh_canary_hashes(changed)
        validate_canary(changed)
        verification = verify_canary_behavior(trace, changed)
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(
            any(check["status"] == "fail" for row in verification["configurations"] for check in row["checks"])
        )

    def test_behavior_verification_distinguishes_statuses_and_ranking_inversion(self):
        trace = adversarial_ranking_trace()
        lossy = compile_trace(trace, timing_sample_limit=2)
        verification = verify_canary_behavior(
            trace,
            lossy,
            configurations=adversarial_ranking_configs(),
            relative_tolerance_pct=1000.0,
            absolute_tolerance_us=1000.0,
            hidden_tolerance_points=100.0,
            tail_recall_threshold=0.0,
            ranking_tie_tolerance_us=0.0,
        )
        self.assertEqual(verification["representation_fidelity_status"], "bounded_approximate")
        self.assertEqual(verification["source_verified_status"], "source_verified")
        self.assertEqual(verification["behavioral_fidelity_status"], "pass")
        self.assertEqual(verification["configuration_ranking_status"], "fail")
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(
            any(
                pair["status"] == "fail" and pair["metric"] == "median_us"
                for pair in verification["ranking"]["pairwise"]
            )
        )

        lossless = compile_trace(trace, timing_sample_limit=32, require_lossless_timing=True)
        verification = verify_canary_behavior(
            trace,
            lossless,
            configurations=adversarial_ranking_configs(),
            ranking_tie_tolerance_us=0.0,
        )
        self.assertEqual(verification["status"], "behaviorally_verified")
        self.assertEqual(verification["configuration_ranking_status"], "pass")

    def test_behavior_verification_replays_full_source_not_candidate_prefix(self):
        trace = small_trace()
        prefix_canary = compile_trace(trace, max_events=1, require_lossless_timing=True)
        verification = verify_canary_behavior(trace, prefix_canary)
        self.assertEqual(verification["source_coverage_status"], "partial_source")
        self.assertEqual(verification["source_verified_status"], "partial_source_verified")
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(
            any(
                check["metric"] == "count" and check["status"] == "fail"
                for check in verification["configurations"][0]["checks"]
            )
        )

    def test_required_behavior_verification_rejects_ranking_losing_compile(self):
        trace = adversarial_ranking_trace()
        with self.assertRaises(SchemaError):
            compile_trace(
                trace,
                timing_sample_limit=2,
                require_behavior_verification=True,
                behavior_configurations=adversarial_ranking_configs(),
            )

        faithful = compile_trace(
            trace,
            timing_sample_limit=32,
            require_lossless_timing=True,
            require_behavior_verification=True,
            behavior_configurations=adversarial_ranking_configs(),
        )
        self.assertEqual(faithful["compiler"]["behavior_verification_status"], "behaviorally_verified")
        self.assertEqual(faithful["compiler"]["configuration_ranking_status"], "pass")

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
        malformed = copy.deepcopy(canary)
        malformed["compiler"].pop("fidelity")
        malformed["compiler"].pop("fidelity_budget", None)
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
