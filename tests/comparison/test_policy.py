from __future__ import annotations

import copy
import math
import os
import tempfile
import unittest

from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace
from commcanary.html_report import render_compare_html
from commcanary.replay import replay_canary
from commcanary.schema import SchemaError, validate_comparison, validate_report, write_json
from tests.builders import small_trace


class ComparisonTests(unittest.TestCase):
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

    def test_allow_mismatch_includes_reasons_even_with_latency_failure(self):
        canary = compile_trace(small_trace())
        baseline = replay_canary(canary, seed=3)
        candidate = replay_canary(canary, seed=99, latency_floor_us=1000.0)
        comparison = compare_reports(baseline, candidate, require_compatible=False, p99_threshold_pct=1.0)
        self.assertEqual(comparison["verdict"], "fail")
        self.assertTrue(any("replay protocol" in reason for reason in comparison["reasons"]))

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

    def test_tiny_absolute_latency_delta_does_not_fail_relative_policy(self):
        baseline = replay_canary(compile_trace(small_trace()), seed=3)
        candidate = copy.deepcopy(baseline)
        for report, value in ((baseline, 0.001), (candidate, 0.002)):
            report["metrics"].update(
                {
                    "median_us": value,
                    "p95_us": value,
                    "p99_us": value,
                    "max_us": value,
                    "mean_us": value,
                }
            )
            for key in ("by_phase", "by_op"):
                for row in report[key]:
                    row.update(
                        {
                            "median_us": value,
                            "p95_us": value,
                            "p99_us": value,
                            "max_us": value,
                            "mean_us": value,
                        }
                    )
        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["verdict"], "pass")
        self.assertEqual(comparison["delta"]["p99_pct"], 100.0)
        self.assertEqual(comparison["evaluations"][2]["threshold_result"], "pass")

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

    def test_localized_p95_regression_fails_comparison(self):
        baseline = replay_canary(compile_trace(small_trace()), seed=3)
        candidate = copy.deepcopy(baseline)
        for report, phase_p95 in ((baseline, 10.0), (candidate, 90.0)):
            report["canary"]["source_events"] = 100
            report["metrics"].update(
                {
                    "count": 100,
                    "median_us": 10.0,
                    "p95_us": 10.0,
                    "p99_us": 100.0,
                    "max_us": 100.0,
                    "mean_us": 20.0,
                    "communication_hidden_pct": 50.0,
                }
            )
            report["by_phase"] = [
                {
                    "name": "rare-middle-tail",
                    "count": 100,
                    "median_us": 10.0,
                    "p95_us": phase_p95,
                    "p99_us": 100.0,
                    "max_us": 100.0,
                    "mean_us": 20.0,
                }
            ]
            report["by_op"] = [
                {
                    "name": "all_reduce",
                    "count": 100,
                    "median_us": 10.0,
                    "p95_us": 10.0,
                    "p99_us": 100.0,
                    "max_us": 100.0,
                    "mean_us": 20.0,
                }
            ]

        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["verdict"], "fail")
        self.assertEqual(comparison["delta"]["p95_pct"], 0.0)
        self.assertTrue(any("phase 'rare-middle-tail' p95" in reason for reason in comparison["reasons"]))
        self.assertIn("<th>P95 Δ</th>", render_compare_html(comparison))

    def test_comparison_artifacts_are_validated(self):
        canary = compile_trace(small_trace())
        baseline = replay_canary(canary, seed=3)
        candidate = replay_canary(canary, seed=3)
        comparison = compare_reports(baseline, candidate)
        validate_comparison(comparison)

        malformed = copy.deepcopy(comparison)
        malformed["delta"]["median_pct"] = 999.0
        with self.assertRaises(SchemaError):
            validate_comparison(malformed)

        failing = compare_reports(
            baseline,
            replay_canary(canary, seed=3, latency_floor_us=1000.0),
            p99_threshold_pct=1.0,
        )
        self.assertEqual(failing["verdict"], "fail")
        malformed = copy.deepcopy(failing)
        malformed["verdict"] = "pass"
        malformed["derived_verdict"] = "pass"
        malformed["reasons"] = ["candidate got much slower"]
        with self.assertRaises(SchemaError):
            validate_comparison(malformed)

        malformed = copy.deepcopy(failing)
        malformed["evaluations"][0]["threshold_result"] = "pass"
        with self.assertRaises(SchemaError):
            validate_comparison(malformed)

        malformed = copy.deepcopy(comparison)
        malformed["compatibility"]["compatible"] = True
        malformed["compatibility"]["reasons"] = ["reports use different replay protocol fingerprints"]
        with self.assertRaises(SchemaError):
            validate_comparison(malformed)

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
