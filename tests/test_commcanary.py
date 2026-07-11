"""Cross-boundary integration characterization.

Capability-level assertions live in the dedicated test packages.  These
remaining workflows intentionally compose CLI, baseline/reduction services,
verification, replay, and presentation, so keeping them together preserves the
end-to-end scenarios without rebuilding the same pipeline in several suites.
"""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

from commcanary.baselines import (
    clustering_representative_baseline_trace,
    frequency_representative_baseline_trace,
    isolated_collective_baseline_trace,
    random_sampling_baseline_trace,
    stratified_sampling_baseline_trace,
)
from commcanary.cli import main as cli_main
from commcanary.compare import compare_reports
from commcanary.compiler import (
    compile_trace,
    verify_canary_behavior,
)
from commcanary.html_report import render_compare_html, render_report_html
from commcanary.reduce import ddmin_ranking_reduction
from commcanary.replay import replay_canary
from commcanary.schema import (
    SchemaError,
    load_json,
    validate_trace,
    write_json,
)
from tests.builders import (
    adversarial_ranking_configs,
    adversarial_ranking_trace,
    small_trace,
)


class CommCanaryTests(unittest.TestCase):
    def test_compile_behavior_search_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "trace.json")
            canary_path = os.path.join(tmp, "canary.json")
            write_json(trace_path, small_trace())
            exit_code = cli_main(
                [
                    "compile",
                    trace_path,
                    "--output",
                    canary_path,
                    "--behavior-search",
                    "--timing-sample-limit",
                    "4",
                ]
            )
            self.assertEqual(exit_code, 0)
            canary = load_json(canary_path)
            self.assertIn("behavior_search", canary["compiler"])
            self.assertEqual(canary["compiler"]["behavior_verification_status"], "behaviorally_verified")

    def test_html_escapes_once_and_allowlists_verdict_class(self):
        canary = compile_trace(small_trace())
        report = replay_canary(canary)
        report["workload"]["name"] = "A&B"
        html = render_report_html(report)
        self.assertIn("<title>CommCanary Report - A&amp;B</title>", html)
        self.assertNotIn("A&amp;amp;B", html)
        self.assertIn('http-equiv="Content-Security-Policy"', html)
        self.assertNotIn("<script", html)
        self.assertIn("Samples unavailable", html)
        self.assertIn("reported count and quantiles", html)

        comparison = compare_reports(report, copy.deepcopy(report))
        comparison["reasons"] = ["A&B"]
        compare_html = render_compare_html(comparison)
        self.assertIn('class="hero pass"', compare_html)
        self.assertNotIn("onclick", compare_html)
        self.assertIn("A&amp;B", compare_html)

        malformed = copy.deepcopy(comparison)
        malformed["verdict"] = 'fail" onclick="bad'
        with self.assertRaises(SchemaError):
            render_compare_html(malformed)

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
            verification_path = os.path.join(tmp, "fidelity.json")
            behavior_path = os.path.join(tmp, "behavior.json")
            report_verification_path = os.path.join(tmp, "report-verification.json")
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
            self.assertEqual(cli_main(["verify-fidelity", trace_path, canary_path, "--output", verification_path]), 0)
            self.assertEqual(cli_main(["verify-behavior", trace_path, canary_path, "--output", behavior_path]), 0)
            self.assertEqual(
                cli_main(["verify-report", report_path, canary_path, "--output", report_verification_path]),
                0,
            )
            self.assertTrue(os.path.exists(html_path))
            self.assertEqual(load_json(verification_path)["status"], "source_verified")
            self.assertEqual(load_json(behavior_path)["status"], "behaviorally_verified")
            self.assertEqual(load_json(report_verification_path)["status"], "model_recomputed")

    def test_clustering_baseline_trace_is_valid_negative_control(self):
        trace = adversarial_ranking_trace()
        baseline = clustering_representative_baseline_trace(trace, cluster_count=4)
        validate_trace(baseline)
        self.assertEqual(len(baseline["events"]), len(trace["events"]))
        self.assertEqual(baseline["workload"]["baseline_method"], "clustering_representative")
        self.assertIn("cluster_count", baseline["events"][0]["metadata"])

        canary = compile_trace(baseline, timing_sample_limit=16)
        verification = verify_canary_behavior(
            trace,
            canary,
            configurations=adversarial_ranking_configs(),
            relative_tolerance_pct=1000.0,
            absolute_tolerance_us=1000.0,
            hidden_tolerance_points=100.0,
            tail_recall_threshold=0.0,
            ranking_tie_tolerance_us=0.0,
        )
        self.assertEqual(verification["source_verified_status"], "failed")
        self.assertNotEqual(verification["status"], "behaviorally_verified")

    def test_research_baseline_traces_are_valid_and_not_source_verified_against_original(self):
        trace = adversarial_ranking_trace()
        random_baseline = random_sampling_baseline_trace(trace, sample_count=8, seed=5)
        frequency_baseline = frequency_representative_baseline_trace(trace)
        cluster_baseline = clustering_representative_baseline_trace(trace, cluster_count=4)
        isolated_baseline = isolated_collective_baseline_trace(trace)

        validate_trace(random_baseline)
        validate_trace(frequency_baseline)
        validate_trace(cluster_baseline)
        validate_trace(isolated_baseline)
        self.assertEqual(len(random_baseline["events"]), len(trace["events"]))
        self.assertEqual(len(frequency_baseline["events"]), len(trace["events"]))
        self.assertEqual(len(cluster_baseline["events"]), len(trace["events"]))
        self.assertLess(len(isolated_baseline["events"]), len(trace["events"]))

        random_canary = compile_trace(random_baseline, timing_sample_limit=16)
        frequency_canary = compile_trace(frequency_baseline, timing_sample_limit=16)
        random_verification = verify_canary_behavior(
            trace,
            random_canary,
            configurations=adversarial_ranking_configs(),
            relative_tolerance_pct=1000.0,
            absolute_tolerance_us=1000.0,
            hidden_tolerance_points=100.0,
            tail_recall_threshold=0.0,
            ranking_tie_tolerance_us=0.0,
        )
        frequency_verification = verify_canary_behavior(
            trace,
            frequency_canary,
            configurations=adversarial_ranking_configs(),
            relative_tolerance_pct=1000.0,
            absolute_tolerance_us=1000.0,
            hidden_tolerance_points=100.0,
            tail_recall_threshold=0.0,
            ranking_tie_tolerance_us=0.0,
        )
        self.assertEqual(random_verification["source_verified_status"], "failed")
        self.assertEqual(frequency_verification["source_verified_status"], "failed")
        self.assertNotEqual(random_verification["status"], "behaviorally_verified")
        self.assertNotEqual(frequency_verification["status"], "behaviorally_verified")

    def test_baseline_cli_generates_frequency_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "trace.json")
            output_path = os.path.join(tmp, "frequency.json")
            write_json(trace_path, small_trace())
            self.assertEqual(
                cli_main(["baseline", trace_path, "--method", "frequency", "--output", output_path]),
                0,
            )
            generated = load_json(output_path)
            validate_trace(generated)
            self.assertEqual(generated["workload"]["baseline_method"], "frequency_representative")

    def test_baseline_cli_generates_cluster_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "trace.json")
            output_path = os.path.join(tmp, "cluster.json")
            write_json(trace_path, adversarial_ranking_trace())
            self.assertEqual(
                cli_main(
                    [
                        "baseline",
                        trace_path,
                        "--method",
                        "cluster",
                        "--cluster-count",
                        "4",
                        "--output",
                        output_path,
                    ]
                ),
                0,
            )
            generated = load_json(output_path)
            validate_trace(generated)
            self.assertEqual(generated["workload"]["baseline_method"], "clustering_representative")

    def test_stratified_baseline_trace_is_valid_negative_control(self):
        trace = adversarial_ranking_trace()
        baseline = stratified_sampling_baseline_trace(trace, strata_per_group=4, seed=7)
        validate_trace(baseline)
        self.assertEqual(len(baseline["events"]), len(trace["events"]))
        self.assertEqual(baseline["workload"]["baseline_method"], "stratified_sampling")
        self.assertIn("stratum_index", baseline["events"][0]["metadata"])
        self.assertEqual(
            baseline,
            stratified_sampling_baseline_trace(trace, strata_per_group=4, seed=7),
        )

        canary = compile_trace(baseline, timing_sample_limit=16)
        verification = verify_canary_behavior(
            trace,
            canary,
            configurations=adversarial_ranking_configs(),
            relative_tolerance_pct=1000.0,
            absolute_tolerance_us=1000.0,
            hidden_tolerance_points=100.0,
            tail_recall_threshold=0.0,
            ranking_tie_tolerance_us=0.0,
        )
        self.assertEqual(verification["source_verified_status"], "failed")
        self.assertNotEqual(verification["status"], "behaviorally_verified")

    def test_baseline_cli_generates_stratified_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "trace.json")
            output_path = os.path.join(tmp, "stratified.json")
            write_json(trace_path, adversarial_ranking_trace())
            self.assertEqual(
                cli_main(
                    [
                        "baseline",
                        trace_path,
                        "--method",
                        "stratified",
                        "--strata-per-group",
                        "3",
                        "--seed",
                        "5",
                        "--output",
                        output_path,
                    ]
                ),
                0,
            )
            generated = load_json(output_path)
            validate_trace(generated)
            self.assertEqual(generated["workload"]["baseline_method"], "stratified_sampling")

    def test_ddmin_reduction_preserves_configuration_rankings(self):
        trace = adversarial_ranking_trace()
        reduced = ddmin_ranking_reduction(
            trace,
            configurations=adversarial_ranking_configs(),
            max_oracle_calls=96,
        )
        validate_trace(reduced)
        reduction = reduced["workload"]["reduction"]
        self.assertEqual(reduction["method"], "ddmin_ranking")
        self.assertEqual(len(reduced["events"]), reduction["reduced_events"])
        self.assertGreater(reduction["oracle_calls"], 0)
        self.assertLess(reduction["reduced_events"], reduction["original_events"])
        self.assertEqual(reduced["workload"]["reduction_method"], "ddmin_ranking")
        self.assertIn(
            "Not source-verified", reduced["workload"]["notes"].replace("not source-verified", "Not source-verified")
        )

        def median_ranking(source_trace):
            canary = compile_trace(
                source_trace,
                timing_sample_limit=max(2, len(source_trace["events"])),
                require_lossless_timing=True,
            )
            medians = {}
            for config in adversarial_ranking_configs():
                settings = dict(config)
                name = settings.pop("name")
                report = replay_canary(canary, backend_label=name, **settings)
                medians[name] = report["metrics"]["median_us"]
            return sorted(medians, key=lambda name: medians[name])

        self.assertEqual(median_ranking(trace), median_ranking(reduced))

    def test_reduce_cli_writes_decision_preserving_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "trace.json")
            output_path = os.path.join(tmp, "reduced.json")
            write_json(trace_path, small_trace())
            self.assertEqual(
                cli_main(
                    [
                        "reduce",
                        trace_path,
                        "--max-oracle-calls",
                        "64",
                        "--output",
                        output_path,
                    ]
                ),
                0,
            )
            reduced = load_json(output_path)
            validate_trace(reduced)
            self.assertEqual(reduced["workload"]["reduction_method"], "ddmin_ranking")
            self.assertLessEqual(len(reduced["events"]), len(small_trace()["events"]))

    def test_ddmin_reduction_rejects_duplicate_configuration_names(self):
        configs = adversarial_ranking_configs()
        configs[1]["name"] = configs[0]["name"]
        with self.assertRaisesRegex(SchemaError, "unique configuration names"):
            ddmin_ranking_reduction(small_trace(), configurations=configs)

    def test_synthetic_trace_script_does_not_overwrite_checked_in_fixture(self):
        script = Path(__file__).resolve().parent.parent / "examples" / "make_synthetic_trace.py"
        source = script.read_text(encoding="utf-8")
        self.assertNotIn('"llama70b_tp8_trace.json"', source)
        self.assertIn("llama70b_tp8_trace_long.json", source)


if __name__ == "__main__":
    unittest.main()
