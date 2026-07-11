from __future__ import annotations

import json
import statistics
import tempfile
import unittest
from pathlib import Path

from benchmarks.fixtures import materialize_fixture_set
from benchmarks.runner import (
    BENCHMARK_RESULT_FORMAT,
    BENCHMARK_SUITE_FORMAT,
    load_fixture_manifest,
    run_case,
    run_manifest,
    run_smoke,
)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_representative_operations_record_measurement_and_semantic_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = materialize_fixture_set(
                Path(temp_dir),
                stored_event_counts=(12,),
                compressed_logical_counts=(16,),
            )
            trace_case, canary_case = load_fixture_manifest(manifest)
            results = [
                run_case(trace_case, "load"),
                run_case(trace_case, "validate"),
                run_case(trace_case, "hash"),
                run_case(canary_case, "replay"),
            ]
            for result in results:
                self.assertEqual(result["format"], BENCHMARK_RESULT_FORMAT)
                self.assertGreaterEqual(result["wall_time_seconds"], 0.0)
                self.assertEqual(len(result["semantic_sha256"]), 64)
                self.assertIn("python_version", result["environment"])
                self.assertIn("peak_rss_bytes", result)
                self.assertIn("peak_rss_method", result)
                self.assertGreaterEqual(result["python_peak_allocated_bytes"], 0)

    def test_semantic_hashes_are_stable_across_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = materialize_fixture_set(
                Path(temp_dir),
                stored_event_counts=(10,),
                compressed_logical_counts=(16,),
            )
            trace_case, canary_case = load_fixture_manifest(manifest)
            for case, operation in ((trace_case, "validate"), (canary_case, "replay")):
                first = run_case(case, operation)
                second = run_case(case, operation)
                self.assertEqual(first["semantic_sha256"], second["semantic_sha256"])

    def test_scale_operation_families_use_uniform_measurements_and_stable_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = materialize_fixture_set(
                Path(temp_dir),
                stored_event_counts=(12,),
                compressed_logical_counts=(16,),
            )
            trace_case, canary_case = load_fixture_manifest(manifest)
            cases = (
                (trace_case, "capture_merge"),
                (trace_case, "behavior_search"),
                (canary_case, "compare"),
                (canary_case, "param_export"),
            )
            for case, operation in cases:
                first = run_case(case, operation)
                second = run_case(case, operation)
                self.assertEqual(first["semantic_sha256"], second["semantic_sha256"])
                self.assertEqual(first["input_sha256"], case.sha256)
                self.assertGreaterEqual(first["wall_time_seconds"], 0.0)
                self.assertGreaterEqual(first["python_peak_allocated_bytes"], 0)
                self.assertIn("environment", first)
                self.assertEqual(first["measurement_scope"], "registered-operation-only")
                self.assertFalse(first["preparation_included_in_wall_time"])
                self.assertTrue(first["rss_baseline_after_preparation"])
                if operation in {"capture_merge", "behavior_search", "compare"}:
                    self.assertEqual(len(first["prepared_input_semantic_sha256"]), 64)

    def test_adversarial_resource_operations_reject_in_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = materialize_fixture_set(
                Path(temp_dir),
                stored_event_counts=(12,),
                compressed_logical_counts=(16,),
            )
            trace_case, canary_case = load_fixture_manifest(manifest)
            results = (
                run_case(trace_case, "capture_merge_preflight"),
                run_case(trace_case, "behavior_search_preflight"),
                run_case(canary_case, "param_export_preflight"),
            )

            self.assertEqual(
                {result["operation"] for result in results},
                {
                    "capture_merge_preflight",
                    "behavior_search_preflight",
                    "param_export_preflight",
                },
            )
            self.assertTrue(all(len(result["semantic_sha256"]) == 64 for result in results))
            self.assertTrue(all(result["wall_time_seconds"] < 1.0 for result in results))

    def test_manifest_runner_can_isolate_a_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = materialize_fixture_set(
                Path(temp_dir),
                stored_event_counts=(8,),
                compressed_logical_counts=(),
            )
            suite = run_manifest(manifest, operations=("validate",), isolate=True, profile="test")
            self.assertEqual(suite["format"], BENCHMARK_SUITE_FORMAT)
            self.assertEqual(suite["result_count"], 1)
            self.assertEqual(suite["results"][0]["operation"], "validate")

    def test_smoke_suite_is_fast_and_covers_required_operation_families(self) -> None:
        suite = run_smoke(isolate=False)
        self.assertEqual(suite["format"], BENCHMARK_SUITE_FORMAT)
        self.assertEqual(suite["profile"], "smoke")
        operations = {result["operation"] for result in suite["results"]}
        self.assertTrue(
            {
                "load",
                "validate",
                "hash",
                "replay",
                "compare",
                "capture_merge",
                "param_export",
                "behavior_search",
                "capture_merge_preflight",
                "param_export_preflight",
                "behavior_search_preflight",
            }.issubset(operations)
        )
        self.assertEqual(len(suite["semantic_set_sha256"]), 64)

    def test_reviewed_baseline_is_compact_observational_evidence_without_thresholds(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "baselines"
            / ("local-arm64-macos-cpython310-20260711.json")
        )
        baseline_text = path.read_text(encoding="utf-8")
        baseline = json.loads(baseline_text)

        self.assertEqual(baseline["format"], "commcanary.benchmark-baseline.v1")
        self.assertIsNone(baseline["regression_thresholds"])
        self.assertEqual(baseline["measurement_policy"]["repeats"], 1)
        self.assertLess(path.stat().st_size, 20_000)
        self.assertNotIn("/Users/", baseline_text)
        self.assertNotIn("/home/", baseline_text)
        self.assertNotIn("\\Users\\", baseline_text)
        families = {row["operation"] for row in baseline["results"]}
        self.assertEqual(families, {"compare", "capture_merge", "param_export", "behavior_search"})
        self.assertTrue(all(len(row["semantic_sha256"]) == 64 for row in baseline["results"]))
        skipped = baseline["campaign"]["skipped"]
        self.assertEqual(skipped[0]["sizes"], [10_000, 100_000])

        review = baseline["optimization_review"]
        self.assertNotIn("environment_executable", review)
        self.assertIsNone(review["regression_thresholds"])
        self.assertEqual(review["measurement_policy"]["repeats"], 3)
        self.assertEqual(len(review["fixture_manifest_sha256"]), 64)
        self.assertEqual(
            {row["operation"] for row in review["results"]},
            {"capture_merge", "param_export"},
        )
        self.assertTrue(all(len(row["semantic_sha256"]) == 64 for row in review["results"]))
        self.assertTrue(all(len(value) == 64 for value in review["semantic_set_sha256"].values()))
        for row in review["results"]:
            for phase in ("before", "after"):
                samples = row[f"{phase}_samples"]
                medians = row[f"{phase}_median"]
                for metric, values in samples.items():
                    self.assertEqual(len(values), 3)
                    self.assertEqual(statistics.median(values), medians[metric])


if __name__ == "__main__":
    unittest.main()
