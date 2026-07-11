from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from benchmarks.fixtures import (
    FIXED_CREATED_AT,
    STANDARD_COMPRESSED_LOGICAL_COUNTS,
    STANDARD_STORED_EVENT_COUNTS,
    canonical_fixture_sha256,
    generate_behavior_search_trace,
    generate_compressed_canary,
    generate_trace,
    materialize_capture_shards,
    materialize_fixture_set,
)
from benchmarks.runner import load_fixture_manifest
from commcanary.capture import merge_trace_shards
from commcanary.schema import canonical_json_bytes, preflight_canary_expansion, validate_canary, validate_trace


class BenchmarkFixtureTests(unittest.TestCase):
    def test_trace_generation_is_exact_and_deterministic(self) -> None:
        first = generate_trace(37)
        second = generate_trace(37)
        validate_trace(first)
        self.assertEqual(len(first["events"]), 37)
        self.assertEqual(canonical_fixture_sha256(first), canonical_fixture_sha256(second))
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))

    def test_compressed_fixture_has_large_logical_expansion_and_stable_bytes(self) -> None:
        first = generate_compressed_canary(40)
        second = generate_compressed_canary(40)
        validate_canary(first)
        expansion = preflight_canary_expansion(first["events"])
        self.assertEqual(expansion.logical_events, 40)
        self.assertLess(expansion.stored_events, expansion.logical_events)
        self.assertEqual(first["created_at"], FIXED_CREATED_AT)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))

    def test_materialized_manifest_is_reproducible_and_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_manifest = materialize_fixture_set(
                Path(first_dir),
                stored_event_counts=(8, 16),
                compressed_logical_counts=(16,),
            )
            second_manifest = materialize_fixture_set(
                Path(second_dir),
                stored_event_counts=(8, 16),
                compressed_logical_counts=(16,),
            )
            self.assertEqual(first_manifest.read_bytes(), second_manifest.read_bytes())
            first_cases = load_fixture_manifest(first_manifest)
            second_cases = load_fixture_manifest(second_manifest)
            self.assertEqual([case.sha256 for case in first_cases], [case.sha256 for case in second_cases])

            target = first_cases[0].path
            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_fixture_manifest(first_manifest)

    def test_standard_sizes_include_1k_10k_and_100k(self) -> None:
        # Assert the full campaign contract without writing or benchmarking the
        # expensive 100K fixture in the fast test suite.
        self.assertEqual(STANDARD_STORED_EVENT_COUNTS, (1_000, 10_000, 100_000))
        self.assertEqual(STANDARD_COMPRESSED_LOGICAL_COUNTS, (10_000, 100_000))
        trace = generate_trace(1_000)
        self.assertEqual(len(trace["events"]), 1_000)
        digest = hashlib.sha256(canonical_json_bytes(trace)).hexdigest()
        self.assertEqual(len(digest), 64)

    def test_behavior_search_fixture_is_purpose_built_and_deterministic(self) -> None:
        first = generate_behavior_search_trace(32)
        second = generate_behavior_search_trace(32)

        self.assertEqual(first["workload"]["motif_pattern_length"], 2)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        with self.assertRaisesRegex(ValueError, "at least four"):
            generate_behavior_search_trace(3)

    def test_capture_merge_preparation_is_deterministic_without_event_inflation(self) -> None:
        trace = generate_trace(12)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = materialize_capture_shards(trace, Path(first_dir))
            second = materialize_capture_shards(trace, Path(second_dir))

            self.assertEqual([path.read_bytes() for path in first], [path.read_bytes() for path in second])
            merged = merge_trace_shards(first_dir, workload_name="benchmark")
            self.assertEqual(len(merged["events"]), 12)
            self.assertEqual(merged["system"]["shards"], 2)


if __name__ == "__main__":
    unittest.main()
