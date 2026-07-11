"""Deterministic local scale fixtures and benchmark runner for CommCanary."""

from .fixtures import (
    FIXTURE_MANIFEST_FORMAT,
    STANDARD_STORED_EVENT_COUNTS,
    generate_behavior_search_trace,
    generate_compressed_canary,
    generate_trace,
    materialize_capture_shards,
    materialize_fixture_set,
)
from .runner import BENCHMARK_RESULT_FORMAT, BENCHMARK_SUITE_FORMAT, run_manifest, run_smoke

__all__ = [
    "BENCHMARK_RESULT_FORMAT",
    "BENCHMARK_SUITE_FORMAT",
    "FIXTURE_MANIFEST_FORMAT",
    "STANDARD_STORED_EVENT_COUNTS",
    "generate_behavior_search_trace",
    "generate_compressed_canary",
    "generate_trace",
    "materialize_capture_shards",
    "materialize_fixture_set",
    "run_manifest",
    "run_smoke",
]
