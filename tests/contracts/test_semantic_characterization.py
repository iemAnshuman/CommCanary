from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace, verify_canary_fidelity
from commcanary.replay import replay_canary
from commcanary.schema import (
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
    canonical_json_bytes,
    validate_canary,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "contracts"


def _load_json(name: str) -> Any:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _refresh_internal_hashes(canary: dict[str, Any]) -> None:
    compiler = canary["compiler"]
    compiler["execution_semantic_sha256"] = canary_execution_sha256(canary)
    compiler["scheduler_execution_sha256"] = canary_scheduler_execution_sha256(canary)
    compiler["calibration_evaluation_sha256"] = canary_calibration_sha256(canary)
    compiler["artifact_provenance_sha256"] = canary_artifact_provenance_sha256(canary)


def _without_created_at(document: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({key: value for key, value in document.items() if key != "created_at"})


def test_flat_and_sequence_motif_encodings_are_golden_semantic_equivalents() -> None:
    vector = _load_json("semantic_vectors.v1.json")["motif_equivalence"]
    trace = vector["trace"]
    expected = vector["expected"]
    flat = compile_trace(trace, enable_sequence_motifs=False)
    motif = compile_trace(trace, enable_sequence_motifs=True)

    assert motif["compiler"]["sequence_motif_count"] == expected["sequence_motif_count"]
    assert motif["events"][0]["source"]["digest"] == expected["motif_source_digest"]
    for field in ("execution_semantic_sha256", "scheduler_execution_sha256", "calibration_evaluation_sha256"):
        assert flat["compiler"][field] == expected[field]
        assert motif["compiler"][field] == expected[field]

    flat_report = replay_canary(flat, include_samples=True, seed=vector["replay_seed"])
    motif_report = replay_canary(motif, include_samples=True, seed=vector["replay_seed"])
    assert flat_report["samples"] == motif_report["samples"]
    samples_digest = hashlib.sha256(canonical_json_bytes({"samples": motif_report["samples"]})).hexdigest()
    assert samples_digest == expected["replay_samples_sha256"]
    assert verify_canary_fidelity(trace, motif)["status"] == "source_verified"


def test_flat_timing_runs_and_repeated_pattern_are_golden_semantic_equivalents() -> None:
    vector = _load_json("semantic_vectors.v1.json")["run_pattern_equivalence"]
    expected = vector["expected"]
    flat = compile_trace(vector["trace"])
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
    _refresh_internal_hashes(patterned)
    validate_canary(patterned)

    assert flat["compiler"]["execution_semantic_sha256"] == expected["execution_semantic_sha256"]
    assert patterned["compiler"]["execution_semantic_sha256"] == expected["execution_semantic_sha256"]
    assert flat["compiler"]["calibration_evaluation_sha256"] == expected["calibration_evaluation_sha256"]
    assert patterned["compiler"]["calibration_evaluation_sha256"] == expected["calibration_evaluation_sha256"]
    flat_report = replay_canary(flat, seed=vector["replay_seed"])
    pattern_report = replay_canary(patterned, seed=vector["replay_seed"])
    assert flat_report["metrics"] == expected["replay_metrics_seed_3"]
    assert pattern_report["metrics"] == expected["replay_metrics_seed_3"]


def test_replay_is_byte_deterministic_after_removing_created_at_for_each_seed() -> None:
    vectors = _load_json("semantic_vectors.v1.json")["replay_determinism"]
    trace = _load_json(vectors["trace_fixture"])
    canary = compile_trace(trace)

    for raw_seed, expected_metrics in vectors["expected_metrics_by_seed"].items():
        seed = int(raw_seed)
        first = replay_canary(canary, seed=seed, include_samples=True)
        second = replay_canary(canary, seed=seed, include_samples=True)
        assert first["metrics"] == expected_metrics
        assert _without_created_at(first) == _without_created_at(second)
        assert canonical_json_bytes(_without_created_at(first)) == canonical_json_bytes(_without_created_at(second))

    observed = {expected["median_us"] for expected in vectors["expected_metrics_by_seed"].values()}
    assert len(observed) == len(vectors["expected_metrics_by_seed"])


def _mutate_source_format(canary: dict[str, Any]) -> None:
    canary["source_format"] = "forged.trace.v1"


def _mutate_workload(canary: dict[str, Any]) -> None:
    canary["workload"]["name"] = "forged-workload"


def _mutate_system(canary: dict[str, Any]) -> None:
    canary["system"]["backend"] = "forged-backend"


def _mutate_source_hash_aliases(canary: dict[str, Any]) -> None:
    canary["compiler"]["source_trace_sha256"] = "f" * 64
    canary["compiler"]["source_normalized_sha256"] = "f" * 64


def _mutate_source_block_digest(canary: dict[str, Any]) -> None:
    canary["events"][0]["source"]["digest"] = "f" * 64


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (_mutate_source_format, "source_format"),
        (_mutate_workload, "workload"),
        (_mutate_system, "system"),
        (_mutate_source_hash_aliases, "source_trace_sha256"),
        (_mutate_source_block_digest, "source_commitments"),
    ],
    ids=["source-format", "workload", "system", "source-hashes", "source-block"],
)
def test_internally_rehashed_producer_mutations_still_disagree_with_source_verifier(
    mutation: Callable[[dict[str, Any]], None],
    failed_check: str,
) -> None:
    trace = _load_json("trace.valid.json")
    canary = compile_trace(trace)
    mutation(canary)
    _refresh_internal_hashes(canary)
    validate_canary(canary)

    verification = verify_canary_fidelity(trace, canary)
    checks = {check["name"]: check["status"] for check in verification["checks"]}
    assert verification["status"] == "failed"
    assert verification["assurance_state"] == "internally_consistent"
    assert checks[failed_check] == "fail"


def test_public_artifact_chain_returns_detached_snapshots() -> None:
    trace = _load_json("trace.valid.json")
    trace["workload"]["tags"] = ["original"]
    canary = compile_trace(trace)
    report = replay_canary(canary)
    sibling = replay_canary(canary)
    comparison = compare_reports(report, sibling)
    fidelity = verify_canary_fidelity(trace, canary)

    trace["workload"]["tags"].append("trace-mutation")
    canary["workload"]["tags"].append("canary-mutation")
    report["workload"]["tags"].append("report-mutation")
    sibling["metrics"]["median_us"] = 999.0

    assert canary["workload"]["tags"] == ["original", "canary-mutation"]
    assert report["workload"]["tags"] == ["original", "report-mutation"]
    assert comparison["baseline"]["metrics"]["median_us"] != 999.0
    workload_check = next(check for check in fidelity["checks"] if check["name"] == "workload")
    assert workload_check["expected"]["tags"] == ["original"]
    assert workload_check["actual"]["tags"] == ["original"]
