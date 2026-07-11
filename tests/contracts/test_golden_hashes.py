from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

import commcanary.compiler as compiler_module
from commcanary.compiler import compile_trace
from commcanary.replay import replay_canary
from commcanary.schema import (
    SchemaError,
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
    canonical_json_bytes,
    replay_protocol_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
VECTORS_PATH = ROOT / "tests" / "fixtures" / "contracts" / "hash_vectors.v1.json"


def _load_vectors() -> Mapping[str, Any]:
    with VECTORS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def reference_python_canonical_json_bytes(value: Any) -> bytes:
    """Independently compose the documented Python canonical JSON encoding."""

    def encode(child: Any) -> str:
        if isinstance(child, dict):
            if any(not isinstance(key, str) for key in child):
                raise TypeError("canonical JSON object keys must be strings")
            return "{" + ",".join(f"{encode(key)}:{encode(child[key])}" for key in sorted(child)) + "}"
        if isinstance(child, list):
            return "[" + ",".join(encode(item) for item in child) + "]"
        return json.dumps(child, ensure_ascii=True, allow_nan=False, separators=(",", ":"))

    return encode(value).encode("utf-8")


def reference_sha256(value: Any) -> str:
    return hashlib.sha256(reference_python_canonical_json_bytes(value)).hexdigest()


def reference_leaf_source_digest(source_ids: Iterable[Any]) -> tuple[bytes, str]:
    records = b"".join(reference_python_canonical_json_bytes(source_id) + b"\0" for source_id in source_ids)
    return records, hashlib.sha256(records).hexdigest()


def reference_artifact_provenance_projection(canary: Mapping[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy({key: value for key, value in canary.items() if key != "created_at"})
    compiler = projection["compiler"]
    projection["compiler"] = {
        key: value
        for key, value in compiler.items()
        if key not in {"artifact_provenance_sha256", "canary_bytes", "byte_compression_ratio"}
    }
    return projection


def reference_report_canary_projection(canary: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({key: value for key, value in canary.items() if key != "created_at"})


def reference_source_segment_projection(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = []
    for sample in samples:
        record = {
            "gap_us": round(float(sample.get("gap_us", 0.0)), 9),
            "arrival_offsets_us": [round(float(value), 9) for value in sample.get("arrival_offsets_us", [])],
            "arrival_skew_us": round(float(sample.get("arrival_skew_us", 0.0)), 9),
            "compute_before_us": round(float(sample.get("compute_before_us", 0.0)), 9),
            "compute_overlap_us": round(float(sample.get("compute_overlap_us", 0.0)), 9),
            "compute_pressure": round(float(sample.get("compute_pressure", 0.5)), 6),
        }
        if "observed_exposed_us" in sample:
            record["observed_exposed_us"] = round(float(sample["observed_exposed_us"]), 9)
        if sample.get("compute_fields_uncertain") is True:
            record["compute_fields_uncertain"] = True
        normalized.append(record)
    return {"samples": normalized}


@pytest.mark.parametrize("vector_index", [0, 1, 2])
def test_literal_canonical_json_bytes_and_digests(vector_index: int) -> None:
    vector = _load_vectors()["canonical_json_vectors"][vector_index]
    expected = vector["canonical_utf8"].encode("utf-8")

    assert bytes.fromhex(vector["canonical_utf8_hex"]) == expected
    assert reference_python_canonical_json_bytes(vector["input"]) == expected
    assert canonical_json_bytes(vector["input"]) == expected
    assert hashlib.sha256(expected).hexdigest() == vector["sha256"]


def test_canonical_json_is_invariant_to_insertion_order() -> None:
    original = _load_vectors()["canonical_json_vectors"][0]["input"]
    expected = canonical_json_bytes(original)
    items = list(original.items())
    generator = random.Random(1729)

    for _ in range(64):
        generator.shuffle(items)
        assert canonical_json_bytes(dict(items)) == expected


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(non_finite: float) -> None:
    with pytest.raises(SchemaError, match="cannot canonicalize JSON"):
        canonical_json_bytes({"value": non_finite})


def test_literal_source_block_motif_and_interval_digest_vectors() -> None:
    vectors = _load_vectors()
    leaf = vectors["artifact_chain"]["leaf_source_digest"]
    records, leaf_digest = reference_leaf_source_digest(leaf["ids"])
    assert records.hex() == leaf["canonical_records_hex"]
    assert leaf_digest == leaf["sha256"]

    motif = vectors["motif_source_digest"]
    motif_projection = {"sources": motif["source_digests"]}
    assert reference_python_canonical_json_bytes(motif_projection).decode("utf-8") == motif["canonical_utf8"]
    assert reference_sha256(motif_projection) == motif["sha256"]

    segment = vectors["source_segment_digest"]
    segment_projection = reference_source_segment_projection(segment["samples"])
    assert reference_python_canonical_json_bytes(segment_projection).decode("utf-8") == segment["canonical_utf8"]
    assert reference_sha256(segment_projection) == segment["sha256"]
    production_digest = getattr(compiler_module, "_source_segment_sha256")(segment["samples"])
    assert production_digest == segment["sha256"]


def test_cross_format_artifact_chain_matches_every_published_hash_projection() -> None:
    chain = _load_vectors()["artifact_chain"]
    expected = chain["expected_hashes"]
    trace = copy.deepcopy(chain["trace"])
    canary = compile_trace(trace)
    compiler = canary["compiler"]

    source_bytes = reference_python_canonical_json_bytes(trace)
    assert source_bytes.decode("utf-8") == chain["source_trace_projection"]["canonical_utf8"]
    assert source_bytes.hex() == chain["source_trace_projection"]["canonical_utf8_hex"]
    assert hashlib.sha256(source_bytes).hexdigest() == chain["source_trace_projection"]["sha256"]

    for field in (
        "source_trace_sha256",
        "source_normalized_sha256",
        "execution_semantic_sha256",
        "scheduler_execution_sha256",
        "calibration_evaluation_sha256",
        "artifact_provenance_sha256",
    ):
        assert compiler[field] == expected[field]

    leaf = chain["leaf_source_digest"]
    assert canary["events"][0]["source"]["digest"] == leaf["sha256"]
    assert canary_execution_sha256(canary) == expected["execution_semantic_sha256"]
    assert canary_scheduler_execution_sha256(canary) == expected["scheduler_execution_sha256"]
    assert canary_calibration_sha256(canary) == expected["calibration_evaluation_sha256"]

    provenance_projection = reference_artifact_provenance_projection(canary)
    assert reference_sha256(provenance_projection) == expected["artifact_provenance_sha256"]
    assert canary_artifact_provenance_sha256(canary) == expected["artifact_provenance_sha256"]

    report = replay_canary(canary, seed=chain["replay_seed"])
    report_projection = reference_report_canary_projection(canary)
    assert reference_sha256(report_projection) == expected["report_canary_sha256"]
    assert report["canary"]["sha256"] == expected["report_canary_sha256"]

    protocol_projection = {
        key: value for key, value in report["replay_protocol"].items() if key not in {"sha256", "max_replay_events"}
    }
    protocol_vector = chain["protocol_projection"]
    assert (
        reference_python_canonical_json_bytes(protocol_projection).decode("utf-8") == protocol_vector["canonical_utf8"]
    )
    assert reference_sha256(protocol_projection) == protocol_vector["sha256"]
    assert replay_protocol_sha256(report["replay_protocol"]) == protocol_vector["sha256"]


def test_literal_execution_and_calibration_projection_bytes_match_runtime_hashes() -> None:
    chain = _load_vectors()["artifact_chain"]
    canary = compile_trace(chain["trace"])

    execution_bytes = chain["execution_projection"]["canonical_utf8"].encode("utf-8")
    calibration_bytes = chain["calibration_projection"]["canonical_utf8"].encode("utf-8")
    assert hashlib.sha256(execution_bytes).hexdigest() == chain["execution_projection"]["sha256"]
    assert hashlib.sha256(calibration_bytes).hexdigest() == chain["calibration_projection"]["sha256"]
    assert canary_execution_sha256(canary) == hashlib.sha256(execution_bytes).hexdigest()
    assert canary_calibration_sha256(canary) == hashlib.sha256(calibration_bytes).hexdigest()


def test_replay_protocol_enforcement_ceiling_is_deliberately_not_hashed() -> None:
    chain = _load_vectors()["artifact_chain"]
    canary = compile_trace(chain["trace"])
    protocol = replay_canary(canary, seed=chain["replay_seed"])["replay_protocol"]
    baseline = replay_protocol_sha256(protocol)

    protocol["max_replay_events"] += 1
    assert replay_protocol_sha256(protocol) == baseline
    protocol["seed"] += 1
    assert replay_protocol_sha256(protocol) != baseline
