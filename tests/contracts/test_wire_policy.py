from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from commcanary.compiler import compile_trace
from commcanary.schema import (
    SchemaError,
    canary_artifact_provenance_sha256,
    canonical_json_bytes,
    load_json,
    validate_trace,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "contracts"
SCHEMA_DIR = ROOT / "schemas"


def _load_json_fixture(name: str) -> Any:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _trace_schema_validator() -> Draft202012Validator:
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in SCHEMA_DIR.glob("*.schema.json")]
    registry = Registry().with_resources((document["$id"], Resource.from_contents(document)) for document in documents)
    trace_schema = next(
        document for document in documents if document["$id"].endswith("commcanary.trace.v1.schema.json")
    )
    return Draft202012Validator(trace_schema, registry=registry)


@pytest.mark.parametrize("case_index", [0, 1, 2], ids=["root-duplicate", "nested-duplicate", "nan"])
def test_literal_noncanonical_json_payloads_are_rejected_during_load(tmp_path: Path, case_index: int) -> None:
    vector = _load_json_fixture("wire_policy_vectors.v1.json")["parse_cases"][case_index]
    path = tmp_path / f"{vector['name']}.json"
    path.write_text(vector["payload"], encoding="utf-8")

    with pytest.raises(SchemaError, match=vector["error_fragment"]):
        load_json(str(path))


def test_parse_serialize_round_trip_is_value_and_byte_idempotent(tmp_path: Path) -> None:
    original = _load_json_fixture("trace.valid.json")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_json(str(first_path), original)
    first = load_json(str(first_path))
    write_json(str(second_path), first)
    second = load_json(str(second_path))

    assert first == original
    assert second == original
    assert first_path.read_bytes() == second_path.read_bytes()
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_runtime_numeric_coercion_does_not_mutate_and_is_not_portable_schema_shape() -> None:
    vectors = _load_json_fixture("wire_policy_vectors.v1.json")
    trace = copy.deepcopy(vectors["canonical_numeric_trace"])
    before = copy.deepcopy(trace)

    validate_trace(trace)

    assert trace == before
    assert list(_trace_schema_validator().iter_errors(trace))
    assert vectors["policy"]["numeric_coercion"] == (
        "runtime_validator_accepts_without_mutating_but_portable_schema_rejects"
    )


def test_unknown_fields_and_extension_named_objects_are_accepted_but_not_given_special_semantics() -> None:
    vectors = _load_json_fixture("wire_policy_vectors.v1.json")
    extended = copy.deepcopy(vectors["unknown_and_extension_trace"])
    baseline = copy.deepcopy(extended)
    baseline.pop("extensions")
    baseline["events"][0].pop("commcanary_future_hint")
    baseline["workload"].pop("extensions")
    validate_trace(extended)
    assert list(_trace_schema_validator().iter_errors(extended)) == []

    baseline_canary = compile_trace(baseline)
    extended_canary = compile_trace(extended)
    assert extended_canary["workload"]["extensions"] == extended["workload"]["extensions"]
    assert extended_canary["compiler"]["source_trace_sha256"] != baseline_canary["compiler"]["source_trace_sha256"]
    assert (
        extended_canary["compiler"]["execution_semantic_sha256"]
        == baseline_canary["compiler"]["execution_semantic_sha256"]
    )
    assert "extensions" not in extended_canary

    before_provenance = canary_artifact_provenance_sha256(extended_canary)
    extended_canary["future_top_level_field"] = {"opaque": True}
    assert canary_artifact_provenance_sha256(extended_canary) != before_provenance
    assert vectors["policy"]["extensions"] == "ordinary_unknown_fields_with_no_reserved_semantics"


def test_unsupported_version_loads_as_json_then_fails_exact_artifact_validator(tmp_path: Path) -> None:
    vectors: Mapping[str, Any] = _load_json_fixture("wire_policy_vectors.v1.json")
    path = tmp_path / "unsupported.trace.json"
    path.write_text(json.dumps(vectors["unsupported_trace"]), encoding="utf-8")
    loaded = load_json(str(path))

    assert loaded["format"] == "commcanary.trace.v2"
    with pytest.raises(SchemaError, match="trace format must be 'commcanary.trace.v1'"):
        validate_trace(loaded)
    assert list(_trace_schema_validator().iter_errors(loaded))
    assert vectors["policy"]["unsupported_format"] == "load_as_json_then_reject_in_artifact_validator"
