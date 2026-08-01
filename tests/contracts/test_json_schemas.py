from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from commcanary.artifacts import (
    validate_qualification_materialization,
    validate_qualification_observation,
    validate_qualification_policy,
    validate_qualification_request,
    validate_qualification_verdict,
)
from commcanary.compare import compare_reports
from commcanary.compiler import (
    compile_trace,
    synthesize_behavioral_canary,
    validate_behavior_search_evidence,
    verify_canary_behavior,
    verify_canary_fidelity,
)
from commcanary.replay import replay_canary, verify_report_against_canary
from commcanary.schema import (
    SchemaError,
    validate_canary,
    validate_comparison,
    validate_report,
    validate_trace,
)
from commcanary.services import prepare_qualification_request
from commcanary.workflows import materialize_qualification
from tests.builders import qualification_policy, qualification_trace, small_trace

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schemas"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "contracts"

SCHEMA_FILES = {
    "trace": "commcanary.trace.v1.schema.json",
    "canary": "commcanary.canary.v2.schema.json",
    "report": "commcanary.report.v2.schema.json",
    "comparison": "commcanary.compare.v2.schema.json",
    "fidelity_verification": "commcanary.fidelity_verification.v1.schema.json",
    "behavior_verification": "commcanary.behavior_verification.v1.schema.json",
    "report_verification": "commcanary.report_verification.v1.schema.json",
    "qualification_request_legacy": "commcanary.qualification_request.v1.schema.json",
    "qualification_request": "commcanary.qualification_request.v2.schema.json",
    "qualification_materialization": "commcanary.qualification_materialization.v1.schema.json",
    "qualification_policy": "commcanary.qualification_policy.v1.schema.json",
    "qualification_observation": "commcanary.qualification_observation.v1.schema.json",
    "qualification_verdict": "commcanary.qualification_verdict.v1.schema.json",
}

RUNTIME_VALIDATORS: Dict[str, Callable[[Mapping[str, Any]], None]] = {
    "trace": validate_trace,
    "canary": validate_canary,
    "report": validate_report,
    "comparison": validate_comparison,
    "qualification_request_legacy": validate_qualification_request,
    "qualification_request": validate_qualification_request,
    "qualification_materialization": validate_qualification_materialization,
    "qualification_policy": validate_qualification_policy,
    "qualification_observation": validate_qualification_observation,
    "qualification_verdict": validate_qualification_verdict,
}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _schema_validators() -> Dict[str, Draft202012Validator]:
    documents = {path.name: _load_json(path) for path in SCHEMA_DIR.glob("*.schema.json")}
    registry = Registry().with_resources(
        (document["$id"], Resource.from_contents(document)) for document in documents.values()
    )
    return {
        artifact: Draft202012Validator(documents[filename], registry=registry)
        for artifact, filename in SCHEMA_FILES.items()
    }


def _apply_mutation(document: Any, mutation: Mapping[str, Any]) -> Any:
    mutated = copy.deepcopy(document)
    target = mutated
    path = mutation["path"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = mutation["value"]
    return mutated


def _schema_errors(validator: Draft202012Validator, document: Any) -> list[str]:
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    return [error.message for error in errors]


@pytest.fixture(scope="module")
def schema_validators() -> Dict[str, Draft202012Validator]:
    return _schema_validators()


@pytest.fixture(scope="module")
def fixture_cases() -> Mapping[str, Any]:
    return _load_json(FIXTURE_DIR / "cases.json")


@pytest.mark.parametrize("schema_path", sorted(SCHEMA_DIR.glob("*.schema.json")), ids=lambda path: path.name)
def test_published_schemas_are_valid_draft_2020_12_documents(schema_path: Path) -> None:
    Draft202012Validator.check_schema(_load_json(schema_path))


@pytest.mark.parametrize("artifact", SCHEMA_FILES)
def test_valid_fixtures_match_portable_schema(
    artifact: str,
    schema_validators: Mapping[str, Draft202012Validator],
) -> None:
    document = _load_json(FIXTURE_DIR / f"{artifact}.valid.json")

    assert _schema_errors(schema_validators[artifact], document) == []


@pytest.mark.parametrize("artifact", SCHEMA_FILES)
def test_invalid_shape_fixtures_fail_portable_schema(
    artifact: str,
    schema_validators: Mapping[str, Draft202012Validator],
    fixture_cases: Mapping[str, Any],
) -> None:
    document = _load_json(FIXTURE_DIR / f"{artifact}.valid.json")
    invalid = _apply_mutation(document, fixture_cases[artifact]["invalid"])

    assert _schema_errors(schema_validators[artifact], invalid)


@pytest.mark.parametrize("artifact", SCHEMA_FILES)
def test_tampered_fixtures_remain_shape_valid(
    artifact: str,
    schema_validators: Mapping[str, Draft202012Validator],
    fixture_cases: Mapping[str, Any],
) -> None:
    document = _load_json(FIXTURE_DIR / f"{artifact}.valid.json")
    tampered = _apply_mutation(document, fixture_cases[artifact]["tampered"])

    assert _schema_errors(schema_validators[artifact], tampered) == []


@pytest.mark.parametrize("artifact", RUNTIME_VALIDATORS)
def test_runtime_validators_accept_valid_fixtures(artifact: str) -> None:
    document = _load_json(FIXTURE_DIR / f"{artifact}.valid.json")

    RUNTIME_VALIDATORS[artifact](document)


@pytest.mark.parametrize("artifact", RUNTIME_VALIDATORS)
@pytest.mark.parametrize("state", ["invalid", "tampered"])
def test_runtime_validators_reject_invalid_and_tampered_fixtures(
    artifact: str,
    state: str,
    fixture_cases: Mapping[str, Any],
) -> None:
    document = _load_json(FIXTURE_DIR / f"{artifact}.valid.json")
    changed = _apply_mutation(document, fixture_cases[artifact][state])

    with pytest.raises(SchemaError):
        RUNTIME_VALIDATORS[artifact](changed)


def test_current_producers_match_published_schemas(
    schema_validators: Mapping[str, Draft202012Validator],
    tmp_path: Path,
) -> None:
    trace = qualification_trace()
    canary = compile_trace(trace)
    report = replay_canary(canary)
    qualification_request = prepare_qualification_request(
        str(tmp_path / "qualification"),
        trace,
        canary,
        qualification_policy(),
    )
    qualification_materialization = materialize_qualification(
        str(tmp_path / "qualification"),
        str(tmp_path / "materialization"),
    )
    produced = {
        "trace": trace,
        "canary": canary,
        "report": report,
        "comparison": compare_reports(report, report),
        "fidelity_verification": verify_canary_fidelity(trace, canary),
        "behavior_verification": verify_canary_behavior(
            trace,
            canary,
            configurations=[
                {"name": "fixture-a"},
                {"name": "fixture-b", "latency_floor_us": 8.0},
            ],
        ),
        "report_verification": verify_report_against_canary(report, canary),
        "qualification_request": qualification_request,
        "qualification_materialization": qualification_materialization,
        "qualification_policy": _load_json(FIXTURE_DIR / "qualification_policy.valid.json"),
        "qualification_observation": _load_json(FIXTURE_DIR / "qualification_observation.valid.json"),
        "qualification_verdict": _load_json(FIXTURE_DIR / "qualification_verdict.valid.json"),
    }

    failures = {
        artifact: _schema_errors(schema_validators[artifact], document)
        for artifact, document in produced.items()
        if _schema_errors(schema_validators[artifact], document)
    }
    assert failures == {}


def test_behavior_search_producer_matches_experimental_sidecar_schema() -> None:
    trace = small_trace()
    evidence: Dict[str, Any] = {}
    canary = synthesize_behavioral_canary(
        trace,
        min_timing_sample_limit=2,
        max_timing_sample_limit=4,
        evidence_output=evidence,
    )
    validate_behavior_search_evidence(evidence, canary)

    documents = {path.name: _load_json(path) for path in SCHEMA_DIR.glob("*.schema.json")}
    registry = Registry().with_resources(
        (document["$id"], Resource.from_contents(document)) for document in documents.values()
    )
    validator = Draft202012Validator(
        documents["commcanary.behavior_search_evidence.experimental.v1.schema.json"],
        registry=registry,
    )

    assert _schema_errors(validator, evidence) == []
