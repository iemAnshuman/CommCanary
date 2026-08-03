from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from experiments.rostam.analysis.schemas import (
    DECISION_FIDELITY_POLICY_SCHEMA,
    DECISION_FIDELITY_POLICY_SCHEMA_V2,
    validate_schema_documents,
)
from experiments.rostam.harness import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = ROOT / "experiments" / "rostam" / "policies" / "decision-fidelity-gate-v1.json"
SCHEMA_PATH = ROOT / "experiments" / "rostam" / "schemas" / "decision-fidelity-policy-v1.schema.json"
POLICY_V2_PATH = ROOT / "experiments" / "rostam" / "policies" / "decision-fidelity-gate-v2.json"
SCHEMA_V2_PATH = ROOT / "experiments" / "rostam" / "schemas" / "decision-fidelity-policy-v2.schema.json"


def test_decision_fidelity_policy_is_schema_valid_and_self_bound() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(policy)
    identity_projection = dict(policy)
    policy_id = identity_projection.pop("policy_id")

    assert policy_id == canonical_sha256(identity_projection)
    assert policy["scope"]["configuration_ids"] == sorted(policy["scope"]["configuration_ids"])
    assert policy["kill_or_reframe"]["condition"] == ("exact_work_pairwise_agreement_not_greater_than_stratified")


def test_decision_policy_schema_is_in_the_committed_analysis_inventory() -> None:
    documents = validate_schema_documents()

    assert DECISION_FIDELITY_POLICY_SCHEMA in {document["schema"] for document in documents}
    assert DECISION_FIDELITY_POLICY_SCHEMA_V2 in {document["schema"] for document in documents}


def test_replicated_decision_fidelity_policy_is_schema_valid_and_self_bound() -> None:
    policy = json.loads(POLICY_V2_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_V2_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(policy)
    identity_projection = dict(policy)
    policy_id = identity_projection.pop("policy_id")

    assert policy_id == canonical_sha256(identity_projection)
    assert policy["measurement"]["configuration_repetitions"] == 8
    assert policy["measurement"]["measured_repetitions"] == 24
    assert len(policy["measurement"]["representation_schedule"]["rows"]) == 6
    assert policy["claim_boundary"]["reduced_canary_claim"] == "not_evaluated"
