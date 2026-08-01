from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import jsonschema
import pytest

from experiments.rostam.analysis.decision_fidelity import (
    DecisionFidelityError,
    evaluate_decision_fidelity,
    validate_decision_fidelity_policy,
    write_decision_fidelity_verdict,
)
from experiments.rostam.analysis.pipeline import ANALYSIS_SCHEMA
from experiments.rostam.analysis.schemas import PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA
from experiments.rostam.harness import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = ROOT / "experiments" / "rostam" / "policies" / "decision-fidelity-gate-v1.json"
VERDICT_SCHEMA_PATH = ROOT / "experiments" / "rostam" / "schemas" / "decision-fidelity-verdict-v1.schema.json"

REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")


def _policy() -> Tuple[Dict[str, Any], bytes]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["comparison"]["uncertainty"]["resamples"] = 100
    projection = dict(policy)
    projection.pop("policy_id")
    policy["policy_id"] = canonical_sha256(projection)
    data = canonical_json_bytes(policy)
    return policy, data


def _aggregate(
    policy: Dict[str, Any],
    policy_sha256: str,
    policy_size: int,
    *,
    stratified_matches_source: bool = False,
    complete: bool = True,
) -> Dict[str, Any]:
    configurations = policy["scope"]["configuration_ids"]
    rows = []
    for index, configuration in enumerate(configurations):
        source = float((index + 1) * 100)
        stratified = source if stratified_matches_source else float((len(configurations) - index) * 100)
        medians = {
            "source": source,
            "exact_work": source * 1.01,
            "stratified": stratified,
            "isolated": 500.0,
            "no_overlap": source * 1.2,
            "no_rank_skew": source * 0.9,
        }
        rows.append(
            {
                "source_run_id": "decision-gate-test",
                "workload_id": "decision-gate",
                "configuration_id": configuration,
                "repetition": 0,
                "measurement_schema": PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA,
                "decision_gate": {
                    "execution": {
                        "iterations": policy["measurement"]["measured_repetitions"],
                        "warmup": policy["measurement"]["warmup"],
                        "order_method": policy["measurement"]["order_method"],
                        "timing_semantics": policy["measurement"]["timing_semantics"],
                    },
                    "request": {"format": "commcanary.qualification_request.v2", "request_id": "a" * 64},
                    "materialization": {"materialization_id": "b" * 64, "program_sha256": "c" * 64},
                    "policy": {"format": "commcanary.qualification_policy.v1", "policy_id": "d" * 64},
                    "representations": {
                        representation: {
                            "timings_us": [medians[representation]] * policy["measurement"]["measured_repetitions"]
                        }
                        for representation in REPRESENTATIONS
                    },
                },
                "decision_gate_runtime": {"hostname": "toranj1.example"},
            }
        )
    campaign = {
        "run_id": "decision-gate-test",
        "manifest_sha256": "1" * 64,
        "selection_sha256": "2" * 64,
        "verdict_sha256": "3" * 64,
        "inputs": [
            {
                "id": "decision-fidelity-policy",
                "sha256": policy_sha256,
                "size_bytes": policy_size,
            }
        ],
    }
    selected = rows if complete else []
    return {
        "schema": ANALYSIS_SCHEMA,
        "completeness": {
            "complete": complete,
            "issue_codes": [] if complete else ["missing_selection"],
        },
        "provenance": {
            "campaigns": [campaign],
            "trusted_join_sha256": canonical_sha256([campaign]),
        },
        "selected_cells": selected,
    }


def _evaluate(*, stratified_matches_source: bool = False, complete: bool = True) -> Dict[str, Any]:
    policy, policy_bytes = _policy()
    return evaluate_decision_fidelity(
        _aggregate(
            policy,
            hashlib.sha256(policy_bytes).hexdigest(),
            len(policy_bytes),
            stratified_matches_source=stratified_matches_source,
            complete=complete,
        ),
        policy_bytes,
    )


def test_decision_fidelity_evaluator_passes_a_decisive_superior_candidate() -> None:
    verdict = _evaluate()
    schema = json.loads(VERDICT_SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(verdict)
    identity_projection = dict(verdict)
    verdict_id = identity_projection.pop("verdict_id")

    assert verdict["outcome"] == "pass"
    assert verdict["representation_metrics"]["exact_work"]["pairwise_ranking_agreement"] == 1.0
    assert verdict["product_interpretation"]["kill_or_reframe_evaluated"] is True
    assert verdict["product_interpretation"]["kill_or_reframe_triggered"] is False
    assert verdict_id == canonical_sha256(identity_projection)


def test_decision_fidelity_evaluator_applies_predeclared_reframe_condition() -> None:
    verdict = _evaluate(stratified_matches_source=True)

    assert verdict["outcome"] == "fail"
    assert verdict["product_interpretation"] == {
        "kill_or_reframe_evaluated": True,
        "kill_or_reframe_triggered": True,
        "positioning": "evidence_and_research_framework",
        "cost_claim": "not_evaluated_by_this_reduced-source-campaign",
        "generality_claim": "not_evaluated_beyond_the_declared_supported_domain",
        "independent_operator_claim": "not_evaluated_by_this_campaign",
    }


def test_incomplete_inventory_is_incomparable_and_cannot_trigger_reframe() -> None:
    verdict = _evaluate(complete=False)

    assert verdict["outcome"] == "incomparable"
    assert [issue["code"] for issue in verdict["issues"]] == [
        "incomplete_evidence",
        "incomplete_configuration_inventory",
    ]
    assert verdict["product_interpretation"]["kill_or_reframe_evaluated"] is False
    assert verdict["product_interpretation"]["kill_or_reframe_triggered"] is False


def test_policy_and_trusted_join_bindings_fail_closed() -> None:
    policy, policy_bytes = _policy()
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    policy_size = len(policy_bytes)
    aggregate = _aggregate(policy, policy_sha256, policy_size)
    aggregate["provenance"]["trusted_join_sha256"] = "f" * 64

    with pytest.raises(DecisionFidelityError, match="trusted join"):
        evaluate_decision_fidelity(
            aggregate,
            policy_bytes,
        )

    aggregate = _aggregate(policy, policy_sha256, policy_size)
    aggregate["provenance"]["campaigns"][0]["inputs"][0]["sha256"] = "e" * 64
    aggregate["provenance"]["trusted_join_sha256"] = canonical_sha256(aggregate["provenance"]["campaigns"])
    with pytest.raises(DecisionFidelityError, match="not bound"):
        evaluate_decision_fidelity(
            aggregate,
            policy_bytes,
        )


def test_policy_document_cannot_be_substituted_behind_a_bound_digest() -> None:
    policy, policy_bytes = _policy()
    aggregate = _aggregate(
        policy,
        hashlib.sha256(policy_bytes).hexdigest(),
        len(policy_bytes),
    )
    substituted = copy.deepcopy(policy)
    substituted["pass_criteria"]["exact_work_min_pairwise_agreement"] = 0.0
    projection = dict(substituted)
    projection.pop("policy_id")
    substituted["policy_id"] = canonical_sha256(projection)

    with pytest.raises(DecisionFidelityError, match="not bound"):
        evaluate_decision_fidelity(aggregate, canonical_json_bytes(substituted))


def test_policy_validator_rejects_unimplemented_declared_semantics() -> None:
    policy, _ = _policy()
    policy["comparison"]["classification_error_counting"]["opposite_direction"] = "false_negative"
    projection = dict(policy)
    projection.pop("policy_id")
    policy["policy_id"] = canonical_sha256(projection)

    with pytest.raises(DecisionFidelityError, match="classification semantics"):
        validate_decision_fidelity_policy(policy)


def test_verdict_publication_is_exclusive(tmp_path: Path) -> None:
    verdict = _evaluate()
    destination = tmp_path / "decision-fidelity-verdict.json"

    write_decision_fidelity_verdict(destination, verdict)

    assert destination.read_bytes() == canonical_json_bytes(verdict)
    with pytest.raises(DecisionFidelityError, match="already exists"):
        write_decision_fidelity_verdict(destination, verdict)
