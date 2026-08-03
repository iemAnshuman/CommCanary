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
)
from experiments.rostam.analysis.decision_fidelity_v2 import _simultaneous_intervals
from experiments.rostam.analysis.pipeline import ANALYSIS_SCHEMA
from experiments.rostam.analysis.schemas import PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2
from experiments.rostam.evaluate_decision_gate import _verdict_summary
from experiments.rostam.harness import canonical_json_bytes, canonical_sha256
from experiments.rostam.lib.executor_artifact import EXECUTOR_ARTIFACT_INPUT_ID, prepare_executor_artifact

ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = ROOT / "experiments" / "rostam" / "policies" / "decision-fidelity-gate-v2.json"
VERDICT_SCHEMA_PATH = ROOT / "experiments" / "rostam" / "schemas" / "decision-fidelity-verdict-v2.schema.json"

REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")


def _environment(block: int, configuration_index: int) -> Dict[str, Any]:
    return {
        "schema": "commcanary.rostam.runtime-observation.v2",
        "driver_version": "550.54.15",
        "nccl_library_sha256": "9" * 64,
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "NVIDIA A100-PCIE-40GB",
                "driver_version": "550.54.15",
                "pci_bus_id": f"00000000:0{index + 1}:00.0",
                "persistence_mode": "Enabled",
                "performance_state": "P0",
                "temperature_c": 50 + index,
                "power_draw_w": 120.0 + index,
                "power_limit_w": 250.0,
                "sm_clock_mhz": 1410,
                "memory_clock_mhz": 1215,
            }
            for index in range(4)
        ],
        "topology": {"method": "nvidia-smi topo -m", "text": "four-GPU topology"},
        "node_state": {
            "method": "scontrol show node --oneliner HOSTNAME",
            "text": "NodeName=toranj State=ALLOCATED",
        },
        "binding": {
            "environment": {},
            "cpu_affinity": [0, 1, 2, 3],
            "cpu_affinity_method": "sched_getaffinity",
        },
        "observation_sha256": canonical_sha256({"allocation_block": block, "configuration_index": configuration_index}),
    }


def _policy() -> Tuple[Dict[str, Any], bytes]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["measurement"]["allocation_blocks"] = 5
    policy["comparison"]["uncertainty"]["resamples"] = 100
    projection = dict(policy)
    projection.pop("policy_id")
    policy["policy_id"] = canonical_sha256(projection)
    data = canonical_json_bytes(policy)
    return policy, data


def _aggregate(policy: Dict[str, Any], policy_bytes: bytes) -> Dict[str, Any]:
    configurations = policy["scope"]["configuration_ids"]
    blocks = policy["measurement"]["allocation_blocks"]
    rows = []
    for block in range(blocks):
        for index, configuration in enumerate(configurations):
            source = float((index + 1) * 100)
            medians = {
                "source": source,
                "exact_work": source * 1.01,
                "stratified": float((len(configurations) - index) * 100),
                "isolated": 500.0,
                "no_overlap": source * 1.2,
                "no_rank_skew": source * 0.9,
            }
            rows.append(
                {
                    "source_run_id": "decision-gate-v2-test",
                    "workload_id": policy["scope"]["workload_id"],
                    "configuration_id": configuration,
                    "repetition": block,
                    "measurement_schema": PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
                    "decision_gate": {
                        "execution": {
                            "allocation_block": block,
                            "iterations": 24,
                            "warmup": policy["measurement"]["warmup"],
                            "order_method": policy["measurement"]["order_method"],
                            "timing_semantics": policy["measurement"]["timing_semantics"],
                        },
                        "request": {"format": "commcanary.qualification_request.v2", "request_id": "a" * 64},
                        "materialization": {"materialization_id": "b" * 64, "program_sha256": "c" * 64},
                        "policy": {"format": "commcanary.qualification_policy.v1", "policy_id": "d" * 64},
                        "representations": {
                            representation: {"timings_us": [medians[representation]] * 24}
                            for representation in REPRESENTATIONS
                        },
                    },
                    "decision_gate_runtime": {
                        "hostname": f"toranj{block % 2}.example",
                        "job_id": f"job-{block:02d}-{index:02d}",
                    },
                    "decision_gate_environment": _environment(block, index),
                }
            )
    campaign = {
        "run_id": "decision-gate-v2-test",
        "manifest_sha256": "1" * 64,
        "selection_sha256": "2" * 64,
        "verdict_sha256": "3" * 64,
        "inputs": [
            {
                "id": "decision-fidelity-policy",
                "sha256": hashlib.sha256(policy_bytes).hexdigest(),
                "size_bytes": len(policy_bytes),
            }
        ],
    }
    return {
        "schema": ANALYSIS_SCHEMA,
        "completeness": {"complete": True, "issue_codes": []},
        "provenance": {
            "campaigns": [campaign],
            "trusted_join_sha256": canonical_sha256([campaign]),
        },
        "selected_cells": rows,
    }


def _evaluate() -> Tuple[Dict[str, Any], Dict[str, Any], bytes, Dict[str, Any]]:
    policy, policy_bytes = _policy()
    aggregate = _aggregate(policy, policy_bytes)
    return evaluate_decision_fidelity(aggregate, policy_bytes), policy, policy_bytes, aggregate


def test_v2_evaluator_uses_complete_blocks_and_simultaneous_intervals() -> None:
    verdict, _policy_value, _policy_bytes, _aggregate_value = _evaluate()
    schema = json.loads(VERDICT_SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(verdict)
    identity_projection = dict(verdict)
    verdict_id = identity_projection.pop("verdict_id")

    assert verdict["outcome"] == "pass"
    assert verdict["evidence"]["allocation_block_count"] == 5
    assert verdict["evidence"]["distinct_job_count"] == 40
    assert verdict["evidence"]["environment_observation_count"] == 40
    assert verdict["uncertainty"]["method"].endswith("standardized-max.v2")
    assert set(verdict["uncertainty"]["metric_intervals"]["exact_work"]) == {
        "pairwise_ranking_agreement",
        "kendall_tau_b",
        "false_negative_count",
        "false_positive_count",
        "median_absolute_relative_error_pct",
        "p95_absolute_relative_error_pct",
        "median_execution_time_ratio_to_source",
    }
    assert all(row["status"] == "pass" for row in verdict["criteria"])
    assert verdict["product_interpretation"]["mode"] == "exact_qualification_capsule"
    assert verdict["product_interpretation"]["reduced_canary_claim"] == "not_evaluated"
    assert verdict_id == canonical_sha256(identity_projection)
    assert _verdict_summary(verdict, Path("verdict.json")) == {
        "mode": "exact_qualification_capsule",
        "outcome": "pass",
        "output": "verdict.json",
        "verdict_id": verdict["verdict_id"],
    }


def test_v2_pair_margin_recomputes_the_relative_tie_band() -> None:
    verdict, policy, policy_bytes, aggregate = _evaluate()
    first_configuration = policy["scope"]["configuration_ids"][0]
    second_configuration = policy["scope"]["configuration_ids"][1]
    for row in aggregate["selected_cells"]:
        if row["configuration_id"] == first_configuration:
            value = 1000.0
        elif row["configuration_id"] == second_configuration:
            value = 1040.0
        else:
            continue
        row["decision_gate"]["representations"]["source"]["timings_us"] = [value] * 24
        row["decision_gate"]["representations"]["exact_work"]["timings_us"] = [value] * 24

    verdict = evaluate_decision_fidelity(aggregate, policy_bytes)
    pair = verdict["pairwise_comparisons"][0]

    assert pair["representations"]["source"]["observed_label"] == "tie"
    assert pair["representations"]["source"]["tie_threshold_us"] == 50.0
    assert pair["representations"]["source"]["policy_margin_us"] == 10.0
    assert pair["representations"]["source"]["simultaneous_margin_interval_us"] == [10.0, 10.0]


def test_v2_constant_statistics_keep_exact_intervals_when_timings_are_noisy() -> None:
    _verdict, policy, policy_bytes, aggregate = _evaluate()
    configuration_indices = {
        configuration: index for index, configuration in enumerate(policy["scope"]["configuration_ids"])
    }
    for row in aggregate["selected_cells"]:
        index = configuration_indices[row["configuration_id"]]
        block = row["repetition"]
        source = float((index + 1) * 100)
        source_timings = [
            source * (1.0 + ((iteration + block + index) % 5 - 2) / 1000.0)
            for iteration in range(24)
        ]
        exact_timings = [
            value * (1.01 + ((iteration + 2 * block + index) % 7 - 3) / 2000.0)
            for iteration, value in enumerate(source_timings)
        ]
        row["decision_gate"]["representations"]["source"]["timings_us"] = source_timings
        row["decision_gate"]["representations"]["exact_work"]["timings_us"] = exact_timings

    verdict = evaluate_decision_fidelity(aggregate, policy_bytes)
    intervals = verdict["uncertainty"]["metric_intervals"]["exact_work"]

    assert verdict["uncertainty"]["standardized_max_critical_value"] > 0.0
    assert intervals["pairwise_ranking_agreement"] == [1.0, 1.0]
    assert intervals["kendall_tau_b"] == [1.0, 1.0]
    assert intervals["false_negative_count"] == [0.0, 0.0]
    assert intervals["false_positive_count"] == [0.0, 0.0]
    assert verdict["outcome"] == "pass"


def test_v2_zero_variance_bootstrap_must_match_the_observation() -> None:
    with pytest.raises(DecisionFidelityError, match="zero variance but disagrees"):
        _simultaneous_intervals(
            {"metric|exact_work|pairwise_ranking_agreement": 1.0},
            [{"metric|exact_work|pairwise_ranking_agreement": 0.5}] * 100,
            confidence=0.95,
            pair_count=28,
        )


def test_v2_evaluator_rejects_reused_allocation_job_ids() -> None:
    _verdict, policy, policy_bytes, aggregate = _evaluate()
    aggregate["selected_cells"][1]["decision_gate_runtime"]["job_id"] = aggregate["selected_cells"][0][
        "decision_gate_runtime"
    ]["job_id"]

    verdict = evaluate_decision_fidelity(aggregate, policy_bytes)

    assert verdict["outcome"] == "incomparable"
    assert [issue["code"] for issue in verdict["issues"]] == ["allocation_job_reuse"]
    assert verdict["uncertainty"]["status"] == "not_evaluated"
    assert verdict["uncertainty"]["standardized_max_critical_value"] is None


def test_v2_evaluator_requires_every_configuration_in_every_block() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    aggregate["selected_cells"].pop()

    verdict = evaluate_decision_fidelity(aggregate, policy_bytes)

    assert verdict["outcome"] == "incomparable"
    assert [issue["code"] for issue in verdict["issues"]] == ["incomplete_allocation_block_inventory"]


def test_v2_evaluator_requires_bound_environment_evidence() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    aggregate["selected_cells"][0].pop("decision_gate_environment")

    with pytest.raises(DecisionFidelityError, match="decision_gate_environment"):
        evaluate_decision_fidelity(aggregate, policy_bytes)


def test_v2_policy_validator_refuses_silent_method_substitution() -> None:
    policy, _ = _policy()
    substituted = copy.deepcopy(policy)
    substituted["comparison"]["uncertainty"]["method"] = "independent-percentile-bootstrap-median-difference"
    projection = dict(substituted)
    projection.pop("policy_id")
    substituted["policy_id"] = canonical_sha256(projection)

    with pytest.raises(DecisionFidelityError, match="uncertainty semantics"):
        validate_decision_fidelity_policy(substituted)


def test_v2_verdict_requires_and_records_the_aggregate_frozen_analyzer(tmp_path: Path) -> None:
    policy, policy_bytes = _policy()
    aggregate = _aggregate(policy, policy_bytes)
    artifact = prepare_executor_artifact(ROOT / "experiments" / "rostam", tmp_path / "executor-artifacts")
    campaign = aggregate["provenance"]["campaigns"][0]
    campaign["inputs"].append(
        {
            "id": EXECUTOR_ARTIFACT_INPUT_ID,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
    )
    aggregate["provenance"]["trusted_join_sha256"] = canonical_sha256([campaign])
    aggregate["provenance"]["analysis_implementation"] = artifact.analyzer_record(
        "experiments.rostam.analyze:main"
    )

    with pytest.raises(DecisionFidelityError, match="frozen evaluator artifact"):
        evaluate_decision_fidelity(aggregate, policy_bytes)

    verdict = evaluate_decision_fidelity(
        aggregate,
        policy_bytes,
        executor_artifact=artifact,
    )

    assert verdict["analyzer"] == artifact.analyzer_record(
        "experiments.rostam.evaluate_decision_gate:main",
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
    )
