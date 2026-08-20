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
from experiments.rostam.analysis.decision_fidelity_v2 import (
    _bootstrap_vector,
    _outcome_for_states,
    _simultaneous_intervals,
)
from experiments.rostam.analysis.pipeline import ANALYSIS_SCHEMA
from experiments.rostam.analysis.schemas import PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2
from experiments.rostam.decision_gate_schedule import frozen_schedule_inventory
from experiments.rostam.evaluate_decision_gate import _verdict_summary
from experiments.rostam.harness import canonical_json_bytes, canonical_sha256
from experiments.rostam.lib.executor_artifact import EXECUTOR_ARTIFACT_INPUT_ID, prepare_executor_artifact

ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = ROOT / "experiments" / "rostam" / "policies" / "decision-fidelity-gate-v2.json"
VERDICT_SCHEMA_PATH = ROOT / "experiments" / "rostam" / "schemas" / "decision-fidelity-verdict-v2.schema.json"

REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")


class _FixedRandom:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def randrange(self, stop: int) -> int:
        value = next(self._values)
        assert 0 <= value < stop
        return value


def _environment(repetition: int, configuration_index: int) -> Dict[str, Any]:
    invariants = {
        "driver_version": "550.54.15",
        "nccl_library_sha256": "9" * 64,
        "gpu_count": 4,
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "NVIDIA A100-PCIE-40GB",
                "driver_version": "550.54.15",
                "pci_bus_id": f"00000000:0{index + 1}:00.0",
                "persistence_mode": "Enabled",
                "power_limit_w": 250.0,
            }
            for index in range(4)
        ],
        "topology": {"method": "nvidia-smi topo -m", "text": "four-GPU topology"},
        "binding": {
            "environment": {
                "CUDA_VISIBLE_DEVICES": None,
                "OMP_NUM_THREADS": None,
                "SLURM_CPUS_PER_TASK": None,
                "SLURM_JOB_GPUS": None,
                "SLURM_LOCALID": None,
                "SLURM_NODEID": None,
                "SLURM_PROCID": None,
                "SLURM_STEP_GPUS": None,
            },
            "cpu_affinity": [0, 1, 2, 3],
            "cpu_affinity_method": "sched_getaffinity",
        },
    }
    platform = {key: value for key, value in invariants.items() if key != "nccl_library_sha256"}

    def telemetry(phase: str, offset: int) -> Dict[str, Any]:
        return {
            "captured_at": f"2026-08-04T00:{repetition:02d}:{configuration_index:02d}.{offset:06d}Z",
            "gpus": [
                {
                    "index": index,
                    "performance_state": "P0",
                    "temperature_c": 50 + index + offset,
                    "power_draw_w": 120.0 + index,
                    "sm_clock_mhz": 1410,
                    "memory_clock_mhz": 1215,
                }
                for index in range(4)
            ],
            "node_state": {
                "method": "scontrol show node --oneliner HOSTNAME",
                "text": f"NodeName=toranj State=ALLOCATED Phase={phase}",
            },
        }

    result = {
        "schema": "commcanary.rostam.runtime-observation.v3",
        "invariants": invariants,
        "telemetry": {
            "method": "bounded-pre-post-nvidia-smi.v1",
            "pre": telemetry("pre", 0),
            "post": telemetry("post", 2),
        },
        "probe_policy": {"timeout_seconds": 10, "max_output_bytes_per_stream": 65_536},
        "platform_sha256": canonical_sha256(platform),
        "observation_sha256": "",
    }
    result["observation_sha256"] = canonical_sha256(
        {
            "invariants": result["invariants"],
            "telemetry": result["telemetry"],
            "probe_policy": result["probe_policy"],
        }
    )
    return result


def _cycle_telemetry(node: str) -> Dict[str, Any]:
    labels = [
        "before_warmup",
        "before_measured_cycle_1",
        "after_measured_cycle_1",
        "after_measured_cycle_2",
        "after_measured_cycle_3",
        "after_measured_cycle_4",
        "final",
    ]
    return {
        "schema": "commcanary.rostam.decision-gate-cycle-telemetry.v1",
        "method": "bounded-between-six-row-cycles.v1",
        "snapshots": [
            {
                "label": label,
                "captured_at": f"2026-08-04T00:00:{snapshot:02d}.000000Z",
                "gpus": [
                    {
                        "index": gpu,
                        "uuid": f"GPU-{gpu}",
                        "performance_state": "P0",
                        "temperature_c": 50 + gpu,
                        "power_draw_w": 120.0 + gpu,
                        "sm_clock_mhz": 1410,
                        "memory_clock_mhz": 1215,
                        "throttle_reasons_active": "0x0000000000000000",
                        "ecc_corrected_volatile_total": 0,
                        "ecc_uncorrected_volatile_total": 0,
                    }
                    for gpu in range(4)
                ],
                "node_state": {
                    "method": "scontrol show node --oneliner HOSTNAME",
                    "node": node,
                    "state": "ALLOCATED",
                },
                "xid": {
                    "method": "journalctl --dmesg --boot --no-pager --grep NVRM.*Xid",
                    "event_count": 0,
                    "window_sha256": "0" * 64,
                },
            }
            for snapshot, label in enumerate(labels)
        ],
    }


def _policy() -> Tuple[Dict[str, Any], bytes]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["measurement"]["configuration_repetitions"] = 5
    policy["measurement"]["configuration_order_by_repetition"] = policy["measurement"][
        "configuration_order_by_repetition"
    ][:5]
    policy["measurement"]["representation_schedule"] = frozen_schedule_inventory(
        configuration_repetitions=5,
        iterations=24,
    )
    policy["comparison"]["uncertainty"]["resamples"] = 100
    projection = dict(policy)
    projection.pop("policy_id")
    policy["policy_id"] = canonical_sha256(projection)
    data = canonical_json_bytes(policy)
    return policy, data


def _aggregate(policy: Dict[str, Any], policy_bytes: bytes) -> Dict[str, Any]:
    configurations = policy["scope"]["configuration_ids"]
    repetitions = policy["measurement"]["configuration_repetitions"]
    schedule = policy["measurement"]["representation_schedule"]
    rows = []
    for repetition in range(repetitions):
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
                    "repetition": repetition,
                    "measurement_schema": PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
                    "decision_gate": {
                        "execution": {
                            "configuration_repetition": repetition,
                            "iterations": 24,
                            "warmup": policy["measurement"]["warmup"],
                            "order_method": policy["measurement"]["order_method"],
                            "representation_order_by_iteration": [
                                schedule["rows"][row_index]
                                for row_index in schedule["row_index_by_configuration_repetition"][repetition]
                            ],
                            "timing_semantics": policy["measurement"]["timing_semantics"],
                        },
                        "request": {"format": "commcanary.qualification_request.v2", "request_id": "a" * 64},
                        "materialization": {"materialization_id": "b" * 64, "program_sha256": "c" * 64},
                        "policy": {"format": "commcanary.qualification_policy.v1", "policy_id": "d" * 64},
                        "representations": {
                            representation: {"timings_us": [medians[representation]] * 24}
                            for representation in REPRESENTATIONS
                        },
                        "telemetry_checkpoints": _cycle_telemetry(f"toranj{repetition % 2}"),
                    },
                    "decision_gate_runtime": {
                        "hostname": f"toranj{repetition % 2}.example",
                        "job_id": f"job-{repetition:02d}-{index:02d}",
                    },
                    "decision_gate_environment": _environment(repetition, index),
                    "decision_gate_scheduler": {
                        "schema": "commcanary.rostam.scheduler-evidence.v1",
                        "method": "scontrol show job --oneliner JOBID",
                        "planned_position": policy["measurement"]["configuration_order_by_repetition"][
                            repetition
                        ].index(configuration),
                        "scheduler_start_time": (
                            "2026-08-04T01:"
                            f"{repetition:02d}:"
                            f"{policy['measurement']['configuration_order_by_repetition'][repetition].index(configuration):02d}"
                        ),
                        "node": f"toranj{repetition % 2}",
                        "elapsed_from_repetition_start_seconds": float(
                            policy["measurement"]["configuration_order_by_repetition"][repetition].index(configuration)
                        ),
                        "chunk_identifier": f"p-{repetition:024x}",
                    },
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


def test_v2_evaluator_uses_independent_repetitions_and_simultaneous_intervals() -> None:
    verdict, _policy_value, _policy_bytes, _aggregate_value = _evaluate()
    schema = json.loads(VERDICT_SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(verdict)
    identity_projection = dict(verdict)
    verdict_id = identity_projection.pop("verdict_id")

    assert verdict["outcome"] == "pass"
    assert verdict["evidence"]["configuration_repetition_count"] == 5
    assert verdict["evidence"]["distinct_job_count"] == 40
    assert verdict["evidence"]["environment_observation_count"] == 40
    assert verdict["evidence"]["cycle_telemetry_summary"] == {
        "cell_count": 40,
        "snapshots_per_cell": [7],
        "minimum_temperature_c": 50,
        "maximum_temperature_c": 53,
        "minimum_power_draw_w": 120.0,
        "maximum_power_draw_w": 123.0,
        "minimum_sm_clock_mhz": 1410,
        "maximum_sm_clock_mhz": 1410,
        "minimum_memory_clock_mhz": 1215,
        "maximum_memory_clock_mhz": 1215,
        "performance_states": ["P0"],
        "maximum_ecc_corrected_delta": 0,
        "maximum_ecc_uncorrected_delta": 0,
        "maximum_xid_event_delta": 0,
    }
    assert verdict["uncertainty"]["method"].endswith("standardized-max.v3")
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
    assert verdict["product_interpretation"]["mode"] == "exact_materialization_conformance_control"
    assert verdict["product_interpretation"]["reduced_canary_claim"] == "not_evaluated"
    assert verdict_id == canonical_sha256(identity_projection)
    assert _verdict_summary(verdict, Path("verdict.json")) == {
        "mode": "exact_materialization_conformance_control",
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
        repetition = row["repetition"]
        source = float((index + 1) * 100)
        source_timings = [
            source * (1.0 + ((iteration + repetition + index) % 5 - 2) / 1000.0) for iteration in range(24)
        ]
        exact_timings = [
            value * (1.01 + ((iteration + 2 * repetition + index) % 7 - 3) / 2000.0)
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


@pytest.mark.parametrize(
    ("issues", "statuses", "unstable", "pairs", "expected"),
    (
        ([], ["fail", "inconclusive"], False, False, "fail"),
        ([], ["fail", "pass"], True, False, "fail"),
        ([{"code": "missing"}], ["fail"], False, False, "incomparable"),
        ([], ["pass", "pass"], False, False, "pass"),
    ),
)
def test_v2_outcome_precedence_is_policy_frozen(issues, statuses, unstable, pairs, expected) -> None:
    assert (
        _outcome_for_states(
            issues=issues,
            criterion_statuses=statuses,
            unstable=unstable,
            inconclusive_pairs=pairs,
        )
        == expected
    )


def test_v2_bootstrap_resamples_repetitions_independently_and_keeps_williams_cycles() -> None:
    samples = {
        repetition: {
            configuration: {representation: (base,) * 6 + (base + 100.0,) * 6 for representation in REPRESENTATIONS}
            for configuration, base in (
                ("configuration-a", 10.0 + 10.0 * repetition),
                ("configuration-b", 30.0 + 10.0 * repetition),
            )
        }
        for repetition in (0, 1)
    }
    rng = _FixedRandom(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
        ]
    )

    vector = _bootstrap_vector(
        samples,
        configuration_repetitions=(0, 1),
        configurations=("configuration-a", "configuration-b"),
        measured_repetitions=12,
        rng=rng,  # type: ignore[arg-type]
    )

    assert vector["configuration-a"]["source"] == 10.0
    assert vector["configuration-b"]["source"] == 40.0


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


def test_v2_evaluator_requires_every_configuration_repetition() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    aggregate["selected_cells"].pop()

    verdict = evaluate_decision_fidelity(aggregate, policy_bytes)

    assert verdict["outcome"] == "incomparable"
    assert [issue["code"] for issue in verdict["issues"]] == ["incomplete_configuration_repetition_inventory"]


def test_v2_evaluator_requires_bound_environment_evidence() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    aggregate["selected_cells"][0].pop("decision_gate_environment")

    with pytest.raises(DecisionFidelityError, match="decision_gate_environment"):
        evaluate_decision_fidelity(aggregate, policy_bytes)


def test_v2_evaluator_rejects_platform_drift_between_repetitions() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    environment = aggregate["selected_cells"][0]["decision_gate_environment"]
    environment["invariants"]["gpus"][0]["uuid"] = "GPU-replaced"
    for snapshot in aggregate["selected_cells"][0]["decision_gate"]["telemetry_checkpoints"]["snapshots"]:
        snapshot["gpus"][0]["uuid"] = "GPU-replaced"
    platform = {key: value for key, value in environment["invariants"].items() if key != "nccl_library_sha256"}
    environment["platform_sha256"] = canonical_sha256(platform)
    environment["observation_sha256"] = canonical_sha256(
        {
            "invariants": environment["invariants"],
            "telemetry": environment["telemetry"],
            "probe_policy": environment["probe_policy"],
        }
    )

    verdict = evaluate_decision_fidelity(aggregate, policy_bytes)

    assert verdict["outcome"] == "incomparable"
    assert [issue["code"] for issue in verdict["issues"]] == ["environment_invariant_mismatch"]
    assert verdict["evidence"]["platform_sha256"] is None


def test_v2_evaluator_enforces_predeclared_telemetry_ranges() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    environment = aggregate["selected_cells"][0]["decision_gate_environment"]
    environment["telemetry"]["post"]["gpus"][0]["temperature_c"] = 96

    with pytest.raises(DecisionFidelityError, match="temperature_c"):
        evaluate_decision_fidelity(aggregate, policy_bytes)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda snapshot: snapshot["gpus"][0].__setitem__("sm_clock_mhz", 100), "sm_clock_mhz"),
        (
            lambda snapshot: snapshot["gpus"][0].__setitem__("throttle_reasons_active", "0x0000000000000001"),
            "identity or state",
        ),
        (lambda snapshot: snapshot["gpus"][0].__setitem__("ecc_corrected_volatile_total", 1), "ECC or Xid"),
        (lambda snapshot: snapshot["node_state"].__setitem__("state", "DRAIN"), "node_state"),
    ),
)
def test_v2_evaluator_enforces_between_cycle_telemetry(mutation, message: str) -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    final = aggregate["selected_cells"][0]["decision_gate"]["telemetry_checkpoints"]["snapshots"][-1]
    mutation(final)

    with pytest.raises(DecisionFidelityError, match=message):
        evaluate_decision_fidelity(aggregate, policy_bytes)


def test_v2_evaluator_rejects_xid_events_between_cycles() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    final = aggregate["selected_cells"][0]["decision_gate"]["telemetry_checkpoints"]["snapshots"][-1]
    final["xid"]["event_count"] = 1
    final["xid"]["window_sha256"] = "1" * 64

    with pytest.raises(DecisionFidelityError, match="ECC or Xid"):
        evaluate_decision_fidelity(aggregate, policy_bytes)


def test_v2_evaluator_recomputes_environment_observation_identity() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    original = copy.deepcopy(aggregate["selected_cells"][0]["decision_gate_environment"])
    for index, row in enumerate(aggregate["selected_cells"]):
        row["decision_gate_environment"] = copy.deepcopy(original)
        row["decision_gate_environment"]["observation_sha256"] = f"{index:064x}"

    with pytest.raises(DecisionFidelityError, match="observation_sha256 does not recompute"):
        evaluate_decision_fidelity(aggregate, policy_bytes)


def test_v2_evaluator_marks_over_window_repetition_incomparable() -> None:
    _verdict, _policy_value, policy_bytes, aggregate = _evaluate()
    target = next(
        row
        for row in aggregate["selected_cells"]
        if row["repetition"] == 0 and row["decision_gate_scheduler"]["planned_position"] == 7
    )
    target["decision_gate_scheduler"]["scheduler_start_time"] = "2026-08-04T03:00:00"
    target["decision_gate_scheduler"]["elapsed_from_repetition_start_seconds"] = 7200.0

    verdict = evaluate_decision_fidelity(aggregate, policy_bytes)

    assert verdict["outcome"] == "incomparable"
    assert "repetition_time_window_exceeded" in [issue["code"] for issue in verdict["issues"]]


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
    aggregate["provenance"]["analysis_implementation"] = artifact.analyzer_record("experiments.rostam.analyze:main")

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
