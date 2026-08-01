from __future__ import annotations

import copy
import json

import pytest

from experiments.rostam import decision_gate_physical
from experiments.rostam.lib.physical_results import (
    DECISION_GATE_MEASUREMENT_SCHEMA,
    DECISION_GATE_PRODUCER_SCHEMA,
    PhysicalResultError,
    adapt_physical_measurement,
)

REQUEST_ID = "1" * 64
MATERIALIZATION_ID = "2" * 64
PROGRAM_SHA256 = "3" * 64
POLICY_ID = "4" * 64


def _parameters() -> dict:
    return {
        "adapter": "torch-json",
        "command": ["python", "-m", "experiments.rostam.decision_gate_physical"],
        "operation": "all_reduce",
        "world_size": 4,
        "global_ranks": [0, 1, 2, 3],
        "iterations": 2,
        "warmup": 1,
        "expected_request_id": REQUEST_ID,
        "expected_materialization_id": MATERIALIZATION_ID,
        "expected_program_sha256": PROGRAM_SHA256,
        "expected_policy_id": POLICY_ID,
        "expected_source_event_count": 8,
        "expected_stratified_source_event_indices": [0, 1],
        "expected_correctness_checks_per_rank": [2, 2, 2, 2],
    }


def _runtime() -> dict:
    return {
        "hostname": "toranj1",
        "job_id": "180001",
        "python_version": "3.12.3",
        "torch_version": "2.4.1",
        "torch_cuda_version": "12.1",
        "runtime_nccl_version_code": 22005,
    }


def _payload() -> dict:
    gathered = [
        {
            "rank": rank,
            "timings_us": {
                representation: [float(100 + rank), float(200 + rank)]
                for representation in decision_gate_physical.REPRESENTATION_IDS
            },
        }
        for rank in range(4)
    ]
    return decision_gate_physical.result_payload(
        request={"format": "commcanary.qualification_request.v2", "request_id": REQUEST_ID},
        materialization_id=MATERIALIZATION_ID,
        program_sha256=PROGRAM_SHA256,
        policy={"format": "commcanary.qualification_policy.v1", "policy_id": POLICY_ID},
        world_size=4,
        iterations=2,
        warmup=1,
        source_event_count=8,
        selected_indices=(0, 1),
        gathered=gathered,
        correctness_checks_per_rank=(2, 2, 2, 2),
        runtime={
            "torch_version": "2.4.1",
            "torch_cuda_version": "12.1",
            "runtime_nccl_version_code": 22005,
            "distributed_backend": "nccl",
        },
    )


def _adapt(payload: dict) -> dict:
    return adapt_physical_measurement(
        measurement_schema=DECISION_GATE_MEASUREMENT_SCHEMA,
        producer_schema=DECISION_GATE_PRODUCER_SCHEMA,
        attempt_id="a-000001",
        parameters=_parameters(),
        stdout=json.dumps(payload),
        stderr="",
        wall_time_s=3.5,
        runtime=_runtime(),
    )


def test_decision_gate_adapter_recomputes_every_representation() -> None:
    measurement = _adapt(_payload())

    assert measurement["samples_us"] == [103.0, 203.0]
    assert measurement["value_us"] == 153.0
    assert measurement["representations"]["exact_work"]["metrics"]["median_us"] == 153.0
    assert measurement["execution"]["stratified_source_event_indices"] == [0, 1]
    assert measurement["correctness_check_count"] == 8
    assert measurement["decision_claims"]["physical_decision_fidelity"] == "not_analyzed"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["representations"]["exact_work"]["timings_us"].__setitem__(0, 999.0), "max-rank"),
        (lambda value: value["execution"].__setitem__("stratified_source_event_indices", [1]), "selection"),
        (lambda value: value["request"].__setitem__("request_id", "f" * 64), "frozen workload"),
        (lambda value: value["claims"].__setitem__("physical_decision_fidelity", "pass"), "claims"),
    ),
)
def test_decision_gate_adapter_refuses_tampered_claims_and_measurements(mutation, message: str) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)

    with pytest.raises(PhysicalResultError, match=message):
        _adapt(payload)
