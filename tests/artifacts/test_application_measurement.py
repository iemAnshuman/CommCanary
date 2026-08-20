from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Mapping, Sequence

import pytest

from commcanary.artifacts.application_measurement import (
    application_regression_decision,
    validate_application_measurement,
)
from commcanary.artifacts.application_oracle import (
    application_evidence_comparable,
    application_evidence_regression_decision,
    build_application_evidence_set,
    build_application_oracle,
    validate_application_evidence_set,
    validate_application_oracle,
)
from commcanary.artifacts.json_codec import canonical_json_bytes
from commcanary.artifacts.physical_execution import telemetry_assessment
from commcanary.errors import SchemaError
from commcanary.formats import APPLICATION_MEASUREMENT_FORMAT


def _percentile(values: Sequence[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.99
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _measurement(
    *,
    subject: str,
    perturbation: str,
    latencies: Sequence[float],
    output_throughputs: Sequence[float],
) -> Dict[str, Any]:
    test_nccl_algorithm = f"fixture-{subject[:8]}"
    model_files = [{"path": "config.json", "bytes": 12, "sha256": hashlib.sha256(b"model").hexdigest()}]
    samples = [
        {
            "batch_latency_ms": latency,
            "output_token_throughput_per_second": throughput,
            "total_token_throughput_per_second": throughput + 100.0,
            "observed_gpu_memory_used_bytes": 1_000_000,
        }
        for latency, throughput in zip(latencies, output_throughputs)
    ]
    environment = {
        "hostname": "toranj0",
        "platform": "Linux",
        "python": "3.12.3",
        "torch": "2.11.0",
        "cuda": "13.0",
        "nccl": [2, 28, 9],
        "visible_gpu_count": 4,
        "slurm_job_id": "1",
        "slurm_node": "toranj0",
        "nccl_configuration": {
            "NCCL_ALGO": test_nccl_algorithm,
            "NCCL_PROTO": None,
            "NCCL_NET": None,
            "NCCL_P2P_DISABLE": None,
            "NCCL_SHM_DISABLE": None,
        },
        "runtime_configuration": {"CUDA_LAUNCH_BLOCKING": None},
    }
    subject_configuration = {
        "runner_oci_digest": f"sha256:{hashlib.sha256(b'runner').hexdigest()}",
        "application_engine": "vllm",
        "application_engine_version": "0.26.0",
        "engine_configuration": {
            "kind": "vllm.v1",
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
            "gpu_memory_utilization": 0.75,
            "kv_cache_memory_bytes": 536870912,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 4,
        },
        "perturbation_environment": {
            **environment["nccl_configuration"],
            **environment["runtime_configuration"],
        },
    }
    telemetry = [
        _telemetry("before_warmup", 1),
        _telemetry("before_measured_cycle_1", 2),
        *(_telemetry(f"after_measured_cycle_{index + 1}", index + 3) for index in range(len(samples))),
        _telemetry("final", len(samples) + 3),
    ]
    document: Dict[str, Any] = {
        "format": APPLICATION_MEASUREMENT_FORMAT,
        "runner": {
            "oci_digest": f"sha256:{hashlib.sha256(b'runner').hexdigest()}",
            "execution_protocol": "vllm-offline-tensor-parallel.v1",
        },
        "subject_sha256": hashlib.sha256(canonical_json_bytes(subject_configuration)).hexdigest(),
        "subject_configuration": subject_configuration,
        "perturbation_id": perturbation,
        "application": {
            "name": "vllm-dummy-llama-tp-dense-decoder",
            "engine": "vllm",
            "engine_version": "0.26.0",
            "ground_truth_kind": "application_ground_truth",
            "model_sha256": hashlib.sha256(canonical_json_bytes(model_files)).hexdigest(),
            "model_files": model_files,
        },
        "workload": {
            "batch_size": 4,
            "input_length": 128,
            "output_length": 8,
            "tensor_parallel_size": 4,
            "warmups": 1,
            "iterations": len(samples),
            "seed": 314159,
            "load_format": "dummy",
            "dtype": "bfloat16",
            "prefix_caching": False,
            "worker_multiprocessing_method": "spawn",
        },
        "samples": samples,
        "summary": {
            "median_batch_latency_ms": sum(latencies) / len(latencies),
            "p99_batch_latency_ms": _percentile(latencies),
            "median_output_token_throughput_per_second": sum(output_throughputs) / len(output_throughputs),
            "median_total_token_throughput_per_second": sum(
                sample["total_token_throughput_per_second"] for sample in samples
            )
            / len(samples),
            "max_observed_gpu_memory_used_bytes": 1_000_000,
        },
        "environment": environment,
        "environment_sha256": hashlib.sha256(canonical_json_bytes(environment)).hexdigest(),
        "profile": {
            "captured": False,
            "excluded_from_timed_samples": True,
            "artifact_sha256": None,
            "files": [],
        },
        "telemetry": telemetry,
        "telemetry_assessment": telemetry_assessment(telemetry, world_size=4),
    }
    document["measurement_id"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    return document


def _telemetry(phase: str, timestamp: int) -> Dict[str, Any]:
    lines: list[str] = []
    return {
        "phase": phase,
        "monotonic_ns": timestamp,
        "gpu_observation": {
            "status": "complete",
            "command": ["nvidia-smi"],
            "returncode": 0,
            "gpus": [
                {
                    "index": rank,
                    "uuid": f"GPU-{rank}",
                    "pstate": "P0",
                    "sm_clock_mhz": 1200.0,
                    "memory_clock_mhz": 1215.0,
                    "temperature_c": 40.0,
                    "power_w": 100.0,
                    "clock_event_reasons_active": "0x0",
                    "corrected_volatile_ecc": 0,
                    "uncorrected_volatile_ecc": 0,
                }
                for rank in range(4)
            ],
            "stderr": "",
        },
        "xid_observation": {
            "status": "complete",
            "command": ["journalctl"],
            "lines": lines,
            "sha256": hashlib.sha256(canonical_json_bytes(lines)).hexdigest(),
            "stderr": "",
        },
        "slurm_node_observation": {
            "status": "complete",
            "command": ["scontrol"],
            "node": "toranj0",
            "state": "ALLOCATED",
            "stderr": "",
        },
    }


def _allocations(measurement: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for repetition in range(8):
        replica = copy.deepcopy(measurement)
        allocation_id = str(1000 + repetition)
        node = f"toranj{repetition % 2}"
        replica["environment"]["slurm_job_id"] = allocation_id
        replica["environment"]["slurm_node"] = node
        for telemetry in replica["telemetry"]:
            telemetry["slurm_node_observation"]["node"] = node
        replica["environment_sha256"] = hashlib.sha256(canonical_json_bytes(replica["environment"])).hexdigest()
        replica["measurement_id"] = hashlib.sha256(
            canonical_json_bytes({key: value for key, value in replica.items() if key != "measurement_id"})
        ).hexdigest()
        day = "05" if repetition < 4 else "06"
        rows.append(
            {
                "repetition": repetition,
                "planned_position": 7 - repetition,
                "chunk_id": f"application-{repetition}",
                "allocation_id": allocation_id,
                "node": node,
                "start_time": f"2026-08-{day}T10:00:00+00:00",
                "end_time": f"2026-08-{day}T10:01:00+00:00",
                "elapsed_seconds": 60,
                "measurement": replica,
            }
        )
    return rows


def test_application_measurement_and_real_oracle_decision_recompute() -> None:
    baseline = _measurement(
        subject=hashlib.sha256(b"baseline").hexdigest(),
        perturbation="baseline",
        latencies=[100.0, 102.0],
        output_throughputs=[1000.0, 1000.0],
    )
    candidate = _measurement(
        subject=hashlib.sha256(b"candidate").hexdigest(),
        perturbation="heldout-nccl",
        latencies=[110.0, 112.0],
        output_throughputs=[900.0, 900.0],
    )

    validate_application_measurement(baseline)
    decision = application_regression_decision(baseline, candidate, threshold_pct=5.0)

    assert decision["decision"] == "fail"
    assert decision["severe"] is True
    assert decision["regression_magnitude_pct"] > 9.0


def test_replicated_application_oracle_preserves_allocation_evidence() -> None:
    measurement = _measurement(
        subject=hashlib.sha256(b"baseline").hexdigest(),
        perturbation="baseline",
        latencies=[100.0, 102.0],
        output_throughputs=[1000.0, 1000.0],
    )

    oracle = build_application_oracle(_allocations(measurement))

    validate_application_oracle(oracle)
    assert application_evidence_comparable(oracle) is True
    assert oracle["allocation_assessment"] == {
        "comparable": True,
        "issues": [],
        "allocation_count": 8,
        "distinct_node_count": 2,
        "distinct_day_count": 2,
        "maximum_allocation_elapsed_seconds": 60,
    }
    assert len(oracle["allocations"]) == 8


def test_replicated_application_oracle_decides_from_aggregate_samples() -> None:
    baseline = build_application_oracle(
        _allocations(
            _measurement(
                subject=hashlib.sha256(b"baseline").hexdigest(),
                perturbation="baseline",
                latencies=[100.0, 102.0],
                output_throughputs=[1000.0, 1000.0],
            )
        )
    )
    candidate = build_application_oracle(
        _allocations(
            _measurement(
                subject=hashlib.sha256(b"candidate").hexdigest(),
                perturbation="candidate",
                latencies=[110.0, 112.0],
                output_throughputs=[900.0, 900.0],
            )
        )
    )

    decision = application_evidence_regression_decision(baseline, candidate, threshold_pct=5.0)

    assert decision["decision"] == "fail"
    assert decision["severe"] is True


def test_application_evidence_set_binds_training_and_unopened_holdout_bytes() -> None:
    baseline = _measurement(
        subject=hashlib.sha256(b"baseline").hexdigest(),
        perturbation="baseline",
        latencies=[100.0, 102.0],
        output_throughputs=[1000.0, 1000.0],
    )
    training = _measurement(
        subject=hashlib.sha256(b"training").hexdigest(),
        perturbation="training-regression",
        latencies=[108.0, 109.0],
        output_throughputs=[920.0, 920.0],
    )
    holdout = _measurement(
        subject=hashlib.sha256(b"holdout").hexdigest(),
        perturbation="holdout-regression",
        latencies=[112.0, 113.0],
        output_throughputs=[880.0, 880.0],
    )

    evidence_set = build_application_evidence_set(
        training_baseline=baseline,
        training_perturbations={"training-regression": training},
        holdout_baseline=copy.deepcopy(baseline),
        holdout_perturbations={"holdout-regression": holdout},
    )

    validate_application_evidence_set(evidence_set)
    assert evidence_set["splits"]["training"]["perturbations"][0]["perturbation_id"] == ("training-regression")
    changed = copy.deepcopy(evidence_set)
    changed["splits"]["holdout"]["perturbations"][0]["evidence"]["summary"]["median_batch_latency_ms"] += 1.0
    with pytest.raises(SchemaError, match="evidence_set_id"):
        validate_application_evidence_set(changed)


def test_replicated_application_oracle_rejects_forged_position_balance() -> None:
    measurement = _measurement(
        subject=hashlib.sha256(b"baseline").hexdigest(),
        perturbation="baseline",
        latencies=[100.0, 102.0],
        output_throughputs=[1000.0, 1000.0],
    )
    oracle = build_application_oracle(_allocations(measurement))
    changed = copy.deepcopy(oracle)
    changed["allocations"][0]["planned_position"] = changed["allocations"][1]["planned_position"]
    changed["environment_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    "repetition": row["repetition"],
                    "planned_position": row["planned_position"],
                    "allocation_id": row["allocation_id"],
                    "node": row["node"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "environment_sha256": row["measurement"]["environment_sha256"],
                }
                for row in changed["allocations"]
            ]
        )
    ).hexdigest()
    changed["oracle_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in changed.items() if key != "oracle_id"})
    ).hexdigest()

    with pytest.raises(SchemaError, match="planned position"):
        validate_application_oracle(changed)


def test_application_measurement_rejects_self_declared_summary() -> None:
    document = _measurement(
        subject=hashlib.sha256(b"baseline").hexdigest(),
        perturbation="baseline",
        latencies=[100.0, 102.0],
        output_throughputs=[1000.0, 1000.0],
    )
    changed = copy.deepcopy(document)
    changed["summary"]["p99_batch_latency_ms"] = 1.0
    changed["measurement_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in changed.items() if key != "measurement_id"})
    ).hexdigest()

    with pytest.raises(SchemaError, match="p99_batch_latency_ms does not recompute"):
        validate_application_measurement(changed)


def test_application_oracle_can_compare_engine_versions_of_the_same_workload() -> None:
    baseline = _measurement(
        subject=hashlib.sha256(b"vllm-0.25").hexdigest(),
        perturbation="historical-baseline",
        latencies=[100.0, 100.0],
        output_throughputs=[1000.0, 1000.0],
    )
    candidate = _measurement(
        subject=hashlib.sha256(b"vllm-0.26").hexdigest(),
        perturbation="historical-candidate",
        latencies=[106.0, 106.0],
        output_throughputs=[940.0, 940.0],
    )
    baseline["application"]["engine_version"] = "0.25.0"
    baseline["subject_configuration"]["application_engine_version"] = "0.25.0"
    baseline["subject_sha256"] = hashlib.sha256(canonical_json_bytes(baseline["subject_configuration"])).hexdigest()
    baseline["measurement_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in baseline.items() if key != "measurement_id"})
    ).hexdigest()

    decision = application_regression_decision(baseline, candidate, threshold_pct=5.0)

    assert decision["decision"] == "fail"


def test_application_measurement_supports_an_independent_sglang_engine() -> None:
    document = _measurement(
        subject=hashlib.sha256(b"sglang").hexdigest(),
        perturbation="sglang-baseline",
        latencies=[100.0, 102.0],
        output_throughputs=[1000.0, 1000.0],
    )
    document["runner"]["execution_protocol"] = "sglang-offline-tensor-parallel.v1"
    document["application"].update(
        {
            "name": "sglang-dummy-llama-tp-dense-decoder",
            "engine": "sglang",
            "engine_version": "0.5.12",
        }
    )
    document["subject_configuration"] = {
        "runner_oci_digest": document["runner"]["oci_digest"],
        "application_engine": "sglang",
        "application_engine_version": "0.5.12",
        "engine_configuration": {
            "kind": "sglang.v1",
            "disable_cuda_graph": False,
            "disable_custom_all_reduce": True,
            "disable_overlap_schedule": False,
            "mem_fraction_static": 0.75,
            "max_running_requests": 4,
            "max_total_tokens": 1024,
            "chunked_prefill_size": 512,
            "max_prefill_tokens": 512,
        },
        "perturbation_environment": {
            **document["environment"]["nccl_configuration"],
            **document["environment"]["runtime_configuration"],
        },
    }
    document["subject_sha256"] = hashlib.sha256(canonical_json_bytes(document["subject_configuration"])).hexdigest()
    document["measurement_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in document.items() if key != "measurement_id"})
    ).hexdigest()

    validate_application_measurement(document)


def test_application_measurement_rejects_engine_protocol_mismatch() -> None:
    document = _measurement(
        subject=hashlib.sha256(b"mismatch").hexdigest(),
        perturbation="mismatch",
        latencies=[100.0, 102.0],
        output_throughputs=[1000.0, 1000.0],
    )
    document["runner"]["execution_protocol"] = "sglang-offline-tensor-parallel.v1"
    document["measurement_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in document.items() if key != "measurement_id"})
    ).hexdigest()

    with pytest.raises(SchemaError, match="engine does not match"):
        validate_application_measurement(document)
