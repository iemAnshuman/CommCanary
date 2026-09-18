from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest

from commcanary.adapters.chakra_capture import commcanary_trace_to_chakra
from commcanary.artifacts import physical_execution as physical_execution_module
from commcanary.artifacts.application_measurement import validate_application_measurement
from commcanary.artifacts.chakra import decode_chakra_execution_trace
from commcanary.artifacts.json_codec import canonical_json_bytes
from commcanary.artifacts.physical_canary import SUPPORTED_PHYSICAL_DOMAIN, with_content_identity
from commcanary.errors import SchemaError
from commcanary.formats import (
    APPLICATION_MEASUREMENT_FORMAT,
    PHYSICAL_CANARY_POLICY_FORMAT,
    PHYSICAL_EXECUTION_MEASUREMENT_FORMAT,
    TRACE_FORMAT,
)
from commcanary.services.active_physical_synthesis import (
    PhysicalCandidateRequest,
    synthesize_active_physical_canary,
    validate_active_physical_study_ledger,
)
from commcanary.workflows.physical_canary import (
    PHYSICAL_CANARY_ACTIVE_LEDGER,
    PHYSICAL_CANARY_APPLICATION_EVIDENCE,
    PHYSICAL_CANARY_PHYSICAL_EVIDENCE,
    build_physical_canary_bundle,
    verify_physical_canary_bundle,
)


def _captured_trace() -> Dict[str, Any]:
    events = []
    for index, tensor_bytes in enumerate((32, 64, 96, 256)):
        recipe = {
            "op": "gemm",
            "dtype": "bfloat16",
            "m": 2 + index,
            "n": 4,
            "k": 3,
            "source_kernel_count": 1,
            "source_kernel_duration_us": 2.0,
        }
        events.append(
            {
                "id": f"collective-{index}",
                "op": "all_reduce",
                "dtype": "bfloat16",
                "bytes": tensor_bytes,
                "ranks": [0, 1, 2, 3],
                "group": "default",
                "start_us": 10.0 + index * 10.0,
                "phase": "decode",
                "reduction_op": "sum",
                "compute_overlap_us": 2.0,
                "rank_arrival_us": {"0": 0.0, "1": 0.5, "2": 1.0, "3": 1.5},
                "compute_recipe_by_rank": {str(rank): [copy.deepcopy(recipe)] for rank in range(4)},
            }
        )
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "active-vllm"},
        "system": {"source_format": "pytorch-kineto"},
        "events": events,
    }


def _policy(runner_digest: str) -> Dict[str, Any]:
    return with_content_identity(
        {
            "format": PHYSICAL_CANARY_POLICY_FORMAT,
            "supported_domain": SUPPORTED_PHYSICAL_DOMAIN,
            "runtime_budget_seconds": 1.0,
            "severe_regression_threshold_pct": 5.0,
            "required_feature_tags": ["rare-tail"],
            "required_baselines": ["communication_only_microbenchmark"],
            "runner": {
                "oci_digest": runner_digest,
                "execution_protocol": "chakra-et-collective-graph.v1",
                "world_size": 4,
            },
            "qualification_gates": {
                "minimum_runtime_reduction_ratio": 10.0,
                "maximum_severe_false_negatives": 0,
                "maximum_false_positive_rate": 0.1,
                "minimum_pairwise_decision_agreement": 1.0,
            },
            "gate_metrics": [
                {
                    "name": "physical_runtime_seconds",
                    "direction": "lower_is_better",
                    "regression_threshold_pct": 5.0,
                    "mandatory": True,
                }
            ],
            "minimum_samples": 5,
            "uncertainty": {
                "method": "percentile_bootstrap_median_difference",
                "confidence": 0.95,
                "bootstrap_resamples": 1000,
                "seed": 0,
            },
            "noise": {"max_relative_iqr_pct": 20.0},
            "privacy": {
                "maximum_leakage_score": 1.0,
                "private_exchange_requires_reviewed_opaque_attributes": True,
            },
        },
        "policy_id",
    )


def _application(
    runner_digest: str,
    *,
    perturbation_id: str,
    subject: str,
    latency_ms: float,
    throughput: float,
) -> Dict[str, Any]:
    test_nccl_algorithm = f"fixture-{subject[:8]}"
    model_files = [{"path": "config.json", "bytes": 12, "sha256": hashlib.sha256(b"model").hexdigest()}]
    samples = [
        {
            "batch_latency_ms": latency_ms,
            "output_token_throughput_per_second": throughput,
            "total_token_throughput_per_second": throughput + 100.0,
            "observed_gpu_memory_used_bytes": 2_000_000_000,
        }
        for _ in range(2)
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
        "runner_oci_digest": runner_digest,
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
        _telemetry("after_measured_cycle_1", 3),
        _telemetry("after_measured_cycle_2", 4),
        _telemetry("final", 5),
    ]
    document: Dict[str, Any] = {
        "format": APPLICATION_MEASUREMENT_FORMAT,
        "runner": {"oci_digest": runner_digest, "execution_protocol": "vllm-offline-tensor-parallel.v1"},
        "subject_sha256": hashlib.sha256(canonical_json_bytes(subject_configuration)).hexdigest(),
        "subject_configuration": subject_configuration,
        "perturbation_id": perturbation_id,
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
            "iterations": 2,
            "seed": 314159,
            "load_format": "dummy",
            "dtype": "bfloat16",
            "prefix_caching": False,
            "worker_multiprocessing_method": "spawn",
        },
        "samples": samples,
        "summary": {
            "median_batch_latency_ms": latency_ms,
            "p99_batch_latency_ms": latency_ms,
            "median_output_token_throughput_per_second": throughput,
            "median_total_token_throughput_per_second": throughput + 100.0,
            "max_observed_gpu_memory_used_bytes": 2_000_000_000,
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
        "telemetry_assessment": physical_execution_module.telemetry_assessment(telemetry, world_size=4),
    }
    document["measurement_id"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    validate_application_measurement(document)
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


def _physical_measurement(
    request: PhysicalCandidateRequest,
    *,
    runner_digest: str,
    projection: Mapping[str, Any],
) -> Dict[str, Any]:
    region_count = len(request.selected_region_ids)
    baseline_runtime = region_count * 0.01
    sensitive = "event-000001" in request.selected_region_ids
    regressed = request.perturbation_id in {"train-regression", "holdout-regression"} and sensitive
    runtime = baseline_runtime * (1.1 if regressed else 1.0)
    telemetry = [
        _telemetry("before_correctness", 1),
        _telemetry("before_measured_cycle_1", 2),
        _telemetry("after_measured_cycle_1", 3),
        _telemetry("after_measured_cycle_2", 4),
        _telemetry("final", 5),
    ]
    environment = {
        "hostname": "toranj0",
        "platform": "Linux",
        "python": "3.12.3",
        "torch": "2.11.0",
        "cuda": "13.0",
        "nccl": [2, 28, 9],
        "world_size": 4,
        "slurm_job_id": "1",
        "slurm_node": "toranj0",
        "nccl_configuration": {
            "NCCL_ALGO": None,
            "NCCL_PROTO": None,
            "NCCL_NET": None,
            "NCCL_P2P_DISABLE": None,
            "NCCL_SHM_DISABLE": None,
        },
    }
    nodes = {int(row["node_id"]): row for row in projection["nodes"]}
    collectives = sum(nodes[node_id]["operation"] == "all_reduce" for node_id in request.selected_node_ids)
    flops = sum(int(nodes[node_id]["executed_flops"]) for node_id in request.selected_node_ids)
    document: Dict[str, Any] = {
        "format": PHYSICAL_EXECUTION_MEASUREMENT_FORMAT,
        "role": request.role,
        "runner": {"oci_digest": runner_digest, "execution_protocol": "chakra-et-collective-graph.v1"},
        "subject_sha256": request.subject_sha256,
        "perturbation_id": request.perturbation_id,
        "source_et_sha256": projection["source_et"]["sha256"],
        "executable_sha256": request.executable_sha256,
        "projection_id": projection["projection_id"],
        "selected_region_ids": list(request.selected_region_ids),
        "selected_node_ids": list(request.selected_node_ids),
        "execution": {
            "world_size": 4,
            "warmups": 1,
            "iterations": 2,
            "disable_overlap": False,
            "communication_only": request.method == "communication_only_microbenchmark",
            "rank_skew_us": 0.0,
            "rank_skew_mechanism": "host_issue_monotonic_spin",
            "collective_buffer_slots": 4,
            "workspace_bytes_max_rank": 1024,
        },
        "correctness": [
            {
                "rank": rank,
                "passed": True,
                "validated_node_ids": list(reversed(request.selected_node_ids)),
                "commitment_sha256": hashlib.sha256(
                    f"{request.candidate_id}:{request.perturbation_id}:{rank}".encode()
                ).hexdigest(),
            }
            for rank in range(4)
        ],
        "cost": {
            "setup_seconds": {
                "program_preparation": 0.01,
                "runtime_initialization": 1.5,
                "allocation_and_correctness": 0.25,
                "warmups": runtime,
                "total": 1.76 + runtime,
            },
            "measured_seconds": runtime * 2.0,
            "instrumentation_seconds": 0.5,
            "total_seconds": 1.76 + runtime + runtime * 2.0,
            "steady_state_seconds_per_iteration": runtime,
        },
        "samples": [
            {
                "iteration": iteration,
                "cuda_seconds_by_rank": [runtime] * 4,
                "host_seconds_by_rank": [runtime + 0.001] * 4,
                "physical_runtime_seconds": runtime,
                "peak_memory_bytes_by_rank": [1024] * 4,
            }
            for iteration in range(2)
        ],
        "physical_metrics": {
            "physical_runtime_seconds": runtime,
            "gpu_seconds": runtime * 4,
            "gpu_count": 4,
            "executed_collectives": collectives,
            "executed_flops": flops,
            "peak_memory_bytes": 1024,
        },
        "environment": environment,
        "environment_sha256": hashlib.sha256(canonical_json_bytes(environment)).hexdigest(),
        "telemetry": telemetry,
        "telemetry_assessment": physical_execution_module.telemetry_assessment(telemetry, world_size=4),
    }
    document["measurement_id"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    return document


def test_active_cegis_adds_counterexample_region_before_opening_holdout(tmp_path: Path) -> None:
    capture = commcanary_trace_to_chakra(_captured_trace(), opaque_attributes_reviewed=True)
    trace = decode_chakra_execution_trace(capture.chakra_et)
    runner_digest = f"sha256:{hashlib.sha256(b'runner').hexdigest()}"
    policy = _policy(runner_digest)
    baseline = _application(
        runner_digest,
        perturbation_id="baseline",
        subject=hashlib.sha256(b"baseline").hexdigest(),
        latency_ms=500.0,
        throughput=1000.0,
    )
    training = {
        "train-ok": _application(
            runner_digest,
            perturbation_id="train-ok",
            subject=hashlib.sha256(b"train-ok").hexdigest(),
            latency_ms=500.0,
            throughput=1000.0,
        ),
        "train-regression": _application(
            runner_digest,
            perturbation_id="train-regression",
            subject=hashlib.sha256(b"train-regression").hexdigest(),
            latency_ms=550.0,
            throughput=900.0,
        ),
    }
    holdout_opened = False
    frozen_selection: Dict[str, Any] = {}

    def selection_freezer(selection: Mapping[str, Any]) -> None:
        assert holdout_opened is False
        assert not frozen_selection
        frozen_selection.update(selection)

    def holdout_loader() -> Mapping[str, Mapping[str, Any]]:
        nonlocal holdout_opened
        assert frozen_selection["selection_sha256"]
        holdout_opened = True
        return {
            "holdout-regression": _application(
                runner_digest,
                perturbation_id="holdout-regression",
                subject=hashlib.sha256(b"holdout-regression").hexdigest(),
                latency_ms=560.0,
                throughput=890.0,
            ),
            # A holdout of nothing but regressions cannot falsify the
            # false-positive bound, so qualification against it would assert an
            # unmeasured claim. The split needs something that should pass.
            "holdout-ok": _application(
                runner_digest,
                perturbation_id="holdout-ok",
                subject=hashlib.sha256(b"holdout-ok").hexdigest(),
                latency_ms=500.0,
                throughput=1000.0,
            ),
        }

    def executor(request: PhysicalCandidateRequest) -> Mapping[str, Any]:
        if request.perturbation_id.startswith("holdout"):
            assert holdout_opened is True
        return _physical_measurement(request, runner_digest=runner_digest, projection=capture.projection)

    result = synthesize_active_physical_canary(
        trace,
        capture.projection,
        policy,
        baseline_application=baseline,
        training_applications=training,
        holdout_perturbation_ids=["holdout-regression", "holdout-ok"],
        holdout_application_loader=holdout_loader,
        executor=executor,
        selection_freezer=selection_freezer,
    )

    assert result.status == "qualified_active_candidate"
    assert set(result.selected_region_ids) == {"event-000001", "event-000003"}
    assert len(result.selected_node_ids) == 4
    assert len(result.selected_node_ids) < len(trace.nodes)
    assert (
        result.ledger["holdout_access"]["first_access_after_selection_sha256"]
        == result.ledger["selection"]["selection_sha256"]
    )
    assert frozen_selection == result.ledger["selection"]
    decisions = {
        observation["perturbation_id"]: observation["decision"]
        for observation in result.corpus["evaluations"][0]["observations"]
    }
    assert decisions["train-regression"] == "fail"
    assert decisions["holdout-regression"] == "fail"
    assert decisions["holdout-ok"] == "pass"
    communication = next(
        row for row in result.corpus["evaluations"] if row["method"] == "communication_only_microbenchmark"
    )
    assert all(observation["physical_metrics"]["executed_flops"] == 0 for observation in communication["observations"])

    source_path = tmp_path / "source.et"
    source_path.write_bytes(capture.chakra_et)
    bundle_path = tmp_path / "bundle"
    manifest = build_physical_canary_bundle(
        str(source_path),
        capture.projection,
        policy,
        str(bundle_path),
        corpus=result.corpus,
        active_ledger=result.ledger,
        application_evidence=result.application_evidence,
        physical_evidence=result.physical_evidence,
    )
    assert manifest["active_ledger_id"] == result.ledger["ledger_id"]
    assert manifest["application_evidence_set_id"] == result.application_evidence["evidence_set_id"]
    assert manifest["physical_evidence_set_id"] == result.physical_evidence["evidence_set_id"]
    assert (bundle_path / PHYSICAL_CANARY_ACTIVE_LEDGER).is_file()
    assert (bundle_path / PHYSICAL_CANARY_APPLICATION_EVIDENCE).is_file()
    assert (bundle_path / PHYSICAL_CANARY_PHYSICAL_EVIDENCE).is_file()
    assert verify_physical_canary_bundle(str(bundle_path)) == manifest

    forged_evidence = copy.deepcopy(result.application_evidence)
    forged_evidence["splits"]["holdout"]["perturbations"][0]["evidence"]["summary"]["median_batch_latency_ms"] += 1.0
    with pytest.raises(SchemaError, match="evidence_set_id"):
        build_physical_canary_bundle(
            str(source_path),
            capture.projection,
            policy,
            str(tmp_path / "forged-bundle"),
            corpus=result.corpus,
            active_ledger=result.ledger,
            application_evidence=forged_evidence,
            physical_evidence=result.physical_evidence,
        )

    forged_physical = copy.deepcopy(result.physical_evidence)
    forged_physical["measurements"][0]["measurement"]["samples"][0]["physical_runtime_seconds"] += 1.0
    with pytest.raises(SchemaError, match="evidence_set_id"):
        build_physical_canary_bundle(
            str(source_path),
            capture.projection,
            policy,
            str(tmp_path / "forged-physical-bundle"),
            corpus=result.corpus,
            active_ledger=result.ledger,
            application_evidence=result.application_evidence,
            physical_evidence=forged_physical,
        )

    if shutil.which("openssl") is not None:
        private_key = tmp_path / "owner-private.pem"
        public_key = tmp_path / "owner-public.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private_key)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["openssl", "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)],
            check=True,
            capture_output=True,
        )
        private_bundle = tmp_path / "private-bundle"
        private_manifest = build_physical_canary_bundle(
            str(source_path),
            capture.projection,
            policy,
            str(private_bundle),
            corpus=result.corpus,
            active_ledger=result.ledger,
            application_evidence=result.application_evidence,
            physical_evidence=result.physical_evidence,
            mode="private_exchange",
            owner_private_key=str(private_key),
            owner_public_key=str(public_key),
        )
        assert private_manifest["application_evidence_set_id"] == result.application_evidence["evidence_set_id"]
        assert private_manifest["physical_evidence_set_id"] == result.physical_evidence["evidence_set_id"]
        assert not (private_bundle / PHYSICAL_CANARY_APPLICATION_EVIDENCE).exists()
        assert not (private_bundle / PHYSICAL_CANARY_PHYSICAL_EVIDENCE).exists()
        assert (
            verify_physical_canary_bundle(
                str(private_bundle),
                trusted_owner_public_key=str(public_key),
            )
            == private_manifest
        )

    forged = copy.deepcopy(result.ledger)
    holdout_index = next(
        index
        for index, request in enumerate(forged["measurement_requests"])
        if request["perturbation_id"] == "holdout-regression"
    )
    forged["measurement_requests"][0], forged["measurement_requests"][holdout_index] = (
        forged["measurement_requests"][holdout_index],
        forged["measurement_requests"][0],
    )
    forged["ledger_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in forged.items() if key != "ledger_id"})
    ).hexdigest()
    with pytest.raises(SchemaError, match="before selection freeze"):
        validate_active_physical_study_ledger(forged)
