from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

from experiments.rostam.product_canary.historical import study
from experiments.rostam.product_canary.historical.build_images import HistoricalImageError, validate_image_lock
from experiments.rostam.product_canary.historical.contract import (
    HistoricalStudyError,
    freeze_historical_study,
    verify_frozen_historical_study,
)
from experiments.rostam.product_canary.historical.stage_model import validate_model_lock

ROOT = Path(__file__).resolve().parents[3]
HISTORICAL = ROOT / "experiments" / "rostam" / "product_canary" / "historical"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _image_build(tmp_path: Path) -> Path:
    lock = validate_image_lock(json.loads((HISTORICAL / "image-lock.json").read_text(encoding="utf-8")))
    root = tmp_path / "images"
    root.mkdir()
    _write_json(root / "image-lock.json", lock)
    rows = []
    for image in lock["images"]:
        identifier = str(image["id"])
        sif = root / f"{identifier}.sif"
        sif.write_bytes(f"sif:{identifier}".encode())
        log = root / f"{identifier}.build.log"
        log.write_bytes(f"build:{identifier}\n".encode())
        rows.append(
            {
                "id": identifier,
                "repository": image["repository"],
                "tag_observed": image["tag_observed"],
                "manifest_digest": image["manifest_digest"],
                "sif_path": str(sif.resolve()),
                "sif_identity": _identity(sif),
                "build_log_identity": _identity(log),
                "versions": {
                    "python": "3.10.0",
                    "torch": "2.1.2",
                    "cuda": "12.1",
                    "vllm": image["expected_vllm_version"],
                },
            }
        )
    descriptor: dict[str, Any] = {
        "format": "commcanary.historical_vllm_image_descriptor.v1",
        "status": "complete",
        "host": "rostam",
        "architecture": "x86_64",
        "image_lock_identity": _identity(root / "image-lock.json"),
        "images": rows,
    }
    descriptor["descriptor_id"] = hashlib.sha256(_canonical(descriptor)).hexdigest()
    _write_json(root / "descriptor.json", descriptor)
    return root / "descriptor.json"


def _model_staging(tmp_path: Path) -> Path:
    lock = validate_model_lock(json.loads((HISTORICAL / "model-lock.json").read_text(encoding="utf-8")))
    root = tmp_path / "model-stage"
    model = root / "model"
    model.mkdir(parents=True)
    _write_json(root / "model-lock.json", lock)
    rows = []
    for name in lock["required_files"]:
        path = model / str(name)
        path.write_bytes(f"historical-model:{name}".encode())
        rows.append({"path": name, **_identity(path)})
    descriptor: dict[str, Any] = {
        "format": "commcanary.historical_vllm_model_descriptor.v1",
        "status": "complete",
        "source": {
            "repository": lock["repository"],
            "revision": lock["revision"],
            "revision_source": lock["revision_source"],
        },
        "model_path": str(model.resolve()),
        "files": rows,
        "model_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }
    descriptor["descriptor_id"] = hashlib.sha256(_canonical(descriptor)).hexdigest()
    _write_json(root / "descriptor.json", descriptor)
    return root / "descriptor.json"


def _freeze(tmp_path: Path) -> Path:
    output = tmp_path / "historical-study"
    freeze_historical_study(
        study_id="issue-2971-a100-tp4",
        output_directory=output,
        image_lock_path=HISTORICAL / "image-lock.json",
        model_lock_path=HISTORICAL / "model-lock.json",
        study_lock_path=HISTORICAL / "study-lock.json",
        image_descriptor_path=_image_build(tmp_path),
        model_descriptor_path=_model_staging(tmp_path),
    )
    return output


def _gpu(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100-PCIE-40GB",
        "pstate": "P0",
        "sm_clock_mhz": 1200.0,
        "memory_clock_mhz": 1215.0,
        "temperature_c": 45.0,
        "power_w": 175.0,
        "clock_event_reasons_active": "0x0",
        "corrected_volatile_ecc": 0,
        "uncorrected_volatile_ecc": 0,
    }


def _measurement(
    manifest: Mapping[str, Any],
    configuration_id: str,
    *,
    job_id: str,
    node: str,
    latency: float,
    throughput: float,
) -> dict[str, Any]:
    conditions = {row["id"]: row for row in manifest["conditions"]}
    images = {row["id"]: row for row in manifest["runtime_images"]}
    condition = conditions[configuration_id]
    image = images[condition["image_id"]]
    samples = [
        {
            "iteration": index,
            "batch_latency_ms": latency,
            "output_token_throughput_per_second": throughput,
            "total_token_throughput_per_second": throughput + 100.0,
            "output_commitment_sha256": hashlib.sha256(f"{configuration_id}:{job_id}:{index}".encode()).hexdigest(),
        }
        for index in range(12)
    ]
    environment = {
        "hostname": node,
        "platform": "Linux",
        "python": "3.10.0",
        "torch": "2.1.2",
        "cuda": "12.1",
        "nccl": [2, 18, 1],
        "visible_gpu_count": 4,
        "device_names": ["NVIDIA A100-PCIE-40GB"] * 4,
        "cuda_visible_devices": "0,1,2,3",
        "slurm_job_id": job_id,
        "slurm_node": node,
        "nccl_configuration": {
            "NCCL_ALGO": None,
            "NCCL_PROTO": None,
            "NCCL_NET": None,
            "NCCL_P2P_DISABLE": None,
            "NCCL_SHM_DISABLE": None,
        },
    }
    phases = [
        "before_engine_initialization",
        "before_warmup",
        "before_measured_cycle_1",
        *(f"after_measured_cycle_{index + 1}" for index in range(12)),
        "final",
    ]
    telemetry = [
        {
            "phase": phase,
            "monotonic_ns": index + 1,
            "command": ["nvidia-smi"],
            "gpus": [_gpu(gpu_index) for gpu_index in range(4)],
        }
        for index, phase in enumerate(phases)
    ]
    document: dict[str, Any] = {
        "format": "commcanary.historical_vllm_measurement.v1",
        "scope": {
            "study": manifest["scope"],
            "application": manifest["claim_boundary"]["application"],
            "hardware": manifest["claim_boundary"]["hardware"],
            "request_stream": manifest["claim_boundary"]["execution"],
        },
        "configuration_id": configuration_id,
        "image_manifest_digest": image["manifest_digest"],
        "sif_sha256": image["sif_identity"]["sha256"],
        "vllm_version": image["versions"]["vllm"],
        "enforce_eager": condition["enforce_eager"],
        "workload": manifest["workload"],
        "engine_initialization_seconds": 1.0,
        "samples": samples,
        "summary": {
            "median_batch_latency_ms": latency,
            "p99_batch_latency_ms": latency,
            "median_output_token_throughput_per_second": throughput,
            "median_total_token_throughput_per_second": throughput + 100.0,
        },
        "environment": environment,
        "environment_sha256": hashlib.sha256(_canonical(environment)).hexdigest(),
        "telemetry": telemetry,
    }
    document["measurement_id"] = hashlib.sha256(_canonical(document)).hexdigest()
    return document


def test_historical_locks_bind_official_versions_and_limit_the_claim() -> None:
    images = validate_image_lock(json.loads((HISTORICAL / "image-lock.json").read_text(encoding="utf-8")))
    model = validate_model_lock(json.loads((HISTORICAL / "model-lock.json").read_text(encoding="utf-8")))
    design = json.loads((HISTORICAL / "study-lock.json").read_text(encoding="utf-8"))

    assert [row["expected_vllm_version"] for row in images["images"]] == ["0.2.7", "0.3.2", "0.3.3"]
    assert model["revision"] == "24c0bea14d53e6f67f1fbe2eca5bfe7cae389b33"
    assert design["claim_boundary"]["interpretation"].endswith("not-an-exact-reproduction-of-the-user-report")


def test_frozen_historical_study_rehashes_external_sif_and_model_bytes(tmp_path: Path) -> None:
    root = _freeze(tmp_path)
    manifest = verify_frozen_historical_study(root)
    assert len(manifest["schedule"]["configuration_order_by_repetition"]) == 8

    sif = Path(manifest["runtime_images"][0]["sif_path"])
    os.chmod(sif, 0o600)
    sif.write_bytes(b"changed")
    with pytest.raises(HistoricalImageError, match="SIF identity changed"):
        verify_frozen_historical_study(root)


def test_historical_submission_spools_exact_wrapper_through_stdin(tmp_path: Path) -> None:
    root = _freeze(tmp_path)
    calls: list[tuple[list[str], Optional[bytes]]] = []

    def runner(argv: Sequence[str], stdin: Optional[bytes]) -> subprocess.CompletedProcess[bytes]:
        calls.append((list(argv), stdin))
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"81234;rostam\n", stderr=b"")

    submitted = study.submit_repetition(root, 4, execute_acknowledged=True, command_runner=runner)

    assert submitted.job_id == "81234"
    assert calls[0][1] == (submitted.attempt_directory / "wrapper.sbatch").read_bytes()
    assert "--begin=now+1day" in calls[0][0]
    assert str(submitted.attempt_directory / "wrapper.sbatch") not in calls[0][0]
    wrapper = calls[0][1].decode("utf-8") if calls[0][1] is not None else ""
    assert "python3 -I" in wrapper
    assert "nvidia-smi" in wrapper


def test_historical_aggregation_can_observe_pattern_without_issuing_causality(tmp_path: Path) -> None:
    root = _freeze(tmp_path)
    manifest = verify_frozen_historical_study(root)
    metrics = {
        "vllm-0.2.7-default": (100.0, 100.0),
        "vllm-0.3.2-default": (110.0, 90.0),
        "vllm-0.3.3-default": (110.0, 90.0),
        "vllm-0.3.3-enforce-eager": (95.0, 105.0),
    }
    for repetition, order in enumerate(manifest["schedule"]["configuration_order_by_repetition"]):
        attempt = root / "state" / "attempts" / f"repetition-{repetition:02d}" / "a-000001"
        measurements = attempt / "measurements"
        measurements.mkdir(parents=True)
        job_id = str(9000 + repetition)
        rows = []
        for position, configuration_id in enumerate(order):
            latency, throughput = metrics[configuration_id]
            document = _measurement(
                manifest,
                configuration_id,
                job_id=job_id,
                node="toranj0",
                latency=latency,
                throughput=throughput,
            )
            path = measurements / f"{configuration_id}.json"
            _write_json(path, document)
            rows.append(
                {
                    "configuration_id": configuration_id,
                    "planned_position": position,
                    "elapsed_from_repetition_start_seconds": float(position),
                    "measurement_id": document["measurement_id"],
                    "path": path.relative_to(attempt).as_posix(),
                    "sha256": _identity(path)["sha256"],
                }
            )
        batch: dict[str, Any] = {
            "format": "commcanary.historical_vllm_repetition.v1",
            "manifest_id": manifest["manifest_id"],
            "repetition": repetition,
            "configuration_order": order,
            "rows": rows,
            "completed_at": "2026-08-05T12:00:00Z",
        }
        batch["batch_id"] = hashlib.sha256(_canonical(batch)).hexdigest()
        _write_json(attempt / "batch-result.json", batch)
        day = "05" if repetition < 4 else "06"
        _write_json(
            attempt / "terminal.json",
            {
                "format": "commcanary.historical_vllm_attempt.v1",
                "status": "success",
                "job_id": job_id,
                "state": "COMPLETED",
                "exit_code": "0:0",
                "node": "toranj0",
                "start_time": f"2026-08-{day}T10:00:00",
                "end_time": f"2026-08-{day}T10:10:00",
                "elapsed_seconds": 600,
                "observed_at": f"2026-08-{day}T10:10:01Z",
            },
        )

    result = study._aggregate(root, manifest)

    assert result["status"] == "complete_comparable"
    assert result["conclusion"] == "reported-version-and-eager-mitigation-pattern-observed"
    assert result["reported_version_comparison"]["threshold_crossed"] is True
    assert result["eager_mitigation_comparison"]["threshold_crossed"] is True
    assert result["causal_claim"] == "not_issued"


def test_historical_measurement_rejects_rehashed_wrong_runtime_binding(tmp_path: Path) -> None:
    root = _freeze(tmp_path)
    manifest = verify_frozen_historical_study(root)
    document = _measurement(
        manifest,
        "vllm-0.2.7-default",
        job_id="99",
        node="toranj0",
        latency=100.0,
        throughput=100.0,
    )
    document["sif_sha256"] = "f" * 64
    document["measurement_id"] = hashlib.sha256(
        _canonical({key: value for key, value in document.items() if key != "measurement_id"})
    ).hexdigest()

    with pytest.raises(HistoricalStudyError, match="frozen sif_sha256"):
        study.validate_measurement(document, manifest=manifest, configuration_id="vllm-0.2.7-default")
