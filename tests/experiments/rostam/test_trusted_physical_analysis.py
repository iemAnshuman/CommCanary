from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest  # type: ignore[import-not-found]

from experiments.rostam.analysis import (
    AnalysisValidationError,
    CampaignEvidence,
    MeasurementValidationError,
    validate_scalar_measurement,
    verify_regenerate_compare,
)
from experiments.rostam.harness import (
    ATTEMPT_SCHEMA,
    CAMPAIGN_SCHEMA,
    CELL_RESULT_SCHEMA,
    ArtifactReference,
    AttemptRecord,
    CampaignSpec,
    CellResult,
    FrozenRun,
    RunManifest,
    SelectionSnapshot,
    build_run_manifest,
    build_selection_snapshot,
    canonical_sha256,
    derive_attempt_id,
    evaluate_completeness,
    freeze_completeness_verdict,
    freeze_run_manifest,
    freeze_selection_snapshot,
    write_attempt_record,
    write_cell_result,
)

MICRO_SCHEMA = "commcanary.rostam.physical.micro-measurement.v1"
FULL_SCHEMA = "commcanary.rostam.physical.full-measurement.v1"
CAPTURE_SCHEMA = "commcanary.rostam.physical.capture-measurement.v1"
PARAM_SCHEMA = "commcanary.rostam.physical.param-measurement.v1"
OVERLAP_SCHEMA = "commcanary.rostam.physical.overlap-measurement.v1"


def _runtime() -> Dict[str, Any]:
    return {
        "hostname": "private-node.example",
        "job_id": "private-job-123",
        "python_version": "3.12.3",
        "torch_version": "2.4.1",
        "torch_cuda_version": "12.1",
        "runtime_nccl_version_code": 22005,
    }


def _configuration(configuration_id: str) -> Dict[str, Any]:
    return {
        "id": configuration_id,
        "environment": {},
        "parameters": {"venv": "experiments/rostam/venvs/test"},
        "expected_runtime": {
            "python_version": "3.12.3",
            "torch_version": "2.4.1",
            "runtime_nccl_version_code": 22005,
        },
    }


def _site() -> Dict[str, Any]:
    return {
        "site_id": "rostam",
        "scheduler": "slurm",
        "partition": "cuda-A100",
        "nodes": 1,
        "exclusive": True,
        "node_constraints": ["private-node"],
        "account": None,
        "resources": {},
    }


def _workload(
    workload_id: str,
    producer_schema: str,
    measurement_schema: str,
    command: list[str],
    *,
    depends_on: Optional[List[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "adapter": "torch-json",
        "command": command,
        "operation": "all_reduce",
        "world_size": 4,
        "global_ranks": [0, 1, 2, 3],
    }
    parameters.update({} if extra is None else extra)
    return {
        "id": workload_id,
        "producer_schema": producer_schema,
        "measurement_schema": measurement_schema,
        "parameters": parameters,
        "depends_on": [] if depends_on is None else depends_on,
    }


def _core_campaign(run_id: str) -> CampaignSpec:
    workloads = [
        _workload(
            "micro",
            "commcanary.rostam.physical.micro-producer.v1",
            MICRO_SCHEMA,
            ["python", "microbench_tp8.py", "--dtype", "bf16", "--msg-sizes", "64K,128K"],
        ),
        _workload(
            "full",
            "commcanary.rostam.physical.full-producer.v1",
            FULL_SCHEMA,
            [
                "python",
                "workload_tp8.py",
                "--layers",
                "32",
                "--tokens",
                "256",
                "--hidden",
                "8192",
                "--gemm-m",
                "256",
                "--dtype",
                "bf16",
            ],
        ),
        _workload(
            "trace-build",
            "commcanary.rostam.physical.capture-producer.v1",
            CAPTURE_SCHEMA,
            ["python", "capture.py"],
            extra={"outputs": {"param_trace": "{workspace}/param_trace.json"}},
        ),
        _workload(
            "canary-param",
            "commcanary.rostam.physical.param-producer.v1",
            PARAM_SCHEMA,
            [
                "python",
                "commsTraceReplay.py",
                "--trace-path",
                "{dependency:trace-build:param_trace}",
            ],
            depends_on=["trace-build"],
            extra={"replay_mode": "timestamp-paced-blocking"},
        ),
    ]
    return CampaignSpec.from_dict(
        {
            "schema": CAMPAIGN_SCHEMA,
            "run_id": run_id,
            "campaign_id": "rostam-core-test",
            "repository": {
                "commit": "1" * 40,
                "dirty": False,
                "patch_sha256": None,
                "source_archive_sha256": "a" * 64,
            },
            "inputs": [],
            "axes": {
                "configurations": [_configuration("nccl-2.19.3-default"), _configuration("nccl-2.20.5-default")],
                "workloads": workloads,
                "repetitions": 1,
            },
            "policy": {"aggregation": "median-of-cell-medians"},
            "expected_site": _site(),
        }
    )


def _shared_campaign(run_id: str, trace_sha256: str, trace_size: int) -> CampaignSpec:
    workload = _workload(
        "shared-overlap",
        "commcanary.rostam.physical.overlap-producer.v1",
        OVERLAP_SCHEMA,
        ["python", "overlap_replay.py", "--trace-path", "{input:shared-param-trace}"],
        extra={"replay_mode": "fixed-input-explicit-wait-overlap"},
    )
    return CampaignSpec.from_dict(
        {
            "schema": CAMPAIGN_SCHEMA,
            "run_id": run_id,
            "campaign_id": "rostam-shared-test",
            "repository": {
                "commit": "1" * 40,
                "dirty": False,
                "patch_sha256": None,
                "source_archive_sha256": "a" * 64,
            },
            "inputs": [{"id": "shared-param-trace", "sha256": trace_sha256, "size_bytes": trace_size}],
            "axes": {
                "configurations": [_configuration("nccl-2.19.3-default"), _configuration("nccl-2.20.5-default")],
                "workloads": [workload],
                "repetitions": 1,
            },
            "policy": {"aggregation": "median-of-cell-medians"},
            "expected_site": _site(),
        }
    )


def _artifact(run_directory: Path, path: Path) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(run_directory).as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


def _base_measurement(attempt_id: str, value_us: float) -> Dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "operation": "all_reduce",
        "world_size": 4,
        "global_ranks": [0, 1, 2, 3],
        "value_us": value_us,
        "samples_us": [value_us - 1.0, value_us, value_us + 1.0],
        "iqr_us": 2.0,
        "count": 3,
        "wall_time_s": value_us / 1000.0,
        "runtime": _runtime(),
    }


def _value(workload_id: str, configuration_id: str) -> float:
    candidate = configuration_id == "nccl-2.20.5-default"
    values = {
        "full": (100.0, 120.0),
        "micro": (90.0, 80.0),
        "canary-param": (110.0, 130.0),
        "trace-build": (50.0, 50.0),
        "shared-overlap": (105.0, 125.0),
    }
    return values[workload_id][1 if candidate else 0]


@dataclass(frozen=True)
class PhysicalFixture:
    manifest: RunManifest
    frozen: FrozenRun
    selection: SelectionSnapshot
    verdict_sha256: str


def _write_physical_attempt(
    manifest: RunManifest,
    frozen: FrozenRun,
    cell: Any,
    measurement: Dict[str, Any],
    *,
    dependency_attempts: list[dict[str, str]],
    partial_outputs: Tuple[ArtifactReference, ...] = (),
) -> AttemptRecord:
    attempt_id = derive_attempt_id(1)
    workspace = frozen.directory / "physical-results" / cell.id / attempt_id
    workspace.mkdir(parents=True)
    workload = next(item for item in manifest.campaign.workloads if item.id == cell.workload_id)
    result = CellResult.from_dict(
        {
            "schema": CELL_RESULT_SCHEMA,
            "cell_id": cell.id,
            "cell_identity_sha256": cell.identity_sha256,
            "producer_schema": workload.producer_schema,
            "measurement_schema": workload.measurement_schema,
            "measurement": measurement,
        }
    )
    result_path = write_cell_result(workspace / "result.json", result)
    stdout_path = workspace / "stdout.log"
    stderr_path = workspace / "stderr.log"
    stdout_path.write_text("private stdout\n", encoding="utf-8")
    stderr_path.write_text("private stderr\n", encoding="utf-8")
    environment_sha256 = canonical_sha256({"cell": cell.id})
    execution_sha256 = canonical_sha256({"private-command-token": cell.id})
    record = AttemptRecord.from_dict(
        {
            "schema": ATTEMPT_SCHEMA,
            "run_id": manifest.run_id,
            "manifest_sha256": frozen.manifest_sha256,
            "cell_id": cell.id,
            "cell_identity_sha256": cell.identity_sha256,
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "status": "success",
            "started_at": "2026-07-11T01:00:00.000000Z",
            "finished_at": "2026-07-11T01:00:01.000000Z",
            "command": ["private-command-token", "--cell", cell.id],
            "observed": {
                "executor": "slurm-cell-entrypoint",
                "site_id": "rostam",
                "hostname": "private-node.example",
                "scheduler": "slurm",
                "job_id": "private-job-123",
                "nodes": ["private-node"],
                "account": "private-account",
                "partition": "cuda-A100",
                "metadata": {
                    "environment_sha256": environment_sha256,
                    "execution_identity_sha256": execution_sha256,
                    "execution_plan_sha256": execution_sha256,
                    "dependency_attempts": dependency_attempts,
                    "input_hashes": {artifact.id: artifact.sha256 for artifact in manifest.campaign.inputs},
                    "runtime_observation": {
                        "schema": "commcanary.rostam.runtime-observation.v1",
                        "runtime": _runtime(),
                    },
                },
            },
            "exit_code": 0,
            "reason": None,
            "stdout": _artifact(frozen.directory, stdout_path).to_dict(),
            "stderr": _artifact(frozen.directory, stderr_path).to_dict(),
            "measurement": _artifact(frozen.directory, result_path).to_dict(),
            "partial_outputs": [item.to_dict() for item in sorted(partial_outputs, key=lambda item: item.path)],
        }
    )
    write_attempt_record(frozen.directory, record)
    return record


def _freeze_physical_campaign(
    campaign: CampaignSpec,
    results_root: Path,
    *,
    tamper: Optional[str] = None,
) -> PhysicalFixture:
    manifest = build_run_manifest(campaign)
    frozen = freeze_run_manifest(manifest, results_root)
    records: Dict[str, AttemptRecord] = {}
    capture_artifacts: Dict[str, ArtifactReference] = {}
    for cell in manifest.cells:
        if cell.workload_id != "trace-build":
            continue
        artifact_path = frozen.directory / "workspaces" / cell.id / "a-000001" / "param_trace.json"
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_text(f"trace:{cell.configuration_id}\n", encoding="utf-8")
        reference = _artifact(frozen.directory, artifact_path)
        capture_artifacts[cell.id] = reference
        value = _value(cell.workload_id, cell.configuration_id)
        measurement = _base_measurement("a-000001", value)
        measurement.update(
            {
                "samples_us": [value],
                "value_us": value,
                "iqr_us": 0.0,
                "count": 1,
                "artifacts": {"param_trace": reference.to_dict()},
            }
        )
        records[cell.id] = _write_physical_attempt(
            manifest,
            frozen,
            cell,
            measurement,
            dependency_attempts=[],
            partial_outputs=(reference,),
        )
    for cell in manifest.cells:
        if cell.workload_id == "trace-build":
            continue
        value = _value(cell.workload_id, cell.configuration_id)
        measurement = _base_measurement("a-000001", value)
        dependency_evidence: list[dict[str, str]] = []
        if cell.workload_id == "micro":
            measurement.update({"dtype": "bf16", "message_sizes_bytes": [65536, 131072]})
        elif cell.workload_id == "full":
            measurement.update(
                {"dtype": "bf16", "layers": 32, "tokens": 256, "hidden": 8192, "gemm_m": 256, "gemm_n": 8192}
            )
            if tamper == "shape" and cell.configuration_id == "nccl-2.19.3-default":
                measurement["tokens"] = 255
            if tamper == "runtime" and cell.configuration_id == "nccl-2.19.3-default":
                measurement["runtime"] = {**_runtime(), "python_version": "3.11.0"}
        elif cell.workload_id == "canary-param":
            dependency_cell_id = cell.dependencies[0]
            dependency = records[dependency_cell_id]
            dependency_evidence = [
                {
                    "cell_id": dependency_cell_id,
                    "attempt_id": "a-000099" if tamper == "dependency" else dependency.attempt_id,
                    "attempt_record_sha256": dependency.sha256,
                }
            ]
            trace_sha256 = capture_artifacts[dependency_cell_id].sha256
            if tamper == "trace" and cell.configuration_id == "nccl-2.19.3-default":
                trace_sha256 = "f" * 64
            measurement.update({"replay_mode": "timestamp-paced-blocking", "trace_sha256": trace_sha256})
        elif cell.workload_id == "shared-overlap":
            trace = next(item for item in manifest.campaign.inputs if item.id == "shared-param-trace")
            measurement.update({"replay_mode": "fixed-input-explicit-wait-overlap", "trace_sha256": trace.sha256})
        records[cell.id] = _write_physical_attempt(
            manifest,
            frozen,
            cell,
            measurement,
            dependency_attempts=dependency_evidence,
        )
    selection = build_selection_snapshot(
        frozen.directory,
        "primary",
        {cell.id: records[cell.id].attempt_id for cell in manifest.cells},
    )
    freeze_selection_snapshot(frozen.directory, selection)
    verdict = evaluate_completeness(frozen.directory, selection)
    stored = freeze_completeness_verdict(frozen.directory, verdict)
    return PhysicalFixture(manifest, frozen, selection, stored.verdict_sha256)


def test_complete_core_and_shared_join_generates_claims_without_private_execution_data(tmp_path: Path) -> None:
    core = _freeze_physical_campaign(_core_campaign("core-run"), tmp_path / "core")
    shared_trace = b"fixed shared trace"
    shared_sha = hashlib.sha256(shared_trace).hexdigest()
    shared = _freeze_physical_campaign(
        _shared_campaign("shared-run", shared_sha, len(shared_trace)),
        tmp_path / "shared",
    )
    publication = verify_regenerate_compare(
        core.frozen.directory,
        core.selection.selection_id,
        core.verdict_sha256,
        tmp_path / "publication",
        regeneration_command="python -m experiments.rostam.analyze verify --trusted-fixture",
        joined_evidence=(CampaignEvidence(shared.frozen.directory, "primary", shared.verdict_sha256),),
        baseline_config="nccl-2.19.3-default",
        candidate_config="nccl-2.20.5-default",
    )
    claims = publication.aggregate["claims"]
    assert claims["status"] == "supported-by-complete-selected-evidence"
    assert set(claims["rankings"]) == {"W-full", "W-micro", "W-canary", "W-shared-overlap"}
    assert claims["agreements"]["W-canary_vs_W-full"]["agreement_pct"] == 100.0
    assert claims["regression_2x2"]["confusion_vs_full"]["W-micro"]["cell"] == "FN"
    assert claims["regression_2x2"]["confusion_vs_full"]["W-shared-overlap"]["cell"] == "TP"
    assert len(publication.aggregate["provenance"]["campaigns"]) == 2
    paper = (publication.output_directory / "paper-fragment.md").read_text(encoding="utf-8")
    assert "### Regression 2x2" in paper
    assert "### Cost" in paper
    encoded = str(publication.aggregate)
    for private_value in (
        "private-node.example",
        "private-job-123",
        "private-account",
        "private-command-token",
        "slurm-cell-entrypoint",
    ):
        assert private_value not in encoded
    assert all("binding_sha256" in row for row in publication.aggregate["selected_cells"])


def test_physical_runtime_nullables_and_replay_enums_match_committed_schema() -> None:
    measurement = _base_measurement("a-000001", 10.0)
    measurement["runtime"] = {**_runtime(), "job_id": "", "torch_cuda_version": ""}
    measurement.update({"replay_mode": "timestamp-paced-blocking", "trace_sha256": "a" * 64})
    scalar = validate_scalar_measurement(
        PARAM_SCHEMA,
        "commcanary.rostam.physical.param-producer.v1",
        "a-000001",
        measurement,
    )
    assert scalar.physical is not None
    assert scalar.physical.runtime.job_id == ""
    assert scalar.physical.runtime.torch_cuda_version == ""

    measurement["replay_mode"] = "explicit-wait-overlap"
    with pytest.raises(MeasurementValidationError, match="mode is not allowed"):
        validate_scalar_measurement(
            PARAM_SCHEMA,
            "commcanary.rostam.physical.param-producer.v1",
            "a-000001",
            measurement,
        )


@pytest.mark.parametrize("tamper", ["shape", "runtime", "trace", "dependency"])  # type: ignore[misc]
def test_physical_manifest_dependency_and_runtime_tampering_is_rejected(tmp_path: Path, tamper: str) -> None:
    fixture = _freeze_physical_campaign(_core_campaign(f"tamper-{tamper}"), tmp_path / tamper, tamper=tamper)
    with pytest.raises(AnalysisValidationError, match="stale|dependency attempt"):
        verify_regenerate_compare(
            fixture.frozen.directory,
            fixture.selection.selection_id,
            fixture.verdict_sha256,
            tmp_path / f"out-{tamper}",
            regeneration_command="python -m experiments.rostam.analyze verify --tamper-test",
        )
