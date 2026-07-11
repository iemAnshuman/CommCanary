from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pytest  # type: ignore[import-not-found]

from experiments.rostam.harness import (
    CAMPAIGN_SCHEMA,
    CELL_RESULT_SCHEMA,
    ArtifactVerificationError,
    AttemptStoreError,
    CampaignSpec,
    CellResult,
    CellRunInterrupted,
    DependencyValidationError,
    ExistingAttemptError,
    FrozenRun,
    IncompleteCampaignError,
    ResultValidationError,
    RunManifest,
    RunnerValidationError,
    StaleExecutionIdentityError,
    WorkspaceCollisionError,
    build_run_manifest,
    build_selection_snapshot,
    evaluate_and_persist_completeness,
    freeze_run_manifest,
    freeze_selection_snapshot,
    load_cell_attempts,
    load_cell_result,
    plan_cell,
    run_cell,
    verify_attempt_artifacts,
    write_cell_result,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "experiments" / "local_cell.py"


def _digest(character: str) -> str:
    return character * 64


def _campaign_dict(state_path: Path) -> Dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "run_id": "local-runner-golden",
        "campaign_id": "local-runner",
        "repository": {
            "commit": "1" * 40,
            "dirty": False,
            "patch_sha256": None,
            "source_archive_sha256": _digest("a"),
        },
        "inputs": [
            {"id": "producer-fixture", "sha256": _digest("b"), "size_bytes": 123},
        ],
        "axes": {
            "configurations": [
                {
                    "id": "local-config",
                    "environment": {
                        "LOCAL_CELL_FAIL_ONCE_STATE": str(state_path),
                        "LOCAL_CONFIG": "from-manifest",
                    },
                    "parameters": {"variant": "local"},
                    "expected_runtime": {"executor": "python"},
                }
            ],
            "workloads": [
                {
                    "id": "prepare",
                    "producer_schema": "commcanary.experiment.prepare.v1",
                    "measurement_schema": "commcanary.experiment.local.prepare-measurement.v1",
                    "parameters": {},
                    "depends_on": [],
                },
                {
                    "id": "consume",
                    "producer_schema": "commcanary.experiment.consume.v1",
                    "measurement_schema": "commcanary.experiment.local.consume-measurement.v1",
                    "parameters": {},
                    "depends_on": ["prepare"],
                },
                {
                    "id": "fail-once",
                    "producer_schema": "commcanary.experiment.fail-once.v1",
                    "measurement_schema": "commcanary.experiment.local.fail-once-measurement.v1",
                    "parameters": {},
                    "depends_on": [],
                },
            ],
            "repetitions": 1,
        },
        "policy": {"aggregation": "median"},
        "expected_site": {
            "site_id": "local",
            "scheduler": "local",
            "partition": "workstation",
            "nodes": 1,
            "exclusive": False,
            "node_constraints": [],
            "account": None,
            "resources": {},
        },
    }


def _frozen_run(tmp_path: Path) -> Tuple[RunManifest, FrozenRun]:
    manifest = build_run_manifest(CampaignSpec.from_dict(_campaign_dict(tmp_path / "fail-once.state")))
    return manifest, freeze_run_manifest(manifest, tmp_path / "results")


def _cell_id(manifest: RunManifest, workload_id: str) -> str:
    return next(cell.id for cell in manifest.cells if cell.workload_id == workload_id)


def _command(mode: str, *extra: str) -> Tuple[str, ...]:
    return (sys.executable, str(FIXTURE), "--mode", mode, *extra)


def _inherited_environment(tmp_path: Path) -> Mapping[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "SECRET_TOKEN": "must-not-leak",
        "UNRELATED_PARENT_STATE": "must-not-leak-either",
    }


def _plan(
    frozen: FrozenRun,
    cell_id: str,
    command: Tuple[str, ...],
    tmp_path: Path,
    **kwargs: Any,
) -> Any:
    return plan_cell(
        frozen.directory,
        cell_id,
        command,
        inherited_environment=_inherited_environment(tmp_path),
        timeout_seconds=5,
        max_output_bytes=64 * 1024,
        max_result_bytes=64 * 1024,
        **kwargs,
    )


def test_golden_local_campaign_covers_planning_retry_and_completeness(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    prepare_id = _cell_id(manifest, "prepare")
    consume_id = _cell_id(manifest, "consume")
    failure_id = _cell_id(manifest, "fail-once")
    prepare_command = _command("success")
    consume_command = _command("success")
    failure_command = _command("fail-once")

    # A dry-run is a real plan but creates neither workspace nor attempt.
    dry_plan = _plan(
        frozen,
        failure_id,
        failure_command,
        tmp_path,
        dry_run=True,
    )
    assert dry_plan.action == "run"
    assert run_cell(dry_plan).workspace is None
    assert load_cell_attempts(frozen.directory, failure_id) == ()
    assert not (frozen.directory / "workspaces").exists()

    prepare_plan = _plan(frozen, prepare_id, prepare_command, tmp_path)
    assert prepare_plan.environment_dict()["LOCAL_CONFIG"] == "from-manifest"
    assert "SECRET_TOKEN" not in prepare_plan.environment_dict()
    assert "UNRELATED_PARENT_STATE" not in prepare_plan.environment_dict()
    prepare = run_cell(prepare_plan)
    assert prepare.record is not None and prepare.record.status == "success"
    assert prepare.result is not None
    prepare_measurement = prepare.result.measurement.to_value()
    assert prepare_measurement["secret_present"] is False
    assert prepare_measurement["config_value"] == "from-manifest"
    assert prepare.workspace is not None
    assert (prepare.workspace / "execution_plan.json").is_file()
    assert verify_attempt_artifacts(frozen.directory, prepare.record)

    with pytest.raises(DependencyValidationError, match="missing"):
        _plan(frozen, consume_id, consume_command, tmp_path)
    assert prepare.record.attempt_id == "a-000001"
    consume_plan = _plan(
        frozen,
        consume_id,
        consume_command,
        tmp_path,
        dependency_attempts={prepare_id: prepare.record.attempt_id},
    )
    assert [binding.cell_id for binding in consume_plan.dependencies] == [prepare_id]
    consume = run_cell(consume_plan)
    assert consume.record is not None and consume.record.status == "success"

    failed = run_cell(_plan(frozen, failure_id, failure_command, tmp_path))
    assert failed.record is not None and failed.record.status == "failed"
    assert failed.record.exit_code == 19
    assert failed.record.reason == "process exited with code 19"
    assert failed.workspace is not None
    assert (failed.workspace / "stdout.log").read_text(encoding="utf-8").startswith("stdout:fail-once")
    assert (failed.workspace / "stderr.log").read_text(encoding="utf-8").startswith("stderr:fail-once")

    incomplete_snapshot = build_selection_snapshot(
        frozen.directory,
        "failed-selection",
        {
            prepare_id: prepare.record.attempt_id,
            consume_id: consume.record.attempt_id,
            failure_id: failed.record.attempt_id,
        },
    )
    freeze_selection_snapshot(frozen.directory, incomplete_snapshot)
    with pytest.raises(IncompleteCampaignError) as captured:
        evaluate_and_persist_completeness(
            frozen.directory,
            incomplete_snapshot.selection_id,
        )
    assert {issue.code for issue in captured.value.verdict.issues} == {"selected-attempt-failed"}

    only_missing = _plan(
        frozen,
        failure_id,
        failure_command,
        tmp_path,
        only_missing=True,
    )
    assert only_missing.action == "skip"
    assert run_cell(only_missing).record is None
    resume_without_retry = _plan(
        frozen,
        failure_id,
        failure_command,
        tmp_path,
        resume=True,
    )
    assert resume_without_retry.action == "skip"

    retry_plan = _plan(
        frozen,
        failure_id,
        failure_command,
        tmp_path,
        retry_failed=True,
    )
    assert retry_plan.action == "run"
    assert retry_plan.attempt_id == "a-000002"
    retried = run_cell(retry_plan)
    assert retried.record is not None and retried.record.status == "success"
    assert [attempt.status for attempt in load_cell_attempts(frozen.directory, failure_id)] == [
        "failed",
        "success",
    ]

    resumed = _plan(
        frozen,
        failure_id,
        failure_command,
        tmp_path,
        resume=True,
        retry_failed=True,
    )
    assert resumed.action == "skip"
    assert resumed.reuse_attempt_id == "a-000002"
    with pytest.raises(ExistingAttemptError):
        _plan(frozen, prepare_id, prepare_command, tmp_path)
    with pytest.raises(StaleExecutionIdentityError):
        _plan(frozen, prepare_id, _command("fail"), tmp_path, resume=True)

    complete_snapshot = build_selection_snapshot(
        frozen.directory,
        "successful-selection",
        {
            prepare_id: prepare.record.attempt_id,
            consume_id: consume.record.attempt_id,
            failure_id: retried.record.attempt_id,
        },
    )
    freeze_selection_snapshot(frozen.directory, complete_snapshot)
    verdict, _stored = evaluate_and_persist_completeness(
        frozen.directory,
        complete_snapshot.selection_id,
    )
    assert verdict.complete is True
    assert verdict.successful_selected_cells == 3


def test_two_precomputed_plans_cannot_claim_the_same_attempt(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    cell_id = _cell_id(manifest, "prepare")
    first = _plan(frozen, cell_id, _command("success"), tmp_path)
    concurrent = _plan(frozen, cell_id, _command("success"), tmp_path)
    run_cell(first)
    with pytest.raises(AttemptStoreError, match="attempt number is stale"):
        run_cell(concurrent)
    assert len(load_cell_attempts(frozen.directory, cell_id)) == 1


def test_workspace_reservation_is_collision_safe_and_never_reused(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    cell_id = _cell_id(manifest, "prepare")
    plan = _plan(frozen, cell_id, _command("success"), tmp_path)
    assert plan.attempt_id is not None
    collision = frozen.directory / "workspaces" / cell_id / plan.attempt_id
    collision.mkdir(parents=True)
    (collision / "foreign.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(WorkspaceCollisionError):
        run_cell(plan)
    assert (collision / "foreign.txt").read_text(encoding="utf-8") == "do not overwrite"
    assert load_cell_attempts(frozen.directory, cell_id) == ()


def test_forged_or_stale_plan_is_rejected_before_workspace_creation(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    cell_id = _cell_id(manifest, "prepare")
    plan = _plan(frozen, cell_id, _command("success"), tmp_path)
    stale_manifest = replace(plan, manifest_sha256=_digest("0"))
    with pytest.raises(StaleExecutionIdentityError, match="stale"):
        run_cell(stale_manifest)

    forged_environment = replace(
        plan,
        environment=tuple(sorted((*plan.environment, ("SECRET_TOKEN", "forged")))),
        environment_sha256=_digest("0"),
    )
    with pytest.raises(StaleExecutionIdentityError, match="outside the allowlist"):
        run_cell(forged_environment)
    assert not (frozen.directory / "workspaces").exists()


def test_dependency_artifacts_are_revalidated_immediately_before_execution(
    tmp_path: Path,
) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    prepare_id = _cell_id(manifest, "prepare")
    consume_id = _cell_id(manifest, "consume")
    prepare = run_cell(_plan(frozen, prepare_id, _command("success"), tmp_path))
    assert prepare.record is not None and prepare.record.stdout is not None
    consume_plan = _plan(
        frozen,
        consume_id,
        _command("success"),
        tmp_path,
        dependency_attempts={prepare_id: prepare.record.attempt_id},
    )
    stdout_path = frozen.directory.joinpath(*prepare.record.stdout.path.split("/"))
    os.chmod(stdout_path, 0o644)
    stdout_path.write_bytes(b"tampered dependency evidence")

    with pytest.raises(ArtifactVerificationError, match="mismatch"):
        run_cell(consume_plan)
    assert load_cell_attempts(frozen.directory, consume_id) == ()


def test_retry_failed_skips_missing_cells_and_flag_conflicts_are_rejected(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    cell_id = _cell_id(manifest, "prepare")
    retry_missing = _plan(
        frozen,
        cell_id,
        _command("success"),
        tmp_path,
        retry_failed=True,
    )
    assert retry_missing.action == "skip"
    with pytest.raises(RunnerValidationError, match="cannot be combined"):
        _plan(
            frozen,
            cell_id,
            _command("success"),
            tmp_path,
            only_missing=True,
            resume=True,
        )


def test_local_runner_refuses_a_non_local_site_manifest(tmp_path: Path) -> None:
    raw = _campaign_dict(tmp_path / "fail-once.state")
    raw["expected_site"]["site_id"] = "rostam"
    raw["expected_site"]["scheduler"] = "slurm"
    manifest = build_run_manifest(CampaignSpec.from_dict(raw))
    frozen = freeze_run_manifest(manifest, tmp_path / "results")
    with pytest.raises(RunnerValidationError, match="strictly local"):
        plan_cell(
            frozen.directory,
            _cell_id(manifest, "prepare"),
            _command("success"),
            inherited_environment=_inherited_environment(tmp_path),
        )


def test_result_writer_is_atomic_canonical_and_schema_bound(tmp_path: Path) -> None:
    manifest, _frozen = _frozen_run(tmp_path)
    cell = next(cell for cell in manifest.cells if cell.workload_id == "prepare")
    workload = next(workload for workload in manifest.campaign.workloads if workload.id == cell.workload_id)
    result = CellResult.from_dict(
        {
            "schema": CELL_RESULT_SCHEMA,
            "cell_id": cell.id,
            "cell_identity_sha256": cell.identity_sha256,
            "producer_schema": workload.producer_schema,
            "measurement_schema": workload.measurement_schema,
            "measurement": {"latency_us": 1.25},
        }
    )
    destination = tmp_path / "writer" / "result.json"
    destination.parent.mkdir()
    assert write_cell_result(destination, result) == destination
    assert destination.read_bytes() == result.to_json_bytes()
    assert os.stat(destination).st_mode & 0o222 == 0
    loaded = load_cell_result(
        destination,
        cell_id=cell.id,
        cell_identity_sha256=cell.identity_sha256,
        producer_schema=workload.producer_schema,
        measurement_schema=workload.measurement_schema,
        max_bytes=1024,
    )
    assert loaded == result
    with pytest.raises(ResultValidationError, match="already exists"):
        write_cell_result(destination, result)
    with pytest.raises(ResultValidationError, match="ownership"):
        load_cell_result(
            destination,
            cell_id="c-wrong-cell",
            cell_identity_sha256=cell.identity_sha256,
            producer_schema=workload.producer_schema,
            measurement_schema=workload.measurement_schema,
            max_bytes=1024,
        )


def test_malformed_result_is_atomic_parse_failed_evidence(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    cell_id = _cell_id(manifest, "prepare")
    outcome = run_cell(_plan(frozen, cell_id, _command("parse-fail"), tmp_path))
    assert outcome.record is not None
    assert outcome.record.status == "parse-failed"
    assert outcome.record.exit_code == 0
    assert outcome.record.measurement is None
    assert len(outcome.record.partial_outputs) == 1
    assert outcome.record.partial_outputs[0].path.endswith("/result.json")
    assert verify_attempt_artifacts(frozen.directory, outcome.record)


def test_timeout_and_output_budgets_preserve_cancelled_terminal_records(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    prepare_id = _cell_id(manifest, "prepare")
    timeout_plan = plan_cell(
        frozen.directory,
        prepare_id,
        _command("sleep", "--sleep-seconds", "5"),
        inherited_environment=_inherited_environment(tmp_path),
        timeout_seconds=1,
        max_output_bytes=1024,
        max_result_bytes=1024,
    )
    timed_out = run_cell(timeout_plan)
    assert timed_out.record is not None and timed_out.record.status == "cancelled"
    assert "timeout" in (timed_out.record.reason or "")
    assert verify_attempt_artifacts(frozen.directory, timed_out.record)

    # Use a different manifest cell because changing execution policy/argv is
    # intentionally stale for an already-attempted cell.
    failure_id = _cell_id(manifest, "fail-once")
    output_plan = plan_cell(
        frozen.directory,
        failure_id,
        _command("output", "--output-bytes", "100000"),
        inherited_environment=_inherited_environment(tmp_path),
        timeout_seconds=5,
        max_output_bytes=1024,
        max_result_bytes=1024,
    )
    flooded = run_cell(output_plan)
    assert flooded.record is not None and flooded.record.status == "cancelled"
    assert "per-stream budget" in (flooded.record.reason or "")
    assert flooded.record.stdout is not None and flooded.record.stdout.size_bytes == 1024
    assert verify_attempt_artifacts(frozen.directory, flooded.record)


def test_keyboard_interrupt_is_recorded_before_it_is_re_raised(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    cell_id = _cell_id(manifest, "prepare")
    plan = _plan(
        frozen,
        cell_id,
        _command("sleep", "--sleep-seconds", "5"),
        tmp_path,
    )

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("experiments.rostam.harness.runner._poll_sleep", interrupt)
    with pytest.raises(CellRunInterrupted) as captured:
        run_cell(plan)
    outcome = captured.value.outcome
    assert outcome.record is not None and outcome.record.status == "cancelled"
    assert outcome.record.reason == "execution interrupted by KeyboardInterrupt"
    assert load_cell_attempts(frozen.directory, cell_id) == (outcome.record,)
    assert verify_attempt_artifacts(frozen.directory, outcome.record)
