from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest  # type: ignore[import-not-found]

from experiments.rostam.harness import (
    ATTEMPT_SCHEMA,
    CAMPAIGN_SCHEMA,
    ArtifactReference,
    ArtifactVerificationError,
    AttemptRecord,
    CampaignSpec,
    CompletenessStoreError,
    FrozenRun,
    IncompleteCampaignError,
    JSONResourceLimits,
    RunManifest,
    SelectionSnapshot,
    SelectionStoreError,
    build_run_manifest,
    build_selection_snapshot,
    derive_attempt_id,
    evaluate_and_persist_completeness,
    evaluate_completeness,
    freeze_run_manifest,
    freeze_selection_snapshot,
    load_completeness_verdict,
    load_selection_snapshot,
    verify_artifact_reference,
    verify_attempt_artifacts,
    write_attempt_record,
)
from experiments.rostam.harness import completeness as completeness_module
from experiments.rostam.harness import selection as selection_module


def _digest(character: str) -> str:
    return character * 64


def _campaign_dict() -> Dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "run_id": "local-completeness-golden",
        "campaign_id": "local-completeness",
        "repository": {
            "commit": "1" * 40,
            "dirty": False,
            "patch_sha256": None,
            "source_archive_sha256": _digest("a"),
        },
        "inputs": [
            {"id": "source-wheel", "sha256": _digest("b"), "size_bytes": 123},
        ],
        "axes": {
            "configurations": [
                {
                    "id": "config-a",
                    "environment": {},
                    "parameters": {"variant": "a"},
                    "expected_runtime": {},
                },
                {
                    "id": "config-b",
                    "environment": {},
                    "parameters": {"variant": "b"},
                    "expected_runtime": {},
                },
            ],
            "workloads": [
                {
                    "id": "probe",
                    "producer_schema": "commcanary.experiment.local-probe.v1",
                    "measurement_schema": "commcanary.experiment.latency-series.v1",
                    "parameters": {},
                    "depends_on": [],
                }
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
    manifest = build_run_manifest(CampaignSpec.from_dict(_campaign_dict()))
    return manifest, freeze_run_manifest(manifest, tmp_path / "results")


def _write_artifact(run_directory: Path, relative_path: str, content: bytes) -> Dict[str, Any]:
    path = run_directory.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _attempt_record(
    manifest: RunManifest,
    frozen: FrozenRun,
    cell_index: int,
    *,
    attempt_number: int = 1,
    status: str = "success",
) -> AttemptRecord:
    cell = manifest.cells[cell_index]
    attempt_id = derive_attempt_id(attempt_number)
    prefix = f"artifacts/{cell.id}/{attempt_id}"
    stdout = _write_artifact(
        frozen.directory,
        f"{prefix}/stdout.log",
        f"stdout:{cell.id}:{attempt_id}".encode("utf-8"),
    )
    stderr = _write_artifact(
        frozen.directory,
        f"{prefix}/stderr.log",
        f"stderr:{cell.id}:{attempt_id}".encode("utf-8"),
    )
    measurement = (
        _write_artifact(
            frozen.directory,
            f"{prefix}/measurement.json",
            f'{{"cell":"{cell.id}","latency_us":1.25}}'.encode("utf-8"),
        )
        if status == "success"
        else None
    )
    partial_outputs: List[Dict[str, Any]] = []
    if status != "success":
        partial_outputs.append(
            _write_artifact(
                frozen.directory,
                f"{prefix}/partial.json",
                f'{{"cell":"{cell.id}","partial":true}}'.encode("utf-8"),
            )
        )
    exit_code: Any = 0 if status == "success" else 9
    reason: Any = None if status == "success" else f"recorded terminal status {status}"
    if status in {"cancelled", "excluded"}:
        exit_code = None
    raw = {
        "schema": ATTEMPT_SCHEMA,
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "cell_id": cell.id,
        "cell_identity_sha256": cell.identity_sha256,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "status": status,
        "started_at": "2026-07-11T01:02:03.000004Z",
        "finished_at": "2026-07-11T01:02:04.000005Z",
        "command": ["python", "local_probe.py", "--cell", cell.id],
        "observed": {
            "executor": "local",
            "site_id": "local",
            "hostname": "test-host",
            "scheduler": None,
            "job_id": None,
            "nodes": ["test-host"],
            "account": None,
            "partition": None,
            "metadata": {"python": "3.12.0"},
        },
        "exit_code": exit_code,
        "reason": reason,
        "stdout": stdout,
        "stderr": stderr,
        "measurement": measurement,
        "partial_outputs": partial_outputs,
    }
    return AttemptRecord.from_dict(raw)


def _write_all_successes(
    manifest: RunManifest,
    frozen: FrozenRun,
) -> Tuple[AttemptRecord, ...]:
    records = tuple(_attempt_record(manifest, frozen, index) for index in range(len(manifest.cells)))
    for record in records:
        write_attempt_record(frozen.directory, record)
    return records


def _selection_mapping(records: Tuple[AttemptRecord, ...]) -> Mapping[str, str]:
    return {record.cell_id: record.attempt_id for record in reversed(records)}


def _frozen_complete_selection(
    tmp_path: Path,
    *,
    selection_id: str = "paper-primary",
) -> Tuple[RunManifest, FrozenRun, Tuple[AttemptRecord, ...], SelectionSnapshot]:
    manifest, frozen = _frozen_run(tmp_path)
    records = _write_all_successes(manifest, frozen)
    snapshot = build_selection_snapshot(
        frozen.directory,
        selection_id,
        _selection_mapping(records),
    )
    freeze_selection_snapshot(frozen.directory, snapshot)
    return manifest, frozen, records, snapshot


def _issue_codes(verdict: Any) -> set[str]:
    return {issue.code for issue in verdict.issues}


def test_golden_complete_campaign_freezes_selection_and_verdict(tmp_path: Path) -> None:
    manifest, frozen, records, snapshot = _frozen_complete_selection(tmp_path)

    verdict, stored = evaluate_and_persist_completeness(
        frozen.directory,
        snapshot.selection_id,
    )

    assert verdict.complete is True
    assert verdict.allow_incomplete is False
    assert verdict.issues == ()
    assert verdict.expected_cells == len(manifest.cells) == 2
    assert verdict.attempted_cells == 2
    assert verdict.selected_cells == 2
    assert verdict.successful_selected_cells == 2
    assert stored.directory == (frozen.directory / "selections" / snapshot.selection_id / "verdicts" / verdict.sha256)
    loaded_snapshot, frozen_snapshot = load_selection_snapshot(
        frozen.directory,
        snapshot.selection_id,
    )
    loaded_verdict, loaded_store = load_completeness_verdict(
        frozen.directory,
        snapshot.selection_id,
        verdict.sha256,
    )
    assert loaded_snapshot == snapshot
    assert frozen_snapshot.selection_sha256 == snapshot.sha256
    assert loaded_verdict == verdict
    assert loaded_store == stored
    assert tuple(verify_attempt_artifacts(frozen.directory, record) for record in records)
    # Golden commitments: changing selection identity, inventory coverage, or
    # verdict semantics requires an explicit schema/contract decision.
    assert snapshot.sha256 == "d72d241fdc4bbbfc594aa36a7be32b73b2ed7623d6174a8e7774027fa8f64e6b"
    assert verdict.attempt_inventory_sha256 == "0229ae0d8d0ab5403485387b0f90e432cccc5bc014f26469ef8b6e87e452c249"
    assert verdict.sha256 == "5a8bfeaf012d036e999b1ed0549e78f3842a49f1cee2b0a810b49adc92420090"


def test_selection_and_verdict_loaders_apply_the_shared_byte_cap(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _manifest, frozen, _records, snapshot = _frozen_complete_selection(tmp_path)
    verdict, _stored = evaluate_and_persist_completeness(frozen.directory, snapshot.selection_id)
    original_selection_limits = selection_module.DEFAULT_JSON_LIMITS
    monkeypatch.setattr(
        selection_module,
        "DEFAULT_JSON_LIMITS",
        JSONResourceLimits(max_document_bytes=8),
    )
    with pytest.raises(SelectionStoreError, match="max_bytes=8"):
        load_selection_snapshot(frozen.directory, snapshot.selection_id)
    monkeypatch.setattr(selection_module, "DEFAULT_JSON_LIMITS", original_selection_limits)
    monkeypatch.setattr(
        completeness_module,
        "DEFAULT_JSON_LIMITS",
        JSONResourceLimits(max_document_bytes=8),
    )
    with pytest.raises(CompletenessStoreError, match="max_bytes=8"):
        load_completeness_verdict(frozen.directory, snapshot.selection_id, verdict.sha256)


def test_missing_and_unselected_cells_fail_closed_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    first = _attempt_record(manifest, frozen, 0)
    write_attempt_record(frozen.directory, first)
    snapshot = build_selection_snapshot(
        frozen.directory,
        "partial-view",
        {first.cell_id: first.attempt_id},
    )
    freeze_selection_snapshot(frozen.directory, snapshot)

    with pytest.raises(IncompleteCampaignError) as captured:
        evaluate_and_persist_completeness(frozen.directory, snapshot.selection_id)
    assert captured.value.verdict.allow_incomplete is False
    assert captured.value.verdict.complete is False
    assert _issue_codes(captured.value.verdict) == {"missing-attempt", "unselected-cell"}
    assert not (frozen.directory / "selections" / snapshot.selection_id / "verdicts").exists()

    verdict, stored = evaluate_and_persist_completeness(
        frozen.directory,
        snapshot.selection_id,
        allow_incomplete=True,
    )
    assert verdict.complete is False
    assert verdict.allow_incomplete is True
    assert stored.verdict_path.read_bytes() == verdict.to_json_bytes()
    assert _issue_codes(verdict) == {"missing-attempt", "unselected-cell"}


def test_duplicate_selection_is_an_incomplete_verdict_not_silent_last_wins(
    tmp_path: Path,
) -> None:
    _manifest, frozen, _records, snapshot = _frozen_complete_selection(
        tmp_path,
        selection_id="base",
    )
    raw = snapshot.to_dict()
    raw["selection_id"] = "duplicate-view"
    raw["entries"].append(copy.deepcopy(raw["entries"][0]))
    raw["entries"].sort(
        key=lambda entry: (
            entry["cell_id"],
            entry["attempt_id"],
            entry["cell_identity_sha256"],
            entry["attempt_record_sha256"],
        )
    )
    duplicate = SelectionSnapshot.from_dict(raw)
    freeze_selection_snapshot(frozen.directory, duplicate)

    with pytest.raises(IncompleteCampaignError) as captured:
        evaluate_completeness(frozen.directory, duplicate)
    assert "duplicate-selection" in _issue_codes(captured.value.verdict)

    verdict = evaluate_completeness(
        frozen.directory,
        duplicate,
        allow_incomplete=True,
    )
    assert "duplicate-selection" in _issue_codes(verdict)
    assert verdict.successful_selected_cells == 1


def test_stale_selection_hash_and_missing_selected_attempt_fail_closed(tmp_path: Path) -> None:
    _manifest, frozen, _records, snapshot = _frozen_complete_selection(
        tmp_path,
        selection_id="base",
    )
    stale_raw = snapshot.to_dict()
    stale_raw["selection_id"] = "stale-view"
    stale_raw["entries"][0]["attempt_record_sha256"] = _digest("0")
    stale = SelectionSnapshot.from_dict(stale_raw)
    stale_verdict = evaluate_completeness(
        frozen.directory,
        stale,
        allow_incomplete=True,
    )
    assert "stale-selection" in _issue_codes(stale_verdict)

    stale_manifest_raw = snapshot.to_dict()
    stale_manifest_raw["selection_id"] = "stale-manifest-view"
    stale_manifest_raw["manifest_sha256"] = _digest("0")
    stale_manifest = SelectionSnapshot.from_dict(stale_manifest_raw)
    stale_manifest_verdict = evaluate_completeness(
        frozen.directory,
        stale_manifest,
        allow_incomplete=True,
    )
    assert "stale-selection" in _issue_codes(stale_manifest_verdict)

    missing_raw = snapshot.to_dict()
    missing_raw["selection_id"] = "missing-selected-view"
    missing_raw["entries"][0]["attempt_id"] = "a-000002"
    missing_raw["entries"][0]["attempt_record_sha256"] = _digest("0")
    missing = SelectionSnapshot.from_dict(missing_raw)
    missing_verdict = evaluate_completeness(
        frozen.directory,
        missing,
        allow_incomplete=True,
    )
    assert "selected-attempt-missing" in _issue_codes(missing_verdict)


def test_failed_selected_attempt_is_evidence_but_not_complete(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    failed = _attempt_record(manifest, frozen, 0, status="failed")
    succeeded = _attempt_record(manifest, frozen, 1)
    write_attempt_record(frozen.directory, failed)
    write_attempt_record(frozen.directory, succeeded)
    snapshot = build_selection_snapshot(
        frozen.directory,
        "failed-view",
        {
            failed.cell_id: failed.attempt_id,
            succeeded.cell_id: succeeded.attempt_id,
        },
    )

    verdict = evaluate_completeness(
        frozen.directory,
        snapshot,
        allow_incomplete=True,
    )
    assert _issue_codes(verdict) == {"selected-attempt-failed"}
    assert verdict.attempted_cells == 2
    assert verdict.selected_cells == 2
    assert verdict.successful_selected_cells == 1


def test_unexpected_attempt_cell_and_selection_are_never_globbed_into_results(
    tmp_path: Path,
) -> None:
    _manifest, frozen, _records, snapshot = _frozen_complete_selection(
        tmp_path,
        selection_id="base",
    )
    unexpected_directory = frozen.directory / "attempts" / "c-unexpected"
    unexpected_directory.mkdir()
    raw = snapshot.to_dict()
    raw["selection_id"] = "unexpected-view"
    raw["entries"].append(
        {
            "cell_id": "c-unexpected",
            "cell_identity_sha256": _digest("0"),
            "attempt_id": "a-000001",
            "attempt_record_sha256": _digest("1"),
        }
    )
    raw["entries"].sort(
        key=lambda entry: (
            entry["cell_id"],
            entry["attempt_id"],
            entry["cell_identity_sha256"],
            entry["attempt_record_sha256"],
        )
    )
    unexpected = SelectionSnapshot.from_dict(raw)

    verdict = evaluate_completeness(
        frozen.directory,
        unexpected,
        allow_incomplete=True,
    )
    assert {"unexpected-attempt-cell", "unexpected-selection"}.issubset(_issue_codes(verdict))


@pytest.mark.parametrize(  # type: ignore[misc]
    ("mutation", "expected_code"),
    [("delete", "artifact-missing"), ("replace", "artifact-stale")],
)
def test_artifact_content_drift_invalidates_completeness(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    _manifest, frozen, records, snapshot = _frozen_complete_selection(tmp_path)
    reference = records[0].measurement
    assert reference is not None
    artifact_path = frozen.directory.joinpath(*reference.path.split("/"))
    if mutation == "delete":
        artifact_path.unlink()
    else:
        artifact_path.write_bytes(b"stale measurement with a different digest")

    with pytest.raises(IncompleteCampaignError) as captured:
        evaluate_completeness(frozen.directory, snapshot)
    matching = [
        issue
        for issue in captured.value.verdict.issues
        if issue.code == expected_code and issue.artifact_path == reference.path
    ]
    assert len(matching) == 1


def test_artifact_verifier_rejects_symlinks_even_when_content_matches(tmp_path: Path) -> None:
    _manifest, frozen = _frozen_run(tmp_path)
    target = frozen.directory / "artifacts" / "target.bin"
    target.parent.mkdir()
    target.write_bytes(b"target")
    link = frozen.directory / "artifacts" / "link.bin"
    link.symlink_to(target)
    reference = ArtifactReference.from_dict(
        {
            "path": "artifacts/link.bin",
            "sha256": hashlib.sha256(b"target").hexdigest(),
            "size_bytes": 6,
        }
    )
    with pytest.raises(ArtifactVerificationError) as captured:
        verify_artifact_reference(frozen.directory, reference)
    assert captured.value.code == "artifact-invalid"


def test_selection_and_verdict_storage_are_immutable_and_tamper_evident(tmp_path: Path) -> None:
    _manifest, frozen, _records, snapshot = _frozen_complete_selection(tmp_path)
    with pytest.raises(SelectionStoreError, match="already exists"):
        freeze_selection_snapshot(frozen.directory, snapshot)
    verdict, stored = evaluate_and_persist_completeness(
        frozen.directory,
        snapshot.selection_id,
    )
    # Persisting identical content is idempotent and never rewrites it.
    repeated_verdict, repeated_store = evaluate_and_persist_completeness(
        frozen.directory,
        snapshot.selection_id,
    )
    assert repeated_verdict == verdict
    assert repeated_store == stored

    os.chmod(stored.verdict_path, 0o644)
    stored.verdict_path.write_bytes(stored.verdict_path.read_bytes() + b" ")
    with pytest.raises(CompletenessStoreError, match="does not match"):
        load_completeness_verdict(
            frozen.directory,
            snapshot.selection_id,
            verdict.sha256,
        )

    _loaded_selection, stored_selection = load_selection_snapshot(
        frozen.directory,
        snapshot.selection_id,
    )
    os.chmod(stored_selection.selection_path, 0o644)
    stored_selection.selection_path.write_bytes(stored_selection.selection_path.read_bytes() + b" ")
    with pytest.raises(SelectionStoreError, match="does not match"):
        load_selection_snapshot(frozen.directory, snapshot.selection_id)
