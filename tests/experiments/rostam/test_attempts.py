from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest  # type: ignore[import-not-found]

from experiments.rostam.harness import (
    ATTEMPT_SCHEMA,
    CAMPAIGN_SCHEMA,
    ArtifactReference,
    AttemptRecord,
    AttemptStoreError,
    AttemptValidationError,
    CampaignSpec,
    FrozenRun,
    JSONResourceLimits,
    RunManifest,
    build_run_manifest,
    derive_attempt_id,
    freeze_run_manifest,
    load_attempt_record,
    load_cell_attempts,
    select_terminal_attempt,
    utc_timestamp,
    write_attempt_record,
)
from experiments.rostam.harness import attempts as attempts_module


def _digest(character: str) -> str:
    return character * 64


def _campaign_dict(*, repetitions: int = 1) -> Dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "run_id": "local-golden-run",
        "campaign_id": "local-golden",
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
                    "id": "local-config",
                    "environment": {},
                    "parameters": {},
                    "expected_runtime": {},
                }
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
            "repetitions": repetitions,
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


def _artifact(path: str, character: str = "c") -> Dict[str, Any]:
    return {"path": path, "sha256": _digest(character), "size_bytes": 17}


def _attempt_dict(
    manifest: RunManifest,
    *,
    attempt_number: int = 1,
    status: str = "success",
    exit_code: Any = 0,
    reason: Any = None,
) -> Dict[str, Any]:
    cell = manifest.cells[0]
    return {
        "schema": ATTEMPT_SCHEMA,
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "cell_id": cell.id,
        "cell_identity_sha256": cell.identity_sha256,
        "attempt_id": derive_attempt_id(attempt_number),
        "attempt_number": attempt_number,
        "status": status,
        "started_at": "2026-07-11T01:02:03.000004Z",
        "finished_at": "2026-07-11T01:02:04.000005Z",
        "command": ["python", "-c", "print('probe')"],
        "observed": {
            "executor": "local",
            "site_id": "local",
            "hostname": "test-host",
            "scheduler": None,
            "job_id": None,
            "nodes": ["test-host"],
            "account": None,
            "partition": None,
            "metadata": {"python": "3.12.0", "pid": 123},
        },
        "exit_code": exit_code,
        "reason": reason,
        "stdout": _artifact("artifacts/a-000001/stdout.log", "d"),
        "stderr": _artifact("artifacts/a-000001/stderr.log", "e"),
        "measurement": _artifact("artifacts/a-000001/measurement.json", "f") if status == "success" else None,
        "partial_outputs": [],
    }


def test_success_record_round_trips_and_freezes_atomically(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    record = AttemptRecord.from_dict(_attempt_dict(manifest))

    stored = write_attempt_record(frozen.directory, record)

    assert stored.directory == frozen.directory / "attempts" / record.cell_id / "a-000001"
    assert stored.record_path.read_bytes() == record.to_json_bytes()
    assert stored.record_sha256 == hashlib.sha256(record.to_json_bytes()).hexdigest()
    assert stored.checksum_path.read_text(encoding="ascii") == (f"{record.sha256}  attempt.json\n")
    assert os.stat(stored.record_path).st_mode & 0o222 == 0
    loaded, loaded_store = load_attempt_record(frozen.directory, record.cell_id, record.attempt_id)
    assert loaded == record
    assert loaded_store == stored
    assert load_cell_attempts(frozen.directory, record.cell_id) == (record,)
    assert select_terminal_attempt((record,), "a-000001") is record


def test_attempt_loader_applies_the_shared_byte_cap(tmp_path: Path, monkeypatch: Any) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    record = AttemptRecord.from_dict(_attempt_dict(manifest))
    write_attempt_record(frozen.directory, record)
    monkeypatch.setattr(
        attempts_module,
        "DEFAULT_JSON_LIMITS",
        JSONResourceLimits(max_document_bytes=8),
    )
    with pytest.raises(AttemptStoreError, match="max_bytes=8"):
        load_attempt_record(frozen.directory, record.cell_id, record.attempt_id)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("status", "exit_code"),
    [
        ("failed", 23),
        ("parse-failed", 0),
        ("cancelled", None),
        ("excluded", None),
    ],
)
def test_every_non_success_terminal_status_retains_failure_evidence(
    tmp_path: Path,
    status: str,
    exit_code: Any,
) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    raw = _attempt_dict(
        manifest,
        status=status,
        exit_code=exit_code,
        reason=f"recorded {status}",
    )
    raw["partial_outputs"] = [_artifact("partial/raw-result.json", "a")]
    record = AttemptRecord.from_dict(raw)

    write_attempt_record(frozen.directory, record)
    loaded = load_cell_attempts(frozen.directory, record.cell_id)

    assert loaded == (record,)
    assert loaded[0].reason == f"recorded {status}"
    assert loaded[0].stdout is not None
    assert loaded[0].stderr is not None
    assert loaded[0].partial_outputs[0].path == "partial/raw-result.json"


@pytest.mark.parametrize(  # type: ignore[misc]
    ("mutation", "message"),
    [
        ({"measurement": None}, "successful attempt requires"),
        ({"reason": "not success"}, "successful attempt requires"),
        ({"status": "failed", "measurement": None, "reason": None, "exit_code": 9}, "requires a reason"),
        ({"status": "failed", "measurement": None, "reason": "bad", "exit_code": 0}, "exit_code 0"),
        ({"status": "excluded", "measurement": None, "reason": "policy", "exit_code": 1}, "exit code"),
        ({"status": "running", "measurement": None}, "must be success"),
    ],
)
def test_status_contract_rejects_ambiguous_terminal_records(
    tmp_path: Path,
    mutation: Dict[str, Any],
    message: str,
) -> None:
    manifest, _frozen = _frozen_run(tmp_path)
    raw = _attempt_dict(manifest)
    raw.update(mutation)
    with pytest.raises(AttemptValidationError, match=message):
        AttemptRecord.from_dict(raw)


def test_retries_append_contiguously_and_preserve_every_prior_attempt(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    failed = AttemptRecord.from_dict(
        _attempt_dict(
            manifest,
            status="failed",
            exit_code=7,
            reason="probe process failed",
        )
    )
    succeeded = AttemptRecord.from_dict(_attempt_dict(manifest, attempt_number=2))
    first_store = write_attempt_record(frozen.directory, failed)
    first_bytes = first_store.record_path.read_bytes()

    write_attempt_record(frozen.directory, succeeded)

    assert first_store.record_path.read_bytes() == first_bytes
    assert load_cell_attempts(frozen.directory, failed.cell_id) == (failed, succeeded)
    assert select_terminal_attempt((failed, succeeded), "a-000001") == failed
    assert select_terminal_attempt((failed, succeeded), "a-000002") == succeeded

    with pytest.raises(AttemptStoreError, match="expected 3"):
        write_attempt_record(frozen.directory, succeeded)
    skipped = AttemptRecord.from_dict(_attempt_dict(manifest, attempt_number=4))
    with pytest.raises(AttemptStoreError, match="expected 3"):
        write_attempt_record(frozen.directory, skipped)


def test_selection_is_explicit_and_must_resolve_exactly_one_record(tmp_path: Path) -> None:
    manifest, _frozen = _frozen_run(tmp_path)
    record = AttemptRecord.from_dict(_attempt_dict(manifest))
    with pytest.raises(AttemptValidationError, match="exactly one"):
        select_terminal_attempt((record,), "a-000002")
    with pytest.raises(AttemptValidationError, match="exactly one"):
        select_terminal_attempt((record, record), "a-000001")
    with pytest.raises(AttemptValidationError, match="invalid selected"):
        select_terminal_attempt((record,), "../escape")


@pytest.mark.parametrize(  # type: ignore[misc]
    ("field", "replacement", "message"),
    [
        ("run_id", "another-run", "run_id"),
        ("manifest_sha256", _digest("0"), "manifest_sha256"),
        ("cell_identity_sha256", _digest("0"), "cell_identity_sha256"),
        ("cell_id", "c-unknown-cell", "unknown cell"),
    ],
)
def test_store_rejects_forged_or_stale_manifest_bindings(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    raw = _attempt_dict(manifest)
    raw[field] = replacement
    record = AttemptRecord.from_dict(raw)
    with pytest.raises(AttemptValidationError, match=message):
        write_attempt_record(frozen.directory, record)


@pytest.mark.parametrize(  # type: ignore[misc]
    "path",
    ["../escape", "/absolute/result.json", "a//b", "a/./b", "a\\b"],
)
def test_artifact_references_reject_non_normalized_or_escaping_paths(path: str) -> None:
    with pytest.raises(AttemptValidationError):
        ArtifactReference.from_dict(_artifact(path))


def test_observed_metadata_and_attempt_identity_are_strict(tmp_path: Path) -> None:
    manifest, _frozen = _frozen_run(tmp_path)

    job_without_scheduler = _attempt_dict(manifest)
    job_without_scheduler["observed"]["job_id"] = "123"
    with pytest.raises(AttemptValidationError, match="requires a scheduler"):
        AttemptRecord.from_dict(job_without_scheduler)

    wrong_attempt_id = _attempt_dict(manifest)
    wrong_attempt_id["attempt_id"] = "a-000002"
    with pytest.raises(AttemptValidationError, match="does not match"):
        AttemptRecord.from_dict(wrong_attempt_id)

    reversed_time = _attempt_dict(manifest)
    reversed_time["finished_at"] = "2026-07-11T01:02:02.000005Z"
    with pytest.raises(AttemptValidationError, match="precedes"):
        AttemptRecord.from_dict(reversed_time)

    unknown = _attempt_dict(manifest)
    unknown["observed"]["slurm_cluster"] = "surprise"
    with pytest.raises(AttemptValidationError, match="unknown fields"):
        AttemptRecord.from_dict(unknown)


def test_duplicate_partial_paths_and_duplicate_json_fields_are_rejected(tmp_path: Path) -> None:
    manifest, _frozen = _frozen_run(tmp_path)
    duplicated = _attempt_dict(manifest)
    duplicated["partial_outputs"] = [
        _artifact("partial/result.json", "a"),
        _artifact("partial/result.json", "b"),
    ]
    with pytest.raises(AttemptValidationError, match="duplicate paths"):
        AttemptRecord.from_dict(duplicated)

    payload = AttemptRecord.from_dict(_attempt_dict(manifest)).to_json_bytes().decode("utf-8")
    malicious = payload.replace('{"attempt_id":', '{"schema":"duplicate","attempt_id":', 1)
    with pytest.raises(Exception, match="duplicate"):
        AttemptRecord.from_json_bytes(malicious.encode("utf-8"))


def test_loader_detects_record_tampering_and_unexpected_entries(tmp_path: Path) -> None:
    manifest, frozen = _frozen_run(tmp_path)
    record = AttemptRecord.from_dict(_attempt_dict(manifest))
    stored = write_attempt_record(frozen.directory, record)
    os.chmod(stored.record_path, 0o644)
    stored.record_path.write_bytes(stored.record_path.read_bytes() + b" ")
    with pytest.raises(AttemptStoreError, match="does not match"):
        load_attempt_record(frozen.directory, record.cell_id, record.attempt_id)

    second_manifest, second_frozen = _frozen_run(tmp_path / "second")
    second_record = AttemptRecord.from_dict(_attempt_dict(second_manifest))
    write_attempt_record(second_frozen.directory, second_record)
    cell_directory = second_frozen.directory / "attempts" / second_record.cell_id
    (cell_directory / "unexpected.txt").write_text("not evidence", encoding="utf-8")
    with pytest.raises(AttemptStoreError, match="unexpected entry"):
        load_cell_attempts(second_frozen.directory, second_record.cell_id)


def test_loader_rejects_attempt_relocated_under_another_known_cell(tmp_path: Path) -> None:
    manifest = build_run_manifest(CampaignSpec.from_dict(_campaign_dict(repetitions=2)))
    frozen = freeze_run_manifest(manifest, tmp_path / "results")
    record = AttemptRecord.from_dict(_attempt_dict(manifest))
    stored = write_attempt_record(frozen.directory, record)
    other_cell = next(cell for cell in manifest.cells if cell.id != record.cell_id)
    other_cell_directory = frozen.directory / "attempts" / other_cell.id
    other_cell_directory.mkdir()
    shutil.copytree(stored.directory, other_cell_directory / record.attempt_id)

    with pytest.raises(AttemptStoreError, match="does not match record cell_id"):
        load_cell_attempts(frozen.directory, other_cell.id)


def test_timestamp_helper_normalizes_offsets_and_rejects_naive_values() -> None:
    from datetime import datetime, timedelta, timezone

    offset = timezone(timedelta(hours=5, minutes=30))
    value = datetime(2026, 7, 11, 12, 30, 0, 1234, tzinfo=offset)
    assert utc_timestamp(value) == "2026-07-11T07:00:00.001234Z"
    with pytest.raises(AttemptValidationError, match="timezone-aware"):
        utc_timestamp(datetime(2026, 7, 11))


def test_record_is_a_detached_immutable_value(tmp_path: Path) -> None:
    manifest, _frozen = _frozen_run(tmp_path)
    raw = _attempt_dict(manifest)
    record = AttemptRecord.from_dict(raw)
    raw["command"][0] = "mutated"
    raw["observed"]["metadata"]["pid"] = 999
    assert record.command[0] == "python"
    assert record.observed.metadata.to_value()["pid"] == 123
    with pytest.raises(Exception):
        record.status = "failed"  # type: ignore[misc]
