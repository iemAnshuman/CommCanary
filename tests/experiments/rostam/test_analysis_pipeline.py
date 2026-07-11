from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest  # type: ignore[import-not-found]

from experiments.rostam.analysis import (
    AGGREGATE_CSV_FILENAME,
    AGGREGATE_JSON_FILENAME,
    PAPER_FRAGMENT_FILENAME,
    ArchiveVerificationError,
    MeasurementValidationError,
    PublicationMismatchError,
    validate_scalar_measurement,
    validate_schema_documents,
    verify_regenerate_compare,
)
from experiments.rostam.analyze import main as analyze_main
from experiments.rostam.harness import (
    ATTEMPT_SCHEMA,
    CAMPAIGN_SCHEMA,
    CELL_RESULT_SCHEMA,
    ArtifactReference,
    AttemptRecord,
    CampaignSpec,
    CellResult,
    FrozenRun,
    IncompleteCampaignError,
    RunManifest,
    SelectionSnapshot,
    build_run_manifest,
    build_selection_snapshot,
    canonical_json_bytes,
    canonical_sha256,
    derive_attempt_id,
    evaluate_completeness,
    freeze_completeness_verdict,
    freeze_run_manifest,
    freeze_selection_snapshot,
    write_attempt_record,
    write_cell_result,
)

GOLDEN_DIRECTORY = Path(__file__).parents[2] / "fixtures" / "experiments" / "golden"
REGENERATION_COMMAND = "python -m experiments.rostam.analyze verify --fixture local-analysis-golden"
MEASUREMENT_SCHEMA = "commcanary.experiment.local.prepare-measurement.v1"
PRODUCER_SCHEMA = "commcanary.experiment.prepare.v1"


def _digest(character: str) -> str:
    return character * 64


def _campaign_dict() -> Dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "run_id": "local-analysis-golden",
        "campaign_id": "local-analysis",
        "repository": {
            "commit": "1" * 40,
            "dirty": False,
            "patch_sha256": None,
            "source_archive_sha256": _digest("a"),
        },
        "inputs": [
            {"id": "golden-producer", "sha256": _digest("b"), "size_bytes": 321},
            {"id": "golden-config", "sha256": _digest("c"), "size_bytes": 123},
        ],
        "axes": {
            "configurations": [
                {
                    "id": "config-a",
                    "environment": {"LOCAL_CONFIG": "A"},
                    "parameters": {"variant": "a"},
                    "expected_runtime": {"python": "3.12"},
                },
                {
                    "id": "config-b",
                    "environment": {"LOCAL_CONFIG": "B"},
                    "parameters": {"variant": "b"},
                    "expected_runtime": {"python": "3.12"},
                },
            ],
            "workloads": [
                {
                    "id": "prepare",
                    "producer_schema": PRODUCER_SCHEMA,
                    "measurement_schema": MEASUREMENT_SCHEMA,
                    "parameters": {"metric": "latency-us"},
                    "depends_on": [],
                }
            ],
            "repetitions": 2,
        },
        "policy": {"aggregation": "median", "tie": {"kind": "max-iqr"}},
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


def _artifact(run_directory: Path, path: Path) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(run_directory).as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


def _value_for_cell(configuration_id: str, repetition: int) -> float:
    base = 10.0 if configuration_id == "config-a" else 20.0
    return base + 2.0 * repetition


def _write_attempt(
    manifest: RunManifest,
    frozen: FrozenRun,
    cell_index: int,
    *,
    attempt_number: int,
    status: str,
) -> AttemptRecord:
    cell = manifest.cells[cell_index]
    configuration = next(
        configuration for configuration in manifest.campaign.configurations if configuration.id == cell.configuration_id
    )
    attempt_id = derive_attempt_id(attempt_number)
    workspace = frozen.directory / "golden-artifacts" / cell.id / attempt_id
    workspace.mkdir(parents=True)
    stdout_path = workspace / "stdout.log"
    stderr_path = workspace / "stderr.log"
    stdout_path.write_text(f"stdout:{cell.id}:{attempt_id}\n", encoding="utf-8")
    stderr_path.write_text(f"stderr:{cell.id}:{attempt_id}\n", encoding="utf-8")
    stdout = _artifact(frozen.directory, stdout_path)
    stderr = _artifact(frozen.directory, stderr_path)
    measurement = None
    if status == "success":
        value_us = _value_for_cell(cell.configuration_id, cell.repetition)
        result = CellResult.from_dict(
            {
                "schema": CELL_RESULT_SCHEMA,
                "cell_id": cell.id,
                "cell_identity_sha256": cell.identity_sha256,
                "producer_schema": PRODUCER_SCHEMA,
                "measurement_schema": MEASUREMENT_SCHEMA,
                "measurement": {
                    "attempt_id": attempt_id,
                    "config_value": configuration.environment.to_value()["LOCAL_CONFIG"],
                    "mode": "success",
                    "samples_us": [value_us - 1.0, value_us, value_us + 1.0],
                    "secret_present": False,
                    "value_us": value_us,
                },
            }
        )
        result_path = write_cell_result(workspace / "result.json", result)
        measurement = _artifact(frozen.directory, result_path)
    environment_sha256 = canonical_sha256(
        {"HOME": "/golden/home", "LOCAL_CONFIG": configuration.environment.to_value()["LOCAL_CONFIG"]}
    )
    execution_identity_sha256 = canonical_sha256(
        {
            "manifest_sha256": manifest.sha256,
            "cell_identity_sha256": cell.identity_sha256,
            "command": ["python", "golden-producer.py", "--cell", cell.id],
            "environment_sha256": environment_sha256,
        }
    )
    raw = {
        "schema": ATTEMPT_SCHEMA,
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.sha256,
        "cell_id": cell.id,
        "cell_identity_sha256": cell.identity_sha256,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "status": status,
        "started_at": f"2026-07-11T01:02:0{attempt_number}.000004Z",
        "finished_at": f"2026-07-11T01:02:0{attempt_number}.500005Z",
        "command": ["python", "golden-producer.py", "--cell", cell.id],
        "observed": {
            "executor": "local",
            "site_id": "local",
            "hostname": "golden-host",
            "scheduler": None,
            "job_id": None,
            "nodes": ["golden-host"],
            "account": None,
            "partition": None,
            "metadata": {
                "environment_sha256": environment_sha256,
                "execution_identity_sha256": execution_identity_sha256,
                "execution_plan_sha256": canonical_sha256({"cell_id": cell.id, "attempt_id": attempt_id}),
            },
        },
        "exit_code": 0 if status == "success" else 7,
        "reason": None if status == "success" else "intentional first-attempt failure",
        "stdout": stdout.to_dict(),
        "stderr": stderr.to_dict(),
        "measurement": None if measurement is None else measurement.to_dict(),
        "partial_outputs": [],
    }
    record = AttemptRecord.from_dict(raw)
    write_attempt_record(frozen.directory, record)
    return record


@dataclass(frozen=True)
class AnalysisFixture:
    manifest: RunManifest
    frozen: FrozenRun
    records: Tuple[AttemptRecord, ...]
    selected_attempts: Dict[str, str]
    selection: SelectionSnapshot
    verdict_sha256: str
    retry_cell_id: str


def _complete_fixture(tmp_path: Path) -> AnalysisFixture:
    manifest = build_run_manifest(CampaignSpec.from_dict(_campaign_dict()))
    frozen = freeze_run_manifest(manifest, tmp_path / "results")
    retry_cell = next(cell for cell in manifest.cells if cell.configuration_id == "config-a" and cell.repetition == 0)
    records = []
    selected_attempts: Dict[str, str] = {}
    for index, cell in enumerate(manifest.cells):
        if cell.id == retry_cell.id:
            records.append(
                _write_attempt(
                    manifest,
                    frozen,
                    index,
                    attempt_number=1,
                    status="failed",
                )
            )
            selected = _write_attempt(
                manifest,
                frozen,
                index,
                attempt_number=2,
                status="success",
            )
            records.append(selected)
        else:
            selected = _write_attempt(
                manifest,
                frozen,
                index,
                attempt_number=1,
                status="success",
            )
            records.append(selected)
        selected_attempts[cell.id] = selected.attempt_id
    selection = build_selection_snapshot(
        frozen.directory,
        "primary",
        selected_attempts,
    )
    freeze_selection_snapshot(frozen.directory, selection)
    verdict = evaluate_completeness(frozen.directory, selection)
    stored = freeze_completeness_verdict(frozen.directory, verdict)
    return AnalysisFixture(
        manifest=manifest,
        frozen=frozen,
        records=tuple(records),
        selected_attempts=selected_attempts,
        selection=selection,
        verdict_sha256=stored.verdict_sha256,
        retry_cell_id=retry_cell.id,
    )


def _freeze_incomplete_selection(
    fixture: AnalysisFixture,
    selection: SelectionSnapshot,
) -> str:
    freeze_selection_snapshot(fixture.frozen.directory, selection)
    verdict = evaluate_completeness(
        fixture.frozen.directory,
        selection,
        allow_incomplete=True,
    )
    return freeze_completeness_verdict(
        fixture.frozen.directory,
        verdict,
    ).verdict_sha256


def test_complete_publication_is_deterministic_deduplicated_and_golden(tmp_path: Path) -> None:
    fixture = _complete_fixture(tmp_path)
    first = verify_regenerate_compare(
        fixture.frozen.directory,
        fixture.selection.selection_id,
        fixture.verdict_sha256,
        tmp_path / "out-a",
        regeneration_command=REGENERATION_COMMAND,
    )
    second = verify_regenerate_compare(
        fixture.frozen.directory,
        fixture.selection.selection_id,
        fixture.verdict_sha256,
        tmp_path / "out-b",
        regeneration_command=REGENERATION_COMMAND,
        golden_directory=GOLDEN_DIRECTORY,
    )

    assert first.aggregate == second.aggregate
    for filename in (AGGREGATE_JSON_FILENAME, AGGREGATE_CSV_FILENAME, PAPER_FRAGMENT_FILENAME):
        assert (tmp_path / "out-a" / filename).read_bytes() == (tmp_path / "out-b" / filename).read_bytes()
    assert second.matched_golden is True
    aggregate_bytes = (tmp_path / "out-a" / AGGREGATE_JSON_FILENAME).read_bytes()
    assert aggregate_bytes == canonical_json_bytes(first.aggregate) + b"\n"
    assert first.aggregate["completeness"]["status"] == "complete"
    assert first.aggregate["selected_cell_count"] == 4
    assert len({row["cell_id"] for row in first.aggregate["selected_cells"]}) == 4
    accounting = first.aggregate["failure_accounting"]
    assert accounting["terminal_attempts"] == 5
    assert accounting["retries"] == 1
    assert accounting["unselected_terminal_attempts"] == 1
    assert accounting["by_status"]["failed"] == 1
    assert accounting["selected_by_status"]["success"] == 4
    retry_rows = [row for row in first.aggregate["selected_cells"] if row["cell_id"] == fixture.retry_cell_id]
    assert len(retry_rows) == 1
    assert retry_rows[0]["attempt_id"] == "a-000002"
    assert [row["median_us"] for row in first.aggregate["aggregates"]] == [11.0, 21.0]
    markdown = (tmp_path / "out-a" / PAPER_FRAGMENT_FILENAME).read_text(encoding="utf-8")
    assert "COMPLETENESS: COMPLETE" in markdown
    assert "## Selected-cell trace" in markdown
    assert retry_rows[0]["environment_sha256"] in markdown
    assert REGENERATION_COMMAND in markdown


def test_incomplete_failed_selection_requires_explicit_opt_in_and_marks_every_output(
    tmp_path: Path,
) -> None:
    fixture = _complete_fixture(tmp_path)
    raw = fixture.selection.to_dict()
    raw["selection_id"] = "failed-selected"
    failed = next(
        record for record in fixture.records if record.cell_id == fixture.retry_cell_id and record.status == "failed"
    )
    for entry in raw["entries"]:
        if entry["cell_id"] == fixture.retry_cell_id:
            entry["attempt_id"] = failed.attempt_id
            entry["attempt_record_sha256"] = failed.sha256
    selection = SelectionSnapshot.from_dict(raw)
    verdict_sha256 = _freeze_incomplete_selection(fixture, selection)

    with pytest.raises(IncompleteCampaignError):
        verify_regenerate_compare(
            fixture.frozen.directory,
            selection.selection_id,
            verdict_sha256,
            tmp_path / "rejected",
            regeneration_command=REGENERATION_COMMAND,
        )
    publication = verify_regenerate_compare(
        fixture.frozen.directory,
        selection.selection_id,
        verdict_sha256,
        tmp_path / "allowed",
        regeneration_command=REGENERATION_COMMAND,
        allow_incomplete=True,
    )
    assert publication.aggregate["completeness"]["status"] == "incomplete"
    assert publication.aggregate["selected_cell_count"] == 3
    csv_text = (tmp_path / "allowed" / AGGREGATE_CSV_FILENAME).read_text(encoding="utf-8")
    markdown = (tmp_path / "allowed" / PAPER_FRAGMENT_FILENAME).read_text(encoding="utf-8")
    assert "campaign,INCOMPLETE,true,selected-attempt-failed" in csv_text
    assert "WARNING — INCOMPLETE EVIDENCE" in markdown
    assert "No performance or ranking claim" in markdown


def test_duplicate_selection_never_duplicates_an_aggregate_row(tmp_path: Path) -> None:
    fixture = _complete_fixture(tmp_path)
    raw = fixture.selection.to_dict()
    raw["selection_id"] = "duplicate"
    raw["entries"].append(copy.deepcopy(raw["entries"][0]))
    raw["entries"].sort(
        key=lambda entry: (
            entry["cell_id"],
            entry["attempt_id"],
            entry["cell_identity_sha256"],
            entry["attempt_record_sha256"],
        )
    )
    selection = SelectionSnapshot.from_dict(raw)
    verdict_sha256 = _freeze_incomplete_selection(fixture, selection)
    publication = verify_regenerate_compare(
        fixture.frozen.directory,
        selection.selection_id,
        verdict_sha256,
        tmp_path / "duplicate-output",
        regeneration_command=REGENERATION_COMMAND,
        allow_incomplete=True,
    )
    assert "duplicate-selection" in publication.aggregate["completeness"]["issue_codes"]
    cell_ids = [row["cell_id"] for row in publication.aggregate["selected_cells"]]
    assert len(cell_ids) == len(set(cell_ids)) == 3


def test_missing_and_unselected_cell_is_excluded_from_allowed_incomplete_output(
    tmp_path: Path,
) -> None:
    fixture = _complete_fixture(tmp_path)
    missing_cell_id = fixture.manifest.cells[-1].id
    shutil.rmtree(fixture.frozen.directory / "attempts" / missing_cell_id)
    selected = {
        cell_id: attempt_id for cell_id, attempt_id in fixture.selected_attempts.items() if cell_id != missing_cell_id
    }
    selection = build_selection_snapshot(
        fixture.frozen.directory,
        "missing-cell",
        selected,
    )
    verdict_sha256 = _freeze_incomplete_selection(fixture, selection)
    publication = verify_regenerate_compare(
        fixture.frozen.directory,
        selection.selection_id,
        verdict_sha256,
        tmp_path / "missing-output",
        regeneration_command=REGENERATION_COMMAND,
        allow_incomplete=True,
    )
    assert {"missing-attempt", "unselected-cell"}.issubset(set(publication.aggregate["completeness"]["issue_codes"]))
    assert missing_cell_id not in {row["cell_id"] for row in publication.aggregate["selected_cells"]}


@pytest.mark.parametrize("tamper_kind", ["artifact", "unexpected-cell"])  # type: ignore[misc]
def test_freshness_recomputation_rejects_tampered_or_unexpected_evidence(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    fixture = _complete_fixture(tmp_path)
    if tamper_kind == "artifact":
        selected_record = next(record for record in fixture.records if record.status == "success")
        assert selected_record.measurement is not None
        path = fixture.frozen.directory.joinpath(*selected_record.measurement.path.split("/"))
        os.chmod(path, 0o644)
        path.write_bytes(b"tampered-result")
    else:
        (fixture.frozen.directory / "attempts" / "c-unexpected").mkdir()
    with pytest.raises(IncompleteCampaignError):
        verify_regenerate_compare(
            fixture.frozen.directory,
            fixture.selection.selection_id,
            fixture.verdict_sha256,
            tmp_path / "tampered-output",
            regeneration_command=REGENERATION_COMMAND,
        )


def test_archive_hash_and_paper_freshness_are_verified(tmp_path: Path) -> None:
    fixture = _complete_fixture(tmp_path)
    archive = tmp_path / "raw.tar"
    archive.write_bytes(b"immutable raw archive")
    correct_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    descriptor = tmp_path / "archive-descriptor.json"
    descriptor_payload = {
        "schema": "commcanary.rostam.raw-archive-descriptor.v1",
        "uri": "urn:commcanary:test-archive",
        "sha256": _digest("0"),
        "size_bytes": archive.stat().st_size,
        "campaigns": [
            {
                "run_id": fixture.manifest.run_id,
                "campaign_id": fixture.manifest.campaign.campaign_id,
                "repository_commit": fixture.manifest.campaign.repository.commit,
                "manifest_sha256": fixture.frozen.manifest_sha256,
                "selection_id": fixture.selection.selection_id,
                "selection_sha256": fixture.selection.sha256,
                "verdict_sha256": fixture.verdict_sha256,
            }
        ],
    }
    descriptor.write_bytes(canonical_json_bytes(descriptor_payload))
    with pytest.raises(ArchiveVerificationError, match="mismatch"):
        verify_regenerate_compare(
            fixture.frozen.directory,
            fixture.selection.selection_id,
            fixture.verdict_sha256,
            tmp_path / "bad-archive",
            regeneration_command=REGENERATION_COMMAND,
            archive_descriptor=descriptor,
            raw_archive=archive,
        )
    descriptor_payload["sha256"] = correct_sha256
    descriptor.write_bytes(canonical_json_bytes(descriptor_payload))
    publication = verify_regenerate_compare(
        fixture.frozen.directory,
        fixture.selection.selection_id,
        fixture.verdict_sha256,
        tmp_path / "fresh-output",
        regeneration_command=REGENERATION_COMMAND,
        archive_descriptor=descriptor,
        raw_archive=archive,
    )
    assert publication.aggregate["provenance"]["raw_archive"]["verified"] is True
    assert any(
        document["schema"] == "commcanary.rostam.raw-archive-descriptor.v1"
        for document in publication.aggregate["provenance"]["schema_documents"]
    )
    assert str(archive) not in canonical_json_bytes(publication.aggregate).decode("utf-8")

    descriptor_payload["campaigns"][0]["selection_id"] = "different-selection"
    descriptor.write_bytes(canonical_json_bytes(descriptor_payload))
    with pytest.raises(ArchiveVerificationError, match="campaign identities"):
        verify_regenerate_compare(
            fixture.frozen.directory,
            fixture.selection.selection_id,
            fixture.verdict_sha256,
            tmp_path / "wrong-evidence-identity",
            regeneration_command=REGENERATION_COMMAND,
            archive_descriptor=descriptor,
            raw_archive=archive,
        )
    descriptor_payload["campaigns"][0]["selection_id"] = fixture.selection.selection_id
    descriptor_payload["uri"] = "file:///private/raw.tar"
    descriptor.write_bytes(canonical_json_bytes(descriptor_payload))
    with pytest.raises(ArchiveVerificationError, match="immutable non-local URI"):
        verify_regenerate_compare(
            fixture.frozen.directory,
            fixture.selection.selection_id,
            fixture.verdict_sha256,
            tmp_path / "local-archive-uri",
            regeneration_command=REGENERATION_COMMAND,
            archive_descriptor=descriptor,
            raw_archive=archive,
        )
    descriptor_payload["uri"] = "urn:commcanary:test-archive"
    descriptor.write_bytes(b"{" + b" " * (1024 * 1024))
    with pytest.raises(ArchiveVerificationError, match="cannot decode archive descriptor"):
        verify_regenerate_compare(
            fixture.frozen.directory,
            fixture.selection.selection_id,
            fixture.verdict_sha256,
            tmp_path / "oversized-descriptor",
            regeneration_command=REGENERATION_COMMAND,
            archive_descriptor=descriptor,
            raw_archive=archive,
        )
    descriptor.write_bytes(canonical_json_bytes(descriptor_payload))

    stale_golden = tmp_path / "stale-golden"
    shutil.copytree(tmp_path / "fresh-output", stale_golden)
    paper = stale_golden / PAPER_FRAGMENT_FILENAME
    paper.write_text(paper.read_text(encoding="utf-8") + "stale edit\n", encoding="utf-8")
    with pytest.raises(PublicationMismatchError, match="paper-fragment.md"):
        verify_regenerate_compare(
            fixture.frozen.directory,
            fixture.selection.selection_id,
            fixture.verdict_sha256,
            tmp_path / "fresh-output-2",
            regeneration_command=REGENERATION_COMMAND,
            archive_descriptor=descriptor,
            raw_archive=archive,
            golden_directory=stale_golden,
        )


def test_legacy_glob_analysis_requires_acknowledgement_and_watermarks_outputs(tmp_path: Path) -> None:
    results = tmp_path / "legacy-results"
    results.mkdir()
    (results / "measurement.json").write_text(
        json.dumps(
            {
                "workload": "full",
                "config": "nccl-2.20.5-default",
                "metrics": {"median_us": 10.0, "iqr_us": 1.0, "wall_time_s": 0.5},
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "legacy.json"
    output_md = tmp_path / "legacy.md"
    arguments = [
        "legacy",
        "--results-dir",
        str(results),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    with pytest.raises(SystemExit) as error:
        analyze_main(arguments)
    assert error.value.code == 2
    assert not output_json.exists()
    assert not output_md.exists()

    assert analyze_main([*arguments, "--unsafe-legacy-glob-analysis"]) == 0
    legacy_json = json.loads(output_json.read_text(encoding="utf-8"))
    assert legacy_json["trust"]["status"] == "unsafe-legacy-unverified"
    assert legacy_json["trust"]["publication_allowed"] is False
    markdown = output_md.read_text(encoding="utf-8")
    assert "UNSAFE LEGACY ANALYSIS" in markdown
    assert "UNTRUSTED / NOT FOR PUBLICATION" in markdown


def test_local_measurement_schemas_are_distinct_strict_and_committed() -> None:
    documents = validate_schema_documents()
    assert len(documents) == 10
    assert len({document["schema"] for document in documents}) == 10
    assert all(len(document["sha256"]) == 64 for document in documents)
    valid = {
        "attempt_id": "a-000001",
        "config_value": "A",
        "mode": "success",
        "samples_us": [9.0, 10.0, 11.0],
        "secret_present": False,
        "value_us": 10.0,
    }
    measurement = validate_scalar_measurement(
        MEASUREMENT_SCHEMA,
        PRODUCER_SCHEMA,
        "a-000001",
        valid,
    )
    assert measurement.value_us == 10.0
    invalid = dict(valid)
    invalid["unknown"] = 1
    with pytest.raises(MeasurementValidationError, match="unknown fields"):
        validate_scalar_measurement(
            MEASUREMENT_SCHEMA,
            PRODUCER_SCHEMA,
            "a-000001",
            invalid,
        )
    invalid_median = dict(valid)
    invalid_median["value_us"] = 99.0
    with pytest.raises(MeasurementValidationError, match="median"):
        validate_scalar_measurement(
            MEASUREMENT_SCHEMA,
            PRODUCER_SCHEMA,
            "a-000001",
            invalid_median,
        )


def test_analyze_cli_defaults_to_the_completeness_gated_path(
    tmp_path: Path,
    capsys: Any,
) -> None:
    from experiments.rostam.analyze import main

    fixture = _complete_fixture(tmp_path)
    output = tmp_path / "cli-output"
    exit_code = main(
        [
            "--run-directory",
            str(fixture.frozen.directory),
            "--selection-id",
            fixture.selection.selection_id,
            "--verdict-sha256",
            fixture.verdict_sha256,
            "--output-directory",
            str(output),
            "--regeneration-command",
            REGENERATION_COMMAND,
        ]
    )
    assert exit_code == 0
    assert "sha256=" in capsys.readouterr().out
    assert {path.name for path in output.iterdir()} == {
        AGGREGATE_JSON_FILENAME,
        AGGREGATE_CSV_FILENAME,
        PAPER_FRAGMENT_FILENAME,
    }
