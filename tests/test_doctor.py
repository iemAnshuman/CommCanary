from __future__ import annotations

import copy
from pathlib import Path

import pytest

from commcanary.cli import main as cli_main
from commcanary.errors import CommCanaryError, SchemaError
from commcanary.formats import DOCTOR_REPORT_FORMAT
from commcanary.resources import DEFAULT_RESOURCE_LIMITS
from commcanary.schema import load_json
from commcanary.services import (
    import_failure_readiness_report,
    qualification_readiness_report,
    validate_doctor_report,
)
from tests.builders import qualification_trace


def test_doctor_reports_exact_work_readiness_without_physical_overclaim() -> None:
    report = qualification_readiness_report(qualification_trace())

    assert report["format"] == DOCTOR_REPORT_FORMAT
    assert report["status"] == "qualification_ready"
    assert report["supported_product_tier"] == "qualification_ready"
    assert report["summary"]["qualification_readiness_pct"] == 100.0
    assert report["claims"] == {
        "source_correspondence": "verified",
        "physical_execution": "not_observed",
        "physical_conformance": "unproven",
        "physical_decision_fidelity": "not_measured",
        "producer_authenticity": "unsigned",
    }
    assert report["estimates"]["qualification_bundle"]["status"] == "lower_bound"
    assert report["estimates"]["execution_memory"]["max_rank_bytes"] > 0
    assert report["estimates"]["physical_runtime"]["status"] == "not_estimable"
    assert report["privacy"]["disclosure_level"] == "high"
    assert "rank_local_gemm_shapes" in report["privacy"]["exposed"]
    validate_doctor_report(report)


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("overlap", "unknown_compute_overlap"),
        ("shape", "unknown_message_shape"),
        ("reduction", "unknown_collective_semantics"),
        ("recipe", "missing_or_unsupported_rank_local_compute_recipe"),
    ],
)
def test_doctor_reports_event_level_readiness_failures(mutation: str, reason_code: str) -> None:
    trace = qualification_trace()
    event = trace["events"][0]
    if mutation == "overlap":
        event.pop("compute_overlap_us")
        event["compute_overlap_unknown"] = True
    elif mutation == "shape":
        event["metadata"]["kineto_message_shape_status"] = "unavailable_missing_size"
    elif mutation == "reduction":
        event.pop("reduction_op")
    else:
        event["metadata"]["kineto_compute_recipe_status"] = "unavailable_missing_explicit_wait"

    report = qualification_readiness_report(trace)

    failed = {check["reason_code"]: check for check in report["checks"] if check["status"] == "fail"}
    assert reason_code in failed
    assert failed[reason_code]["locations"][0] == {
        "event_index": 0,
        "event_id": "decode-0",
        "ranks": [0, 1, 2, 3],
    }
    assert report["status"] == "not_qualification_ready"
    assert report["summary"]["qualification_readiness_pct"] < 100.0


def test_doctor_records_complete_source_profile_inventory() -> None:
    trace = qualification_trace()
    trace["system"]["kineto_imported_ranks"] = [0, 1, 2, 3]
    trace["system"]["kineto_source_profiles"] = [
        {"rank": rank, "sha256": str(rank + 1) * 64, "size_bytes": 10} for rank in range(4)
    ]

    report = qualification_readiness_report(trace)

    check = next(check for check in report["checks"] if check["check_id"] == "source_profile_inventory")
    assert check["status"] == "pass"
    assert check["reason_code"] == "ready"


def test_import_failure_report_uses_stable_reason_and_validates() -> None:
    report = import_failure_readiness_report(
        CommCanaryError("multi-rank Kineto import requires an explicit clock alignment claim")
    )

    assert report["status"] == "not_importable"
    assert report["checks"][0]["reason_code"] == "clock_alignment_unproven"
    validate_doctor_report(report)
    malformed = copy.deepcopy(report)
    malformed["checks"][0]["status"] = "maybe"
    with pytest.raises(SchemaError, match="status is unsupported"):
        validate_doctor_report(malformed)


def test_doctor_cli_prints_readiness_and_can_write_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "doctor.json"
    monkeypatch.setattr(
        "commcanary.command_line.commands._import_kineto_profiles",
        lambda _args: (qualification_trace(), DEFAULT_RESOURCE_LIMITS),
    )

    assert cli_main(["doctor", "rank0.json", "rank1.json", "--assume-shared-clock", "-o", str(output)]) == 0

    report = load_json(str(output))
    assert report["status"] == "qualification_ready"
    rendered = capsys.readouterr().out
    assert "Qualification readiness: 100.0%" in rendered
    assert "Result: qualification_ready" in rendered
    assert "physical" not in rendered.lower()


def test_doctor_cli_returns_negative_result_for_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_args: object) -> object:
        raise CommCanaryError("clock alignment is missing")

    monkeypatch.setattr("commcanary.command_line.commands._import_kineto_profiles", fail)

    assert cli_main(["doctor", "rank0.json", "rank1.json"]) == 1
    rendered = capsys.readouterr().out
    assert "not_importable" in rendered
    assert "clock_alignment_unproven" in rendered
