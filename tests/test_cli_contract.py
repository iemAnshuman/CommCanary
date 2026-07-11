from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import commcanary.cli as cli_module
from commcanary.cli import (
    EXIT_APPLICATION_ERROR,
    EXIT_CHILD_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_NEGATIVE_RESULT,
    EXIT_SUCCESS,
    EXIT_USAGE,
    _build_parser,
    main,
)
from commcanary.formats import CANONICAL_JSON_VERSION, format_capabilities
from commcanary.replay import SIMULATION_MODEL_VERSION
from commcanary.version import __version__


def test_cli_exit_code_table_is_stable() -> None:
    assert (
        EXIT_SUCCESS,
        EXIT_NEGATIVE_RESULT,
        EXIT_USAGE,
        EXIT_APPLICATION_ERROR,
        EXIT_CHILD_FAILURE,
        EXIT_INTERRUPTED,
    ) == (0, 1, 2, 3, 4, 130)


def test_version_reports_package_formats_canonicalization_and_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        _build_parser().parse_args(["--version"])

    assert raised.value.code == EXIT_SUCCESS
    output = capsys.readouterr().out
    assert f"commcanary {__version__}" in output
    assert CANONICAL_JSON_VERSION in output
    assert SIMULATION_MODEL_VERSION in output
    for capability in format_capabilities():
        assert capability.format_id in output


def test_help_lists_the_documented_command_surface(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        _build_parser().parse_args(["--help"])

    assert raised.value.code == EXIT_SUCCESS
    output = capsys.readouterr().out
    for command in (
        "compile",
        "replay",
        "compare",
        "verify-fidelity",
        "verify-behavior",
        "baseline",
        "reduce",
        "import-kineto",
        "export-param",
        "verify-report",
        "capture",
        "render-html",
        "report",
    ):
        assert command in output


def test_application_error_is_distinct_from_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = str(tmp_path / "missing.trace.json")
    result = main(["compile", missing, "--output", str(tmp_path / "output.json")])

    assert result == EXIT_APPLICATION_ERROR
    assert "does not exist" in capsys.readouterr().err


def test_capture_child_failure_reports_tool_and_child_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "capture",
            "--output",
            str(tmp_path / "unused.trace.json"),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(23)",
        ]
    )

    assert result == EXIT_CHILD_FAILURE
    assert "child code 23" in capsys.readouterr().err


def test_json_diagnostics_keep_stderr_machine_readable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--diagnostics-json",
            "compile",
            str(tmp_path / "missing.trace.json"),
            "--output",
            str(tmp_path / "output.json"),
        ]
    )

    assert result == EXIT_APPLICATION_ERROR
    rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert rows[0] == {
        "command": "compile",
        "event": "started",
        "exit_code": EXIT_SUCCESS,
        "format": "commcanary.diagnostic.v1",
    }
    assert rows[1]["command"] == "compile"
    assert rows[1]["event"] == "error"
    assert rows[1]["exit_code"] == EXIT_APPLICATION_ERROR
    assert rows[1]["format"] == "commcanary.diagnostic.v1"
    assert rows[1]["message"] == f"{tmp_path / 'missing.trace.json'} does not exist"
    assert rows[1]["elapsed_seconds"] >= 0.0


def test_json_child_diagnostic_records_original_code_and_tool_completion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--diagnostics-json",
            "capture",
            "--output",
            str(tmp_path / "unused.trace.json"),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(29)",
        ]
    )

    assert result == EXIT_CHILD_FAILURE
    rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [row["event"] for row in rows] == ["started", "child_failure", "completed"]
    assert rows[1]["child_returncode"] == 29
    assert rows[1]["exit_code"] == EXIT_CHILD_FAILURE
    assert rows[2]["exit_code"] == EXIT_CHILD_FAILURE
    assert rows[2]["outcome"] == "error"
    assert rows[2]["elapsed_seconds"] >= 0.0


def test_baseline_rejects_method_inapplicable_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = Path(__file__).parent / "fixtures" / "contracts" / "trace.valid.json"
    output = tmp_path / "baseline.json"

    result = main(
        [
            "baseline",
            str(trace),
            "--output",
            str(output),
            "--method",
            "isolated",
            "--seed",
            "7",
        ]
    )

    assert result == EXIT_APPLICATION_ERROR
    assert "does not accept --seed" in capsys.readouterr().err
    assert not output.exists()


def test_render_html_is_primary_and_report_alias_is_deprecated(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = Path(__file__).parent / "fixtures" / "contracts" / "report.valid.json"
    primary = tmp_path / "primary.html"
    compatible = tmp_path / "compatible.html"

    assert main(["render-html", str(report), "--output", str(primary)]) == EXIT_SUCCESS
    primary_capture = capsys.readouterr()
    assert primary_capture.err == ""
    assert main(["report", str(report), "--output", str(compatible)]) == EXIT_SUCCESS
    compatible_capture = capsys.readouterr()

    assert primary.read_bytes() == compatible.read_bytes()
    assert "deprecated" in compatible_capture.err
    assert "render-html" in compatible_capture.err


def test_expensive_commands_emit_bounded_work_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = Path(__file__).parent / "fixtures" / "contracts" / "trace.valid.json"
    canary = tmp_path / "searched.canary.json"
    reduced = tmp_path / "reduced.trace.json"

    assert (
        main(
            [
                "--diagnostics-json",
                "compile",
                str(trace),
                "--output",
                str(canary),
                "--behavior-search",
                "--behavior-search-min-sample-limit",
                "2",
                "--timing-sample-limit",
                "2",
            ]
        )
        == EXIT_SUCCESS
    )
    search_rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    search_progress = [row for row in search_rows if row["event"] == "progress"]
    assert [row["status"] for row in search_progress] == ["started", "completed"]
    assert search_progress[0]["uniform_candidates_planned"] == 1
    assert search_progress[1]["uniform_candidates_evaluated"] == 1
    assert search_progress[1]["accepted_candidates"] == 1

    assert (
        main(
            [
                "--diagnostics-json",
                "reduce",
                str(trace),
                "--output",
                str(reduced),
                "--max-oracle-calls",
                "1",
            ]
        )
        == EXIT_SUCCESS
    )
    reduce_rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    reduce_progress = [row for row in reduce_rows if row["event"] == "progress"]
    assert [row["status"] for row in reduce_progress] == ["started", "completed"]
    assert reduce_progress[0]["oracle_call_budget"] == 1
    assert reduce_progress[1]["oracle_calls"] == 0
    assert reduce_progress[1]["budget_exhausted"] is False


def test_interrupt_diagnostic_is_terminal_and_records_elapsed_time(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "_cmd_compile", interrupt)
    result = cli_module.main(
        [
            "--diagnostics-json",
            "compile",
            str(tmp_path / "unused.trace.json"),
            "--output",
            str(tmp_path / "unused.canary.json"),
        ]
    )

    assert result == EXIT_INTERRUPTED
    rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [row["event"] for row in rows] == ["started", "interrupted"]
    assert rows[-1]["exit_code"] == EXIT_INTERRUPTED
    assert rows[-1]["elapsed_seconds"] >= 0.0
