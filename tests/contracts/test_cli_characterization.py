from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from commcanary.formats import CANONICAL_JSON_VERSION, format_capabilities
from commcanary.replay import SIMULATION_MODEL_VERSION

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "contracts"
COMMANDS = (
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
)


def _load_vectors() -> Mapping[str, Any]:
    with (FIXTURE_DIR / "cli_exit_vectors.v1.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_cli(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "commcanary", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_process(completed: subprocess.CompletedProcess[str], expected: Mapping[str, Any]) -> None:
    assert completed.returncode == expected["exit_code"]
    if "stdout" in expected:
        assert completed.stdout == expected["stdout"]
    if "stderr" in expected:
        assert completed.stderr == expected["stderr"]
    if "stdout_prefix" in expected:
        assert completed.stdout.startswith(expected["stdout_prefix"])
    if "stdout_lines" in expected:
        assert completed.stdout.splitlines() == expected["stdout_lines"]
    if "stderr_prefix" in expected:
        assert completed.stderr.startswith(expected["stderr_prefix"])
    for fragment in expected.get("stderr_contains", []):
        assert fragment in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_usage_exit_and_stderr_are_characterized() -> None:
    expected = _load_vectors()["cases"]["usage_error"]
    _assert_process(_run_cli([]), expected)


def test_cli_input_error_exit_and_stderr_are_characterized(tmp_path: Path) -> None:
    expected = _load_vectors()["cases"]["commcanary_input_error"]
    missing = tmp_path / "missing.trace.json"
    _assert_process(_run_cli(["compile", str(missing), "--output", str(tmp_path / "out.json")]), expected)


def test_cli_success_exit_stdout_and_silent_stderr_are_characterized(tmp_path: Path) -> None:
    expected = _load_vectors()["cases"]["successful_compile"]
    completed = _run_cli(
        [
            "compile",
            str(FIXTURE_DIR / "trace.valid.json"),
            "--output",
            str(tmp_path / "out.canary.json"),
        ]
    )
    _assert_process(completed, expected)


def test_cli_valid_negative_verification_uses_exit_one(tmp_path: Path) -> None:
    expected = _load_vectors()["cases"]["valid_negative_verification"]
    completed = _run_cli(
        [
            "verify-fidelity",
            str(FIXTURE_DIR / "trace.valid.json"),
            str(FIXTURE_DIR / "canary.valid.json"),
            "--output",
            str(tmp_path / "verification.json"),
        ]
    )
    _assert_process(completed, expected)


def test_cli_capture_maps_child_failure_and_preserves_original_code_in_stderr(tmp_path: Path) -> None:
    expected = _load_vectors()["cases"]["capture_child_failure"]
    completed = _run_cli(
        [
            "capture",
            "--output",
            str(tmp_path / "unused.trace.json"),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ]
    )
    _assert_process(completed, expected)


def test_cli_version_exposes_exact_capabilities_canonicalization_and_model() -> None:
    expected = _load_vectors()["cases"]["version_capability_summary"]
    completed = _run_cli(["--version"])
    _assert_process(completed, expected)
    compact_stdout = "".join(completed.stdout.split())
    for capability in format_capabilities():
        assert capability.format_id in compact_stdout
    assert f"canonicalization:{CANONICAL_JSON_VERSION}" in compact_stdout
    assert f"replay-model:{SIMULATION_MODEL_VERSION}" in compact_stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_every_subcommand_has_help_and_usage_contract(command: str) -> None:
    help_result = _run_cli([command, "--help"])
    assert help_result.returncode == 0
    assert help_result.stdout.startswith(f"usage: commcanary {command}")
    assert help_result.stderr == ""

    usage_result = _run_cli([command])
    assert usage_result.returncode == 2
    assert usage_result.stderr.startswith(f"usage: commcanary {command}")
    assert "Traceback" not in usage_result.stderr


@pytest.mark.parametrize(
    "arguments",
    (
        ("compile", "{missing}", "--output", "{output}"),
        ("replay", "{missing}", "--output", "{output}"),
        ("compare", "{missing}", "{missing2}", "--output", "{output}"),
        ("verify-fidelity", "{missing}", "{missing2}", "--output", "{output}"),
        ("verify-behavior", "{missing}", "{missing2}", "--output", "{output}"),
        ("baseline", "{missing}", "--method", "frequency", "--output", "{output}"),
        ("reduce", "{missing}", "--output", "{output}"),
        ("import-kineto", "{missing}", "--output", "{output}"),
        ("export-param", "{missing}", "--output", "{output}"),
        ("verify-report", "{missing}", "{missing2}", "--output", "{output}"),
        ("render-html", "{missing}", "--output", "{output}"),
        ("report", "{missing}", "--output", "{output}"),
    ),
)
def test_path_consuming_commands_map_missing_inputs_to_application_error(
    tmp_path: Path,
    arguments: Sequence[str],
) -> None:
    replacements = {
        "{missing}": str(tmp_path / "missing-a.json"),
        "{missing2}": str(tmp_path / "missing-b.json"),
        "{output}": str(tmp_path / "output.json"),
    }
    rendered = [replacements.get(argument, argument) for argument in arguments]

    completed = _run_cli(rendered)

    assert completed.returncode == 3
    assert "does not exist" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_capture_without_child_command_is_application_error(tmp_path: Path) -> None:
    completed = _run_cli(["capture", "--output", str(tmp_path / "trace.json")])

    assert completed.returncode == 3
    assert "requires a command after --" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_remaining_command_success_paths_are_subprocess_contracts(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    reduced = tmp_path / "reduced.json"
    captured = tmp_path / "captured.json"

    compare_result = _run_cli(
        [
            "compare",
            str(FIXTURE_DIR / "report.valid.json"),
            str(FIXTURE_DIR / "report.valid.json"),
            "--output",
            str(comparison),
        ]
    )
    reduce_result = _run_cli(
        [
            "reduce",
            str(FIXTURE_DIR / "trace.valid.json"),
            "--max-oracle-calls",
            "1",
            "--output",
            str(reduced),
        ]
    )
    capture_result = _run_cli(
        [
            "capture",
            "--output",
            str(captured),
            "--",
            sys.executable,
            str(ROOT / "examples" / "instrumented_decode.py"),
        ]
    )

    for completed in (compare_result, reduce_result, capture_result):
        assert completed.returncode == 0, completed.stderr
        assert "Traceback" not in completed.stderr
    assert comparison.is_file()
    assert reduced.is_file()
    assert captured.is_file()
