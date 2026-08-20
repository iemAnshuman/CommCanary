from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from commcanary.artifacts.chakra import decode_chakra_execution_trace
from commcanary.cli import main
from commcanary.command_line import capture as capture_module
from commcanary.errors import CommCanaryError, SchemaError
from commcanary.formats import TRACE_FORMAT
from commcanary.schema import load_json


def _captured_trace() -> Dict[str, Any]:
    recipe = {
        "op": "gemm",
        "dtype": "bfloat16",
        "m": 2,
        "n": 4,
        "k": 3,
        "source_kernel_count": 1,
        "source_kernel_duration_us": 2.0,
    }
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "captured"},
        "system": {"source_format": "pytorch-kineto"},
        "events": [
            {
                "id": "collective-0",
                "op": "all_reduce",
                "dtype": "bfloat16",
                "bytes": 32,
                "ranks": [0, 1, 2, 3],
                "group": "default",
                "start_us": 10.0,
                "phase": "decode",
                "reduction_op": "sum",
                "compute_overlap_us": 2.0,
                "rank_arrival_us": {"0": 0.0, "1": 0.5, "2": 1.0, "3": 1.5},
                "compute_recipe_by_rank": {str(rank): [copy.deepcopy(recipe)] for rank in range(4)},
            }
        ],
    }


def _args(tmp_path: Path, **updates: Any) -> SimpleNamespace:
    values: Dict[str, Any] = {
        "output": str(tmp_path / "trace.json"),
        "chakra_output": str(tmp_path / "trace.et"),
        "projection_output": str(tmp_path / "projection.json"),
        "command": ["--", "workload"],
        "workload_name": "captured",
        "preserve_on_failure": None,
        "diagnostics_json": False,
        "allow_empty": False,
        "opaque_attributes_reviewed": True,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_capture_command_writes_trace_chakra_and_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda command, env: SimpleNamespace(returncode=0))
    monkeypatch.setattr(capture_module, "merge_trace_shards", lambda *args, **kwargs: _captured_trace())

    result = capture_module.capture_command(
        args,
        failure_preserver=lambda *args, **kwargs: None,
        diagnostic_emitter=lambda *args, **kwargs: None,
    )

    assert result == 0
    assert load_json(args.output)["events"][0]["op"] == "all_reduce"
    trace = decode_chakra_execution_trace(Path(args.chakra_output).read_bytes())
    projection = load_json(args.projection_output)
    assert trace.node_ids == (1, 2)
    assert projection["privacy_review"]["opaque_chakra_attributes_reviewed"] is True


def test_capture_command_rejects_ambiguous_outputs_and_missing_command(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="supplied together"):
        capture_module.capture_command(
            _args(tmp_path, projection_output=None),
            failure_preserver=lambda *args, **kwargs: None,
            diagnostic_emitter=lambda *args, **kwargs: None,
        )
    with pytest.raises(SchemaError, match="different paths"):
        capture_module.capture_command(
            _args(tmp_path, projection_output=str(tmp_path / "trace.json")),
            failure_preserver=lambda *args, **kwargs: None,
            diagnostic_emitter=lambda *args, **kwargs: None,
        )
    with pytest.raises(CommCanaryError, match="requires a command"):
        capture_module.capture_command(
            _args(tmp_path, chakra_output=None, projection_output=None, command=["--"]),
            failure_preserver=lambda *args, **kwargs: None,
            diagnostic_emitter=lambda *args, **kwargs: None,
        )


def test_capture_command_preserves_and_reports_child_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preserved: list[tuple[Any, ...]] = []
    diagnostics: list[Dict[str, Any]] = []
    args = _args(
        tmp_path,
        chakra_output=None,
        projection_output=None,
        preserve_on_failure=str(tmp_path / "failed"),
        diagnostics_json=True,
    )
    monkeypatch.setattr(subprocess, "run", lambda command, env: SimpleNamespace(returncode=17))

    result = capture_module.capture_command(
        args,
        failure_preserver=lambda *values, **kwargs: preserved.append((*values, kwargs)),
        diagnostic_emitter=lambda _args, **kwargs: diagnostics.append(kwargs),
    )

    assert result == capture_module.EXIT_CHILD_FAILURE
    assert preserved[0][-1]["child_returncode"] == 17
    assert diagnostics == [
        {
            "event": "child_failure",
            "exit_code": capture_module.EXIT_CHILD_FAILURE,
            "child_returncode": 17,
        }
    ]


def test_capture_command_converts_launch_errors_and_supports_explicit_empty_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, chakra_output=None, projection_output=None)

    def launch_error(command: list[str], env: Dict[str, str]) -> Any:
        raise OSError("not executable")

    monkeypatch.setattr(subprocess, "run", launch_error)
    with pytest.raises(CommCanaryError, match="could not run capture command"):
        capture_module.capture_command(
            args,
            failure_preserver=lambda *args, **kwargs: None,
            diagnostic_emitter=lambda *args, **kwargs: None,
        )

    args.allow_empty = True
    monkeypatch.setattr(subprocess, "run", lambda command, env: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        capture_module,
        "merge_trace_shards",
        lambda *args, **kwargs: {"format": TRACE_FORMAT, "workload": {"name": "captured"}, "events": []},
    )
    assert (
        capture_module.capture_command(
            args,
            failure_preserver=lambda *args, **kwargs: None,
            diagnostic_emitter=lambda *args, **kwargs: None,
        )
        == 0
    )
    assert load_json(args.output)["events"] == []


def test_example_instrumented_decode_emits_chakra_and_projection(tmp_path: Path) -> None:
    examples = Path(__file__).resolve().parents[1] / "examples"
    output = tmp_path / "trace.json"
    chakra = tmp_path / "trace.et"
    projection = tmp_path / "trace.projection.json"
    result = main(
        [
            "capture",
            "--output",
            str(output),
            "--chakra-output",
            str(chakra),
            "--projection-output",
            str(projection),
            "--workload-name",
            "decode",
            "--",
            sys.executable,
            str(examples / "instrumented_decode.py"),
        ]
    )

    assert result == 0
    trace = decode_chakra_execution_trace(chakra.read_bytes())
    assert trace.node_ids
    projected = load_json(str(projection))
    collectives = [node for node in projected["nodes"] if node["operation"] == "all_reduce"]
    gemms = [node for node in projected["nodes"] if node["operation"] == "gemm"]
    assert projected["format"] == "commcanary.chakra_projection.v1"
    assert len(collectives) == 24
    assert len(gemms) == 24
    assert all(node["execution"]["reduction"] == "sum" for node in collectives)
    assert all(node["execution"]["dtype"] == "bfloat16" for node in collectives)
