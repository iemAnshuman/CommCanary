"""Compatibility facade and stable entry point for the CommCanary CLI."""

from __future__ import annotations

import argparse
from typing import Any, List, Optional

from .command_line.capture import capture_command
from .command_line.capture_failure import preserve_capture_failure
from .command_line.codes import (
    EXIT_APPLICATION_ERROR as _EXIT_APPLICATION_ERROR,
)
from .command_line.codes import (
    EXIT_CHILD_FAILURE as _EXIT_CHILD_FAILURE,
)
from .command_line.codes import (
    EXIT_INTERRUPTED as _EXIT_INTERRUPTED,
)
from .command_line.codes import (
    EXIT_NEGATIVE_RESULT as _EXIT_NEGATIVE_RESULT,
)
from .command_line.codes import (
    EXIT_SUCCESS as _EXIT_SUCCESS,
)
from .command_line.codes import (
    EXIT_USAGE as _EXIT_USAGE,
)
from .command_line.commands import (
    baseline_command,
    compare_command,
    compile_command,
    export_param_command,
    import_kineto_command,
    reduce_command,
    replay_command,
    report_command,
    split_ablations,
    verify_behavior_command,
    verify_fidelity_command,
    verify_report_command,
)
from .command_line.diagnostics import elapsed_seconds, emit_diagnostic, version_text
from .command_line.lifecycle import run_cli
from .command_line.parser import CommandHandlers, build_parser

EXIT_SUCCESS = _EXIT_SUCCESS
EXIT_NEGATIVE_RESULT = _EXIT_NEGATIVE_RESULT
EXIT_USAGE = _EXIT_USAGE
EXIT_APPLICATION_ERROR = _EXIT_APPLICATION_ERROR
EXIT_CHILD_FAILURE = _EXIT_CHILD_FAILURE
EXIT_INTERRUPTED = _EXIT_INTERRUPTED


def _version_text() -> str:
    return version_text()


def _emit_diagnostic(args: Any, *, event: str, exit_code: int, **fields: Any) -> None:
    emit_diagnostic(args, event=event, exit_code=exit_code, **fields)


def _elapsed_seconds(started: float) -> float:
    return elapsed_seconds(started)


def _preserve_capture_failure(
    trace_dir: str,
    destination: str,
    *,
    workload_name: str,
    session_id: str,
    child_returncode: int,
) -> None:
    preserve_capture_failure(
        trace_dir,
        destination,
        workload_name=workload_name,
        session_id=session_id,
        child_returncode=child_returncode,
    )


def _split_ablations(values: List[str]) -> List[str]:
    return split_ablations(values)


def _cmd_compile(args: Any) -> int:
    return compile_command(
        args,
        diagnostic_emitter=_emit_diagnostic,
        elapsed_clock=_elapsed_seconds,
    )


def _cmd_baseline(args: Any) -> int:
    return baseline_command(args)


def _cmd_reduce(args: Any) -> int:
    return reduce_command(
        args,
        diagnostic_emitter=_emit_diagnostic,
        elapsed_clock=_elapsed_seconds,
    )


def _cmd_import_kineto(args: Any) -> int:
    return import_kineto_command(args)


def _cmd_export_param(args: Any) -> int:
    return export_param_command(args)


def _cmd_replay(args: Any) -> int:
    return replay_command(args, ablation_splitter=_split_ablations)


def _cmd_compare(args: Any) -> int:
    return compare_command(args)


def _cmd_verify_fidelity(args: Any) -> int:
    return verify_fidelity_command(args)


def _cmd_verify_behavior(args: Any) -> int:
    return verify_behavior_command(args)


def _cmd_verify_report(args: Any) -> int:
    return verify_report_command(args)


def _cmd_capture(args: Any) -> int:
    return capture_command(
        args,
        failure_preserver=_preserve_capture_failure,
        diagnostic_emitter=_emit_diagnostic,
    )


def _cmd_report(args: Any) -> int:
    return report_command(args, diagnostic_emitter=_emit_diagnostic)


def _build_parser() -> argparse.ArgumentParser:
    return build_parser(
        handlers=CommandHandlers(
            compile=_cmd_compile,
            replay=_cmd_replay,
            compare=_cmd_compare,
            verify_fidelity=_cmd_verify_fidelity,
            verify_behavior=_cmd_verify_behavior,
            baseline=_cmd_baseline,
            reduce=_cmd_reduce,
            import_kineto=_cmd_import_kineto,
            export_param=_cmd_export_param,
            verify_report=_cmd_verify_report,
            capture=_cmd_capture,
            report=_cmd_report,
        ),
        version=_version_text(),
    )


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(
        argv,
        parser_factory=_build_parser,
        diagnostic_emitter=_emit_diagnostic,
        elapsed_clock=_elapsed_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
