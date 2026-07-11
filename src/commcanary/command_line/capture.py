"""Instrumented child-process capture command."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Callable

from ..adapters.capture import TraceRecorder, merge_trace_shards
from ..artifacts import write_json
from ..errors import CommCanaryError
from .codes import EXIT_CHILD_FAILURE

CaptureFailurePreserver = Callable[..., None]
DiagnosticEmitter = Callable[..., None]


def capture_command(
    args: Any,
    *,
    failure_preserver: CaptureFailurePreserver,
    diagnostic_emitter: DiagnosticEmitter,
) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise CommCanaryError("capture requires a command after --")

    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="commcanary-capture-") as trace_dir:
        manual_trace = os.path.join(trace_dir, "manual.trace.json")
        session_id = str(uuid.uuid4())
        env["COMMCANARY_TRACE_DIR"] = trace_dir
        env["COMMCANARY_TRACE_OUT"] = manual_trace
        env["COMMCANARY_WORKLOAD_NAME"] = args.workload_name
        env["COMMCANARY_CAPTURE_SESSION_ID"] = session_id
        try:
            completed = subprocess.run(command, env=env)
        except OSError as exc:
            raise CommCanaryError(f"could not run capture command {command[0]!r}: {exc}") from exc
        if completed.returncode != 0:
            if args.preserve_on_failure:
                failure_preserver(
                    trace_dir,
                    args.preserve_on_failure,
                    workload_name=args.workload_name,
                    session_id=session_id,
                    child_returncode=completed.returncode,
                )
            if args.diagnostics_json:
                diagnostic_emitter(
                    args,
                    event="child_failure",
                    exit_code=EXIT_CHILD_FAILURE,
                    child_returncode=completed.returncode,
                )
            else:
                print(f"commcanary: workload exited with child code {completed.returncode}", file=sys.stderr)
            return EXIT_CHILD_FAILURE

        merged = merge_trace_shards(trace_dir, workload_name=args.workload_name)
        if not merged["events"]:
            if not args.allow_empty:
                raise CommCanaryError(
                    "target command did not write a trace; import commcanary.capture.record_collective "
                    "or pass --allow-empty"
                )
            merged = TraceRecorder(args.output, workload={"name": args.workload_name}).to_trace()
        write_json(args.output, merged)
    print(f"captured trace: {args.output}")
    return 0


__all__ = ["CaptureFailurePreserver", "DiagnosticEmitter", "capture_command"]
