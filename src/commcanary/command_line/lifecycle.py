"""CLI parse/dispatch/error lifecycle."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List, Optional

from ..errors import CommCanaryError
from .codes import (
    EXIT_APPLICATION_ERROR,
    EXIT_INTERRUPTED,
    EXIT_NEGATIVE_RESULT,
    EXIT_SUCCESS,
)

ParserFactory = Callable[[], argparse.ArgumentParser]
DiagnosticEmitter = Callable[..., None]
ElapsedClock = Callable[[float], float]


def run_cli(
    argv: Optional[List[str]] = None,
    *,
    parser_factory: ParserFactory,
    diagnostic_emitter: DiagnosticEmitter,
    elapsed_clock: ElapsedClock,
) -> int:
    parser = parser_factory()
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.diagnostics_json:
        diagnostic_emitter(args, event="started", exit_code=EXIT_SUCCESS)
    try:
        exit_code = int(args.func(args))
    except CommCanaryError as exc:
        if args.diagnostics_json:
            diagnostic_emitter(
                args,
                event="error",
                exit_code=EXIT_APPLICATION_ERROR,
                elapsed_seconds=elapsed_clock(started),
                message=str(exc),
            )
        else:
            print(f"commcanary: {exc}", file=sys.stderr)
        return EXIT_APPLICATION_ERROR
    except KeyboardInterrupt:
        if args.diagnostics_json:
            diagnostic_emitter(
                args,
                event="interrupted",
                exit_code=EXIT_INTERRUPTED,
                elapsed_seconds=elapsed_clock(started),
                message="interrupted",
            )
        else:
            print("commcanary: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    if args.diagnostics_json:
        if exit_code == EXIT_SUCCESS:
            outcome = "success"
        elif exit_code == EXIT_NEGATIVE_RESULT:
            outcome = "negative_result"
        else:
            outcome = "error"
        diagnostic_emitter(
            args,
            event="completed",
            exit_code=exit_code,
            elapsed_seconds=elapsed_clock(started),
            outcome=outcome,
        )
    return exit_code


__all__ = ["DiagnosticEmitter", "ElapsedClock", "ParserFactory", "run_cli"]
