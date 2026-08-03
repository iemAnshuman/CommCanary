"""Dispatch commands from the privately staged Rostam executor artifact."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from .executor_artifact import ExecutorArtifact, ExecutorArtifactError, load_executor_artifact


def _running_artifact() -> ExecutorArtifact:
    raw_path = os.environ.get("COMMCANARY_EXECUTOR_PATH")
    expected_sha256 = os.environ.get("COMMCANARY_EXECUTOR_SHA256")
    if not raw_path or not expected_sha256:
        raise ExecutorArtifactError("frozen executor commands require the private bootstrap")
    artifact = load_executor_artifact(Path(raw_path))
    if artifact.sha256 != expected_sha256:
        raise ExecutorArtifactError("running executor identity disagrees with the private bootstrap")
    return artifact


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else "execute-cell"
    if command == "analyze":
        from ..analyze import main as analyze_main

        artifact = _running_artifact()
        return analyze_main(arguments[1:], executor_artifact=artifact)
    if command == "evaluate-decision-gate":
        from ..evaluate_decision_gate import main as evaluate_main

        artifact = _running_artifact()
        return evaluate_main(arguments[1:], executor_artifact=artifact)
    if command == "execute-cell":
        arguments = arguments[1:]
    from .cell_entrypoint import main as cell_main

    return cell_main(arguments)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
