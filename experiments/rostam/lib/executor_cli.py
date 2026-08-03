"""Dispatch commands from the privately staged Rostam executor artifact."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Optional, Sequence

from .executor_artifact import ExecutorArtifact, ExecutorArtifactError, load_executor_artifact

_CHILD_IMPORT_PATHS = "COMMCANARY_CHILD_IMPORT_PATHS"
_ALLOWED_SCRIPT_ROOT = "COMMCANARY_ALLOWED_SCRIPT_ROOT"


def _running_artifact() -> ExecutorArtifact:
    raw_path = os.environ.get("COMMCANARY_EXECUTOR_PATH")
    expected_sha256 = os.environ.get("COMMCANARY_EXECUTOR_SHA256")
    if not raw_path or not expected_sha256:
        raise ExecutorArtifactError("frozen executor commands require the private bootstrap")
    artifact = load_executor_artifact(Path(raw_path))
    if artifact.sha256 != expected_sha256:
        raise ExecutorArtifactError("running executor identity disagrees with the private bootstrap")
    return artifact


def run_python(arguments: Sequence[str]) -> int:
    """Run one module or reviewed external script after isolated startup."""

    artifact = _running_artifact()
    raw_paths = os.environ.get(_CHILD_IMPORT_PATHS, "")
    import_paths = []
    for raw in raw_paths.split(os.pathsep):
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ExecutorArtifactError("child import path is missing or unsafe")
        import_paths.append(str(path.resolve()))
    if len(import_paths) != len(set(import_paths)):
        raise ExecutorArtifactError("child import path inventory contains duplicates")
    standard_library_paths = [
        item for item in sys.path if item and Path(item).resolve() != artifact.path and item not in import_paths
    ]
    sys.path[:] = [str(artifact.path), *standard_library_paths, *import_paths]
    if len(arguments) >= 2 and arguments[0] == "-m":
        module = arguments[1]
        if not module or module.startswith(".") or any(part in {"", ".", ".."} for part in module.split(".")):
            raise ExecutorArtifactError("child module name is unsafe")
        sys.argv = [module, *arguments[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    if not arguments:
        raise ExecutorArtifactError("isolated child runner requires a module or script")
    allowed_root_raw = os.environ.get(_ALLOWED_SCRIPT_ROOT)
    if not allowed_root_raw:
        raise ExecutorArtifactError("isolated child runner does not permit external scripts")
    allowed_root = Path(allowed_root_raw)
    script = Path(arguments[0])
    if allowed_root.is_symlink() or not allowed_root.is_dir() or script.is_symlink() or not script.is_file():
        raise ExecutorArtifactError("external child script is missing or unsafe")
    root = allowed_root.resolve()
    candidate = script.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ExecutorArtifactError("external child script escapes its reviewed artifact") from exc
    sys.argv = [str(candidate), *arguments[1:]]
    runpy.run_path(str(candidate), run_name="__main__")
    return 0


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
    if command == "run-python":
        return run_python(arguments[1:])
    if command == "execute-cell":
        arguments = arguments[1:]
    from .cell_entrypoint import main as cell_main

    return cell_main(arguments)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
