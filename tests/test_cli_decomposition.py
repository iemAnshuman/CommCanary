from __future__ import annotations

import ast
import inspect
from pathlib import Path

import commcanary.cli as legacy_cli
from commcanary.command_line import capture as capture_boundary
from commcanary.command_line import capture_failure, codes, commands, diagnostics, lifecycle, parser

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "commcanary"
COMMAND_LINE = PACKAGE / "command_line"


def _definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}


def test_legacy_cli_surface_and_private_handler_signatures_are_preserved() -> None:
    assert (
        legacy_cli.EXIT_SUCCESS,
        legacy_cli.EXIT_NEGATIVE_RESULT,
        legacy_cli.EXIT_USAGE,
        legacy_cli.EXIT_APPLICATION_ERROR,
        legacy_cli.EXIT_CHILD_FAILURE,
        legacy_cli.EXIT_INTERRUPTED,
    ) == (
        codes.EXIT_SUCCESS,
        codes.EXIT_NEGATIVE_RESULT,
        codes.EXIT_USAGE,
        codes.EXIT_APPLICATION_ERROR,
        codes.EXIT_CHILD_FAILURE,
        codes.EXIT_INTERRUPTED,
    )
    assert list(inspect.signature(legacy_cli.main).parameters) == ["argv"]
    assert list(inspect.signature(legacy_cli._build_parser).parameters) == []
    for name in (
        "_cmd_compile",
        "_cmd_baseline",
        "_cmd_reduce",
        "_cmd_import_kineto",
        "_cmd_export_param",
        "_cmd_replay",
        "_cmd_compare",
        "_cmd_verify_fidelity",
        "_cmd_verify_behavior",
        "_cmd_verify_report",
        "_cmd_capture",
        "_cmd_report",
    ):
        assert list(inspect.signature(getattr(legacy_cli, name)).parameters) == ["args"]


def test_cli_responsibilities_live_in_distinct_modules() -> None:
    assert _definitions(COMMAND_LINE / "parser.py") == {"CommandHandlers", "build_parser"}
    assert _definitions(COMMAND_LINE / "lifecycle.py") == {"run_cli"}
    assert _definitions(COMMAND_LINE / "diagnostics.py") == {
        "elapsed_seconds",
        "emit_diagnostic",
        "version_text",
    }
    assert _definitions(COMMAND_LINE / "capture.py") == {"capture_command"}
    assert _definitions(COMMAND_LINE / "capture_failure.py") == {"preserve_capture_failure"}
    assert capture_boundary.capture_command.__module__.endswith(".capture")
    assert capture_failure.preserve_capture_failure.__module__.endswith(".capture_failure")
    assert commands.compile_command.__module__.endswith(".commands")
    assert parser.build_parser.__module__.endswith(".parser")
    assert lifecycle.run_cli.__module__.endswith(".lifecycle")
    assert diagnostics.emit_diagnostic.__module__.endswith(".diagnostics")


def test_command_line_dependency_graph_is_acyclic_and_has_no_private_edges() -> None:
    files = {path.stem: path for path in sorted(COMMAND_LINE.glob("*.py")) if path.name != "__init__.py"}
    edges: dict[str, set[str]] = {module: set() for module in files}
    for module, path in files.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level >= 1:
                assert all(not alias.name.startswith("_") for alias in node.names), (
                    path,
                    node.lineno,
                    node.module,
                )
            if node.level == 1 and node.module in files:
                edges[module].add(node.module)
            if node.level >= 2:
                assert node.module != "cli", (path, node.lineno)
            if node.level == 0 and node.module is not None:
                assert node.module != "commcanary.cli", (path, node.lineno)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        assert module not in visiting, f"command-line dependency cycle reaches {module}"
        if module in visited:
            return
        visiting.add(module)
        for dependency in edges[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in edges:
        visit(module)


def test_parser_and_lifecycle_do_not_import_domain_engines() -> None:
    permitted_parent_modules = {
        "errors",
    }
    for name in ("parser.py", "lifecycle.py"):
        path = COMMAND_LINE / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 2 and node.module is not None:
                assert node.module in permitted_parent_modules, (path, node.lineno, node.module)


def test_legacy_cli_file_contains_wiring_not_domain_implementation() -> None:
    path = PACKAGE / "cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.level == 1:
            assert node.module.startswith("command_line"), (node.lineno, node.module)
