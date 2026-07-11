"""Architecture contracts for the Phase 5 package decomposition."""

from __future__ import annotations

import ast
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import commcanary.compiler as compiler_facade
import commcanary.reduce as reduction_facade
import commcanary.replay as replay_facade
from commcanary.compilation.core import compile_trace_core
from commcanary.replay.core import replay_canary
from commcanary.services.compile import compile_trace
from commcanary.services.reduction import ddmin_ranking_reduction
from commcanary.verification.behavior import verify_canary_behavior
from commcanary.verification.fidelity import verify_canary_fidelity

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "commcanary"
BOUNDARIES = ("compilation", "replay", "verification", "services")
LAYER = {"compilation": 1, "replay": 1, "verification": 2, "services": 3}


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("commcanary", *parts))


def _boundary_sources() -> Iterable[Path]:
    for boundary in BOUNDARIES:
        yield from sorted((PACKAGE / boundary).glob("*.py"))


def _import_targets(path: Path) -> set[str]:
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    targets: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * node.level + (node.module or "")
                target = importlib.util.resolve_name(relative, package)
            else:
                target = node.module or ""
            if target:
                targets.add(target)
    return targets


def _boundary(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "commcanary" and parts[1] in BOUNDARIES:
        return parts[1]
    return None


def test_boundary_dependency_graph_is_directional_and_acyclic() -> None:
    graph: dict[str, set[str]] = defaultdict(set)
    for path in _boundary_sources():
        source = _module_name(path)
        source_boundary = _boundary(source)
        assert source_boundary is not None
        for target in _import_targets(path):
            target_boundary = _boundary(target)
            if target_boundary is None:
                continue
            # replay/__init__.py is the documented legacy facade; its lazy
            # verification import does not belong to the replay core DAG.
            if source == "commcanary.replay" and target_boundary == "verification":
                continue
            assert LAYER[target_boundary] <= LAYER[source_boundary], f"upward import: {source} -> {target}"
            graph[source].add(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, trail: tuple[str, ...]) -> None:
        if module in visiting:
            raise AssertionError("dependency cycle: " + " -> ".join((*trail, module)))
        if module in visited:
            return
        visiting.add(module)
        for target in graph[module]:
            if target in graph:
                visit(target, (*trail, module))
        visiting.remove(module)
        visited.add(module)

    for module in sorted(graph):
        visit(module, ())


def test_new_boundaries_do_not_depend_on_legacy_schema_facade() -> None:
    violations = {
        _module_name(path): sorted(target for target in _import_targets(path) if target == "commcanary.schema")
        for path in _boundary_sources()
        if "commcanary.schema" in _import_targets(path)
    }
    assert violations == {}


def test_no_private_names_cross_package_boundaries() -> None:
    violations: list[str] = []
    for path in _boundary_sources():
        source = _module_name(path)
        source_boundary = _boundary(source)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = source if path.name == "__init__.py" else source.rpartition(".")[0]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                target = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            else:
                target = node.module or ""
            target_boundary = _boundary(target)
            if target_boundary is None or target_boundary == source_boundary:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(f"{source} imports {target}.{alias.name}")
    assert violations == []


def test_compile_core_has_no_verification_or_service_orchestration() -> None:
    path = PACKAGE / "compilation" / "core.py"
    imports = _import_targets(path)
    assert not any(target.startswith("commcanary.verification") for target in imports)
    assert not any(target.startswith("commcanary.services") for target in imports)
    functions = {
        node.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions == {"compile_trace_core"}


def test_fidelity_reference_does_not_import_producer_private_calculations() -> None:
    path = PACKAGE / "verification" / "fidelity.py"
    imports = _import_targets(path)
    assert "commcanary.compilation.compression" not in imports
    assert "commcanary.compilation.metrics" not in imports
    assert "commcanary.compilation.normalization" not in imports
    source = path.read_text(encoding="utf-8")
    assert "def _recompute_interval_commitment(" in source
    assert "def _source_segment_sha256(" in source


def test_reduction_uses_intentional_ranking_service_abstraction() -> None:
    path = PACKAGE / "services" / "reduction.py"
    source = path.read_text(encoding="utf-8")
    assert "from ._ranking import RANKING_METRICS, ranking_relation" in source
    assert "commcanary.compiler" not in source
    assert "from ..compiler import" not in source


def test_legacy_facades_preserve_characterized_entry_points() -> None:
    assert compiler_facade.compile_trace is compile_trace
    assert compiler_facade.verify_canary_behavior is verify_canary_behavior
    assert compiler_facade.verify_canary_fidelity is verify_canary_fidelity
    assert replay_facade.replay_canary is replay_canary
    assert reduction_facade.ddmin_ranking_reduction is ddmin_ranking_reduction
    assert callable(compile_trace_core)
