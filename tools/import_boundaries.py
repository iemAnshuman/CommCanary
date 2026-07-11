"""Enforce CommCanary's dependency DAG with only the Python standard library."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

BOUNDARY_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("commcanary.artifacts", "artifacts"),
    ("commcanary.schemas", "artifacts"),
    ("commcanary.operation_identity", "artifacts"),
    ("commcanary.behavior_config", "artifacts"),
    ("commcanary.compilation", "compilation"),
    ("commcanary.replay", "replay"),
    ("commcanary.comparison", "comparison"),
    ("commcanary.verification", "verification"),
    ("commcanary.services", "services"),
    ("commcanary.adapters", "adapters"),
    ("commcanary.reporting", "reporting"),
    ("commcanary.experimental", "experimental"),
    ("commcanary.baselines", "experimental"),
    ("commcanary.command_line", "cli"),
    ("commcanary.cli", "cli"),
    ("commcanary.__main__", "cli"),
)

FOUNDATION_MODULES = {
    "commcanary.errors",
    "commcanary.formats",
    "commcanary.resources",
    "commcanary.statistics",
    "commcanary.version",
}

COMPATIBILITY_TARGETS: Mapping[str, Set[str]] = {
    "commcanary.schema": {"foundation", "artifacts"},
    "commcanary.compiler": {"artifacts", "compilation", "verification", "services"},
    "commcanary.compare": {"comparison"},
    "commcanary.capture": {"adapters"},
    "commcanary.interop": {"adapters"},
    "commcanary.html_report": {"reporting"},
    "commcanary.reduce": {"services"},
}

PACKAGE_FACADE_TARGETS: Mapping[str, Set[str]] = {
    # ``commcanary.replay`` is both the replay package and the historical
    # public module that re-exports report verification.
    "commcanary.replay": {"foundation", "artifacts", "replay", "verification"},
}

PUBLIC_UNDERSCORE_NAMES = {"__version__"}

ALLOWED_BOUNDARIES: Mapping[str, Set[str]] = {
    "foundation": {"foundation"},
    "artifacts": {"foundation", "artifacts"},
    "compilation": {"foundation", "artifacts", "compilation"},
    "replay": {"foundation", "artifacts", "replay"},
    "comparison": {"foundation", "artifacts", "comparison"},
    "verification": {"foundation", "artifacts", "compilation", "replay", "comparison", "verification"},
    "services": {
        "foundation",
        "artifacts",
        "compilation",
        "replay",
        "comparison",
        "verification",
        "services",
    },
    "adapters": {"foundation", "artifacts", "adapters"},
    "reporting": {"foundation", "artifacts", "comparison", "reporting"},
    "experimental": {
        "foundation",
        "artifacts",
        "compilation",
        "replay",
        "comparison",
        "verification",
        "services",
        "adapters",
        "reporting",
        "experimental",
    },
    "cli": {
        "foundation",
        "artifacts",
        "compilation",
        "replay",
        "comparison",
        "verification",
        "services",
        "adapters",
        "reporting",
        "experimental",
        "compatibility",
        "cli",
    },
    "facade": {
        "foundation",
        "artifacts",
        "compilation",
        "replay",
        "comparison",
        "verification",
        "services",
        "adapters",
        "reporting",
        "experimental",
        "compatibility",
        "cli",
        "facade",
    },
}


@dataclass(frozen=True)
class Module:
    name: str
    path: Path
    boundary: str
    is_package: bool


@dataclass(frozen=True)
class ImportEdge:
    source: str
    target: str
    imported_name: str
    line: int

    @property
    def is_private(self) -> bool:
        imported = self.imported_name.rsplit(".", 1)[-1]
        target_parts = self.target.split(".")[1:]
        return (imported.startswith("_") and imported not in PUBLIC_UNDERSCORE_NAMES) or any(
            part.startswith("_") for part in target_parts
        )


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    line: int
    message: str


def _module_name(source_root: Path, path: Path) -> Tuple[str, bool]:
    relative = path.relative_to(source_root)
    parts = list(relative.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _classify(module: str) -> Optional[str]:
    if module == "commcanary":
        return "facade"
    if module in FOUNDATION_MODULES:
        return "foundation"
    if module in COMPATIBILITY_TARGETS:
        return "compatibility"
    for prefix, boundary in BOUNDARY_PREFIXES:
        if module == prefix or module.startswith(prefix + "."):
            return boundary
    return None


def discover_modules(source_root: Path) -> Tuple[Dict[str, Module], List[Violation]]:
    modules: Dict[str, Module] = {}
    violations: List[Violation] = []
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name, is_package = _module_name(source_root, path)
        boundary = _classify(name)
        if boundary is None:
            violations.append(
                Violation(
                    path=path.relative_to(source_root).as_posix(),
                    line=1,
                    message=f"module {name!r} has no declared dependency boundary",
                )
            )
            boundary = "unclassified"
        modules[name] = Module(name=name, path=path, boundary=boundary, is_package=is_package)
    return modules, violations


def _relative_base(source: Module, level: int, module: Optional[str]) -> Optional[str]:
    package = source.name if source.is_package else source.name.rpartition(".")[0]
    parts = package.split(".") if package else []
    if level > len(parts):
        return None
    keep = len(parts) - max(0, level - 1)
    base = parts[:keep]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _known_target(candidate: str, modules: Mapping[str, Module]) -> Optional[str]:
    current = candidate
    while current.startswith("commcanary"):
        if current in modules:
            return current
        if "." not in current:
            break
        current = current.rpartition(".")[0]
    return None


def _edges_for_module(module: Module, modules: Mapping[str, Module]) -> Tuple[List[ImportEdge], List[Violation]]:
    try:
        tree = ast.parse(module.path.read_text(encoding="utf-8"), filename=str(module.path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [], [Violation(str(module.path), 1, f"cannot parse module: {exc}")]
    edges: List[ImportEdge] = []
    violations: List[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("commcanary"):
                    continue
                target = _known_target(alias.name, modules)
                if target is not None:
                    edges.append(ImportEdge(module.name, target, alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _relative_base(module, node.level, node.module)
                if base is None:
                    violations.append(
                        Violation(
                            module.path.as_posix(),
                            node.lineno,
                            "relative import escapes the commcanary package",
                        )
                    )
                    continue
            else:
                base = node.module or ""
            if not base.startswith("commcanary"):
                continue
            for alias in node.names:
                candidate = f"{base}.{alias.name}" if node.module is None else base
                target = _known_target(candidate, modules)
                if target is not None:
                    edges.append(ImportEdge(module.name, target, alias.name, node.lineno))
    return edges, violations


def _find_cycles(modules: Mapping[str, Module], edges: Iterable[ImportEdge]) -> List[Tuple[str, ...]]:
    graph: Dict[str, Set[str]] = {name: set() for name in modules}
    for edge in edges:
        if edge.source != edge.target:
            graph[edge.source].add(edge.target)
    index = 0
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    cycles: List[Tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(graph[node]):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] != indices[node]:
            return
        component: List[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1:
            cycles.append(tuple(sorted(component)))

    for name in sorted(graph):
        if name not in indices:
            visit(name)
    return sorted(cycles)


def check_boundaries(source_root: Path) -> List[Violation]:
    modules, violations = discover_modules(source_root)
    edges: List[ImportEdge] = []
    for module in modules.values():
        module_edges, parse_violations = _edges_for_module(module, modules)
        edges.extend(module_edges)
        violations.extend(parse_violations)
    for edge in edges:
        source = modules[edge.source]
        target = modules[edge.target]
        if source.boundary == "unclassified" or target.boundary == "unclassified":
            continue
        if source.name in PACKAGE_FACADE_TARGETS:
            allowed = PACKAGE_FACADE_TARGETS[source.name]
        elif source.boundary == "compatibility":
            allowed = COMPATIBILITY_TARGETS.get(source.name, set())
        else:
            allowed = ALLOWED_BOUNDARIES.get(source.boundary, set())
        if target.boundary not in allowed:
            violations.append(
                Violation(
                    source.path.relative_to(source_root).as_posix(),
                    edge.line,
                    f"{source.boundary} module imports upward from {target.boundary}: {edge.target}",
                )
            )
        if (
            edge.is_private
            and source.boundary not in {"compatibility", "facade"}
            and source.boundary != target.boundary
        ):
            violations.append(
                Violation(
                    source.path.relative_to(source_root).as_posix(),
                    edge.line,
                    f"cross-boundary private import {edge.imported_name!r} from {edge.target}",
                )
            )
    for cycle in _find_cycles(modules, edges):
        first = modules[cycle[0]]
        violations.append(
            Violation(
                first.path.relative_to(source_root).as_posix(),
                1,
                "dependency cycle: " + " -> ".join((*cycle, cycle[0])),
            )
        )
    return sorted(set(violations))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    args = parser.parse_args(argv)
    violations = check_boundaries(args.source_root)
    if violations:
        for violation in violations:
            print(f"{violation.path}:{violation.line}: {violation.message}", file=sys.stderr)
        return 1
    print("import boundary policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
