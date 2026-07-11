from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Iterable

import commcanary.artifacts as artifacts
import commcanary.artifacts.canary as canary_contract
import commcanary.artifacts.comparison as comparison_contract
import commcanary.artifacts.report as report_contract
import commcanary.artifacts.trace as trace_contract
import commcanary.artifacts.wire as wire_contract
import commcanary.schema as schema
from commcanary.artifacts.json_codec import canonical_json_bytes
from commcanary.errors import CommCanaryError, SchemaError
from commcanary.statistics import median, percentile, percentile_from_sorted, summarize_latencies

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "src" / "commcanary" / "artifacts"
STDLIB_IMPORT_ROOTS = {
    "__future__",
    "dataclasses",
    "enum",
    "hashlib",
    "importlib",
    "json",
    "math",
    "os",
    "pathlib",
    "stat",
    "tempfile",
    "typing",
}

LEGACY_FACADE_EXPORTS = [
    "ARTIFACT_PROVENANCE_ALGORITHM",
    "ASSURANCE_STATES",
    "CANARY_FORMAT",
    "CANARY_HASH_FIELD_NAMES",
    "CANARY_INTEGRITY_PROFILE",
    "COMPARE_FORMAT",
    "CanaryExpansionCounts",
    "CommCanaryError",
    "DEFAULT_RESOURCE_LIMITS",
    "FIDELITY_ERROR_FIELDS",
    "JsonDict",
    "JsonResourceError",
    "MAX_ABS_INTEGER",
    "MAX_RANK_COUNT",
    "MAX_TIME_US",
    "PROTOCOL_FINGERPRINT_EXCLUDE",
    "REPORT_FORMAT",
    "ResourceLimits",
    "SUPPORTED_OPS",
    "SchemaError",
    "TRACE_FORMAT",
    "arrival_skew_us",
    "as_float",
    "as_int",
    "average_wait_us",
    "canary_artifact_provenance_sha256",
    "canary_calibration_sha256",
    "canary_execution_sha256",
    "canary_scheduler_execution_sha256",
    "canonical_json_bytes",
    "clean_private_keys",
    "comparison_policy_evaluations",
    "derive_comparison_verdict",
    "iter_canary_logical_events",
    "iter_canary_stored_leaf_events",
    "load_json",
    "load_json_document",
    "median",
    "merge_metadata",
    "normalize_arrival_offsets",
    "normalize_ranks",
    "percentile",
    "percentile_from_sorted",
    "preflight_canary_expansion",
    "replay_protocol_sha256",
    "require_format",
    "summarize_latencies",
    "validate_canary",
    "validate_comparison",
    "validate_report",
    "validate_trace",
    "write_json",
]

ARTIFACT_PUBLIC_EXPORTS = [
    "ASSURANCE_STATES",
    "AtomicWritePolicy",
    "CANARY_HASH_FIELD_NAMES",
    "CanaryExpansionCounts",
    "FIDELITY_ERROR_FIELDS",
    "JsonDict",
    "MAX_ABS_INTEGER",
    "MAX_RANK_COUNT",
    "MAX_TIME_US",
    "PARAM_TRACE_POLICY",
    "PROTOCOL_FINGERPRINT_EXCLUDE",
    "SENSITIVE_JSON_POLICY",
    "SHAREABLE_HTML_POLICY",
    "SUPPORTED_OPS",
    "SymlinkPolicy",
    "TempPlacement",
    "arrival_skew_us",
    "as_float",
    "as_int",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "average_wait_us",
    "canary_artifact_provenance_sha256",
    "canary_calibration_sha256",
    "canary_execution_sha256",
    "canary_scheduler_execution_sha256",
    "canonical_json_bytes",
    "clean_private_keys",
    "comparison_policy_evaluations",
    "derive_comparison_verdict",
    "formatted_json_bytes",
    "iter_canary_logical_events",
    "iter_canary_stored_leaf_events",
    "load_json",
    "load_json_document",
    "load_schema_bytes",
    "merge_metadata",
    "normalize_arrival_offsets",
    "normalize_ranks",
    "preflight_canary_expansion",
    "replay_protocol_sha256",
    "require_format",
    "validate_canary",
    "validate_comparison",
    "validate_report",
    "validate_trace",
    "write_json",
]


def _python_files() -> Iterable[Path]:
    for relative in ("src", "tests", "benchmarks", "examples"):
        root = ROOT / relative
        if root.exists():
            yield from sorted(root.rglob("*.py"))


def _documented_python_blocks() -> Iterable[tuple[Path, str]]:
    markdown_paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for path in markdown_paths:
        if not path.is_file():
            continue
        document = path.read_text(encoding="utf-8")
        for match in re.finditer(r"```python\n(.*?)```", document, flags=re.DOTALL):
            source = match.group(1)
            if "commcanary.schema" in source:
                yield path, source


def _artifact_module(path: Path) -> str:
    suffix = "" if path.name == "__init__.py" else f".{path.stem}"
    return f"commcanary.artifacts{suffix}"


def _artifact_edges(path: Path) -> set[str]:
    edges: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is None:
            continue
        target = ARTIFACT_DIR / f"{node.module}.py"
        if target.is_file():
            edges.add(_artifact_module(target))
    return edges


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        assert module not in visiting, f"artifact dependency cycle reaches {module}"
        if module in visited:
            return
        visiting.add(module)
        for dependency in graph.get(module, set()):
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)


def test_schema_is_a_complete_identity_compatible_facade() -> None:
    assert schema.__all__ == LEGACY_FACADE_EXPORTS
    expected_by_module: dict[ModuleType, tuple[str, ...]] = {
        wire_contract: (
            "JsonDict",
            "MAX_ABS_INTEGER",
            "MAX_RANK_COUNT",
            "MAX_TIME_US",
            "PROTOCOL_FINGERPRINT_EXCLUDE",
            "SUPPORTED_OPS",
            "arrival_skew_us",
            "as_float",
            "as_int",
            "average_wait_us",
            "clean_private_keys",
            "load_json",
            "load_json_document",
            "merge_metadata",
            "normalize_arrival_offsets",
            "normalize_ranks",
            "replay_protocol_sha256",
            "require_format",
            "write_json",
        ),
        canary_contract: (
            "ASSURANCE_STATES",
            "CANARY_HASH_FIELD_NAMES",
            "CanaryExpansionCounts",
            "FIDELITY_ERROR_FIELDS",
            "canary_artifact_provenance_sha256",
            "canary_calibration_sha256",
            "canary_execution_sha256",
            "canary_scheduler_execution_sha256",
            "iter_canary_logical_events",
            "iter_canary_stored_leaf_events",
            "preflight_canary_expansion",
            "validate_canary",
        ),
        trace_contract: ("validate_trace",),
        report_contract: ("validate_report",),
        comparison_contract: (
            "comparison_policy_evaluations",
            "derive_comparison_verdict",
            "validate_comparison",
        ),
    }
    for module, names in expected_by_module.items():
        for name in names:
            assert getattr(schema, name) is getattr(module, name), name

    assert schema.CommCanaryError is CommCanaryError
    assert schema.SchemaError is SchemaError
    assert schema.canonical_json_bytes is canonical_json_bytes
    assert schema.median is median
    assert schema.percentile is percentile
    assert schema.percentile_from_sorted is percentile_from_sorted
    assert schema.summarize_latencies is summarize_latencies
    assert schema._expand_sequence_motif is canary_contract.expand_sequence_motif


def test_artifacts_package_has_a_deliberate_engine_facing_contract_surface() -> None:
    assert artifacts.__all__ == ARTIFACT_PUBLIC_EXPORTS
    for name in ARTIFACT_PUBLIC_EXPORTS:
        assert getattr(artifacts, name) is not None
    for name in LEGACY_FACADE_EXPORTS:
        if name in ARTIFACT_PUBLIC_EXPORTS:
            assert getattr(artifacts, name) is getattr(schema, name)


def test_every_repository_schema_import_resolves_through_the_facade() -> None:
    imported_names: set[str] = set()
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            is_absolute = node.level == 0 and node.module == "commcanary.schema"
            is_package_relative = (
                node.level == 1 and node.module == "schema" and path.is_relative_to(ROOT / "src" / "commcanary")
            )
            if is_absolute or is_package_relative:
                imported_names.update(alias.name for alias in node.names if alias.name != "*")

    for path, source in _documented_python_blocks():
        tree = ast.parse(source, filename=f"{path} documented Python block")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "commcanary.schema":
                imported_names.update(alias.name for alias in node.names if alias.name != "*")

    assert imported_names
    assert not sorted(name for name in imported_names if not hasattr(schema, name))


def test_artifact_contract_dependency_graph_is_downward_acyclic_and_stdlib_only() -> None:
    artifact_paths = sorted(ARTIFACT_DIR.glob("*.py"))
    graph = {_artifact_module(path): _artifact_edges(path) for path in artifact_paths}
    _assert_acyclic(graph)

    permitted_parent_modules = {
        "commcanary.errors",
        "commcanary.formats",
        "commcanary.resources",
        "commcanary.statistics",
    }
    forbidden_engine_modules = {
        "commcanary.baselines",
        "commcanary.behavior_config",
        "commcanary.capture",
        "commcanary.cli",
        "commcanary.compare",
        "commcanary.compiler",
        "commcanary.html_report",
        "commcanary.interop",
        "commcanary.operation_identity",
        "commcanary.reduce",
        "commcanary.replay",
        "commcanary.schema",
    }

    for path in artifact_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(alias.name.startswith("_") for alias in node.names), (
                    f"{path.name} imports a private cross-module name"
                )
                if node.level == 2 and node.module is not None:
                    target = f"commcanary.{node.module}"
                    assert target in permitted_parent_modules, f"upward artifact dependency: {target}"
                if node.level == 0 and node.module is not None:
                    root_name = node.module.split(".", 1)[0]
                    assert root_name in STDLIB_IMPORT_ROOTS, f"third-party artifact dependency: {node.module}"
                    assert node.module not in forbidden_engine_modules
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", 1)[0]
                    assert root_name in STDLIB_IMPORT_ROOTS, f"third-party artifact dependency: {alias.name}"


def test_schema_facade_contains_no_contract_implementation() -> None:
    path = ROOT / "src" / "commcanary" / "schema.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert definitions == []
    assert importlib.import_module("commcanary.schema") is schema
