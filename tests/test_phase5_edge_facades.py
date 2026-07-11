from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import commcanary.adapters.capture as adapter_capture
import commcanary.adapters.interop as adapter_interop
import commcanary.capture as legacy_capture
import commcanary.interop as legacy_interop
from commcanary.adapters.capture_merge import merge_trace_shards as direct_merge_trace_shards
from commcanary.adapters.kineto import (
    kineto_trace_to_commcanary_trace as direct_kineto_trace_to_commcanary_trace,
)
from commcanary.adapters.kineto import load_kineto_trace as direct_load_kineto_trace
from commcanary.adapters.param import canary_to_param_comms_trace as direct_canary_to_param_comms_trace
from commcanary.adapters.param import write_param_comms_trace as direct_write_param_comms_trace
from commcanary.compare import (
    ComparisonReasonCode as LegacyReasonCode,
)
from commcanary.compare import (
    ComparisonThresholdPolicy as LegacyThresholdPolicy,
)
from commcanary.compare import (
    compare_reports as legacy_compare_reports,
)
from commcanary.compare import (
    comparison_reason_codes as legacy_reason_codes,
)
from commcanary.comparison import (
    ComparisonReasonCode,
    ComparisonThresholdPolicy,
    compare_reports,
    comparison_reason_codes,
)
from commcanary.html_report import (
    render_compare_html as legacy_render_compare_html,
)
from commcanary.html_report import (
    render_report_html as legacy_render_report_html,
)
from commcanary.html_report import (
    write_compare_html as legacy_write_compare_html,
)
from commcanary.html_report import (
    write_report_html as legacy_write_report_html,
)
from commcanary.reporting import (
    render_compare_html,
    render_report_html,
    write_compare_html,
    write_report_html,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "commcanary"


def test_comparison_and_reporting_compatibility_facades_preserve_objects() -> None:
    assert LegacyReasonCode is ComparisonReasonCode
    assert LegacyThresholdPolicy is ComparisonThresholdPolicy
    assert legacy_compare_reports is compare_reports
    assert legacy_reason_codes is comparison_reason_codes
    assert legacy_render_compare_html is render_compare_html
    assert legacy_render_report_html is render_report_html
    assert legacy_write_compare_html is write_compare_html
    assert legacy_write_report_html is write_report_html


def test_stateful_adapter_facades_preserve_module_patch_points() -> None:
    assert legacy_capture is adapter_capture
    assert legacy_interop is adapter_interop
    assert sys.modules["commcanary.capture"] is adapter_capture
    assert sys.modules["commcanary.interop"] is adapter_interop
    assert getattr(legacy_capture, "load_json") is getattr(adapter_capture, "load_json")
    assert getattr(legacy_interop, "iter_canary_logical_events") is getattr(
        adapter_interop,
        "iter_canary_logical_events",
    )


def test_capture_size_alias_signature_is_identical_at_new_and_old_paths() -> None:
    legacy_signature = inspect.signature(legacy_capture.record_collective)
    adapter_signature = inspect.signature(adapter_capture.record_collective)

    assert legacy_signature == adapter_signature
    assert "byte_count" in legacy_signature.parameters
    assert "bytes" in legacy_signature.parameters
    assert "kwargs" not in legacy_signature.parameters


def test_split_adapter_signatures_match_the_combined_compatibility_boundary() -> None:
    assert inspect.signature(legacy_capture.merge_trace_shards) == inspect.signature(direct_merge_trace_shards)
    assert inspect.signature(legacy_interop.canary_to_param_comms_trace) == inspect.signature(
        direct_canary_to_param_comms_trace
    )
    assert legacy_interop.load_kineto_trace is direct_load_kineto_trace
    assert legacy_interop.kineto_trace_to_commcanary_trace is direct_kineto_trace_to_commcanary_trace
    assert legacy_interop.write_param_comms_trace is direct_write_param_comms_trace


def test_edge_packages_do_not_import_legacy_facades_or_high_level_layers() -> None:
    forbidden_modules = {
        "baselines",
        "capture",
        "cli",
        "compare",
        "html_report",
        "interop",
        "schema",
    }
    package_directories = ("comparison", "adapters", "reporting")

    for directory in package_directories:
        for path in sorted((PACKAGE / directory).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                top_level = node.module.split(".", 1)[0]
                if node.level >= 2:
                    assert top_level not in forbidden_modules, (path, node.lineno, node.module)
                if node.level == 0:
                    assert node.module not in {f"commcanary.{module}" for module in forbidden_modules}, (
                        path,
                        node.lineno,
                        node.module,
                    )
                if node.level >= 1:
                    assert all(not alias.name.startswith("_") for alias in node.names), (
                        path,
                        node.lineno,
                        node.module,
                    )


def test_historical_edge_modules_contain_only_facade_wiring() -> None:
    for name in ("compare.py", "capture.py", "interop.py", "html_report.py"):
        path = PACKAGE / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        implementations = [
            node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert implementations == []


def test_edge_package_dependency_graph_is_acyclic() -> None:
    files = {
        f"{directory}.{path.stem}": path
        for directory in ("comparison", "adapters", "reporting")
        for path in sorted((PACKAGE / directory).glob("*.py"))
    }
    edges: dict[str, set[str]] = {module: set() for module in files}
    for module, path in files.items():
        directory = module.split(".", 1)[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is None:
                continue
            target = f"{directory}.{node.module}"
            if target in files:
                edges[module].add(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        assert module not in visiting, f"edge-package dependency cycle reaches {module}"
        if module in visited:
            return
        visiting.add(module)
        for dependency in edges[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in edges:
        visit(module)


def test_adapter_implementations_have_one_change_boundary_each() -> None:
    capture_tree = ast.parse((PACKAGE / "adapters" / "capture.py").read_text(encoding="utf-8"))
    merge_tree = ast.parse((PACKAGE / "adapters" / "capture_merge.py").read_text(encoding="utf-8"))
    kineto_tree = ast.parse((PACKAGE / "adapters" / "kineto.py").read_text(encoding="utf-8"))
    param_tree = ast.parse((PACKAGE / "adapters" / "param.py").read_text(encoding="utf-8"))

    def definitions(tree: ast.Module) -> set[str]:
        return {
            node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }

    assert "TraceRecorder" in definitions(capture_tree)
    assert "_coalesce_events" not in definitions(capture_tree)
    assert "TraceRecorder" not in definitions(merge_tree)
    assert "_coalesce_events" in definitions(merge_tree)
    assert "kineto_trace_to_commcanary_trace" in definitions(kineto_tree)
    assert "canary_to_param_comms_trace" not in definitions(kineto_tree)
    assert "canary_to_param_comms_trace" in definitions(param_tree)
    assert "kineto_trace_to_commcanary_trace" not in definitions(param_tree)
