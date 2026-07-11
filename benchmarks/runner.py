"""Reproducible, manifest-driven local benchmark runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from commcanary.capture import merge_trace_shards
from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace, synthesize_behavioral_canary
from commcanary.interop import canary_to_param_comms_trace
from commcanary.replay import replay_canary, verify_report_against_canary
from commcanary.resources import DEFAULT_RESOURCE_LIMITS
from commcanary.schema import (
    JsonDict,
    SchemaError,
    canary_execution_sha256,
    canonical_json_bytes,
    load_json,
    preflight_canary_expansion,
    validate_canary,
    validate_trace,
    write_json,
)
from commcanary.version import package_version

from .fixtures import (
    FIXTURE_MANIFEST_FORMAT,
    generate_behavior_search_trace,
    materialize_capture_shards,
    materialize_fixture_set,
)

BENCHMARK_RESULT_FORMAT = "commcanary.benchmark-result.v1"
BENCHMARK_SUITE_FORMAT = "commcanary.benchmark-suite.v1"
RESULT_HASH_ALGORITHM = "sha256-canonical-json-v1"

# These capability-oriented operations all use the same registry, isolation,
# measurement, environment, and semantic-hash path as the original families.
EXTENSION_OPERATION_FAMILIES = (
    "compare",
    "capture_merge",
    "param_export",
    "behavior_search",
)


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    kind: str
    path: Path
    stored_events: int
    logical_events: int
    sha256: str


@dataclass(frozen=True)
class OperationContext:
    case: FixtureCase
    document: Mapping[str, Any]
    prepared: Optional[Mapping[str, Any]] = None


Operation = Callable[[OperationContext], Any]
_OPERATIONS: Dict[str, Tuple[Tuple[str, ...], Operation]] = {}


def benchmark_operation(name: str, *, kinds: Sequence[str]) -> Callable[[Operation], Operation]:
    """Register one timed operation while keeping result handling uniform."""

    supported = tuple(kinds)

    def register(function: Operation) -> Operation:
        if name in _OPERATIONS:
            raise RuntimeError(f"duplicate benchmark operation {name!r}")
        _OPERATIONS[name] = (supported, function)
        return function

    return register


@benchmark_operation("load", kinds=("trace", "canary"))
def _load(context: OperationContext) -> Mapping[str, Any]:
    return load_json(str(context.case.path))


@benchmark_operation("validate", kinds=("trace", "canary"))
def _validate(context: OperationContext) -> None:
    if context.case.kind == "trace":
        validate_trace(context.document)
    else:
        validate_canary(context.document)


@benchmark_operation("hash", kinds=("trace", "canary"))
def _hash(context: OperationContext) -> str:
    if context.case.kind == "canary":
        return canary_execution_sha256(context.document)
    return hashlib.sha256(canonical_json_bytes(context.document)).hexdigest()


@benchmark_operation("compile", kinds=("trace",))
def _compile(context: OperationContext) -> Mapping[str, Any]:
    return compile_trace(context.document)


@benchmark_operation("replay", kinds=("canary",))
def _replay(context: OperationContext) -> Mapping[str, Any]:
    return replay_canary(context.document)


@benchmark_operation("verify", kinds=("canary",))
def _verify(context: OperationContext) -> Mapping[str, Any]:
    if context.prepared is None:
        raise RuntimeError("verify benchmark requires a prepared replay report")
    return verify_report_against_canary(context.prepared, context.document)


@benchmark_operation("compare", kinds=("canary",))
def _compare(context: OperationContext) -> Mapping[str, Any]:
    prepared = _require_prepared(context, "compare")
    baseline = prepared.get("baseline")
    candidate = prepared.get("candidate")
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise RuntimeError("compare benchmark preparation is invalid")
    return compare_reports(baseline, candidate)


@benchmark_operation("capture_merge", kinds=("trace",))
def _capture_merge(context: OperationContext) -> Mapping[str, Any]:
    prepared = _require_prepared(context, "capture_merge")
    shard_dir = prepared.get("shard_dir")
    workload_name = prepared.get("workload_name")
    if not isinstance(shard_dir, str) or not isinstance(workload_name, str):
        raise RuntimeError("capture-merge benchmark preparation is invalid")
    return merge_trace_shards(shard_dir, workload_name=workload_name)


@benchmark_operation("param_export", kinds=("canary",))
def _param_export(context: OperationContext) -> Any:
    return canary_to_param_comms_trace(context.document)


@benchmark_operation("behavior_search", kinds=("trace",))
def _behavior_search(context: OperationContext) -> Mapping[str, Any]:
    prepared = _require_prepared(context, "behavior_search")
    trace = prepared.get("trace")
    if not isinstance(trace, Mapping):
        raise RuntimeError("behavior-search benchmark preparation is invalid")
    return synthesize_behavioral_canary(
        trace,
        min_timing_sample_limit=2,
        max_timing_sample_limit=3,
    )


@benchmark_operation("capture_merge_preflight", kinds=("trace",))
def _capture_merge_preflight(context: OperationContext) -> Mapping[str, Any]:
    prepared = _require_prepared(context, "capture_merge_preflight")
    shard_dir = prepared.get("shard_dir")
    workload_name = prepared.get("workload_name")
    if not isinstance(shard_dir, str) or not isinstance(workload_name, str):
        raise RuntimeError("capture-merge preflight preparation is invalid")
    return _expected_preflight_rejection(
        "max_capture_shards",
        "capture shards=2 exceeds limit=1",
        lambda: merge_trace_shards(
            shard_dir,
            workload_name=workload_name,
            limits=replace(DEFAULT_RESOURCE_LIMITS, max_capture_shards=1),
        ),
    )


@benchmark_operation("param_export_preflight", kinds=("canary",))
def _param_export_preflight(context: OperationContext) -> Mapping[str, Any]:
    return _expected_preflight_rejection(
        "max_param_entries",
        "exceeds limit=1",
        lambda: canary_to_param_comms_trace(
            context.document,
            limits=replace(DEFAULT_RESOURCE_LIMITS, max_param_entries=1),
        ),
    )


@benchmark_operation("behavior_search_preflight", kinds=("trace",))
def _behavior_search_preflight(context: OperationContext) -> Mapping[str, Any]:
    prepared = _require_prepared(context, "behavior_search_preflight")
    trace = prepared.get("trace")
    if not isinstance(trace, Mapping):
        raise RuntimeError("behavior-search preflight preparation is invalid")
    return _expected_preflight_rejection(
        "max_behavior_candidates",
        "would evaluate 2 candidates",
        lambda: synthesize_behavioral_canary(
            trace,
            min_timing_sample_limit=2,
            max_timing_sample_limit=3,
            limits=replace(DEFAULT_RESOURCE_LIMITS, max_behavior_candidates=1),
        ),
    )


DEFAULT_OPERATIONS: Mapping[str, Tuple[str, ...]] = {
    "trace": ("load", "validate", "hash", "compile", "capture_merge", "behavior_search"),
    "canary": ("load", "validate", "hash", "replay", "verify", "compare", "param_export"),
}


def run_case(case: FixtureCase, operation: str, *, iteration: int = 0) -> JsonDict:
    """Run one operation in the current process and return a self-contained result."""

    if operation not in _OPERATIONS:
        raise ValueError(f"unknown benchmark operation {operation!r}")
    kinds, function = _OPERATIONS[operation]
    if case.kind not in kinds:
        raise ValueError(f"operation {operation!r} does not support {case.kind!r} fixtures")
    _verify_fixture_hash(case)

    with tempfile.TemporaryDirectory(prefix=f"commcanary-benchmark-{operation}-") as temp_dir:
        document: Mapping[str, Any] = {}
        if operation != "load":
            document = load_json(str(case.path))
        prepared, prepared_sha256 = _prepare_operation(
            case,
            operation,
            document,
            Path(temp_dir),
        )
        context = OperationContext(case=case, document=document, prepared=prepared)

        rss_before, rss_method = _peak_rss_bytes()
        tracemalloc.start()
        started_ns = time.perf_counter_ns()
        try:
            output = function(context)
            elapsed_ns = time.perf_counter_ns() - started_ns
            _current_python_bytes, peak_python_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        rss_after, _ = _peak_rss_bytes()
        semantic_sha256 = _semantic_output_sha256(operation, context, output)

        result: JsonDict = {
            "format": BENCHMARK_RESULT_FORMAT,
            "case_id": case.case_id,
            "input_kind": case.kind,
            "input_sha256": case.sha256,
            "operation": operation,
            "iteration": iteration,
            "stored_events": case.stored_events,
            "logical_events": case.logical_events,
            "measurement_scope": "registered-operation-only",
            "preparation_included_in_wall_time": False,
            "rss_baseline_after_preparation": True,
            "wall_time_seconds": elapsed_ns / 1_000_000_000.0,
            "peak_rss_bytes": rss_after,
            "peak_rss_baseline_bytes": rss_before,
            "peak_rss_method": rss_method,
            "python_peak_allocated_bytes": peak_python_bytes,
            "semantic_sha256": semantic_sha256,
            "semantic_hash_algorithm": RESULT_HASH_ALGORITHM,
            "environment": benchmark_environment(),
        }
        if prepared_sha256 is not None:
            result["prepared_input_semantic_sha256"] = prepared_sha256
        return result


def _prepare_operation(
    case: FixtureCase,
    operation: str,
    document: Mapping[str, Any],
    workspace: Path,
) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    if operation in {"verify", "compare"}:
        report = replay_canary(document)
        stable_report = _stable_report_projection(report)
        if operation == "verify":
            return report, _mapping_sha256(stable_report)
        prepared_compare: JsonDict = {"baseline": report, "candidate": report}
        return prepared_compare, _mapping_sha256({"baseline": stable_report, "candidate": stable_report})

    if operation in {"capture_merge", "capture_merge_preflight"}:
        shard_dir = workspace / "capture-shards"
        shard_paths = materialize_capture_shards(document, shard_dir)
        workload = document.get("workload")
        workload_name = "commcanary-benchmark-capture"
        if isinstance(workload, Mapping) and isinstance(workload.get("name"), str):
            workload_name = str(workload["name"])
        prepared_capture: JsonDict = {
            "shard_dir": str(shard_dir),
            "workload_name": workload_name,
        }
        shard_hashes = [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in shard_paths
        ]
        return prepared_capture, _mapping_sha256({"shards": shard_hashes})

    if operation in {"behavior_search", "behavior_search_preflight"}:
        behavior_trace = generate_behavior_search_trace(case.stored_events)
        prepared_behavior: JsonDict = {"trace": behavior_trace}
        return prepared_behavior, _mapping_sha256(behavior_trace)

    return None, None


def _require_prepared(context: OperationContext, operation: str) -> Mapping[str, Any]:
    if context.prepared is None:
        raise RuntimeError(f"{operation} benchmark requires prepared input")
    return context.prepared


def _expected_preflight_rejection(
    limit_name: str,
    expected_message: str,
    operation: Callable[[], Any],
) -> JsonDict:
    try:
        operation()
    except SchemaError as exc:
        if expected_message not in str(exc):
            raise RuntimeError(f"{limit_name} adversarial benchmark rejected for an unexpected reason: {exc}") from exc
        return {
            "status": "rejected_preflight",
            "limit": limit_name,
            "error_type": "SchemaError",
        }
    raise RuntimeError(f"{limit_name} adversarial benchmark did not reject oversized work")


def run_manifest(
    manifest_path: Path,
    *,
    repeats: int = 1,
    operations: Optional[Sequence[str]] = None,
    isolate: bool = True,
    profile: str = "full",
) -> JsonDict:
    """Run verified fixtures, isolating each operation for meaningful peak RSS."""

    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    cases = load_fixture_manifest(manifest_path)
    requested = tuple(operations) if operations is not None else None
    if requested is not None:
        unknown = sorted(set(requested) - set(_OPERATIONS))
        if unknown:
            raise ValueError(f"unknown benchmark operation {unknown[0]!r}")
    results: List[JsonDict] = []
    for case in cases:
        case_operations = requested if requested is not None else DEFAULT_OPERATIONS[case.kind]
        for operation in case_operations:
            if operation not in _OPERATIONS or case.kind not in _OPERATIONS[operation][0]:
                continue
            for iteration in range(repeats):
                result = (
                    _run_case_isolated(case, operation, iteration=iteration)
                    if isolate
                    else run_case(case, operation, iteration=iteration)
                )
                results.append(result)
    return _suite_result(profile=profile, results=results)


def run_smoke(*, isolate: bool = True) -> JsonDict:
    """Run the fast, deterministic local benchmark intended for a future PR gate."""

    with tempfile.TemporaryDirectory(prefix="commcanary-benchmark-smoke-") as temp_dir:
        manifest = materialize_fixture_set(
            Path(temp_dir),
            stored_event_counts=(64,),
            compressed_logical_counts=(64,),
        )
        return run_manifest(
            manifest,
            operations=(
                "load",
                "validate",
                "hash",
                "replay",
                "compare",
                "capture_merge",
                "param_export",
                "behavior_search",
                "capture_merge_preflight",
                "param_export_preflight",
                "behavior_search_preflight",
            ),
            isolate=isolate,
            profile="smoke",
        )


def load_fixture_manifest(manifest_path: Path) -> List[FixtureCase]:
    """Load a manifest, enforce path containment, and verify every file hash."""

    path = Path(manifest_path).resolve()
    manifest = load_json(str(path))
    if manifest.get("format") != FIXTURE_MANIFEST_FORMAT:
        raise ValueError(f"unsupported fixture manifest format {manifest.get('format')!r}")
    raw_fixtures = manifest.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise ValueError("fixture manifest must contain a non-empty fixtures list")
    root = path.parent
    cases: List[FixtureCase] = []
    seen = set()
    for index, raw in enumerate(raw_fixtures):
        if not isinstance(raw, Mapping):
            raise ValueError(f"fixture manifest entry {index} must be an object")
        case_id = _nonempty_text(raw.get("case_id"), f"fixture {index} case_id")
        if case_id in seen:
            raise ValueError(f"duplicate fixture case_id {case_id!r}")
        seen.add(case_id)
        kind = _nonempty_text(raw.get("kind"), f"fixture {index} kind")
        if kind not in DEFAULT_OPERATIONS:
            raise ValueError(f"fixture {case_id!r} has unsupported kind {kind!r}")
        raw_relative = _nonempty_text(raw.get("path"), f"fixture {index} path")
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"fixture {case_id!r} path must stay within the manifest directory")
        fixture_path = (root / relative).resolve()
        try:
            fixture_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"fixture {case_id!r} path escapes the manifest directory") from exc
        case = FixtureCase(
            case_id=case_id,
            kind=kind,
            path=fixture_path,
            stored_events=_non_negative_int(raw.get("stored_events"), f"fixture {case_id} stored_events"),
            logical_events=_non_negative_int(raw.get("logical_events"), f"fixture {case_id} logical_events"),
            sha256=_sha256_text(raw.get("sha256"), f"fixture {case_id} sha256"),
        )
        _verify_fixture_hash(case)
        _verify_fixture_metadata(case)
        cases.append(case)
    return cases


def benchmark_environment() -> JsonDict:
    """Capture the execution environment needed to interpret a measurement."""

    return {
        "commcanary_version": package_version(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "executable": sys.executable,
    }


def _run_case_isolated(case: FixtureCase, operation: str, *, iteration: int) -> JsonDict:
    command = [
        sys.executable,
        "-m",
        "benchmarks",
        "_worker",
        "--case-id",
        case.case_id,
        "--kind",
        case.kind,
        "--input",
        str(case.path),
        "--stored-events",
        str(case.stored_events),
        "--logical-events",
        str(case.logical_events),
        "--sha256",
        case.sha256,
        "--operation",
        operation,
        "--iteration",
        str(iteration),
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"isolated benchmark failed for {case.case_id}/{operation}: {completed.stderr.strip()}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"isolated benchmark emitted invalid JSON: {exc}") from exc
    if not isinstance(result, dict) or result.get("format") != BENCHMARK_RESULT_FORMAT:
        raise RuntimeError("isolated benchmark emitted an invalid result object")
    return result


def _semantic_output_sha256(operation: str, context: OperationContext, output: Any) -> str:
    if operation == "hash":
        if not isinstance(output, str):
            raise RuntimeError("hash operation did not return a digest")
        return output
    if operation in {"compile", "behavior_search"}:
        if not isinstance(output, Mapping):
            raise RuntimeError(f"{operation} operation did not return a canary")
        return canary_execution_sha256(output)
    if operation == "replay":
        if not isinstance(output, Mapping):
            raise RuntimeError("replay operation did not return a report")
        return _mapping_sha256(_stable_report_projection(output))
    if operation == "verify":
        if not isinstance(output, Mapping):
            raise RuntimeError("verify operation did not return a verdict")
        return _mapping_sha256(output)
    if operation == "compare":
        if not isinstance(output, Mapping):
            raise RuntimeError("compare operation did not return a comparison")
        return _mapping_sha256({key: value for key, value in output.items() if key != "created_at"})
    if operation == "capture_merge":
        if not isinstance(output, Mapping):
            raise RuntimeError("capture_merge operation did not return a trace")
        return _mapping_sha256({key: value for key, value in output.items() if key != "created_at"})
    if operation == "param_export":
        if not isinstance(output, list):
            raise RuntimeError("param_export operation did not return an entry list")
        return _json_sha256(output)
    if operation.endswith("_preflight"):
        if not isinstance(output, Mapping):
            raise RuntimeError(f"{operation} operation did not return a rejection record")
        return _mapping_sha256(output)
    document = output if operation == "load" else context.document
    if not isinstance(document, Mapping):
        raise RuntimeError(f"{operation} operation did not preserve a JSON document")
    return _mapping_sha256(document)


def _stable_report_projection(report: Mapping[str, Any]) -> JsonDict:
    keys = (
        "format",
        "canary",
        "simulation_model",
        "replay_protocol",
        "backend",
        "workload",
        "canary_summary",
        "metrics",
        "by_phase",
        "by_op",
        "calibration",
        "samples",
    )
    return {key: report[key] for key in keys if key in report}


def _suite_result(*, profile: str, results: Sequence[Mapping[str, Any]]) -> JsonDict:
    semantic_rows = [
        {
            "case_id": result.get("case_id"),
            "operation": result.get("operation"),
            "iteration": result.get("iteration"),
            "semantic_sha256": result.get("semantic_sha256"),
        }
        for result in results
    ]
    return {
        "format": BENCHMARK_SUITE_FORMAT,
        "profile": profile,
        "result_count": len(results),
        "environment": benchmark_environment(),
        "semantic_set_sha256": _mapping_sha256({"results": semantic_rows}),
        "results": list(results),
    }


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return _json_sha256(value)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _verify_fixture_hash(case: FixtureCase) -> None:
    try:
        actual = hashlib.sha256(case.path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot read fixture {case.path}: {exc}") from exc
    if actual != case.sha256:
        raise ValueError(f"fixture {case.case_id!r} sha256 mismatch")


def _verify_fixture_metadata(case: FixtureCase) -> None:
    document = load_json(str(case.path))
    if case.kind == "trace":
        validate_trace(document)
        events = document.get("events")
        if not isinstance(events, list):
            raise ValueError(f"fixture {case.case_id!r} trace events are invalid")
        stored_events = len(events)
        logical_events = stored_events
    else:
        validate_canary(document)
        expansion = preflight_canary_expansion(document.get("events", []))
        stored_events = expansion.stored_events
        logical_events = expansion.logical_events
    if stored_events != case.stored_events:
        raise ValueError(f"fixture {case.case_id!r} stored_events metadata mismatch")
    if logical_events != case.logical_events:
        raise ValueError(f"fixture {case.case_id!r} logical_events metadata mismatch")


def _peak_rss_bytes() -> Tuple[Optional[int], str]:
    try:
        import resource
    except ImportError:
        return None, "unavailable"
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    operating_system = platform.system().lower()
    if operating_system == "darwin":
        return int(raw), "resource.getrusage.ru_maxrss-bytes"
    if operating_system in {"linux", "freebsd"}:
        return int(raw * 1024), "resource.getrusage.ru_maxrss-kibibytes"
    return None, "resource.getrusage.ru_maxrss-unknown-units"


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha256_text(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _write_suite(path: Path, suite: Mapping[str, Any]) -> None:
    write_json(str(path), suite)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixtures = subparsers.add_parser("fixtures", help="generate the standard 1K/10K/100K fixture set")
    fixtures.add_argument("output_dir", type=Path)

    run = subparsers.add_parser("run", help="run a verified fixture manifest")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--operation", action="append", dest="operations")
    run.add_argument("--in-process", action="store_true", help="disable per-operation process isolation")

    smoke = subparsers.add_parser("smoke", help="run the fast local benchmark smoke suite")
    smoke.add_argument("--output", type=Path)
    smoke.add_argument("--in-process", action="store_true", help="disable per-operation process isolation")

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--case-id", required=True)
    worker.add_argument("--kind", choices=("trace", "canary"), required=True)
    worker.add_argument("--input", type=Path, required=True)
    worker.add_argument("--stored-events", type=int, required=True)
    worker.add_argument("--logical-events", type=int, required=True)
    worker.add_argument("--sha256", required=True)
    worker.add_argument("--operation", required=True)
    worker.add_argument("--iteration", type=int, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fixtures":
        manifest = materialize_fixture_set(args.output_dir)
        print(manifest)
        return 0
    if args.command == "run":
        suite = run_manifest(
            args.manifest,
            repeats=args.repeats,
            operations=args.operations,
            isolate=not args.in_process,
        )
        _write_suite(args.output, suite)
        print(args.output)
        return 0
    if args.command == "smoke":
        suite = run_smoke(isolate=not args.in_process)
        if args.output is not None:
            _write_suite(args.output, suite)
            print(args.output)
        else:
            print(json.dumps(suite, sort_keys=True, separators=(",", ":")))
        return 0
    case = FixtureCase(
        case_id=args.case_id,
        kind=args.kind,
        path=args.input.resolve(),
        stored_events=args.stored_events,
        logical_events=args.logical_events,
        sha256=args.sha256,
    )
    print(json.dumps(run_case(case, args.operation, iteration=args.iteration), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
