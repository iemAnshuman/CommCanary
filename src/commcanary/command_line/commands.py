"""Pure CLI command handlers for domain services and presentation."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from ..adapters.kineto import (
    kineto_trace_to_commcanary_trace,
    kineto_traces_to_commcanary_trace,
    load_kineto_trace_with_identity,
)
from ..adapters.param import canary_to_param_comms_trace, write_param_comms_trace
from ..artifacts import (
    SENSITIVE_JSON_POLICY,
    atomic_write_json,
    load_json,
    validate_report,
    validate_trace,
    write_json,
)
from ..artifacts.wire import JsonDict, as_float, as_int
from ..comparison import compare_reports
from ..errors import CommCanaryError, SchemaError
from ..execution import (
    distributed_execution_environment,
    execute_qualification_materialization,
)
from ..experimental import (
    clustering_representative_baseline_trace,
    frequency_representative_baseline_trace,
    isolated_collective_baseline_trace,
    random_sampling_baseline_trace,
    stratified_sampling_baseline_trace,
)
from ..replay import replay_canary
from ..reporting import write_compare_html, write_report_html
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ..services import (
    compile_trace,
    ddmin_ranking_reduction,
    prepare_qualification_request,
    synthesize_behavioral_canary,
    verify_qualification_request,
)
from ..verification.canary import verify_canary_behavior, verify_canary_fidelity
from ..verification.report import verify_report_against_canary
from ..workflows import materialize_qualification, verify_qualification_materialization
from .codes import EXIT_SUCCESS

DiagnosticEmitter = Callable[..., None]
ElapsedClock = Callable[[float], float]
AblationSplitter = Callable[[List[str]], List[str]]


def split_ablations(values: List[str]) -> List[str]:
    result: List[str] = []
    for value in values or []:
        result.extend(item.strip() for item in str(value).split(",") if item.strip())
    return result


def compile_command(
    args: Any,
    *,
    diagnostic_emitter: DiagnosticEmitter,
    elapsed_clock: ElapsedClock,
) -> int:
    if args.behavior_search and not args.search_evidence_output:
        raise SchemaError("--search-evidence-output is required with --behavior-search")
    if not args.behavior_search and args.search_evidence_output:
        raise SchemaError("--search-evidence-output requires --behavior-search")
    if args.search_evidence_output and os.path.abspath(args.search_evidence_output) == os.path.abspath(args.output):
        raise SchemaError("canary and search evidence outputs must be different paths")
    trace = load_json(args.trace)
    common_kwargs = {
        "max_events": args.max_events,
        "max_gap_error_us": args.max_gap_error_us,
        "max_skew_error_us": args.max_skew_error_us,
        "max_arrival_offset_error_us": args.max_arrival_offset_error_us,
        "max_compute_before_error_us": args.max_compute_before_error_us,
        "max_overlap_error_us": args.max_overlap_error_us,
        "max_pressure_error": args.max_pressure_error,
        "max_observed_exposed_error_us": args.max_observed_exposed_error_us,
        "max_prefix_gap_error_us": args.max_prefix_gap_error_us,
        "require_lossless_timing": args.lossless_timing,
        "enable_sequence_motifs": not args.disable_sequence_motifs,
    }
    if args.behavior_search:
        phase_started = time.monotonic()
        candidates_planned = max(0, args.timing_sample_limit - args.behavior_search_min_sample_limit + 1)
        if args.diagnostics_json:
            diagnostic_emitter(
                args,
                event="progress",
                exit_code=EXIT_SUCCESS,
                phase="behavior_search",
                status="started",
                uniform_candidates_planned=candidates_planned,
            )
        else:
            print(
                f"behavior search: evaluating up to {candidates_planned} uniform candidates plus per-group refinement",
                file=sys.stderr,
            )
        search_evidence: JsonDict = {}
        canary = synthesize_behavioral_canary(
            trace,
            min_timing_sample_limit=args.behavior_search_min_sample_limit,
            max_timing_sample_limit=args.timing_sample_limit,
            evidence_output=search_evidence,
            **common_kwargs,
        )
        search = canary.get("compiler", {}).get("behavior_search", {})
        if args.diagnostics_json:
            diagnostic_emitter(
                args,
                event="progress",
                exit_code=EXIT_SUCCESS,
                phase="behavior_search",
                status="completed",
                elapsed_seconds=elapsed_clock(phase_started),
                uniform_candidates_evaluated=search.get("search_space", {}).get("uniform_candidate_count"),
                accepted_candidates=search.get("accepted_candidates"),
                selected_timing_sample_limit=search.get("selected_candidate", {}).get("timing_sample_limit"),
            )
    else:
        canary = compile_trace(
            trace,
            timing_sample_limit=args.timing_sample_limit,
            require_behavior_verification=args.require_behavior_verification,
            **common_kwargs,
        )
    if args.behavior_search:
        write_json(args.search_evidence_output, search_evidence)
    write_json(args.output, canary)
    compiler = canary["compiler"]
    fidelity = compiler.get("fidelity", {})
    print(
        "compiled "
        f"{compiler['source_events']} trace events into "
        f"{compiler['canary_events']} canary events; "
        f"event ratio={compiler['event_compression_ratio']}x, "
        f"byte ratio={compiler['byte_compression_ratio']}x, "
        f"timing={fidelity.get('mode', 'unknown')}"
    )
    if args.behavior_search:
        evidence_identity = compiler["behavior_search"]["evidence"]
        print(
            "behavior search sizes: "
            f"executable_canary_bytes={compiler['canary_bytes']} "
            f"search_evidence_bytes={evidence_identity['canonical_bytes']} "
            f"source_trace_bytes={compiler['source_bytes']} "
            "physical_execution_duration=not_observed"
        )
    if fidelity.get("mode") == "bounded_approximate":
        print(
            "approximation: "
            f"gap<={fidelity.get('max_gap_error_us', 0.0)} us, "
            f"skew<={fidelity.get('max_skew_error_us', 0.0)} us, "
            f"compute-before<={fidelity.get('max_compute_before_error_us', 0.0)} us, "
            f"pressure<={fidelity.get('max_pressure_error', 0.0)}, "
            f"prefix-gap<={fidelity.get('max_prefix_gap_error_us', 0.0)} us"
        )
    return 0


def baseline_command(args: Any) -> int:
    trace = load_json(args.trace)
    option_values = {
        "sample_count": args.sample_count,
        "cluster_count": args.cluster_count,
        "strata_per_group": args.strata_per_group,
        "seed": args.seed,
        "partial": args.partial,
    }
    allowed_options = {
        "isolated": set(),
        "random": {"sample_count", "seed", "partial"},
        "frequency": set(),
        "cluster": {"cluster_count"},
        "stratified": {"strata_per_group", "seed"},
    }[args.method]
    inapplicable = sorted(
        name for name, value in option_values.items() if name not in allowed_options and value not in (None, False)
    )
    if inapplicable:
        flags = ", ".join("--" + name.replace("_", "-") for name in inapplicable)
        raise CommCanaryError(f"baseline method {args.method!r} does not accept {flags}")
    if args.method == "isolated":
        baseline = isolated_collective_baseline_trace(trace)
    elif args.method == "random":
        baseline = random_sampling_baseline_trace(
            trace,
            sample_count=8 if args.sample_count is None else args.sample_count,
            seed=0 if args.seed is None else args.seed,
            preserve_source_event_count=not args.partial,
        )
    elif args.method == "frequency":
        baseline = frequency_representative_baseline_trace(trace)
    elif args.method == "cluster":
        baseline = clustering_representative_baseline_trace(
            trace,
            cluster_count=8 if args.cluster_count is None else args.cluster_count,
        )
    elif args.method == "stratified":
        baseline = stratified_sampling_baseline_trace(
            trace,
            strata_per_group=4 if args.strata_per_group is None else args.strata_per_group,
            seed=0 if args.seed is None else args.seed,
        )
    else:  # pragma: no cover - argparse constrains this.
        raise CommCanaryError(f"unknown baseline method {args.method!r}")
    write_json(args.output, baseline)
    print(f"wrote {args.method} baseline trace with {len(baseline['events'])} events: {args.output}")
    return 0


def reduce_command(
    args: Any,
    *,
    diagnostic_emitter: DiagnosticEmitter,
    elapsed_clock: ElapsedClock,
) -> int:
    trace = load_json(args.trace)
    phase_started = time.monotonic()
    raw_events = trace.get("events")
    source_events = len(raw_events) if isinstance(raw_events, list) else None
    if args.diagnostics_json:
        diagnostic_emitter(
            args,
            event="progress",
            exit_code=EXIT_SUCCESS,
            phase="reduction",
            status="started",
            source_events=source_events,
            oracle_call_budget=args.max_oracle_calls,
        )
    else:
        print(f"reduction: oracle-call budget {args.max_oracle_calls}", file=sys.stderr)
    reduced = ddmin_ranking_reduction(
        trace,
        ranking_tie_tolerance_us=args.ranking_tie_tolerance_us,
        timing_sample_limit=args.timing_sample_limit,
        max_oracle_calls=args.max_oracle_calls,
    )
    write_json(args.output, reduced)
    reduction = reduced["workload"]["reduction"]
    if args.diagnostics_json:
        diagnostic_emitter(
            args,
            event="progress",
            exit_code=EXIT_SUCCESS,
            phase="reduction",
            status="completed",
            elapsed_seconds=elapsed_clock(phase_started),
            source_events=reduction["original_events"],
            reduced_events=reduction["reduced_events"],
            oracle_calls=reduction["oracle_calls"],
            budget_exhausted=reduction["budget_exhausted"],
        )
    print(
        "ddmin reduced {original} -> {reduced} events in {calls} oracle calls: {output}".format(
            original=reduction["original_events"],
            reduced=reduction["reduced_events"],
            calls=reduction["oracle_calls"],
            output=args.output,
        )
    )
    if reduction["budget_exhausted"]:
        print("warning: oracle call budget exhausted; result may not be 1-minimal", file=sys.stderr)
    return 0


def import_kineto_command(args: Any) -> int:
    trace, _limits = _import_kineto_profiles(args)
    write_json(args.output, trace)
    workload = trace["workload"]
    print(
        "imported {events} collective events; overlap {derived} derived, {unknown} unknown; "
        "message shapes {shape_derived} derived, {shape_unknown} unavailable; "
        "reduction operators {reduction_derived} derived, {reduction_unknown} unknown; "
        "broadcast roots {root_derived} derived, {root_unknown} unknown "
        "(skipped {control} control, {empty} empty): {output}".format(
            events=workload["imported_events"],
            derived=workload["overlap_derived_events"],
            unknown=workload["overlap_unknown_events"],
            shape_derived=workload["message_shapes_derived_events"],
            shape_unknown=workload["message_shapes_unknown_events"],
            reduction_derived=workload["reduction_ops_derived_events"],
            reduction_unknown=workload["reduction_ops_unknown_events"],
            root_derived=workload["broadcast_roots_derived_events"],
            root_unknown=workload["broadcast_roots_unknown_events"],
            control=workload["skipped_control_events"],
            empty=workload["skipped_empty_events"],
            output=args.output,
        )
    )
    return 0


def prepare_qualification_command(args: Any) -> int:
    trace, limits = _import_kineto_profiles(args)
    canary = compile_trace(
        trace,
        timing_sample_limit=args.timing_sample_limit,
        require_lossless_timing=not args.allow_bounded_timing,
        limits=limits,
    )
    request = prepare_qualification_request(
        args.output_directory,
        trace,
        canary,
        limits=limits,
    )
    print(f"prepared source-verified qualification request {request['request_id']}: {args.output_directory}")
    print("exact rank-local compute work is bound; physical measurement and verdict are not included")
    return 0


def verify_qualification_command(args: Any) -> int:
    request = verify_qualification_request(args.bundle_directory)
    print(f"verified portable qualification request: {request['request_id']}")
    print("assurance: source correspondence verified; physical fidelity unproven; qualification verdict not issued")
    return 0


def materialize_qualification_command(args: Any) -> int:
    materialization = materialize_qualification(
        args.bundle_directory,
        args.output_directory,
    )
    print(
        f"materialized request-bound exact-work program {materialization['materialization_id']}: {args.output_directory}"
    )
    print(
        "source-bound work: {events} events, {operations} rank-local GEMMs, {flops} mathematical FLOPs".format(
            events=materialization["compute_work"]["event_count"],
            operations=materialization["compute_work"]["operation_count"],
            flops=materialization["compute_work"]["matmul_flop_count"],
        )
    )
    print("target timing calibration is not used; physical measurement and qualification verdict are not included")
    return 0


def verify_materialization_command(args: Any) -> int:
    materialization = verify_qualification_materialization(
        args.bundle_directory,
        args.materialization_directory,
    )
    print(f"verified request-bound qualification materialization: {materialization['materialization_id']}")
    print(
        "assurance: exact rank-local work and program recomputed from the source trace; "
        "physical execution and verdict not included"
    )
    return 0


def execute_materialization_command(args: Any) -> int:
    rank, world_size, local_rank = distributed_execution_environment(os.environ)
    result = execute_qualification_materialization(
        args.bundle_directory,
        args.materialization_directory,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=args.device,
        backend=args.backend,
        iterations=args.iterations,
        warmup=args.warmup,
        distributed_timeout_seconds=args.distributed_timeout_seconds,
    )
    if result is None:
        return 0
    atomic_write_json(
        args.output,
        result,
        indent=2,
        policy=replace(
            SENSITIVE_JSON_POLICY,
            artifact_label="reference execution diagnostic",
            overwrite=False,
        ),
    )
    print(f"wrote bound reference-execution diagnostic: {args.output}")
    print(
        "assurance: self-reported physical execution; reference executor conformance "
        "and physical fidelity unproven; qualification verdict not issued"
    )
    return 0


def _import_kineto_profiles(args: Any) -> tuple[Dict[str, Any], ResourceLimits]:
    limits = DEFAULT_RESOURCE_LIMITS
    if args.max_input_bytes is not None:
        if args.max_input_bytes < 1:
            raise CommCanaryError("--max-input-bytes must be a positive integer")
        limits = replace(limits, max_input_bytes=args.max_input_bytes)
    if args.max_json_items is not None:
        if args.max_json_items < 1:
            raise CommCanaryError("--max-json-items must be a positive integer")
        limits = replace(limits, max_json_items=args.max_json_items)
    loaded_profiles = [load_kineto_trace_with_identity(path, limits=limits) for path in args.kineto_trace]
    kinetos = [profile for profile, _identity in loaded_profiles]
    if len(kinetos) == 1:
        if args.assume_shared_clock or args.clock_offset_us:
            raise CommCanaryError("clock-alignment options require at least two Kineto rank profiles")
        trace = kineto_trace_to_commcanary_trace(
            kinetos[0],
            workload_name=args.workload_name,
            phase=args.phase,
            process_group=args.process_group,
            limits=limits,
        )
    else:
        trace = kineto_traces_to_commcanary_trace(
            kinetos,
            workload_name=args.workload_name,
            phase=args.phase,
            process_group=args.process_group,
            clock_offsets_us=_parse_clock_offsets(args.clock_offset_us),
            assume_shared_clock=args.assume_shared_clock,
            limits=limits,
        )
    trace["system"]["kineto_source_profiles"] = _kineto_source_profiles(
        loaded_profiles,
        multi_rank=len(kinetos) > 1,
    )
    validate_trace(trace, limits=limits)
    return trace, limits


def _kineto_source_profiles(
    loaded_profiles: List[tuple[Dict[str, Any], Dict[str, Any]]],
    *,
    multi_rank: bool,
) -> List[Dict[str, Any]]:
    """Return path-free, exact-byte identities for successfully imported profiles."""

    identities: List[Dict[str, Any]] = []
    for profile, source_identity in loaded_profiles:
        identity = dict(source_identity)
        distributed = profile.get("distributedInfo")
        if isinstance(distributed, dict) and "rank" in distributed:
            identity["rank"] = as_int(distributed.get("rank"))
        elif multi_rank:
            # The multi-rank converter reports the richer profile-index error.
            # This guard keeps the helper independently fail closed.
            raise CommCanaryError("multi-rank Kineto source identity requires distributedInfo.rank")
        identities.append(identity)
    return sorted(
        identities,
        key=lambda identity: (
            identity.get("rank", -1),
            identity["sha256"],
        ),
    )


def _parse_clock_offsets(values: List[str]) -> Optional[Dict[int, float]]:
    if not values:
        return None
    offsets: Dict[int, float] = {}
    for value in values:
        if not isinstance(value, str) or value.count("=") != 1:
            raise CommCanaryError("--clock-offset-us must use RANK=OFFSET")
        raw_rank, raw_offset = value.split("=", 1)
        rank = as_int(raw_rank)
        if rank < 0:
            raise CommCanaryError("--clock-offset-us rank must be non-negative")
        if rank in offsets:
            raise CommCanaryError(f"--clock-offset-us repeats rank {rank}")
        offsets[rank] = as_float(raw_offset)
    return offsets


def export_param_command(args: Any) -> int:
    canary = load_json(args.canary)
    entries = canary_to_param_comms_trace(
        canary,
        dtype=args.dtype,
        skip_unsupported=args.skip_unsupported,
        compute_fill_us_per_gemm=args.compute_fill_us_per_gemm,
        compute_fill_gemm_dim=args.compute_fill_gemm_dim,
        compute_fill_dtype=args.compute_fill_dtype,
        overlap_structure=args.overlap_structure,
    )
    write_param_comms_trace(args.output, entries)
    print(f"exported {len(entries)} legacy PARAM-basic-derived entries: {args.output}")
    return 0


def replay_command(
    args: Any,
    *,
    ablation_splitter: AblationSplitter,
) -> int:
    canary = load_json(args.canary)
    report = replay_canary(
        canary,
        backend_label=args.backend_label,
        bandwidth_gbps=args.bandwidth_gbps,
        latency_floor_us=args.latency_floor_us,
        compute_pressure=args.compute_pressure,
        overlap_efficiency=args.overlap_efficiency,
        iterations=args.iterations,
        seed=args.seed,
        include_samples=args.include_samples,
        max_replay_events=args.max_replay_events,
        ablations=ablation_splitter(args.ablate),
    )
    write_json(args.output, report)
    if args.html:
        write_report_html(args.html, report)
    metrics = report["metrics"]
    print(
        f"replayed {metrics['count']} events: "
        f"median={metrics['median_us']} us p95={metrics['p95_us']} us "
        f"p99={metrics['p99_us']} us hidden={metrics['communication_hidden_pct']}%"
    )
    return 0


def compare_command(args: Any) -> int:
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    comparison = compare_reports(
        baseline,
        candidate,
        p99_threshold_pct=args.p99_threshold_pct,
        p95_threshold_pct=args.p95_threshold_pct,
        median_threshold_pct=args.median_threshold_pct,
        p99_absolute_threshold_us=args.p99_absolute_threshold_us,
        p95_absolute_threshold_us=args.p95_absolute_threshold_us,
        median_absolute_threshold_us=args.median_absolute_threshold_us,
        hidden_drop_threshold_points=args.hidden_drop_threshold_points,
        breakdown_threshold_pct=args.breakdown_threshold_pct,
        breakdown_absolute_threshold_us=args.breakdown_absolute_threshold_us,
        require_compatible=not args.allow_mismatch,
    )
    write_json(args.output, comparison)
    if args.html:
        write_compare_html(args.html, comparison)
    print(f"comparison verdict: {comparison['verdict']}")
    for reason in comparison["reasons"]:
        print(f"- {reason}")
    return 0 if comparison["verdict"] != "fail" else 1


def verify_fidelity_command(args: Any) -> int:
    trace = load_json(args.trace)
    canary = load_json(args.canary)
    verification = verify_canary_fidelity(trace, canary)
    write_json(args.output, verification)
    print(f"fidelity verification: {verification['status']}")
    for check in verification["checks"]:
        print(f"- {check['name']}: {check['status']}")
    return 0 if verification["status"] == "source_verified" else 1


def verify_behavior_command(args: Any) -> int:
    trace = load_json(args.trace)
    canary = load_json(args.canary)
    verification = verify_canary_behavior(
        trace,
        canary,
        relative_tolerance_pct=args.relative_tolerance_pct,
        absolute_tolerance_us=args.absolute_tolerance_us,
        hidden_tolerance_points=args.hidden_tolerance_points,
        tail_recall_threshold=args.tail_recall_threshold,
        ranking_tie_tolerance_us=args.ranking_tie_tolerance_us,
    )
    write_json(args.output, verification)
    print(f"behavior verification: {verification['status']}")
    print(f"- representation fidelity: {verification['representation_fidelity_status']}")
    print(f"- source verified: {verification['source_verified_status']}")
    print(f"- deterministic-model behavior: {verification['model_behavior_preservation_status']}")
    print(f"- configuration ranking: {verification['configuration_ranking_status']}")
    for row in verification["configurations"]:
        print(f"- {row['name']}: {row['status']}")
    print(f"- ranking: {verification['ranking']['status']}")
    return 0 if verification["status"] == "model_behavior_preserved" else 1


def verify_report_command(args: Any) -> int:
    report = load_json(args.report)
    canary = load_json(args.canary)
    verification = verify_report_against_canary(report, canary)
    write_json(args.output, verification)
    print(f"report verification: {verification['status']}")
    for check in verification["checks"]:
        print(f"- {check['name']}: {check['status']}")
    return 0 if verification["status"] == "model_recomputed" else 1


def report_command(
    args: Any,
    *,
    diagnostic_emitter: DiagnosticEmitter,
) -> int:
    report = load_json(args.report)
    validate_report(report)
    write_report_html(args.output, report)
    if args.deprecated_report_alias:
        if args.diagnostics_json:
            diagnostic_emitter(
                args,
                event="deprecation",
                exit_code=EXIT_SUCCESS,
                replacement="render-html",
                removal_version="0.5.0",
            )
        else:
            print("commcanary: 'report' is deprecated; use 'render-html' (removal in 0.5.0)", file=sys.stderr)
    print(f"wrote HTML report: {args.output}")
    return 0


__all__ = [
    "AblationSplitter",
    "DiagnosticEmitter",
    "ElapsedClock",
    "baseline_command",
    "compare_command",
    "compile_command",
    "export_param_command",
    "import_kineto_command",
    "reduce_command",
    "replay_command",
    "report_command",
    "split_ablations",
    "verify_behavior_command",
    "verify_fidelity_command",
    "verify_report_command",
]
