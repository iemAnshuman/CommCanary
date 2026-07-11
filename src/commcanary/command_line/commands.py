"""Pure CLI command handlers for domain services and presentation."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, List

from ..adapters.kineto import kineto_trace_to_commcanary_trace, load_kineto_trace
from ..adapters.param import canary_to_param_comms_trace, write_param_comms_trace
from ..artifacts import load_json, validate_report, write_json
from ..comparison import compare_reports
from ..errors import CommCanaryError
from ..experimental import (
    clustering_representative_baseline_trace,
    frequency_representative_baseline_trace,
    isolated_collective_baseline_trace,
    random_sampling_baseline_trace,
    stratified_sampling_baseline_trace,
)
from ..replay import replay_canary
from ..reporting import write_compare_html, write_report_html
from ..services import compile_trace, ddmin_ranking_reduction, synthesize_behavioral_canary
from ..verification.canary import verify_canary_behavior, verify_canary_fidelity
from ..verification.report import verify_report_against_canary
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
        canary = synthesize_behavioral_canary(
            trace,
            min_timing_sample_limit=args.behavior_search_min_sample_limit,
            max_timing_sample_limit=args.timing_sample_limit,
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
                uniform_candidates_evaluated=search.get("candidate_count"),
                accepted_candidates=search.get("accepted_candidates"),
                selected_timing_sample_limit=search.get("selected_timing_sample_limit"),
            )
    else:
        canary = compile_trace(
            trace,
            timing_sample_limit=args.timing_sample_limit,
            require_behavior_verification=args.require_behavior_verification,
            **common_kwargs,
        )
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
    kineto = load_kineto_trace(args.kineto_trace)
    trace = kineto_trace_to_commcanary_trace(
        kineto,
        workload_name=args.workload_name,
        phase=args.phase,
        process_group=args.process_group,
    )
    write_json(args.output, trace)
    workload = trace["workload"]
    print(
        "imported {events} collective events (skipped {control} control, {empty} empty): {output}".format(
            events=workload["imported_events"],
            control=workload["skipped_control_events"],
            empty=workload["skipped_empty_events"],
            output=args.output,
        )
    )
    return 0


def export_param_command(args: Any) -> int:
    canary = load_json(args.canary)
    entries = canary_to_param_comms_trace(
        canary,
        dtype=args.dtype,
        skip_unsupported=args.skip_unsupported,
        compute_fill_us_per_gemm=args.compute_fill_us_per_gemm,
        compute_fill_gemm_dim=args.compute_fill_gemm_dim,
        overlap_structure=args.overlap_structure,
    )
    write_param_comms_trace(args.output, entries)
    print(f"exported {len(entries)} PARAM comms-replay entries: {args.output}")
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
    print(f"- behavioral fidelity: {verification['behavioral_fidelity_status']}")
    print(f"- configuration ranking: {verification['configuration_ranking_status']}")
    for row in verification["configurations"]:
        print(f"- {row['name']}: {row['status']}")
    print(f"- ranking: {verification['ranking']['status']}")
    return 0 if verification["status"] == "behaviorally_verified" else 1


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
