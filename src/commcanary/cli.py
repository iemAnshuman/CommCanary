from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import uuid
from typing import Any, List, Optional

from .baselines import (
    clustering_representative_baseline_trace,
    frequency_representative_baseline_trace,
    isolated_collective_baseline_trace,
    random_sampling_baseline_trace,
    stratified_sampling_baseline_trace,
)
from .capture import TraceRecorder, merge_trace_shards
from .compare import compare_reports
from .compiler import compile_trace, synthesize_behavioral_canary, verify_canary_behavior, verify_canary_fidelity
from .html_report import write_compare_html, write_report_html
from .interop import (
    canary_to_param_comms_trace,
    kineto_trace_to_commcanary_trace,
    load_kineto_trace,
    write_param_comms_trace,
)
from .reduce import ddmin_ranking_reduction
from .replay import replay_canary, verify_report_against_canary
from .schema import CommCanaryError, load_json, validate_report, write_json


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CommCanaryError as exc:
        print(f"commcanary: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("commcanary: interrupted", file=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="commcanary", description="Workload-derived communication canaries.")
    sub = parser.add_subparsers(dest="command", required=True)

    compile_parser = sub.add_parser("compile", help="compile a trace into a compact canary")
    compile_parser.add_argument("trace")
    compile_parser.add_argument("--output", "-o", required=True)
    compile_parser.add_argument("--max-events", type=int)
    compile_parser.add_argument("--timing-sample-limit", type=int, default=128)
    compile_parser.add_argument("--max-gap-error-us", type=float)
    compile_parser.add_argument("--max-skew-error-us", type=float)
    compile_parser.add_argument("--max-arrival-offset-error-us", type=float)
    compile_parser.add_argument("--max-compute-before-error-us", type=float)
    compile_parser.add_argument("--max-overlap-error-us", type=float)
    compile_parser.add_argument("--max-pressure-error", type=float)
    compile_parser.add_argument("--max-observed-exposed-error-us", type=float)
    compile_parser.add_argument("--max-prefix-gap-error-us", type=float)
    compile_parser.add_argument(
        "--lossless-timing",
        action="store_true",
        help="fail rather than emit bounded approximate timing intervals",
    )
    compile_parser.add_argument(
        "--disable-sequence-motifs",
        action="store_true",
        help="emit only flat events instead of replay-equivalent multi-event sequence motifs",
    )
    compile_parser.add_argument(
        "--require-behavior-verification",
        action="store_true",
        help="fail compilation unless verify-behavior passes on the generated canary",
    )
    compile_parser.add_argument(
        "--behavior-search",
        action="store_true",
        help=(
            "search timing sample limits up to --timing-sample-limit and choose the smallest "
            "behaviorally verified canary"
        ),
    )
    compile_parser.add_argument(
        "--behavior-search-min-sample-limit",
        type=int,
        default=2,
        help="lowest timing sample limit to try in --behavior-search mode",
    )
    compile_parser.set_defaults(func=_cmd_compile)

    replay_parser = sub.add_parser("replay", help="replay a canary and emit a report")
    replay_parser.add_argument("canary")
    replay_parser.add_argument("--output", "-o", required=True)
    replay_parser.add_argument("--html")
    replay_parser.add_argument("--backend-label", default="simulated-nccl")
    replay_parser.add_argument("--bandwidth-gbps", type=float, default=55.0)
    replay_parser.add_argument("--latency-floor-us", type=float, default=7.5)
    replay_parser.add_argument("--compute-pressure", type=float, default=0.55)
    replay_parser.add_argument("--overlap-efficiency", type=float, default=0.72)
    replay_parser.add_argument("--iterations", type=int, default=1)
    replay_parser.add_argument("--seed", type=int, default=7)
    replay_parser.add_argument("--include-samples", action="store_true")
    replay_parser.add_argument("--max-replay-events", type=int, default=1_000_000)
    replay_parser.add_argument(
        "--ablate",
        action="append",
        default=[],
        help=(
            "replay ablation to apply; repeat or comma-separate values: arrival_skew, "
            "compute_overlap, operation_ordering, rare_tail_windows, queue_reset_gaps, "
            "pressure, observed_exposed_us"
        ),
    )
    replay_parser.set_defaults(func=_cmd_replay)

    compare_parser = sub.add_parser("compare", help="compare baseline and candidate reports")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    compare_parser.add_argument("--output", "-o", required=True)
    compare_parser.add_argument("--html")
    compare_parser.add_argument("--p99-threshold-pct", type=float, default=15.0)
    compare_parser.add_argument("--p95-threshold-pct", type=float, default=10.0)
    compare_parser.add_argument("--median-threshold-pct", type=float, default=8.0)
    compare_parser.add_argument("--p99-absolute-threshold-us", type=float, default=1.0)
    compare_parser.add_argument("--p95-absolute-threshold-us", type=float, default=1.0)
    compare_parser.add_argument("--median-absolute-threshold-us", type=float, default=1.0)
    compare_parser.add_argument("--hidden-drop-threshold-points", type=float, default=5.0)
    compare_parser.add_argument("--breakdown-threshold-pct", type=float)
    compare_parser.add_argument("--breakdown-absolute-threshold-us", type=float)
    compare_parser.add_argument("--allow-mismatch", action="store_true")
    compare_parser.set_defaults(func=_cmd_compare)

    verify_parser = sub.add_parser("verify-fidelity", help="verify canary fidelity against a source trace")
    verify_parser.add_argument("trace")
    verify_parser.add_argument("canary")
    verify_parser.add_argument("--output", "-o", required=True)
    verify_parser.set_defaults(func=_cmd_verify_fidelity)

    behavior_parser = sub.add_parser("verify-behavior", help="verify canary replay behavior against a source trace")
    behavior_parser.add_argument("trace")
    behavior_parser.add_argument("canary")
    behavior_parser.add_argument("--output", "-o", required=True)
    behavior_parser.add_argument("--relative-tolerance-pct", type=float, default=10.0)
    behavior_parser.add_argument("--absolute-tolerance-us", type=float, default=1.0)
    behavior_parser.add_argument("--hidden-tolerance-points", type=float, default=5.0)
    behavior_parser.add_argument("--tail-recall-threshold", type=float, default=0.80)
    behavior_parser.add_argument("--ranking-tie-tolerance-us", type=float, default=0.001)
    behavior_parser.set_defaults(func=_cmd_verify_behavior)

    baseline_parser = sub.add_parser("baseline", help="generate research baseline traces for comparison experiments")
    baseline_parser.add_argument("trace")
    baseline_parser.add_argument("--output", "-o", required=True)
    baseline_parser.add_argument(
        "--method",
        choices=("isolated", "random", "frequency", "cluster", "stratified"),
        required=True,
        help=(
            "baseline generator: isolated collective, random sampling, "
            "frequency representative, clustering representative, or "
            "stratified sampling"
        ),
    )
    baseline_parser.add_argument("--sample-count", type=int, default=8)
    baseline_parser.add_argument("--cluster-count", type=int, default=8)
    baseline_parser.add_argument("--strata-per-group", type=int, default=4)
    baseline_parser.add_argument("--seed", type=int, default=0)
    baseline_parser.add_argument(
        "--partial",
        action="store_true",
        help="for random sampling, emit only selected events instead of tiling to the source event count",
    )
    baseline_parser.set_defaults(func=_cmd_baseline)

    reduce_parser = sub.add_parser(
        "reduce",
        help="ddmin-style decision-preserving event reduction (research baseline)",
    )
    reduce_parser.add_argument("trace")
    reduce_parser.add_argument("--output", "-o", required=True)
    reduce_parser.add_argument("--ranking-tie-tolerance-us", type=float, default=0.001)
    reduce_parser.add_argument("--max-oracle-calls", type=int, default=256)
    reduce_parser.add_argument(
        "--timing-sample-limit",
        type=int,
        default=None,
        help="compile oracle candidates with this timing sample limit instead of lossless timing",
    )
    reduce_parser.set_defaults(func=_cmd_reduce)

    import_parser = sub.add_parser(
        "import-kineto",
        help="import record_param_comms collectives from a PyTorch Kineto profiler trace",
    )
    import_parser.add_argument("kineto_trace")
    import_parser.add_argument("--output", "-o", required=True)
    import_parser.add_argument("--workload-name", default="kineto-import")
    import_parser.add_argument("--phase", default=None)
    import_parser.add_argument(
        "--process-group",
        default=None,
        help="import only events from this Process Group Name",
    )
    import_parser.set_defaults(func=_cmd_import_kineto)

    export_parser = sub.add_parser(
        "export-param",
        help="export a canary as a PARAM comms-replay basic JSON trace",
    )
    export_parser.add_argument("canary")
    export_parser.add_argument("--output", "-o", required=True)
    export_parser.add_argument("--dtype", default="float32")
    export_parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="drop events without a PARAM equivalent instead of failing",
    )
    export_parser.add_argument(
        "--compute-fill-us-per-gemm",
        type=float,
        default=None,
        help=(
            "export inter-collective gaps as PARAM gemm compute entries, one "
            "per this many microseconds (calibrate per device); replay the "
            "result WITHOUT --use-timestamp"
        ),
    )
    export_parser.add_argument("--compute-fill-gemm-dim", type=int, default=1024)
    export_parser.add_argument(
        "--overlap-structure",
        action="store_true",
        help=(
            "emit collectives as async-issue plus explicit wait entries placed "
            "after the next gap's gemm entries, reconstructing compute/"
            "communication overlap; requires --compute-fill-us-per-gemm"
        ),
    )
    export_parser.set_defaults(func=_cmd_export_param)

    report_verify_parser = sub.add_parser("verify-report", help="recompute a report from a canary and backend settings")
    report_verify_parser.add_argument("report")
    report_verify_parser.add_argument("canary")
    report_verify_parser.add_argument("--output", "-o", required=True)
    report_verify_parser.set_defaults(func=_cmd_verify_report)

    capture_parser = sub.add_parser("capture", help="run an instrumented command and collect a trace")
    capture_parser.add_argument("--output", "-o", required=True)
    capture_parser.add_argument("--workload-name", default="instrumented-workload")
    capture_parser.add_argument("--allow-empty", action="store_true")
    capture_parser.add_argument("command", nargs=argparse.REMAINDER)
    capture_parser.set_defaults(func=_cmd_capture)

    report_parser = sub.add_parser("report", help="render an existing JSON report as standalone HTML")
    report_parser.add_argument("report")
    report_parser.add_argument("--output", "-o", required=True)
    report_parser.set_defaults(func=_cmd_report)

    return parser


def _split_ablations(values: List[str]) -> List[str]:
    result: List[str] = []
    for value in values or []:
        result.extend(item.strip() for item in str(value).split(",") if item.strip())
    return result


def _cmd_compile(args: Any) -> int:
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
        canary = synthesize_behavioral_canary(
            trace,
            min_timing_sample_limit=args.behavior_search_min_sample_limit,
            max_timing_sample_limit=args.timing_sample_limit,
            **common_kwargs,
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


def _cmd_baseline(args: Any) -> int:
    trace = load_json(args.trace)
    if args.method == "isolated":
        baseline = isolated_collective_baseline_trace(trace)
    elif args.method == "random":
        baseline = random_sampling_baseline_trace(
            trace,
            sample_count=args.sample_count,
            seed=args.seed,
            preserve_source_event_count=not args.partial,
        )
    elif args.method == "frequency":
        baseline = frequency_representative_baseline_trace(trace)
    elif args.method == "cluster":
        baseline = clustering_representative_baseline_trace(
            trace,
            cluster_count=args.cluster_count,
        )
    elif args.method == "stratified":
        baseline = stratified_sampling_baseline_trace(
            trace,
            strata_per_group=args.strata_per_group,
            seed=args.seed,
        )
    else:  # pragma: no cover - argparse constrains this.
        raise CommCanaryError(f"unknown baseline method {args.method!r}")
    write_json(args.output, baseline)
    print(f"wrote {args.method} baseline trace with {len(baseline['events'])} events: {args.output}")
    return 0


def _cmd_reduce(args: Any) -> int:
    trace = load_json(args.trace)
    reduced = ddmin_ranking_reduction(
        trace,
        ranking_tie_tolerance_us=args.ranking_tie_tolerance_us,
        timing_sample_limit=args.timing_sample_limit,
        max_oracle_calls=args.max_oracle_calls,
    )
    write_json(args.output, reduced)
    reduction = reduced["workload"]["reduction"]
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


def _cmd_import_kineto(args: Any) -> int:
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
        "imported {events} collective events "
        "(skipped {control} control, {empty} empty): {output}".format(
            events=workload["imported_events"],
            control=workload["skipped_control_events"],
            empty=workload["skipped_empty_events"],
            output=args.output,
        )
    )
    return 0


def _cmd_export_param(args: Any) -> int:
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


def _cmd_replay(args: Any) -> int:
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
        ablations=_split_ablations(args.ablate),
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


def _cmd_compare(args: Any) -> int:
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


def _cmd_verify_fidelity(args: Any) -> int:
    trace = load_json(args.trace)
    canary = load_json(args.canary)
    verification = verify_canary_fidelity(trace, canary)
    write_json(args.output, verification)
    print(f"fidelity verification: {verification['status']}")
    for check in verification["checks"]:
        print(f"- {check['name']}: {check['status']}")
    return 0 if verification["status"] == "source_verified" else 1


def _cmd_verify_behavior(args: Any) -> int:
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


def _cmd_verify_report(args: Any) -> int:
    report = load_json(args.report)
    canary = load_json(args.canary)
    verification = verify_report_against_canary(report, canary)
    write_json(args.output, verification)
    print(f"report verification: {verification['status']}")
    for check in verification["checks"]:
        print(f"- {check['name']}: {check['status']}")
    return 0 if verification["status"] == "model_recomputed" else 1


def _cmd_capture(args: Any) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise CommCanaryError("capture requires a command after --")

    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="commcanary-capture-") as trace_dir:
        manual_trace = os.path.join(trace_dir, "manual.trace.json")
        env["COMMCANARY_TRACE_DIR"] = trace_dir
        env["COMMCANARY_TRACE_OUT"] = manual_trace
        env["COMMCANARY_WORKLOAD_NAME"] = args.workload_name
        env["COMMCANARY_CAPTURE_SESSION_ID"] = str(uuid.uuid4())
        try:
            completed = subprocess.run(command, env=env)
        except OSError as exc:
            raise CommCanaryError(f"could not run capture command {command[0]!r}: {exc}") from exc
        if completed.returncode != 0:
            return completed.returncode

        merged = merge_trace_shards(trace_dir, workload_name=args.workload_name)
        if not merged["events"]:
            if not args.allow_empty:
                raise CommCanaryError(
                    "target command did not write a trace; import commcanary.capture.record_collective "
                    "or pass --allow-empty"
                )
            merged = TraceRecorder(args.output, workload={"name": args.workload_name}).to_trace()
        write_json(args.output, merged)
    print(f"captured trace: {args.output}")
    return 0


def _cmd_report(args: Any) -> int:
    report = load_json(args.report)
    validate_report(report)
    write_report_html(args.output, report)
    print(f"wrote HTML report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
