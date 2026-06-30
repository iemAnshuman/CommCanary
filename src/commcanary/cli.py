from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import uuid
from typing import Any, List, Optional

from .capture import TraceRecorder, merge_trace_shards
from .compare import compare_reports
from .compiler import compile_trace, verify_canary_behavior, verify_canary_fidelity
from .html_report import write_compare_html, write_report_html
from .replay import replay_canary, verify_report_against_canary
from .schema import CommCanaryError, load_json, validate_report, write_json


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
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
    canary = compile_trace(
        trace,
        max_events=args.max_events,
        timing_sample_limit=args.timing_sample_limit,
        max_gap_error_us=args.max_gap_error_us,
        max_skew_error_us=args.max_skew_error_us,
        max_arrival_offset_error_us=args.max_arrival_offset_error_us,
        max_compute_before_error_us=args.max_compute_before_error_us,
        max_overlap_error_us=args.max_overlap_error_us,
        max_pressure_error=args.max_pressure_error,
        max_observed_exposed_error_us=args.max_observed_exposed_error_us,
        max_prefix_gap_error_us=args.max_prefix_gap_error_us,
        require_lossless_timing=args.lossless_timing,
        enable_sequence_motifs=not args.disable_sequence_motifs,
        require_behavior_verification=args.require_behavior_verification,
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
