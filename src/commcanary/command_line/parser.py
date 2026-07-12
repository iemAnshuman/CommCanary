"""Argparse construction with handler injection and no engine imports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

CommandHandler = Callable[[Any], int]


@dataclass(frozen=True)
class CommandHandlers:
    compile: CommandHandler
    replay: CommandHandler
    compare: CommandHandler
    verify_fidelity: CommandHandler
    verify_behavior: CommandHandler
    baseline: CommandHandler
    reduce: CommandHandler
    import_kineto: CommandHandler
    export_param: CommandHandler
    verify_report: CommandHandler
    capture: CommandHandler
    report: CommandHandler


def build_parser(*, handlers: CommandHandlers, version: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commcanary",
        description="Workload-derived communication canaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=version)
    parser.add_argument(
        "--diagnostics-json",
        action="store_true",
        help="emit machine-readable JSON Lines diagnostics on stderr",
    )
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
    compile_parser.set_defaults(func=handlers.compile)

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
    replay_parser.set_defaults(func=handlers.replay)

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
    compare_parser.set_defaults(func=handlers.compare)

    verify_parser = sub.add_parser("verify-fidelity", help="verify canary fidelity against a source trace")
    verify_parser.add_argument("trace")
    verify_parser.add_argument("canary")
    verify_parser.add_argument("--output", "-o", required=True)
    verify_parser.set_defaults(func=handlers.verify_fidelity)

    behavior_parser = sub.add_parser("verify-behavior", help="verify canary replay behavior against a source trace")
    behavior_parser.add_argument("trace")
    behavior_parser.add_argument("canary")
    behavior_parser.add_argument("--output", "-o", required=True)
    behavior_parser.add_argument("--relative-tolerance-pct", type=float, default=10.0)
    behavior_parser.add_argument("--absolute-tolerance-us", type=float, default=1.0)
    behavior_parser.add_argument("--hidden-tolerance-points", type=float, default=5.0)
    behavior_parser.add_argument("--tail-recall-threshold", type=float, default=0.80)
    behavior_parser.add_argument("--ranking-tie-tolerance-us", type=float, default=0.001)
    behavior_parser.set_defaults(func=handlers.verify_behavior)

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
    baseline_parser.add_argument("--sample-count", type=int)
    baseline_parser.add_argument("--cluster-count", type=int)
    baseline_parser.add_argument("--strata-per-group", type=int)
    baseline_parser.add_argument("--seed", type=int)
    baseline_parser.add_argument(
        "--partial",
        action="store_true",
        help="for random sampling, emit only selected events instead of tiling to the source event count",
    )
    baseline_parser.set_defaults(func=handlers.baseline)

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
    reduce_parser.set_defaults(func=handlers.reduce)

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
    import_parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=None,
        help="raise the bounded-JSON input budget for a trusted, locally produced profile",
    )
    import_parser.add_argument(
        "--max-json-items",
        type=int,
        default=None,
        help="raise the bounded-JSON structural item budget for a trusted, locally produced profile",
    )
    import_parser.set_defaults(func=handlers.import_kineto)

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
    export_parser.set_defaults(func=handlers.export_param)

    report_verify_parser = sub.add_parser("verify-report", help="recompute a report from a canary and backend settings")
    report_verify_parser.add_argument("report")
    report_verify_parser.add_argument("canary")
    report_verify_parser.add_argument("--output", "-o", required=True)
    report_verify_parser.set_defaults(func=handlers.verify_report)

    capture_parser = sub.add_parser("capture", help="run an instrumented command and collect a trace")
    capture_parser.add_argument("--output", "-o", required=True)
    capture_parser.add_argument("--workload-name", default="instrumented-workload")
    capture_parser.add_argument("--allow-empty", action="store_true")
    capture_parser.add_argument(
        "--preserve-on-failure",
        metavar="DIR",
        help="write a bounded partial-shard bundle and failure manifest when the child fails",
    )
    capture_parser.add_argument("command", nargs=argparse.REMAINDER)
    capture_parser.set_defaults(func=handlers.capture)

    render_parser = sub.add_parser("render-html", help="render an existing JSON report as standalone HTML")
    render_parser.add_argument("report")
    render_parser.add_argument("--output", "-o", required=True)
    render_parser.set_defaults(func=handlers.report, deprecated_report_alias=False)

    report_parser = sub.add_parser("report", help="deprecated alias for render-html")
    report_parser.add_argument("report")
    report_parser.add_argument("--output", "-o", required=True)
    report_parser.set_defaults(func=handlers.report, deprecated_report_alias=True)

    return parser


__all__ = ["CommandHandler", "CommandHandlers", "build_parser"]
