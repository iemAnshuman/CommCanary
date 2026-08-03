#!/usr/bin/env python3
"""Evaluate one trusted physical decision-gate aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .analysis.decision_fidelity import (
    evaluate_decision_fidelity,
    load_decision_fidelity_policy,
    write_decision_fidelity_verdict,
)
from .harness import ContractError, JSONResourceLimits, canonical_json_bytes, read_bounded_bytes, strict_json_loads
from .lib.executor_artifact import ExecutorArtifact

_AGGREGATE_LIMITS = JSONResourceLimits(max_document_bytes=64 * 1024 * 1024, max_items=4_000_000)


def _verdict_summary(verdict: object, output: Path) -> dict[str, object]:
    if not isinstance(verdict, dict):
        raise ContractError("decision-gate evaluator returned an invalid verdict")
    interpretation = verdict.get("product_interpretation")
    if not isinstance(interpretation, dict):
        raise ContractError("decision-gate verdict lacks product interpretation")
    summary: dict[str, object] = {
        "outcome": verdict.get("outcome"),
        "output": str(output),
        "verdict_id": verdict.get("verdict_id"),
    }
    if "kill_or_reframe_triggered" in interpretation:
        summary["reframe"] = interpretation["kill_or_reframe_triggered"]
    elif "mode" in interpretation:
        summary["mode"] = interpretation["mode"]
    else:
        raise ContractError("decision-gate verdict has no supported product interpretation")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="require the frozen evaluator to reproduce these existing verdict bytes before writing output",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    executor_artifact: Optional[ExecutorArtifact] = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        aggregate_bytes = read_bounded_bytes(
            args.aggregate,
            max_bytes=_AGGREGATE_LIMITS.max_document_bytes,
            field="trusted decision-gate aggregate",
        )
        aggregate = strict_json_loads(aggregate_bytes, limits=_AGGREGATE_LIMITS)
        policy_bytes = load_decision_fidelity_policy(args.policy)
        verdict = evaluate_decision_fidelity(
            aggregate,
            policy_bytes,
            executor_artifact=executor_artifact,
        )
        if args.verify_against is not None:
            expected = canonical_json_bytes(verdict)
            golden = read_bounded_bytes(
                args.verify_against,
                max_bytes=len(expected),
                field="golden decision-gate verdict",
            )
            if golden != expected:
                raise ContractError("frozen evaluator does not reproduce the golden decision-gate verdict")
        write_decision_fidelity_verdict(args.output, verdict)
        summary = _verdict_summary(verdict, args.output)
    except (ContractError, OSError, UnicodeError) as exc:
        raise SystemExit(f"decision-gate evaluation error: {exc}") from exc
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
