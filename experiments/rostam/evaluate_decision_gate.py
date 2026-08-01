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
from .harness import ContractError, JSONResourceLimits, read_bounded_bytes, strict_json_loads

_AGGREGATE_LIMITS = JSONResourceLimits(max_document_bytes=64 * 1024 * 1024, max_items=4_000_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        aggregate_bytes = read_bounded_bytes(
            args.aggregate,
            max_bytes=_AGGREGATE_LIMITS.max_document_bytes,
            field="trusted decision-gate aggregate",
        )
        aggregate = strict_json_loads(aggregate_bytes, limits=_AGGREGATE_LIMITS)
        policy_bytes = load_decision_fidelity_policy(args.policy)
        verdict = evaluate_decision_fidelity(aggregate, policy_bytes)
        write_decision_fidelity_verdict(args.output, verdict)
    except (ContractError, OSError, UnicodeError) as exc:
        raise SystemExit(f"decision-gate evaluation error: {exc}") from exc
    print(
        json.dumps(
            {
                "outcome": verdict["outcome"],
                "output": str(args.output),
                "reframe": verdict["product_interpretation"]["kill_or_reframe_triggered"],
                "verdict_id": verdict["verdict_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
