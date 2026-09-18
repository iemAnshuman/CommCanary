#!/usr/bin/env python3
"""Uncertainty and sensitivity for the trusted-join agreement counts.

The trusted join reports point agreement with W-full and no interval. This
script recomputes those counts from the committed aggregate CSV with the
analyzer's own rule, then resamples each cell's selected repetitions to show
which comparisons survive the repetition-to-repetition spread, and repeats the
analysis with nccl-2.20.5-tree-ll excluded.

Rule, copied from experiments/rostam/analysis: per (workload, configuration)
the selected repetition values give a median and a Tukey-hinge IQR, and a pair
is a tie when |median difference| < max(IQR_a, IQR_b). The script refuses to
write anything if its point counts disagree with the published aggregate.

Five repetitions per cell make the percentile bootstrap optimistic; the output
says so. Usage:

    python3 paper/arxiv/scripts/trusted_join_uncertainty.py          # write
    python3 paper/arxiv/scripts/trusted_join_uncertainty.py --check  # verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parents[1] / "evidence" / "trusted_join_uncertainty.json"
PUBLICATION = "experiments/rostam/results/publications/trusted-join-core-shared-overlap-primary"
GATE_VERDICT = (
    "experiments/rostam/results/publications/decision-gate-20260801-r3-analysis-1300d12-primary/"
    "decision-fidelity-verdict.json"
)
LABELS = {
    "micro": "W-micro",
    "full": "W-full",
    "canary-param": "W-canary",
    "canary-overlap": "W-canary-overlap",
    "shared-overlap": "W-shared-overlap",
}
PROXIES = ("W-micro", "W-canary", "W-canary-overlap", "W-shared-overlap")
EXCLUDED = "nccl-2.20.5-tree-ll"
DRAWS = 4000
SEED = 20260919
DIFFERENCES = (
    ("W-canary", "W-micro"),
    ("W-canary-overlap", "W-canary"),
    ("W-shared-overlap", "W-micro"),
    ("W-shared-overlap", "W-canary"),
)

Cells = Dict[str, Dict[str, List[float]]]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    lower, upper = (
        (ordered[:middle], ordered[middle + 1 :]) if len(ordered) % 2 else (ordered[:middle], ordered[middle:])
    )
    return _median(upper) - _median(lower)


def _labels(cells: Dict[str, List[float]], pairs: Sequence[Tuple[str, str]]) -> List[int]:
    stats = {name: (_median(values), _iqr(values)) for name, values in cells.items()}
    labels = []
    for left, right in pairs:
        difference = stats[left][0] - stats[right][0]
        tolerance = max(stats[left][1], stats[right][1])
        labels.append(0 if difference == 0.0 or abs(difference) < tolerance else (-1 if difference < 0 else 1))
    return labels


def _agreement(proxy: Sequence[int], reference: Sequence[int]) -> int:
    return sum(1 for left, right in zip(proxy, reference) if left == right)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _load_cells(path: Path) -> Cells:
    cells: Cells = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["record_kind"] == "measurement"]
    for row in sorted(rows, key=lambda item: item["cell_id"]):
        workload = LABELS.get(row["workload_id"])
        if workload is not None:
            cells[workload][row["configuration_id"]].append(float(row["value_us"]))
    return cells


def _analyse(cells: Cells, configurations: Sequence[str]) -> Dict[str, object]:
    pairs = list(itertools.combinations(sorted(configurations), 2))
    subset = {workload: {name: cells[workload][name] for name in configurations} for workload in cells}
    reference = _labels(subset["W-full"], pairs)
    point = {name: _agreement(_labels(subset[name], pairs), reference) for name in PROXIES}
    rng = random.Random(SEED)
    draws: Dict[str, List[int]] = {name: [] for name in PROXIES}
    reference_kept = [0] * len(pairs)
    for _ in range(DRAWS):
        resampled_reference = _labels(
            {name: [rng.choice(values) for _ in values] for name, values in subset["W-full"].items()}, pairs
        )
        for index, (resampled_label, point_label) in enumerate(zip(resampled_reference, reference)):
            reference_kept[index] += resampled_label == point_label
        for name in PROXIES:
            resampled = {config: [rng.choice(values) for _ in values] for config, values in subset[name].items()}
            draws[name].append(_agreement(_labels(resampled, pairs), resampled_reference))

    def interval(values: Sequence[float]) -> List[float]:
        return [round(_percentile(values, 0.025), 2), round(_percentile(values, 0.975), 2)]

    differences = []
    for left, right in DIFFERENCES:
        gaps = [a - b for a, b in zip(draws[left], draws[right])]
        differences.append(
            {
                "comparison": f"{left} minus {right}",
                "point": point[left] - point[right],
                "interval_95": interval(gaps),
                "share_of_draws_below_zero": round(sum(1 for gap in gaps if gap < 0) / len(gaps), 3),
                "share_of_draws_at_zero": round(sum(1 for gap in gaps if gap == 0) / len(gaps), 3),
            }
        )
    unstable_reference = sorted(
        (
            {"pair": f"{left}|{right}", "share_of_draws_keeping_label": round(kept / DRAWS, 3)}
            for (left, right), kept in zip(pairs, reference_kept)
            if kept / DRAWS < 0.95
        ),
        key=lambda item: (item["share_of_draws_keeping_label"], item["pair"]),
    )
    return {
        "pairs": len(pairs),
        "reference_labels_kept_in_under_95pct_of_draws": unstable_reference,
        "agreement": {name: {"point": point[name], "interval_95": interval(draws[name])} for name in PROXIES},
        "differences": differences,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> Dict[str, object]:
    csv_path = REPOSITORY_ROOT / PUBLICATION / "aggregate.csv"
    json_path = REPOSITORY_ROOT / PUBLICATION / "aggregate.json"
    verdict_path = REPOSITORY_ROOT / GATE_VERDICT
    cells = _load_cells(csv_path)
    aggregate = json.loads(json_path.read_text(encoding="utf-8"))
    published = aggregate["claims"]["agreements"]
    configurations = sorted(cells["W-full"])
    everything = _analyse(cells, configurations)
    for name in PROXIES:
        expected = published[f"{name}_vs_W-full"]["agree"]
        observed = everything["agreement"][name]["point"]  # type: ignore[index]
        if observed != expected:
            raise SystemExit(f"{name}: recomputed {observed} agreeing pairs, published {expected}")
    ties = {
        name: {
            key: published[f"{name}_vs_W-full"][key]
            for key in ("agree", "concordant", "discordant", "ties_workload_only", "ties_reference_only", "both_tie")
        }
        for name in PROXIES
    }
    relative_iqr = {
        workload: {name: round(100.0 * _iqr(values) / _median(values), 2) for name, values in sorted(configs.items())}
        for workload, configs in sorted(cells.items())
    }
    largest = max(
        ((workload, name, value) for workload, configs in relative_iqr.items() for name, value in configs.items()),
        key=lambda item: item[2],
    )
    rankings = aggregate["claims"]["rankings"]
    ll_ranks = {
        config: {
            workload: next(row["rank"] for row in rows if row["config"] == config)
            for workload, rows in sorted(rankings.items())
        }
        for config in ("nccl-2.20.5-ring-ll", EXCLUDED)
    }
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    gate = {
        name: metrics["pairwise_agreement_count"] for name, metrics in sorted(verdict["representation_metrics"].items())
    }
    return {
        "schema": "commcanary.paper.trusted_join_uncertainty.v1",
        "inputs": {
            f"{PUBLICATION}/aggregate.csv": _sha256(csv_path),
            f"{PUBLICATION}/aggregate.json": _sha256(json_path),
            GATE_VERDICT: _sha256(verdict_path),
        },
        "method": {
            "rule": "median and Tukey-hinge IQR over selected repetitions; tie when |median difference| < max IQR",
            "resampling": "each cell's selected repetitions resampled with replacement, independently per cell",
            "draws": DRAWS,
            "seed": SEED,
            "interval": "percentile, 95%",
            "caveat": "five repetitions per cell make this bootstrap optimistic; read intervals as lower bounds on uncertainty",
        },
        "all_configurations": everything,
        "excluding": {"configuration": EXCLUDED, **_analyse(cells, [c for c in configurations if c != EXCLUDED])},
        "published_label_structure": ties,
        "stability": {
            "relative_iqr_pct": relative_iqr,
            "largest": {"workload": largest[0], "configuration": largest[1], "relative_iqr_pct": largest[2]},
            "exact_work_gate_limit_pct": 20.0,
        },
        "ll_protocol_ranks": ll_ranks,
        "exact_work_gate_agreement": gate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail unless the committed output regenerates exactly")
    args = parser.parse_args(argv)
    text = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{OUTPUT} does not match a fresh regeneration")
        print(f"verified {OUTPUT.relative_to(REPOSITORY_ROOT)}")
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
