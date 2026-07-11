"""Legacy glob-based Rostam analysis, available only through explicit legacy mode."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

JsonDict = Dict[str, Any]


def median(values: Sequence[float]) -> float:
    xs = sorted(float(value) for value in values)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def iqr(values: Sequence[float]) -> float:
    xs = sorted(float(value) for value in values)
    if len(xs) < 2:
        return 0.0
    mid = len(xs) // 2
    if len(xs) % 2:
        lower = xs[:mid]
        upper = xs[mid + 1 :]
    else:
        lower = xs[:mid]
        upper = xs[mid:]
    if not lower or not upper:
        return 0.0
    return median(upper) - median(lower)


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def canonical_workload(name: Any) -> str:
    text = str(name or "").strip()
    lowered = text.lower().replace("_", "-")
    aliases = {
        "micro": "W-micro",
        "w-micro": "W-micro",
        "full": "W-full",
        "w-full": "W-full",
        "canary": "W-canary",
        "w-canary": "W-canary",
        "baseline-stratified": "W-baseline-stratified",
        "w-baseline-stratified": "W-baseline-stratified",
        "baseline-random": "W-baseline-random",
        "w-baseline-random": "W-baseline-random",
        "baseline-ddmin": "W-baseline-ddmin",
        "w-baseline-ddmin": "W-baseline-ddmin",
    }
    return aliases.get(lowered, text)


def config_name(result: Mapping[str, Any]) -> str:
    config = result.get("config")
    if isinstance(config, Mapping) and isinstance(config.get("name"), str):
        return str(config["name"])
    if isinstance(config, str):
        return config
    if isinstance(result.get("config_name"), str):
        return str(result["config_name"])
    raise ValueError("result is missing config name")


def result_scalar_us(result: Mapping[str, Any]) -> Optional[float]:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        value = as_float(metrics.get("median_us"))
        if value is not None:
            return value
    for key in ("timings_us", "timing_distribution_us", "latencies_us"):
        values = numeric_list(result.get(key))
        if values:
            return median(values)
    # No wall-time fallback by design: a replay that produced no per-op
    # latency is a measurement failure, not a data point.
    return None


def result_iqr_us(result: Mapping[str, Any]) -> Optional[float]:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        value = as_float(metrics.get("iqr_us"))
        if value is not None:
            return value
    for key in ("timings_us", "timing_distribution_us", "latencies_us"):
        values = numeric_list(result.get(key))
        if values:
            return iqr(values)
    return None


def numeric_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    numbers = []
    for item in value:
        number = as_float(item)
        if number is not None:
            numbers.append(number)
    return numbers


def wall_time_s(result: Mapping[str, Any]) -> Optional[float]:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        value = as_float(metrics.get("wall_time_s"))
        if value is not None:
            return value
    return as_float(result.get("wall_time_s"))


def artifact_number(result: Mapping[str, Any], key: str) -> Optional[float]:
    artifacts = result.get("artifacts")
    if isinstance(artifacts, Mapping):
        return as_float(artifacts.get(key))
    return as_float(result.get(key))


def load_results(results_dir: Path) -> List[JsonDict]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "results.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data["_source_path"] = str(path)
            results.append(data)
    return results


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> Tuple[JsonDict, JsonDict]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    configs: JsonDict = {}
    for result in results:
        # A parse-failed replay carries no real latency; never let it into an
        # aggregate. status != 0 is dropped for the same reason.
        status = result.get("status")
        if result.get("parse_failed") is True or (isinstance(status, int) and status != 0):
            continue
        workload = canonical_workload(result.get("workload"))
        name = config_name(result)
        grouped.setdefault((workload, name), []).append(result)
        if name not in configs:
            raw_config = result.get("config")
            configs[name] = raw_config if isinstance(raw_config, Mapping) else {"name": name}

    workloads: JsonDict = {}
    for (workload, name), rows in sorted(grouped.items()):
        values = [value for row in rows for value in [result_scalar_us(row)] if value is not None]
        if not values:
            continue
        per_run_iqrs = [value for row in rows for value in [result_iqr_us(row)] if value is not None]
        aggregate_iqr = iqr(values) if len(values) > 1 else (median(per_run_iqrs) if per_run_iqrs else 0.0)
        workloads.setdefault(workload, {})[name] = {
            "config": name,
            "median_us": round(median(values), 6),
            "iqr_us": round(aggregate_iqr, 6),
            "repetitions": len(values),
            "source_files": [str(row.get("_source_path", "")) for row in rows],
        }
    return workloads, configs


def build_rankings(workloads: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> JsonDict:
    rankings: JsonDict = {}
    for workload, configs in sorted(workloads.items()):
        rows = sorted(configs.values(), key=lambda row: (row["median_us"], row["config"]))
        rankings[workload] = [
            {
                "rank": index + 1,
                "config": row["config"],
                "median_us": row["median_us"],
                "iqr_us": row["iqr_us"],
                "repetitions": row["repetitions"],
            }
            for index, row in enumerate(rows)
        ]
    return rankings


def relation(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> Tuple[int, str]:
    diff = float(row_a["median_us"]) - float(row_b["median_us"])
    tolerance = max(float(row_a.get("iqr_us", 0.0)), float(row_b.get("iqr_us", 0.0)))
    if abs(diff) < tolerance:
        return 0, "tie"
    if diff < 0:
        return -1, "a_faster"
    return 1, "b_faster"


def build_pairwise(workloads: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> JsonDict:
    all_pairwise: JsonDict = {}
    for workload, configs in sorted(workloads.items()):
        pairs: JsonDict = {}
        names = sorted(configs)
        for left, right in itertools.combinations(names, 2):
            cmp_value, label = relation(configs[left], configs[right])
            pairs[f"{left}|{right}"] = {
                "config_a": left,
                "config_b": right,
                "relation": label,
                "relation_value": cmp_value,
                "median_a_us": configs[left]["median_us"],
                "median_b_us": configs[right]["median_us"],
                "iqr_a_us": configs[left]["iqr_us"],
                "iqr_b_us": configs[right]["iqr_us"],
                "tie_tolerance_us": max(configs[left]["iqr_us"], configs[right]["iqr_us"]),
            }
        all_pairwise[workload] = pairs
    return all_pairwise


def agreement(pairwise: Mapping[str, Mapping[str, Mapping[str, Any]]], workload: str, reference: str) -> JsonDict:
    left = pairwise.get(workload, {})
    right = pairwise.get(reference, {})
    common = sorted(set(left) & set(right))
    agree = 0
    concordant = 0
    discordant = 0
    ties_left = 0
    ties_right = 0
    both_tie = 0
    examples: List[JsonDict] = []
    for key in common:
        a = int(left[key]["relation_value"])
        b = int(right[key]["relation_value"])
        if a == b:
            agree += 1
        elif len(examples) < 5:
            examples.append(
                {
                    "pair": key,
                    workload: left[key]["relation"],
                    reference: right[key]["relation"],
                }
            )
        if a == 0 and b == 0:
            both_tie += 1
        elif a == 0:
            ties_left += 1
        elif b == 0:
            ties_right += 1
        elif a == b:
            concordant += 1
        else:
            discordant += 1
    total = len(common)
    denominator = math.sqrt((concordant + discordant + ties_left) * (concordant + discordant + ties_right))
    if denominator:
        tau = (concordant - discordant) / denominator
    elif total and agree == total:
        tau = 1.0
    else:
        tau = 0.0
    return {
        "workload": workload,
        "reference": reference,
        "pairs": total,
        "agree": agree,
        "disagree": total - agree,
        "agreement_pct": round((100.0 * agree / total) if total else 0.0, 6),
        "kendall_tau": round(tau, 6),
        "concordant": concordant,
        "discordant": discordant,
        "ties_workload_only": ties_left,
        "ties_reference_only": ties_right,
        "both_tie": both_tie,
        "examples": examples,
    }


def build_agreements(pairwise: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> JsonDict:
    if "W-full" not in pairwise:
        return {}
    # every observed workload is compared against W-full; a hardcoded list
    # silently dropped W-canary-compute in sweep 2.
    candidates = sorted(name for name in pairwise if name != "W-full")
    rows: JsonDict = {}
    for workload in candidates:
        if workload in pairwise and workload != "W-full":
            rows[f"{workload}_vs_W-full"] = agreement(pairwise, workload, "W-full")
    return rows


def find_version_config(configs: Mapping[str, Any], version: str) -> Optional[str]:
    names = sorted(configs)
    default_matches = [name for name in names if version in name and "default" in name.lower()]
    if default_matches:
        return default_matches[0]
    matches = [name for name in names if version in name]
    return matches[0] if matches else None


def build_regression_table(
    workloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
    configs: Mapping[str, Any],
    baseline_config: Optional[str],
    candidate_config: Optional[str],
    rel_threshold_pct: float,
    abs_threshold_us: float,
) -> JsonDict:
    # 2.19.3, not 2.18.3: the version pair was retargeted when torch 2.4.1
    # became the required interpreter (2.18.5 never existed on PyPI either).
    baseline = baseline_config or find_version_config(configs, "2.19.3")
    candidate = candidate_config or find_version_config(configs, "2.20.5")
    table: JsonDict = {
        "baseline_config": baseline,
        "candidate_config": candidate,
        "thresholds": {
            "median_threshold_pct": rel_threshold_pct,
            "median_absolute_threshold_us": abs_threshold_us,
            "rule": "candidate_median - baseline_median > max(abs_us, baseline_median * pct / 100)",
        },
        "workloads": {},
        "confusion_vs_full": {},
    }
    if not baseline or not candidate:
        return table
    for workload, rows in sorted(workloads.items()):
        if baseline not in rows or candidate not in rows:
            continue
        base = float(rows[baseline]["median_us"])
        cand = float(rows[candidate]["median_us"])
        delta = cand - base
        threshold = max(abs_threshold_us, abs(base) * rel_threshold_pct / 100.0)
        regression = delta > threshold
        table["workloads"][workload] = {
            "baseline_median_us": round(base, 6),
            "candidate_median_us": round(cand, 6),
            "delta_us": round(delta, 6),
            "delta_pct": round((100.0 * delta / base) if base else 0.0, 6),
            "threshold_us": round(threshold, 6),
            "regression": regression,
        }
    full_flag = table["workloads"].get("W-full", {}).get("regression")
    if isinstance(full_flag, bool):
        for workload, row in sorted(table["workloads"].items()):
            if workload == "W-full":
                continue
            other_flag = bool(row["regression"])
            if full_flag and other_flag:
                cell = "TP"
            elif full_flag and not other_flag:
                cell = "FN"
            elif not full_flag and other_flag:
                cell = "FP"
            else:
                cell = "TN"
            table["confusion_vs_full"][workload] = {
                "full_regression": full_flag,
                "workload_regression": other_flag,
                "cell": cell,
            }
    return table


def build_costs(results: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for result in results:
        grouped.setdefault(canonical_workload(result.get("workload")), []).append(result)
    rows: JsonDict = {}
    artifact_keys = ("raw_trace_bytes", "canary_bytes", "param_trace_bytes", "profile_bytes")
    for workload, items in sorted(grouped.items()):
        wall_values = [value for item in items for value in [wall_time_s(item)] if value is not None]
        row: JsonDict = {
            "runs": len(items),
            "wall_time_s_median": round(median(wall_values), 6) if wall_values else None,
        }
        for key in artifact_keys:
            values = [value for item in items for value in [artifact_number(item, key)] if value is not None]
            row[f"{key}_median"] = round(median(values), 6) if values else None
        rows[workload] = row
    return rows


def render_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Rostam Results — UNSAFE LEGACY ANALYSIS",
        "",
        "> **UNTRUSTED / NOT FOR PUBLICATION.** These rows came from globbed JSON without manifest, selection, "
        "or completeness verification.",
        "",
    ]
    lines.append("## Rankings")
    for workload, rows in sorted(analysis["rankings"].items()):
        lines.append(f"### {workload}")
        lines.append("| rank | config | median us | IQR us | reps |")
        lines.append("|---:|---|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['rank']} | {row['config']} | {row['median_us']:.3f} | "
                f"{row['iqr_us']:.3f} | {row['repetitions']} |"
            )
        lines.append("")

    lines.append("## Pairwise Agreement vs W-full")
    lines.append("| workload | pairs | agreement | Kendall tau | disagree |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in analysis["agreements"].values():
        lines.append(
            f"| {row['workload']} | {row['pairs']} | {row['agreement_pct']:.2f}% | "
            f"{row['kendall_tau']:.3f} | {row['disagree']} |"
        )
    lines.append("")

    regression = analysis["regression_2x2"]
    lines.append("## NCCL Version Regression")
    lines.append(f"Baseline: `{regression.get('baseline_config')}`; candidate: `{regression.get('candidate_config')}`.")
    lines.append("| workload | baseline us | candidate us | delta pct | regression | vs full |")
    lines.append("|---|---:|---:|---:|---|---|")
    confusion = regression.get("confusion_vs_full", {})
    for workload, row in sorted(regression.get("workloads", {}).items()):
        cell = "reference" if workload == "W-full" else confusion.get(workload, {}).get("cell", "")
        lines.append(
            f"| {workload} | {row['baseline_median_us']:.3f} | {row['candidate_median_us']:.3f} | "
            f"{row['delta_pct']:.2f}% | {str(row['regression']).lower()} | {cell} |"
        )
    lines.append("")

    lines.append("## Cost")
    lines.append("| workload | runs | median wall s | raw trace bytes | canary bytes | param trace bytes |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for workload, row in sorted(analysis["costs"].items()):
        lines.append(
            f"| {workload} | {row['runs']} | {fmt(row['wall_time_s_median'])} | "
            f"{fmt(row['raw_trace_bytes_median'])} | {fmt(row['canary_bytes_median'])} | "
            f"{fmt(row['param_trace_bytes_median'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def analyze(
    results_dir: Path,
    *,
    baseline_config: Optional[str],
    candidate_config: Optional[str],
    rel_threshold_pct: float,
    abs_threshold_us: float,
) -> JsonDict:
    results = load_results(results_dir)
    workloads, configs = aggregate_results(results)
    rankings = build_rankings(workloads)
    pairwise = build_pairwise(workloads)
    return {
        "schema": "commcanary.rostam.analysis.v1",
        "trust": {
            "status": "unsafe-legacy-unverified",
            "publication_allowed": False,
            "warning": "globbed inputs are not manifest/selection/completeness bound",
        },
        "results_dir": str(results_dir),
        "input_files": len(results),
        "configs": configs,
        "workload_metrics": workloads,
        "rankings": rankings,
        "pairwise_relations": pairwise,
        "agreements": build_agreements(pairwise),
        "regression_2x2": build_regression_table(
            workloads,
            configs,
            baseline_config,
            candidate_config,
            rel_threshold_pct,
            abs_threshold_us,
        ),
        "costs": build_costs(results),
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    default_results = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description="Analyze Rostam experiment JSON result files.")
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--baseline-config")
    parser.add_argument("--candidate-config")
    parser.add_argument("--median-threshold-pct", type=float, default=8.0)
    parser.add_argument("--median-absolute-threshold-us", type=float, default=1.0)
    parser.add_argument(
        "--unsafe-legacy-glob-analysis",
        action="store_true",
        help="required acknowledgement that legacy glob output is untrusted and not publication evidence",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.unsafe_legacy_glob_analysis:
        parser.error("legacy mode requires --unsafe-legacy-glob-analysis")
    results_dir = args.results_dir
    output_json = args.output_json or (results_dir / "results.json")
    output_md = args.output_md or (results_dir / "results.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    analysis = analyze(
        results_dir,
        baseline_config=args.baseline_config,
        candidate_config=args.candidate_config,
        rel_threshold_pct=args.median_threshold_pct,
        abs_threshold_us=args.median_absolute_threshold_us,
    )
    write_json(output_json, analysis)
    output_md.write_text(render_markdown(analysis), encoding="utf-8")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
