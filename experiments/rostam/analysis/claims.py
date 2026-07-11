"""Research claims computed only from manifest-bound selected measurements."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

JsonDict = Dict[str, Any]

_WORKLOAD_LABELS = {
    "micro": "W-micro",
    "full": "W-full",
    "canary-param": "W-canary",
    "canary-overlap": "W-canary-overlap",
    "shared-overlap": "W-shared-overlap",
}
_CAPTURE_SCHEMA = "commcanary.rostam.physical.capture-measurement.v1"


def _label(workload_id: str) -> str:
    return _WORKLOAD_LABELS.get(workload_id, workload_id)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a median over no trusted values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _metrics(aggregates: Sequence[Mapping[str, Any]]) -> JsonDict:
    result: JsonDict = {}
    for row in aggregates:
        if row["measurement_schema"] == _CAPTURE_SCHEMA:
            continue
        workload = _label(str(row["workload_id"]))
        configuration = str(row["configuration_id"])
        if configuration in result.setdefault(workload, {}):
            raise ValueError(f"duplicate trusted aggregate for {workload}/{configuration}")
        result[workload][configuration] = {
            "config": configuration,
            "median_us": float(row["median_us"]),
            "iqr_us": float(row["iqr_us"]),
            "repetitions": int(row["selected_repetitions"]),
        }
    return result


def _rankings(workloads: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> JsonDict:
    return {
        workload: [
            {
                "rank": index + 1,
                "config": row["config"],
                "median_us": row["median_us"],
                "iqr_us": row["iqr_us"],
                "repetitions": row["repetitions"],
            }
            for index, row in enumerate(
                sorted(configurations.values(), key=lambda item: (item["median_us"], item["config"]))
            )
        ]
        for workload, configurations in sorted(workloads.items())
    }


def _relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> Tuple[int, str]:
    difference = float(left["median_us"]) - float(right["median_us"])
    tolerance = max(float(left["iqr_us"]), float(right["iqr_us"]))
    if difference == 0.0 or abs(difference) < tolerance:
        return 0, "tie"
    return (-1, "a_faster") if difference < 0 else (1, "b_faster")


def _pairwise(workloads: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> JsonDict:
    result: JsonDict = {}
    for workload, configurations in sorted(workloads.items()):
        pairs: JsonDict = {}
        for left, right in itertools.combinations(sorted(configurations), 2):
            relation_value, relation = _relation(configurations[left], configurations[right])
            pairs[f"{left}|{right}"] = {
                "config_a": left,
                "config_b": right,
                "relation": relation,
                "relation_value": relation_value,
                "median_a_us": configurations[left]["median_us"],
                "median_b_us": configurations[right]["median_us"],
                "iqr_a_us": configurations[left]["iqr_us"],
                "iqr_b_us": configurations[right]["iqr_us"],
                "tie_tolerance_us": max(configurations[left]["iqr_us"], configurations[right]["iqr_us"]),
            }
        result[workload] = pairs
    return result


def _agreement(pairwise: Mapping[str, Mapping[str, Mapping[str, Any]]], workload: str) -> JsonDict:
    reference = "W-full"
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
        observed = int(left[key]["relation_value"])
        expected = int(right[key]["relation_value"])
        if observed == expected:
            agree += 1
        elif len(examples) < 5:
            examples.append(
                {"pair": key, "workload_relation": left[key]["relation"], "reference_relation": right[key]["relation"]}
            )
        if observed == 0 and expected == 0:
            both_tie += 1
        elif observed == 0:
            ties_left += 1
        elif expected == 0:
            ties_right += 1
        elif observed == expected:
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


def _agreements(pairwise: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> JsonDict:
    if "W-full" not in pairwise:
        return {}
    return {
        f"{workload}_vs_W-full": _agreement(pairwise, workload) for workload in sorted(pairwise) if workload != "W-full"
    }


def _find_version_config(configurations: Sequence[str], version: str) -> Optional[str]:
    defaults = [name for name in sorted(configurations) if version in name and "default" in name.lower()]
    if defaults:
        return defaults[0]
    matches = [name for name in sorted(configurations) if version in name]
    return matches[0] if matches else None


def _regression(
    workloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    baseline_config: Optional[str],
    candidate_config: Optional[str],
    relative_threshold_pct: float,
    absolute_threshold_us: float,
) -> JsonDict:
    all_configurations = sorted({configuration for rows in workloads.values() for configuration in rows})
    baseline = baseline_config or _find_version_config(all_configurations, "2.19.3")
    candidate = candidate_config or _find_version_config(all_configurations, "2.20.5")
    table: JsonDict = {
        "baseline_config": baseline,
        "candidate_config": candidate,
        "thresholds": {
            "median_threshold_pct": relative_threshold_pct,
            "median_absolute_threshold_us": absolute_threshold_us,
            "rule": "candidate_median - baseline_median > max(abs_us, baseline_median * pct / 100)",
        },
        "workloads": {},
        "confusion_vs_full": {},
    }
    if baseline is None or candidate is None:
        return table
    for workload, rows in sorted(workloads.items()):
        if baseline not in rows or candidate not in rows:
            continue
        base = float(rows[baseline]["median_us"])
        observed = float(rows[candidate]["median_us"])
        delta = observed - base
        threshold = max(absolute_threshold_us, abs(base) * relative_threshold_pct / 100.0)
        table["workloads"][workload] = {
            "baseline_median_us": round(base, 6),
            "candidate_median_us": round(observed, 6),
            "delta_us": round(delta, 6),
            "delta_pct": round((100.0 * delta / base) if base else 0.0, 6),
            "threshold_us": round(threshold, 6),
            "regression": delta > threshold,
        }
    full = table["workloads"].get("W-full", {}).get("regression")
    if isinstance(full, bool):
        for workload, row in sorted(table["workloads"].items()):
            if workload == "W-full":
                continue
            observed = bool(row["regression"])
            table["confusion_vs_full"][workload] = {
                "full_regression": full,
                "workload_regression": observed,
                "cell": "TP" if full and observed else "FN" if full else "FP" if observed else "TN",
            }
    return table


def _costs(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("wall_time_s") is not None:
            grouped.setdefault(_label(str(row["workload_id"])), []).append(row)
    result: JsonDict = {}
    for workload, items in sorted(grouped.items()):
        wall_times = [float(item["wall_time_s"]) for item in items]
        artifact_values: Dict[str, List[float]] = {}
        for item in items:
            artifacts = item.get("artifacts", [])
            if isinstance(artifacts, list):
                for artifact in artifacts:
                    if isinstance(artifact, Mapping):
                        artifact_values.setdefault(str(artifact["artifact_id"]), []).append(
                            float(artifact["size_bytes"])
                        )
        result[workload] = {
            "runs": len(items),
            "wall_time_s_median": round(_median(wall_times), 6),
            "artifact_size_bytes_median": {
                artifact_id: round(_median(values), 6) for artifact_id, values in sorted(artifact_values.items())
            },
        }
    return result


def build_trusted_claims(
    aggregates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    baseline_config: Optional[str],
    candidate_config: Optional[str],
    relative_threshold_pct: float,
    absolute_threshold_us: float,
) -> JsonDict:
    """Build rankings and verdicts only after the caller proves completeness."""

    policy = {
        "baseline_config": baseline_config,
        "candidate_config": candidate_config,
        "median_threshold_pct": relative_threshold_pct,
        "median_absolute_threshold_us": absolute_threshold_us,
        "tie_policy": "difference-below-either-config-iqr",
    }
    if not complete:
        return {"status": "withheld-incomplete", "policy": policy}
    workloads = _metrics(aggregates)
    if "W-full" not in workloads:
        return {"status": "not-applicable-no-full-workload", "policy": policy, "costs": _costs(rows)}
    pairwise = _pairwise(workloads)
    return {
        "status": "supported-by-complete-selected-evidence",
        "policy": policy,
        "workload_metrics": workloads,
        "rankings": _rankings(workloads),
        "pairwise_relations": pairwise,
        "agreements": _agreements(pairwise),
        "regression_2x2": _regression(
            workloads,
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            relative_threshold_pct=relative_threshold_pct,
            absolute_threshold_us=absolute_threshold_us,
        ),
        "costs": _costs(rows),
    }
