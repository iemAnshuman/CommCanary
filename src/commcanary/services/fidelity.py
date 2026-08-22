"""Decision-fidelity scoring for arbitrary measurement proxies."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.measurement_set import validate_measurement_set
from ..errors import SchemaError
from ..statistics import median, percentile

FIDELITY_REPORT_FORMAT = "commcanary.fidelity_report.v1"
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_MINIMUM_REPLICATES = 3
MAX_BOOTSTRAP_SAMPLE_DRAWS = 10_000_000


def score_fidelity(
    measurement_sets: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float = 0.0,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    minimum_replicates: int = DEFAULT_MINIMUM_REPLICATES,
) -> Dict[str, Any]:
    """Score one reference and one or more proxies without tool-specific paths."""

    _validate_method(
        tie_tolerance=tie_tolerance,
        confidence=confidence,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        minimum_replicates=minimum_replicates,
    )
    if len(measurement_sets) < 2:
        raise SchemaError("fidelity scoring requires one reference and at least one proxy")
    for measurement_set in measurement_sets:
        validate_measurement_set(measurement_set)
    references = [row for row in measurement_sets if row["role"] == "reference"]
    proxies = [row for row in measurement_sets if row["role"] == "proxy"]
    if len(references) != 1 or not proxies:
        raise SchemaError("fidelity scoring requires exactly one reference and at least one proxy")
    names = [str(row["name"]) for row in measurement_sets]
    if len(set(names)) != len(names):
        raise SchemaError("fidelity measurement set names must be unique")

    reference = references[0]
    configurations = set(_measurements(reference))
    for proxy in proxies:
        proxy_configurations = set(_measurements(proxy))
        if proxy_configurations != configurations:
            missing = sorted(configurations - proxy_configurations)
            extra = sorted(proxy_configurations - configurations)
            raise SchemaError(
                f"proxy {proxy['name']!r} configuration set does not match the reference "
                f"(missing={missing}, extra={extra})"
            )
    for measurement_set in measurement_sets:
        insufficient = sorted(
            configuration
            for configuration, samples in _measurements(measurement_set).items()
            if len(samples) < minimum_replicates
        )
        if insufficient:
            raise SchemaError(
                f"measurement set {measurement_set['name']!r} has fewer than {minimum_replicates} "
                f"replicates for configurations {insufficient}; refusing bootstrap interval"
            )
    reference_sample_count = sum(len(samples) for samples in _measurements(reference).values())
    for proxy in proxies:
        proxy_sample_count = sum(len(samples) for samples in _measurements(proxy).values())
        bootstrap_work = bootstrap_resamples * (reference_sample_count + proxy_sample_count)
        if bootstrap_work > MAX_BOOTSTRAP_SAMPLE_DRAWS:
            raise SchemaError(
                f"fidelity bootstrap sample draws {bootstrap_work} exceed limit={MAX_BOOTSTRAP_SAMPLE_DRAWS}"
            )

    reference_centers = _centers(reference)
    reference_ranking = _ranking(reference_centers, str(reference["direction"]))
    true_optimum = reference_ranking[0]
    true_worst = reference_ranking[-1]
    scored = [
        _score_proxy(
            reference,
            proxy,
            reference_centers=reference_centers,
            reference_ranking=reference_ranking,
            tie_tolerance=tie_tolerance,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        for proxy in proxies
    ]
    by_regret = sorted(scored, key=lambda row: (float(row["regret_at_1_pct"]), str(row["name"])))
    by_agreement = sorted(
        scored,
        key=lambda row: (
            -float(_mapping(row["ranking_quality"])["pairwise_decision_agreement"]),
            str(row["name"]),
        ),
    )
    regret_names = [str(row["name"]) for row in by_regret]
    agreement_names = [str(row["name"]) for row in by_agreement]
    regret_ranks = {name: index + 1 for index, name in enumerate(regret_names)}
    agreement_ranks = {name: index + 1 for index, name in enumerate(agreement_names)}
    for row in scored:
        name = str(row["name"])
        row["regret_rank"] = regret_ranks[name]
        row["agreement_rank"] = agreement_ranks[name]

    return {
        "format": FIDELITY_REPORT_FORMAT,
        "reference": {
            "name": reference["name"],
            "measurement_set_sha256": hashlib.sha256(canonical_json_bytes(reference)).hexdigest(),
            "metric": reference["metric"],
            "direction": reference["direction"],
            "configuration_count": len(configurations),
            "true_optimum": true_optimum,
            "true_worst": true_worst,
            "ranking": reference_ranking,
        },
        "method": {
            "point_estimate": "median",
            "regret_unit": "percent_of_reference_optimum",
            "shortlist_k": [2, 3],
            "tie_tolerance": {
                "method": "maximum_configuration_iqr_with_absolute_floor",
                "absolute_floor": tie_tolerance,
                "boundary": "difference_less_than_tolerance_is_tie",
            },
            "bootstrap": {
                "method": "percentile",
                "confidence": confidence,
                "resamples": bootstrap_resamples,
                "seed": seed,
                "seed_derivation": "sha256_canonical_names_first_8_bytes_big_endian",
                "minimum_replicates": minimum_replicates,
            },
        },
        "decision_ranking": regret_names,
        "ranking_quality_ranking": agreement_names,
        "rankings_diverge": regret_names != agreement_names,
        "proxies": sorted(scored, key=lambda row: int(row["regret_rank"])),
    }


def _score_proxy(
    reference: Mapping[str, Any],
    proxy: Mapping[str, Any],
    *,
    reference_centers: Mapping[str, float],
    reference_ranking: Sequence[Mapping[str, Any]],
    tie_tolerance: float,
    confidence: float,
    bootstrap_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    proxy_centers = _centers(proxy)
    proxy_ranking = _ranking(proxy_centers, str(proxy["direction"]))
    chosen = str(proxy_ranking[0]["configuration"])
    regret_at_1 = _regret_pct(
        reference_centers[chosen],
        float(reference_ranking[0]["value"]),
        str(reference["direction"]),
    )
    shortlist = []
    for requested_k in (2, 3):
        effective_k = min(requested_k, len(proxy_ranking))
        candidates = [str(row["configuration"]) for row in proxy_ranking[:effective_k]]
        best = _best_configuration(candidates, reference_centers, str(reference["direction"]))
        shortlist.append(
            {
                "k": requested_k,
                "effective_k": effective_k,
                "configurations": candidates,
                "best_reference_configuration": best,
                "regret_pct": _regret_pct(
                    reference_centers[best],
                    float(reference_ranking[0]["value"]),
                    str(reference["direction"]),
                ),
            }
        )
    lower, upper = _bootstrap_regret_interval(
        reference,
        proxy,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    derived_seed = _bootstrap_seed(reference, proxy, seed)
    worst_configuration = str(reference_ranking[-1]["configuration"])
    proxy_position = next(
        index for index, row in enumerate(proxy_ranking, start=1) if row["configuration"] == worst_configuration
    )
    bottom_count = max(1, math.ceil(len(proxy_ranking) / 4.0))
    bottom_start = len(proxy_ranking) - bottom_count + 1
    ranking_quality = _ranking_quality(
        reference,
        proxy,
        reference_centers=reference_centers,
        proxy_centers=proxy_centers,
        absolute_tie_tolerance=tie_tolerance,
    )
    return {
        "name": proxy["name"],
        "measurement_set_sha256": hashlib.sha256(canonical_json_bytes(proxy)).hexdigest(),
        "metric": proxy["metric"],
        "direction": proxy["direction"],
        "recommended_configuration": chosen,
        "recommended_reference_value": reference_centers[chosen],
        "regret_at_1_pct": regret_at_1,
        "regret_at_k": shortlist,
        "regret_at_1_confidence_interval": {
            "confidence": confidence,
            "lower_pct": lower,
            "upper_pct": upper,
            "resamples": bootstrap_resamples,
            "seed": seed,
            "derived_seed": derived_seed,
        },
        "disaster_detection": {
            "configuration": worst_configuration,
            "reference_value": reference_centers[worst_configuration],
            "true_shortfall_pct": _regret_pct(
                reference_centers[worst_configuration],
                float(reference_ranking[0]["value"]),
                str(reference["direction"]),
            ),
            "proxy_rank": proxy_position,
            "bottom_quartile_rank_start": bottom_start,
            "in_bottom_quartile": proxy_position >= bottom_start,
        },
        "ranking_quality": ranking_quality,
        "proxy_ranking": proxy_ranking,
    }


def _ranking_quality(
    reference: Mapping[str, Any],
    proxy: Mapping[str, Any],
    *,
    reference_centers: Mapping[str, float],
    proxy_centers: Mapping[str, float],
    absolute_tie_tolerance: float,
) -> Dict[str, Any]:
    reference_iqrs = _iqrs(reference)
    proxy_iqrs = _iqrs(proxy)
    pairs = list(itertools.combinations(sorted(reference_centers), 2))
    agreement = 0
    concordant = 0
    discordant = 0
    proxy_ties_only = 0
    reference_ties_only = 0
    both_tie = 0
    for left, right in pairs:
        reference_relation = _relation(
            reference_centers[left],
            reference_centers[right],
            direction=str(reference["direction"]),
            tolerance=max(absolute_tie_tolerance, reference_iqrs[left], reference_iqrs[right]),
        )
        proxy_relation = _relation(
            proxy_centers[left],
            proxy_centers[right],
            direction=str(proxy["direction"]),
            tolerance=max(absolute_tie_tolerance, proxy_iqrs[left], proxy_iqrs[right]),
        )
        if reference_relation == proxy_relation:
            agreement += 1
        if reference_relation == 0 and proxy_relation == 0:
            both_tie += 1
        elif proxy_relation == 0:
            proxy_ties_only += 1
        elif reference_relation == 0:
            reference_ties_only += 1
        elif reference_relation == proxy_relation:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + proxy_ties_only) * (concordant + discordant + reference_ties_only)
    )
    tau: Any
    if denominator:
        tau = (concordant - discordant) / denominator
    else:
        tau = None
    return {
        "metric_class": "ranking_quality",
        "pair_count": len(pairs),
        "pairwise_decision_agreement": agreement / len(pairs),
        "pairwise_decision_agreement_pct": agreement / len(pairs) * 100.0,
        "kendall_tau_b": tau,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "proxy_ties_only": proxy_ties_only,
        "reference_ties_only": reference_ties_only,
        "both_tie": both_tie,
    }


def _bootstrap_regret_interval(
    reference: Mapping[str, Any],
    proxy: Mapping[str, Any],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> Tuple[float, float]:
    reference_measurements = _measurements(reference)
    proxy_measurements = _measurements(proxy)
    configurations = sorted(reference_measurements)
    derived_seed = _bootstrap_seed(reference, proxy, seed)
    generator = random.Random(derived_seed)
    regrets: List[float] = []
    for _ in range(resamples):
        reference_draw = {
            configuration: median(samples[generator.randrange(len(samples))] for _ in samples)
            for configuration, samples in reference_measurements.items()
        }
        proxy_draw = {
            configuration: median(samples[generator.randrange(len(samples))] for _ in samples)
            for configuration, samples in proxy_measurements.items()
        }
        chosen = _best_configuration(configurations, proxy_draw, str(proxy["direction"]))
        optimum = _best_configuration(configurations, reference_draw, str(reference["direction"]))
        regrets.append(_regret_pct(reference_draw[chosen], reference_draw[optimum], str(reference["direction"])))
    alpha_pct = (1.0 - confidence) * 50.0
    return percentile(regrets, alpha_pct), percentile(regrets, 100.0 - alpha_pct)


def _bootstrap_seed(reference: Mapping[str, Any], proxy: Mapping[str, Any], seed: int) -> int:
    seed_material = {
        "seed": seed,
        "reference": reference["name"],
        "proxy": proxy["name"],
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(seed_material)).digest()[:8], "big")


def _measurements(measurement_set: Mapping[str, Any]) -> Mapping[str, List[float]]:
    raw = measurement_set["measurements"]
    if not isinstance(raw, Mapping):
        raise SchemaError("measurement set measurements must be an object")
    return raw


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError("fidelity result contains a non-object field")
    return value


def _centers(measurement_set: Mapping[str, Any]) -> Dict[str, float]:
    return {configuration: median(samples) for configuration, samples in _measurements(measurement_set).items()}


def _iqrs(measurement_set: Mapping[str, Any]) -> Dict[str, float]:
    return {
        configuration: _interquartile_range(samples)
        for configuration, samples in _measurements(measurement_set).items()
    }


def _interquartile_range(samples: Sequence[float]) -> float:
    ordered = sorted(samples)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        lower = ordered[:middle]
        upper = ordered[middle + 1 :]
    else:
        lower = ordered[:middle]
        upper = ordered[middle:]
    return median(upper) - median(lower)


def _ranking(centers: Mapping[str, float], direction: str) -> List[Dict[str, Any]]:
    ordered = sorted(
        centers,
        key=lambda configuration: (
            centers[configuration] if direction == "lower_is_better" else -centers[configuration],
            configuration,
        ),
    )
    return [
        {"rank": rank, "configuration": configuration, "value": centers[configuration]}
        for rank, configuration in enumerate(ordered, start=1)
    ]


def _best_configuration(
    configurations: Sequence[str],
    centers: Mapping[str, float],
    direction: str,
) -> str:
    return min(
        configurations,
        key=lambda configuration: (
            centers[configuration] if direction == "lower_is_better" else -centers[configuration],
            configuration,
        ),
    )


def _regret_pct(value: float, optimum: float, direction: str) -> float:
    difference = value - optimum if direction == "lower_is_better" else optimum - value
    return max(0.0, difference / optimum * 100.0)


def _relation(left: float, right: float, *, direction: str, tolerance: float) -> int:
    difference = left - right
    if difference == 0.0 or abs(difference) < tolerance:
        return 0
    if direction == "lower_is_better":
        return -1 if difference < 0.0 else 1
    return -1 if difference > 0.0 else 1


def _validate_method(
    *,
    tie_tolerance: float,
    confidence: float,
    bootstrap_resamples: int,
    seed: int,
    minimum_replicates: int,
) -> None:
    if isinstance(tie_tolerance, bool) or not isinstance(tie_tolerance, (int, float)):
        raise SchemaError("fidelity tie tolerance must be numeric")
    if not math.isfinite(float(tie_tolerance)) or tie_tolerance < 0.0:
        raise SchemaError("fidelity tie tolerance must be finite and non-negative")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise SchemaError("fidelity bootstrap confidence must be numeric")
    if not 0.5 <= float(confidence) < 1.0:
        raise SchemaError("fidelity bootstrap confidence must be at least 0.5 and below 1")
    for value, name, lower, upper in (
        (bootstrap_resamples, "bootstrap resamples", 100, 1_000_000),
        (seed, "bootstrap seed", 0, (1 << 63) - 1),
        (minimum_replicates, "minimum replicates", 2, 1_000_000),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
            raise SchemaError(f"fidelity {name} must be an integer from {lower} to {upper}")


__all__ = [
    "DEFAULT_BOOTSTRAP_CONFIDENCE",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_MINIMUM_REPLICATES",
    "FIDELITY_REPORT_FORMAT",
    "MAX_BOOTSTRAP_SAMPLE_DRAWS",
    "score_fidelity",
]
