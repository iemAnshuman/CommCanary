"""Compilation accounting, fidelity summaries, and artifact finalization."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from ..artifacts.canary import (
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
)
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.wire import JsonDict, as_float, as_int
from ..errors import SchemaError
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from ._constants import FIDELITY_FIELDS, US_TOLERANCE


def _update_source_digest(hasher: Any, source_id: Any) -> None:
    try:
        encoded = canonical_json_bytes(source_id)
    except SchemaError as exc:
        raise SchemaError(f"source id is not JSON serializable: {exc}") from exc
    hasher.update(encoded)
    hasher.update(b"\0")


def _recursive_timing_record_count(records: Any) -> int:
    if not isinstance(records, list):
        return 0
    total = 0
    for record in records:
        if isinstance(record, Mapping):
            total += 1 + _recursive_timing_record_count(record.get("timing_pattern"))
    return total


def _approximate_record_count(records: Any) -> int:
    if not isinstance(records, list):
        return 0
    total = 0
    for record in records:
        if isinstance(record, Mapping):
            total += int(record.get("approximation") == "bounded_interval")
            total += _approximate_record_count(record.get("timing_pattern"))
    return total


def _timing_record_gap_sum(record: Mapping[str, Any]) -> float:
    if "gap_sum_us" in record:
        return as_float(record.get("gap_sum_us"))
    return as_float(record.get("gap_us"), 0.0) * as_int(record.get("weight"), 1)


def _timing_records_gap_sum(records: Any) -> float:
    if not isinstance(records, list):
        return 0.0
    return sum(_timing_record_gap_sum(record) for record in records if isinstance(record, Mapping))


def _timing_records_uncertain_weight(records: Any) -> int:
    if not isinstance(records, list):
        return 0
    return sum(_timing_record_logical_uncertain_weight(record) for record in records if isinstance(record, Mapping))


def _summarize_fidelity(
    events: Sequence[Mapping[str, Any]],
    *,
    source_gap_total: float,
    encoded_gap_total: float,
    total_gap_error: float,
) -> JsonDict:
    maxima = {field: 0.0 for field in FIDELITY_FIELDS}
    approximate = 0
    for event in events:
        for record in _walk_timing_records(event.get("timing_samples")):
            if record.get("approximation") == "bounded_interval":
                approximate += 1
            for field in FIDELITY_FIELDS:
                if field in record:
                    maxima[field] = max(maxima[field], as_float(record.get(field)))
    return {
        "mode": "bounded_approximate" if approximate else "lossless_timing",
        "approximate_timing_records": approximate,
        **{key: _round_us(value) for key, value in maxima.items()},
        "source_gap_total_us": _round_us(source_gap_total),
        "encoded_gap_total_us": _round_us(encoded_gap_total),
        "total_gap_error_us": _round_us(total_gap_error),
    }


def _walk_timing_records(records: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, Mapping):
            continue
        yield record
        yield from _walk_timing_records(record.get("timing_pattern"))


def _source_sample_uncertain_weight(sample: Mapping[str, Any]) -> int:
    return 1 if sample.get("compute_fields_uncertain") is True else 0


def _timing_record_uncertain_weight(record: Mapping[str, Any]) -> int:
    if "uncertain_weight" in record:
        return as_int(record.get("uncertain_weight"))
    if record.get("compute_fields_uncertain") is True:
        return as_int(record.get("weight"), 1)
    return 0


def _timing_record_logical_uncertain_weight(record: Mapping[str, Any]) -> int:
    if "uncertain_weight" in record:
        return as_int(record.get("uncertain_weight"))
    pattern = record.get("timing_pattern")
    if isinstance(pattern, list) and pattern:
        repeats = as_int(record.get("pattern_repeats"), 1)
        return (
            sum(_timing_record_logical_uncertain_weight(child) for child in pattern if isinstance(child, Mapping))
            * repeats
        )
    return _timing_record_uncertain_weight(record)


def _enforce_fidelity_budgets(fidelity: Mapping[str, Any], budgets: Mapping[str, Optional[float]]) -> None:
    for field, budget in budgets.items():
        if budget is None:
            continue
        actual = as_float(fidelity.get(field), 0.0)
        if actual > budget + US_TOLERANCE:
            raise SchemaError(f"timing fidelity {field}={actual} us exceeds budget {budget} us")


def _normalize_timing_group_limits(
    raw_limits: Optional[Mapping[Any, int]],
    default_limit: int,
) -> Dict[int, int]:
    if raw_limits is None:
        return {}
    if not isinstance(raw_limits, Mapping):
        raise SchemaError("timing_sample_limits_by_group must be an object")
    parsed_default = as_int(default_limit)
    if parsed_default < 2:
        raise SchemaError("timing_sample_limit must be at least 2")
    result: Dict[int, int] = {}
    for raw_group, raw_limit in raw_limits.items():
        group_id = as_int(raw_group)
        if group_id < 0:
            raise SchemaError("timing_sample_limits_by_group keys must be non-negative")
        limit = as_int(raw_limit)
        if limit < 2:
            raise SchemaError("timing_sample_limits_by_group values must be at least 2")
        if limit > parsed_default:
            raise SchemaError("timing_sample_limits_by_group values must not exceed timing_sample_limit")
        if limit != parsed_default:
            result[group_id] = limit
    return result


def _compiler_timing_group_limits(
    compiler: Mapping[str, Any],
    default_limit: int,
) -> Dict[int, int]:
    raw = compiler.get("timing_sample_limits_by_group")
    if raw is None:
        return {}
    return _normalize_timing_group_limits(raw, default_limit)


def _optional_non_negative(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    parsed = as_float(value)
    if parsed < 0.0:
        raise SchemaError(f"{name} must be non-negative")
    return parsed


def _round_us(value: float) -> float:
    return round(as_float(value), 9)


def _source_segment_sha256(samples: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for sample in samples:
        normalized.append(
            {
                "gap_us": _round_us(as_float(sample.get("gap_us"), 0.0)),
                "arrival_offsets_us": [_round_us(as_float(value)) for value in sample.get("arrival_offsets_us", [])],
                "arrival_skew_us": _round_us(as_float(sample.get("arrival_skew_us"), 0.0)),
                "compute_before_us": _round_us(as_float(sample.get("compute_before_us"), 0.0)),
                "compute_overlap_us": _round_us(as_float(sample.get("compute_overlap_us"), 0.0)),
                "compute_pressure": round(as_float(sample.get("compute_pressure"), 0.5), 6),
                **(
                    {"observed_exposed_us": _round_us(as_float(sample.get("observed_exposed_us")))}
                    if "observed_exposed_us" in sample
                    else {}
                ),
                **({"compute_fields_uncertain": True} if sample.get("compute_fields_uncertain") is True else {}),
            }
        )
    return hashlib.sha256(canonical_json_bytes({"samples": normalized})).hexdigest()


def _refresh_canary_hashes_and_size(
    canary: JsonDict,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    compiler = canary["compiler"]
    compiler["execution_semantic_sha256"] = canary_execution_sha256(
        canary,
        limits=limits,
    )
    compiler["scheduler_execution_sha256"] = canary_scheduler_execution_sha256(
        canary,
        limits=limits,
    )
    compiler["calibration_evaluation_sha256"] = canary_calibration_sha256(
        canary,
        limits=limits,
    )
    compiler["artifact_provenance_sha256"] = canary_artifact_provenance_sha256(canary)
    _update_size_metrics(canary)


def _json_size(data: Mapping[str, Any]) -> int:
    return len(canonical_json_bytes(data))


def _update_size_metrics(
    canary: JsonDict,
    *,
    size_calculator: Optional[Callable[[Mapping[str, Any]], int]] = None,
) -> None:
    compiler = canary["compiler"]
    calculate_size = _json_size if size_calculator is None else size_calculator
    source_bytes = as_int(compiler.get("source_bytes"), 0)
    candidate_size = 0
    seen_sizes = set()
    while candidate_size not in seen_sizes:
        seen_sizes.add(candidate_size)
        _set_size_metrics(compiler, source_bytes=source_bytes, canary_bytes=candidate_size, ratio_precision=3)
        current_size = calculate_size(canary)
        if current_size == candidate_size:
            return
        candidate_size = current_size

    # A ratio can cross a decimal-rendering boundary (for example 0.2905),
    # making the three-decimal mapping alternate between adjacent byte counts.
    # Solve that finite cycle at a lower displayed precision, but only accept a
    # state whose declared byte count exactly equals its canonical byte length.
    cycle = sorted(seen_sizes | {candidate_size})
    for ratio_precision in (2, 1, 0):
        for cycle_size in cycle:
            if cycle_size <= 0:
                continue
            _set_size_metrics(
                compiler,
                source_bytes=source_bytes,
                canary_bytes=cycle_size,
                ratio_precision=ratio_precision,
            )
            if calculate_size(canary) == cycle_size:
                return
    raise SchemaError(f"serialized-size accounting did not converge; repeated size cycle {cycle}")


def _set_size_metrics(
    compiler: JsonDict,
    *,
    source_bytes: int,
    canary_bytes: int,
    ratio_precision: int,
) -> None:
    compiler["canary_bytes"] = canary_bytes
    compiler["byte_compression_ratio"] = (
        round(source_bytes / canary_bytes, ratio_precision) if source_bytes and canary_bytes > 0 else 0.0
    )


# Intentional package interfaces used by orchestration and compatibility facades.
compiler_timing_group_limits = _compiler_timing_group_limits
json_size = _json_size
refresh_canary_hashes_and_size = _refresh_canary_hashes_and_size
source_segment_sha256 = _source_segment_sha256
update_size_metrics = _update_size_metrics

__all__ = [
    "compiler_timing_group_limits",
    "json_size",
    "refresh_canary_hashes_and_size",
    "source_segment_sha256",
    "update_size_metrics",
]
