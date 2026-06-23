from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import (
    CANARY_FORMAT,
    TRACE_FORMAT,
    JsonDict,
    SchemaError,
    as_float,
    as_int,
    arrival_skew_us,
    canary_execution_sha256,
    median,
    normalize_arrival_offsets,
    normalize_ranks,
    percentile,
    validate_canary,
    validate_trace,
)

DEFAULT_TIMING_SAMPLE_LIMIT = 128
_US_TOLERANCE = 1e-6

_FIDELITY_FIELDS = (
    "max_gap_error_us",
    "max_skew_error_us",
    "max_arrival_offset_error_us",
    "max_compute_before_error_us",
    "max_overlap_error_us",
    "max_pressure_error",
    "max_observed_exposed_error_us",
    "max_prefix_gap_error_us",
)


def compile_trace(
    trace: Mapping[str, Any],
    *,
    max_events: Optional[int] = None,
    timing_sample_limit: int = DEFAULT_TIMING_SAMPLE_LIMIT,
    max_gap_error_us: Optional[float] = None,
    max_skew_error_us: Optional[float] = None,
    max_arrival_offset_error_us: Optional[float] = None,
    max_compute_before_error_us: Optional[float] = None,
    max_overlap_error_us: Optional[float] = None,
    max_pressure_error: Optional[float] = None,
    max_observed_exposed_error_us: Optional[float] = None,
    max_prefix_gap_error_us: Optional[float] = None,
    require_lossless_timing: bool = False,
    allow_empty: bool = False,
) -> JsonDict:
    """Compile a communication trace into a compact, fidelity-audited canary.

    Exact run/pattern encodings are preferred. If the timing stream cannot fit
    within ``timing_sample_limit``, ordered bounded intervals are emitted with
    explicit approximation errors. Optional budgets make compilation fail
    closed rather than silently exceeding an acceptable error.
    """

    validate_trace(trace)
    if not isinstance(require_lossless_timing, bool):
        raise SchemaError("require_lossless_timing must be a boolean")
    if not isinstance(allow_empty, bool):
        raise SchemaError("allow_empty must be a boolean")

    parsed_max_events: Optional[int]
    if max_events is None:
        parsed_max_events = None
    else:
        parsed_max_events = as_int(max_events)
        if parsed_max_events < 0:
            raise SchemaError("max_events must be non-negative")

    timing_sample_limit = as_int(timing_sample_limit)
    if timing_sample_limit < 2:
        raise SchemaError("timing_sample_limit must be at least 2")

    budgets = {
        "max_gap_error_us": _optional_non_negative(max_gap_error_us, "max_gap_error_us"),
        "max_skew_error_us": _optional_non_negative(max_skew_error_us, "max_skew_error_us"),
        "max_arrival_offset_error_us": _optional_non_negative(
            max_arrival_offset_error_us, "max_arrival_offset_error_us"
        ),
        "max_compute_before_error_us": _optional_non_negative(
            max_compute_before_error_us, "max_compute_before_error_us"
        ),
        "max_overlap_error_us": _optional_non_negative(max_overlap_error_us, "max_overlap_error_us"),
        "max_pressure_error": _optional_non_negative(max_pressure_error, "max_pressure_error"),
        "max_observed_exposed_error_us": _optional_non_negative(
            max_observed_exposed_error_us, "max_observed_exposed_error_us"
        ),
        "max_prefix_gap_error_us": _optional_non_negative(
            max_prefix_gap_error_us, "max_prefix_gap_error_us"
        ),
    }

    ordered_events, ordered_gaps, timing_mode = _ordered_trace_events(list(trace.get("events", [])))
    if parsed_max_events is not None:
        ordered_events = ordered_events[:parsed_max_events]
        ordered_gaps = ordered_gaps[:parsed_max_events]
    if not ordered_events and not allow_empty:
        raise SchemaError("cannot compile an empty trace without allow_empty=True")

    observed_flags = ["observed_exposed_us" in event for event in ordered_events]
    if any(observed_flags) and not all(observed_flags):
        raise SchemaError("observed_exposed_us must be present on every selected trace event or none")
    has_observed_tail = bool(observed_flags and all(observed_flags))

    canary_events: List[JsonDict] = []
    for source_index, (event, gap_us) in enumerate(zip(ordered_events, ordered_gaps)):
        step = _event_to_step(
            event,
            source_index=source_index,
            gap_us=gap_us,
            sample_limit=timing_sample_limit,
        )
        signature = _signature(step)
        if canary_events and canary_events[-1].get("_signature") == signature:
            _append_sample(canary_events[-1], step)
        else:
            step["_signature"] = signature
            canary_events.append(step)

    finalized = [_finalize_step(step) for step in canary_events]
    # Release the uncompressed timing streams before serialising the source
    # trace. This avoids holding two large canonical representations at once.
    canary_events.clear()

    source_count = len(ordered_events)
    compiled_count = len(finalized)
    event_ratio = round(source_count / compiled_count, 3) if compiled_count else 0.0
    recursive_records = sum(_recursive_timing_record_count(event.get("timing_samples")) for event in finalized)
    approximate_records = sum(_approximate_record_count(event.get("timing_samples")) for event in finalized)
    compute_uncertain_events = sum(
        _timing_records_uncertain_weight(event.get("timing_samples"))
        for event in finalized
    )

    source_gap_total = _round_us(sum(ordered_gaps))
    encoded_gap_total = _round_us(
        sum(
            _timing_records_gap_sum(event.get("timing_samples", []))
            for event in finalized
        )
    )
    total_gap_error = _round_us(abs(source_gap_total - encoded_gap_total))
    if total_gap_error > _US_TOLERANCE:
        raise SchemaError(
            f"timing compression changed total gap duration by {total_gap_error} us"
        )

    fidelity = _summarize_fidelity(
        finalized,
        source_gap_total=source_gap_total,
        encoded_gap_total=encoded_gap_total,
        total_gap_error=total_gap_error,
    )
    if require_lossless_timing and fidelity["mode"] != "lossless_timing":
        raise SchemaError("lossless timing was requested but bounded approximation was required")
    _enforce_fidelity_budgets(fidelity, budgets)

    measured_trace = dict(trace)
    measured_trace["events"] = ordered_events
    source_bytes = _json_size(measured_trace)
    source_sha = hashlib.sha256(_canonical_json_bytes(measured_trace)).hexdigest()

    canary: JsonDict = {
        "format": CANARY_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_format": trace.get("format", TRACE_FORMAT),
        "workload": dict(trace.get("workload", {})),
        "system": dict(trace.get("system", {})),
        "compiler": {
            "compression": "ordered exact patterns with fidelity-audited bounded intervals",
            "timing_sample_limit": timing_sample_limit,
            "timing_mode": timing_mode,
            "tail_signal": "observed_exposed_us" if has_observed_tail else "structural-proxy",
            "source_events": source_count,
            "canary_events": compiled_count,
            "compression_ratio": event_ratio,
            "event_compression_ratio": event_ratio,
            "recursive_timing_records": recursive_records,
            "approximate_timing_records": approximate_records,
            "source_bytes": source_bytes,
            "source_trace_sha256": source_sha,
            "fidelity": fidelity,
            "fidelity_budget": {key: value for key, value in budgets.items() if value is not None},
        },
        "events": finalized,
    }
    if compute_uncertain_events:
        canary["compiler"]["capture_uncertainty"] = {
            "compute_fields_uncertain_events": compute_uncertain_events,
            "status": "rank_local_compute_fields_uncertain",
        }
    canary["compiler"]["execution_semantic_sha256"] = canary_execution_sha256(canary)
    _update_size_metrics(canary)
    validate_canary(canary)
    return canary


def _ordered_trace_events(events: List[Mapping[str, Any]]) -> Tuple[List[Mapping[str, Any]], List[float], str]:
    if not events:
        return [], [], "empty"
    start_flags = ["start_us" in event for event in events]
    if all(start_flags):
        ordered = sorted(enumerate(events), key=lambda pair: (as_float(pair[1].get("start_us")), pair[0]))
        result = [event for _position, event in ordered]
        gaps: List[float] = []
        previous_start: Optional[float] = None
        for index, event in enumerate(result):
            start_us = as_float(event.get("start_us"))
            derived_gap = 0.0 if previous_start is None else start_us - previous_start
            if derived_gap < -_US_TOLERANCE:
                raise SchemaError("start_us values must be non-decreasing after ordering")
            derived_gap = max(0.0, derived_gap)
            if "gap_us" in event:
                explicit_gap = as_float(event.get("gap_us"))
                if index == 0:
                    gap_us = explicit_gap
                else:
                    if abs(explicit_gap - derived_gap) > 0.001:
                        raise SchemaError("gap_us conflicts with the difference between start_us values")
                    gap_us = derived_gap
            else:
                gap_us = derived_gap
            gaps.append(_round_us(gap_us))
            previous_start = start_us
        return result, gaps, "absolute_start_us"

    if not any(start_flags):
        result = list(events)
        gaps = [
            _round_us(
                as_float(event.get("gap_us"))
                if "gap_us" in event
                else as_float(event.get("compute_before_us"), 0.0)
            )
            for event in result
        ]
        return result, gaps, "relative_gap_us"

    # A partial absolute clock cannot be ordered safely. It is only usable when
    # every event also carries an explicit relative gap, in which case input
    # order is the authoritative sequence.
    if not all("gap_us" in event for event in events):
        raise SchemaError(
            "mixed timestamped and untimestamped events require explicit gap_us on every event"
        )
    return list(events), [_round_us(as_float(event.get("gap_us"))) for event in events], "explicit_relative_mixed"


def _event_to_step(
    event: Mapping[str, Any],
    *,
    source_index: int,
    gap_us: float,
    sample_limit: int,
) -> JsonDict:
    if event.get("arrival_skew_unknown"):
        raise SchemaError("cannot compile uncalibrated cross-rank arrival skew")
    ranks = normalize_ranks(event.get("ranks"))
    offsets = normalize_arrival_offsets(event, ranks)
    skew = arrival_skew_us(offsets)
    overlap_us = as_float(event.get("compute_overlap_us"), 0.0)
    compute_before_us = as_float(event.get("compute_before_us"), 0.0)
    compute_pressure = as_float(event.get("compute_pressure"), 0.5)
    observed_exposed = event.get("observed_exposed_us")
    source_id = event.get("id", source_index)

    timing_sample: JsonDict = {
        "gap_us": _round_us(gap_us),
        "arrival_offsets_us": [_round_us(value) for value in offsets],
        "arrival_skew_us": _round_us(skew),
        "compute_before_us": _round_us(compute_before_us),
        "compute_overlap_us": _round_us(overlap_us),
        "compute_pressure": round(compute_pressure, 6),
        "source_index": 0,
        "weight": 1,
    }
    if observed_exposed is not None:
        timing_sample["observed_exposed_us"] = _round_us(as_float(observed_exposed))
    if event.get("compute_fields_uncertain") is True:
        timing_sample["compute_fields_uncertain"] = True
        timing_sample["uncertain_weight"] = 1

    hasher = hashlib.sha256()
    _update_source_digest(hasher, source_id)
    step: JsonDict = {
        "phase": str(event.get("phase", "unknown")),
        "op": str(event.get("op")),
        "bytes": as_int(event.get("bytes")),
        "ranks": ranks,
        "rank_count": len(ranks),
        "group": str(event.get("group", "default")),
        "repeat": 1,
        "gap_us": _round_us(gap_us),
        "arrival_skew_us": _round_us(skew),
        "arrival_offsets_us": [_round_us(value) for value in offsets],
        "compute_before_us": _round_us(compute_before_us),
        "compute_overlap_us": _round_us(overlap_us),
        "compute_pressure": round(compute_pressure, 6),
        "concurrent_groups": as_int(event.get("concurrent_groups"), 1),
        "timing_samples": [timing_sample],
        "source": {"count": 1, "first_id": source_id, "last_id": source_id},
        "_source_hasher": hasher,
        "_all_timing_samples": [timing_sample],
        "_sample_limit": sample_limit,
    }
    if event.get("custom_op") is True:
        step["custom_op"] = True
    return step


def _append_sample(target: Dict[str, Any], sample: Mapping[str, Any]) -> None:
    current_repeat = as_int(target.get("repeat"), 1)
    source = target["source"]
    sample_source = sample["source"]
    source["count"] = as_int(source.get("count"), 1) + 1
    source["last_id"] = sample_source.get("last_id")
    _update_source_digest(target["_source_hasher"], sample_source.get("last_id"))
    timing_sample = dict(sample["timing_samples"][0])
    timing_sample["source_index"] = current_repeat
    target["_all_timing_samples"].append(timing_sample)
    target["repeat"] = current_repeat + 1


def _finalize_step(step: Dict[str, Any]) -> JsonDict:
    all_samples: List[JsonDict] = step.get("_all_timing_samples", step.get("timing_samples", []))
    timing_samples = _compress_timing_samples(
        all_samples,
        as_int(step.get("_sample_limit"), DEFAULT_TIMING_SAMPLE_LIMIT),
    )
    result = {key: value for key, value in step.items() if not key.startswith("_")}
    result["timing_samples"] = timing_samples
    result["gap_us"] = _round_us(median(as_float(sample.get("gap_us")) for sample in all_samples))
    result["arrival_skew_us"] = _round_us(
        median(as_float(sample.get("arrival_skew_us")) for sample in all_samples)
    )
    result["compute_overlap_us"] = _round_us(
        median(as_float(sample.get("compute_overlap_us")) for sample in all_samples)
    )
    result["compute_before_us"] = _round_us(
        median(as_float(sample.get("compute_before_us")) for sample in all_samples)
    )
    result["compute_pressure"] = round(
        median(as_float(sample.get("compute_pressure"), 0.5) for sample in all_samples), 6
    )
    result["arrival_offsets_us"] = list(all_samples[0].get("arrival_offsets_us", [])) if all_samples else []
    if all_samples and "observed_exposed_us" in all_samples[0]:
        result["observed_exposed_us"] = _round_us(
            median(as_float(sample.get("observed_exposed_us")) for sample in all_samples)
        )
    result["source"]["digest"] = step["_source_hasher"].hexdigest()
    result["source"]["sampled_timing_records"] = _recursive_timing_record_count(timing_samples)
    if any(_timing_record_uncertain_weight(record) for record in _walk_timing_records(timing_samples)):
        result["compute_fields_uncertain"] = True
    else:
        result.pop("compute_fields_uncertain", None)
    return result


def _signature(step: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        step.get("phase"),
        step.get("op"),
        step.get("bytes"),
        tuple(step.get("ranks", [])),
        step.get("group"),
        as_int(step.get("concurrent_groups"), 1),
        step.get("custom_op") is True,
    )


def _update_source_digest(hasher: Any, source_id: Any) -> None:
    try:
        encoded = json.dumps(source_id, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"source id is not JSON serializable: {exc}") from exc
    hasher.update(encoded)
    hasher.update(b"\0")


def _compress_timing_samples(samples: List[JsonDict], sample_limit: int) -> List[JsonDict]:
    if len(samples) <= sample_limit:
        return [_timing_record(samples[index], index, index) for index in range(len(samples))]

    pattern_length = _short_repeated_pattern_length(samples, sample_limit)
    if pattern_length:
        return [_pattern_record(samples, pattern_length)]

    prefix_pattern = _prefix_pattern_records(samples, sample_limit)
    if prefix_pattern:
        return prefix_pattern

    runs = _run_length_timing_samples(samples)
    if _recursive_timing_record_count(runs) <= sample_limit:
        return runs

    return _bounded_interval_records(samples, sample_limit)


def _bounded_interval_records(samples: List[JsonDict], sample_limit: int) -> List[JsonDict]:
    anchors = _important_timing_indices(samples, sample_limit)
    records = _records_for_anchors(samples, anchors)
    if len(records) > sample_limit:  # Defensive; the selector should already guarantee this.
        records = _stratified_interval_records(samples, sample_limit)
    return records


def _important_timing_indices(samples: List[JsonDict], sample_limit: int) -> List[int]:
    if not samples:
        return []
    if sample_limit <= 1:
        return [0]

    last_index = len(samples) - 1
    scores: Dict[int, float] = {0: float("inf"), last_index: float("inf")}
    gaps = [as_float(sample.get("gap_us"), 0.0) for sample in samples]
    positive_gaps = [gap for gap in gaps if gap > 0.0]
    baseline_gap = median(positive_gaps) if positive_gaps else 0.0

    for index, gap_us in enumerate(gaps):
        previous_gap = gaps[index - 1] if index else gap_us
        if (gap_us == 0.0) != (previous_gap == 0.0):
            scores[index] = max(scores.get(index, 0.0), 1e9 + abs(gap_us - previous_gap))
        if gap_us > 0.0 and (baseline_gap == 0.0 or gap_us >= baseline_gap * 4.0):
            scores[index] = max(scores.get(index, 0.0), 1e10 + gap_us)

    observed = [
        as_float(sample.get("observed_exposed_us"))
        for sample in samples
        if "observed_exposed_us" in sample
    ]
    observed_p95 = percentile(observed, 95.0) if observed else None

    for index in range(1, len(samples)):
        delta = _timing_delta(samples[index - 1], samples[index])
        if delta > 0.0:
            scores[index] = max(scores.get(index, 0.0), delta)
        if observed_p95 is not None:
            value = as_float(samples[index].get("observed_exposed_us"))
            if value >= observed_p95:
                scores[index] = max(scores.get(index, 0.0), 1e12 + value)

    candidates = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    anchors: List[int] = []
    for index, _score in candidates:
        position = bisect_left(anchors, index)
        if position < len(anchors) and anchors[position] == index:
            continue
        candidate = anchors[:position] + [index] + anchors[position:]
        if _record_count_for_anchors(len(samples), candidate) <= sample_limit:
            anchors = candidate
    return anchors or [0]


def _record_count_for_anchors(sample_count: int, anchors: Sequence[int]) -> int:
    count = 0
    previous = 0
    for index in anchors:
        if index < previous:
            continue
        if previous < index:
            count += 1
        count += 1
        previous = index + 1
    if previous < sample_count:
        count += 1
    return count


def _records_for_anchors(samples: List[JsonDict], anchors: Sequence[int]) -> List[JsonDict]:
    records: List[JsonDict] = []
    previous = 0
    for index in sorted(set(anchors)):
        if index < previous:
            continue
        if previous < index:
            records.append(_aggregate_interval_record(samples, previous, index))
        records.append(_timing_record(samples[index], index, index))
        previous = index + 1
    if previous < len(samples):
        records.append(_aggregate_interval_record(samples, previous, len(samples)))
    return records


def _timing_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    gap_delta = abs(as_float(left.get("gap_us"), 0.0) - as_float(right.get("gap_us"), 0.0))
    skew_delta = abs(
        as_float(left.get("arrival_skew_us"), 0.0) - as_float(right.get("arrival_skew_us"), 0.0)
    )
    before_delta = abs(
        as_float(left.get("compute_before_us"), 0.0) - as_float(right.get("compute_before_us"), 0.0)
    )
    overlap_delta = abs(
        as_float(left.get("compute_overlap_us"), 0.0) - as_float(right.get("compute_overlap_us"), 0.0)
    )
    pressure_delta = abs(
        as_float(left.get("compute_pressure"), 0.5) - as_float(right.get("compute_pressure"), 0.5)
    )
    observed_delta = 0.0
    if "observed_exposed_us" in left and "observed_exposed_us" in right:
        observed_delta = abs(
            as_float(left.get("observed_exposed_us")) - as_float(right.get("observed_exposed_us"))
        )
    return gap_delta + skew_delta + before_delta + overlap_delta + pressure_delta + observed_delta


def _aggregate_interval_record(samples: List[JsonDict], start: int, end: int) -> JsonDict:
    segment = samples[start:end]
    weight = len(segment)
    gap_sum_us = sum(as_float(sample.get("gap_us"), 0.0) for sample in segment)
    average_gap_us = gap_sum_us / weight
    representative = _joint_medoid_sample(segment, average_gap_us)
    offsets = [_round_us(as_float(value)) for value in representative.get("arrival_offsets_us", [])]
    representative_skew = arrival_skew_us(offsets)

    max_offset_error = 0.0
    for sample in segment:
        source_offsets = [as_float(value) for value in sample.get("arrival_offsets_us", [])]
        if len(source_offsets) == len(offsets):
            max_offset_error = max(
                max_offset_error,
                max((abs(left - right) for left, right in zip(source_offsets, offsets)), default=0.0),
            )

    prefix_source = 0.0
    prefix_encoded = 0.0
    max_prefix_error = 0.0
    for sample in segment:
        prefix_source += as_float(sample.get("gap_us"), 0.0)
        prefix_encoded += average_gap_us
        max_prefix_error = max(max_prefix_error, abs(prefix_source - prefix_encoded))

    record: JsonDict = {
        "gap_us": _round_us(average_gap_us),
        "arrival_offsets_us": offsets,
        "arrival_skew_us": _round_us(representative_skew),
        "compute_before_us": _round_us(as_float(representative.get("compute_before_us"), 0.0)),
        "compute_overlap_us": _round_us(as_float(representative.get("compute_overlap_us"), 0.0)),
        "compute_pressure": round(as_float(representative.get("compute_pressure"), 0.5), 6),
        "source_index": start,
        "source_start": start,
        "source_end": end - 1,
        "weight": weight,
        "gap_sum_us": _round_us(gap_sum_us),
        "approximation": "bounded_interval",
        "max_gap_error_us": _round_us(
            max(abs(as_float(sample.get("gap_us"), 0.0) - average_gap_us) for sample in segment)
        ),
        "max_skew_error_us": _round_us(
            max(
                abs(as_float(sample.get("arrival_skew_us"), 0.0) - representative_skew)
                for sample in segment
            )
        ),
        "max_arrival_offset_error_us": _round_us(max_offset_error),
        "max_compute_before_error_us": _round_us(
            max(
                abs(
                    as_float(sample.get("compute_before_us"), 0.0)
                    - as_float(representative.get("compute_before_us"), 0.0)
                )
                for sample in segment
            )
        ),
        "max_overlap_error_us": _round_us(
            max(
                abs(
                    as_float(sample.get("compute_overlap_us"), 0.0)
                    - as_float(representative.get("compute_overlap_us"), 0.0)
                )
                for sample in segment
            )
        ),
        "max_pressure_error": round(
            max(
                abs(
                    as_float(sample.get("compute_pressure"), 0.5)
                    - as_float(representative.get("compute_pressure"), 0.5)
                )
                for sample in segment
            ),
            6,
        ),
        "representative_gap_error_us": _round_us(
            abs(as_float(representative.get("gap_us"), 0.0) - average_gap_us)
        ),
        "max_prefix_gap_error_us": _round_us(max_prefix_error),
    }
    uncertain_weight = sum(_source_sample_uncertain_weight(sample) for sample in segment)
    if uncertain_weight:
        record["compute_fields_uncertain"] = True
        record["uncertain_weight"] = uncertain_weight
    if "observed_exposed_us" in representative:
        representative_observed = as_float(representative.get("observed_exposed_us"))
        record["observed_exposed_us"] = _round_us(representative_observed)
        record["max_observed_exposed_error_us"] = _round_us(
            max(
                abs(as_float(sample.get("observed_exposed_us")) - representative_observed)
                for sample in segment
            )
        )
    return record


def _joint_medoid_sample(samples: List[JsonDict], average_gap_us: float) -> JsonDict:
    gap_scale = max(1.0, max(as_float(sample.get("gap_us"), 0.0) for sample in samples))
    skew_scale = max(1.0, max(as_float(sample.get("arrival_skew_us"), 0.0) for sample in samples))
    before_scale = max(1.0, max(as_float(sample.get("compute_before_us"), 0.0) for sample in samples))
    overlap_scale = max(1.0, max(as_float(sample.get("compute_overlap_us"), 0.0) for sample in samples))
    pressure_scale = max(1.0, max(as_float(sample.get("compute_pressure"), 0.5) for sample in samples))
    observed_values = [
        as_float(sample.get("observed_exposed_us"))
        for sample in samples
        if "observed_exposed_us" in sample
    ]
    observed_scale = max(1.0, max(observed_values, default=1.0))

    median_skew = median(as_float(sample.get("arrival_skew_us"), 0.0) for sample in samples)
    median_before = median(as_float(sample.get("compute_before_us"), 0.0) for sample in samples)
    median_overlap = median(as_float(sample.get("compute_overlap_us"), 0.0) for sample in samples)
    median_pressure = median(as_float(sample.get("compute_pressure"), 0.5) for sample in samples)
    median_observed = median(observed_values) if observed_values else 0.0

    def distance(sample: Mapping[str, Any]) -> Tuple[float, int]:
        value = (
            abs(as_float(sample.get("arrival_skew_us"), 0.0) - median_skew) / skew_scale
            + abs(as_float(sample.get("gap_us"), 0.0) - average_gap_us) / gap_scale
            + abs(as_float(sample.get("compute_before_us"), 0.0) - median_before) / before_scale
            + abs(as_float(sample.get("compute_overlap_us"), 0.0) - median_overlap) / overlap_scale
            + abs(as_float(sample.get("compute_pressure"), 0.5) - median_pressure) / pressure_scale
        )
        if observed_values:
            value += abs(as_float(sample.get("observed_exposed_us")) - median_observed) / observed_scale
        return value, as_int(sample.get("source_index"), 0)

    return min(samples, key=distance)


def _stratified_interval_records(samples: List[JsonDict], sample_limit: int) -> List[JsonDict]:
    records: List[JsonDict] = []
    total = len(samples)
    for bucket in range(sample_limit):
        start = bucket * total // sample_limit
        end = (bucket + 1) * total // sample_limit
        if end <= start:
            end = start + 1
        records.append(_aggregate_interval_record(samples, start, end))
    return records


def _timing_record(sample: Mapping[str, Any], source_start: int, source_end: int) -> JsonDict:
    weight = source_end - source_start + 1
    gap_us = as_float(sample.get("gap_us"), 0.0)
    record: JsonDict = {
        "gap_us": _round_us(gap_us),
        "arrival_offsets_us": [_round_us(as_float(value)) for value in sample.get("arrival_offsets_us", [])],
        "arrival_skew_us": _round_us(as_float(sample.get("arrival_skew_us"), 0.0)),
        "compute_before_us": _round_us(as_float(sample.get("compute_before_us"), 0.0)),
        "compute_overlap_us": _round_us(as_float(sample.get("compute_overlap_us"), 0.0)),
        "compute_pressure": round(as_float(sample.get("compute_pressure"), 0.5), 6),
        "source_index": source_start,
        "source_start": source_start,
        "source_end": source_end,
        "weight": weight,
        "gap_sum_us": _round_us(gap_us * weight),
    }
    uncertain_weight = _source_sample_uncertain_weight(sample) * weight
    if uncertain_weight:
        record["compute_fields_uncertain"] = True
        record["uncertain_weight"] = uncertain_weight
    if "observed_exposed_us" in sample:
        record["observed_exposed_us"] = _round_us(as_float(sample.get("observed_exposed_us")))
    return record


def _pattern_record(
    samples: List[JsonDict],
    pattern_length: int,
    *,
    start: int = 0,
    end: Optional[int] = None,
) -> JsonDict:
    end = len(samples) if end is None else end
    pattern = [_timing_record(samples[index], index, index) for index in range(start, start + pattern_length)]
    repeats = (end - start) // pattern_length
    gap_sum_us = sum(as_float(sample.get("gap_us"), 0.0) for sample in samples[start:end])
    record = dict(pattern[0])
    record.update(
        {
            "gap_us": _round_us(gap_sum_us / (end - start)),
            "source_index": start,
            "source_start": start,
            "source_end": end - 1,
            "weight": end - start,
            "gap_sum_us": _round_us(gap_sum_us),
            "timing_pattern": pattern,
            "pattern_repeats": repeats,
        }
    )
    uncertain_weight = sum(_timing_record_uncertain_weight(child) for child in pattern) * repeats
    if uncertain_weight:
        record["compute_fields_uncertain"] = True
        record["uncertain_weight"] = uncertain_weight
    return record


def _run_length_timing_samples(samples: List[JsonDict]) -> List[JsonDict]:
    runs: List[JsonDict] = []
    start = 0
    while start < len(samples):
        end = start + 1
        while end < len(samples) and _timing_equal(samples[start], samples[end]):
            end += 1
        runs.append(_timing_record(samples[start], start, end - 1))
        start = end
    return runs


def _short_repeated_pattern_length(samples: List[JsonDict], sample_limit: int) -> Optional[int]:
    max_pattern = min(max(1, sample_limit - 1), len(samples) // 2)
    for pattern_length in range(1, max_pattern + 1):
        if len(samples) % pattern_length:
            continue
        if all(_timing_equal(samples[index], samples[index % pattern_length]) for index in range(len(samples))):
            return pattern_length
    return None


def _prefix_pattern_records(samples: List[JsonDict], sample_limit: int) -> Optional[List[JsonDict]]:
    if sample_limit < 4 or len(samples) < 4:
        return None
    for start in range(1, min(4, len(samples) - 1)):
        remaining = len(samples) - start
        max_pattern = min(max(1, sample_limit - start - 1), remaining // 2)
        for pattern_length in range(1, max_pattern + 1):
            if remaining % pattern_length:
                continue
            if all(
                _timing_equal(samples[index], samples[start + ((index - start) % pattern_length)])
                for index in range(start, len(samples))
            ):
                records = [_timing_record(samples[index], index, index) for index in range(start)]
                records.append(_pattern_record(samples, pattern_length, start=start, end=len(samples)))
                if _recursive_timing_record_count(records) <= sample_limit:
                    return records
    return None


def _timing_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    scalar_keys = (
        "gap_us",
        "arrival_skew_us",
        "compute_before_us",
        "compute_overlap_us",
        "compute_pressure",
    )
    if any(_round_us(as_float(left.get(key), 0.0)) != _round_us(as_float(right.get(key), 0.0)) for key in scalar_keys):
        return False
    left_offsets = tuple(_round_us(as_float(value)) for value in left.get("arrival_offsets_us", []))
    right_offsets = tuple(_round_us(as_float(value)) for value in right.get("arrival_offsets_us", []))
    if left_offsets != right_offsets:
        return False
    if bool(left.get("compute_fields_uncertain")) != bool(right.get("compute_fields_uncertain")):
        return False
    left_observed = "observed_exposed_us" in left
    right_observed = "observed_exposed_us" in right
    if left_observed != right_observed:
        return False
    if left_observed:
        return _round_us(as_float(left.get("observed_exposed_us"))) == _round_us(
            as_float(right.get("observed_exposed_us"))
        )
    return True


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
    return sum(
        _timing_record_logical_uncertain_weight(record)
        for record in records
        if isinstance(record, Mapping)
    )


def _summarize_fidelity(
    events: Sequence[Mapping[str, Any]],
    *,
    source_gap_total: float,
    encoded_gap_total: float,
    total_gap_error: float,
) -> JsonDict:
    maxima = {field: 0.0 for field in _FIDELITY_FIELDS}
    approximate = 0
    for event in events:
        for record in _walk_timing_records(event.get("timing_samples")):
            if record.get("approximation") == "bounded_interval":
                approximate += 1
            for field in _FIDELITY_FIELDS:
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
        return sum(
            _timing_record_logical_uncertain_weight(child)
            for child in pattern
            if isinstance(child, Mapping)
        ) * repeats
    return _timing_record_uncertain_weight(record)


def _enforce_fidelity_budgets(fidelity: Mapping[str, Any], budgets: Mapping[str, Optional[float]]) -> None:
    for field, budget in budgets.items():
        if budget is None:
            continue
        actual = as_float(fidelity.get(field), 0.0)
        if actual > budget + _US_TOLERANCE:
            raise SchemaError(f"timing fidelity {field}={actual} us exceeds budget {budget} us")


def _optional_non_negative(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    parsed = as_float(value)
    if parsed < 0.0:
        raise SchemaError(f"{name} must be non-negative")
    return parsed


def _round_us(value: float) -> float:
    return round(as_float(value), 9)


def _canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"cannot canonicalize JSON: {exc}") from exc


def _json_size(data: Mapping[str, Any]) -> int:
    return len(_canonical_json_bytes(data))


def _update_size_metrics(canary: JsonDict) -> None:
    compiler = canary["compiler"]
    last_size = -1
    for _ in range(12):
        compiler["canary_bytes"] = max(0, last_size)
        source_bytes = as_int(compiler.get("source_bytes"), 0)
        compiler["byte_compression_ratio"] = (
            round(source_bytes / last_size, 3) if source_bytes and last_size > 0 else 0.0
        )
        current_size = _json_size(canary)
        if current_size == last_size:
            break
        last_size = current_size
    compiler["canary_bytes"] = _json_size(canary)
    compiler["byte_compression_ratio"] = (
        round(as_int(compiler.get("source_bytes")) / compiler["canary_bytes"], 3)
        if compiler["canary_bytes"]
        else 0.0
    )
