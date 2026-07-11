"""Deterministic timing compression and bounded-interval evidence production."""

from __future__ import annotations

from bisect import bisect_left
from enum import IntEnum
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from ..artifacts.wire import JsonDict, arrival_skew_us, as_float, as_int
from ..statistics import median, percentile
from .metrics import (
    _recursive_timing_record_count,
    _round_us,
    _source_sample_uncertain_weight,
    _source_segment_sha256,
    _timing_record_uncertain_weight,
)


class TimingPriorityTier(IntEnum):
    """Named importance tiers ordered exactly like the characterized selector."""

    CHANGE_MAGNITUDE = 0
    ZERO_GAP_TRANSITION = 10
    LONG_GAP = 20
    SENSITIVITY_TRANSITION = 40
    HIGH_PRESSURE = 50
    HIGH_OVERLAP = 60
    HIGH_SKEW = 70
    SKEW_WITH_OVERLAP = 75
    BACKLOG_TRANSITION = 80
    OBSERVED_TAIL = 90
    ENDPOINT = 100


class TimingPriority(NamedTuple):
    """Explicit tier, within-tier magnitude, then source-index tie-breaking."""

    tier: TimingPriorityTier
    magnitude: float


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
    scores: Dict[int, TimingPriority] = {
        0: TimingPriority(TimingPriorityTier.ENDPOINT, 0.0),
        last_index: TimingPriority(TimingPriorityTier.ENDPOINT, 0.0),
    }
    gaps = [as_float(sample.get("gap_us"), 0.0) for sample in samples]
    positive_gaps = [gap for gap in gaps if gap > 0.0]
    baseline_gap = median(positive_gaps) if positive_gaps else 0.0

    for index, gap_us in enumerate(gaps):
        previous_gap = gaps[index - 1] if index else gap_us
        if (gap_us == 0.0) != (previous_gap == 0.0):
            _offer_timing_priority(
                scores,
                index,
                TimingPriorityTier.ZERO_GAP_TRANSITION,
                abs(gap_us - previous_gap),
            )
        if gap_us > 0.0 and (baseline_gap == 0.0 or gap_us >= baseline_gap * 4.0):
            _offer_timing_priority(scores, index, TimingPriorityTier.LONG_GAP, gap_us)

    observed = [as_float(sample.get("observed_exposed_us")) for sample in samples if "observed_exposed_us" in sample]
    observed_p95 = percentile(observed, 95.0) if observed else None
    skews = [as_float(sample.get("arrival_skew_us"), 0.0) for sample in samples]
    overlaps = [as_float(sample.get("compute_overlap_us"), 0.0) for sample in samples]
    pressures = [as_float(sample.get("compute_pressure"), 0.5) for sample in samples]
    skew_p95 = percentile(skews, 95.0) if skews else 0.0
    overlap_p95 = percentile(overlaps, 95.0) if overlaps else 0.0
    pressure_p95 = percentile(pressures, 95.0) if pressures else 0.0

    backlog = 0.0
    previous_backlog = 0.0
    for index, sample in enumerate(samples):
        skew = skews[index]
        overlap = overlaps[index]
        pressure = pressures[index]
        # This is a sensitivity proxy, not a physical backend model: high skew,
        # pressure, and overlap are the windows most likely to change exposed
        # latency or backend ranking under different replay configurations.
        proxy_service_us = 8.0 + skew * 0.15 + pressure * 8.0
        backlog = max(0.0, backlog + proxy_service_us - gaps[index])
        if (previous_backlog == 0.0) != (backlog == 0.0):
            _offer_timing_priority(
                scores,
                index,
                TimingPriorityTier.BACKLOG_TRANSITION,
                abs(backlog - previous_backlog),
            )
        if skew_p95 > 0.0 and skew >= skew_p95:
            _offer_timing_priority(scores, index, TimingPriorityTier.HIGH_SKEW, skew)
        if overlap_p95 > 0.0 and overlap >= overlap_p95:
            _offer_timing_priority(scores, index, TimingPriorityTier.HIGH_OVERLAP, overlap)
        if pressure_p95 > 0.0 and pressure >= pressure_p95:
            _offer_timing_priority(scores, index, TimingPriorityTier.HIGH_PRESSURE, pressure)
        if skew > 0.0 and overlap > 0.0:
            _offer_timing_priority(
                scores,
                index,
                TimingPriorityTier.SKEW_WITH_OVERLAP,
                skew + overlap,
            )
        previous_backlog = backlog

    for index in range(1, len(samples)):
        delta = _timing_delta(samples[index - 1], samples[index])
        if delta > 0.0:
            _offer_timing_priority(scores, index, TimingPriorityTier.CHANGE_MAGNITUDE, delta)
        previous_high = skews[index - 1] >= skew_p95 or overlaps[index - 1] >= overlap_p95
        current_high = skews[index] >= skew_p95 or overlaps[index] >= overlap_p95
        if previous_high != current_high:
            _offer_timing_priority(scores, index, TimingPriorityTier.SENSITIVITY_TRANSITION, delta)
        if observed_p95 is not None:
            value = as_float(samples[index].get("observed_exposed_us"))
            if value >= observed_p95:
                _offer_timing_priority(scores, index, TimingPriorityTier.OBSERVED_TAIL, value)

    candidates = sorted(
        scores.items(),
        key=lambda item: (-int(item[1].tier), -item[1].magnitude, item[0]),
    )
    anchors: List[int] = []
    for index, _score in candidates:
        position = bisect_left(anchors, index)
        if position < len(anchors) and anchors[position] == index:
            continue
        candidate = anchors[:position] + [index] + anchors[position:]
        if _record_count_for_anchors(len(samples), candidate) <= sample_limit:
            anchors = candidate
    return anchors or [0]


def _offer_timing_priority(
    priorities: Dict[int, TimingPriority],
    index: int,
    tier: TimingPriorityTier,
    magnitude: float,
) -> None:
    candidate = TimingPriority(tier, magnitude)
    current = priorities.get(index)
    if current is None or candidate > current:
        priorities[index] = candidate


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
    skew_delta = abs(as_float(left.get("arrival_skew_us"), 0.0) - as_float(right.get("arrival_skew_us"), 0.0))
    before_delta = abs(as_float(left.get("compute_before_us"), 0.0) - as_float(right.get("compute_before_us"), 0.0))
    overlap_delta = abs(as_float(left.get("compute_overlap_us"), 0.0) - as_float(right.get("compute_overlap_us"), 0.0))
    pressure_delta = abs(as_float(left.get("compute_pressure"), 0.5) - as_float(right.get("compute_pressure"), 0.5))
    observed_delta = 0.0
    if "observed_exposed_us" in left and "observed_exposed_us" in right:
        observed_delta = abs(as_float(left.get("observed_exposed_us")) - as_float(right.get("observed_exposed_us")))
    return gap_delta + skew_delta + before_delta + overlap_delta + pressure_delta + observed_delta


def _aggregate_interval_record(samples: List[JsonDict], start: int, end: int) -> JsonDict:
    segment = samples[start:end]
    weight = len(segment)
    gap_sum_us = sum(as_float(sample.get("gap_us"), 0.0) for sample in segment)
    average_gap_us = gap_sum_us / weight
    representative = _joint_medoid_sample(segment, average_gap_us)
    representative_source_index = start + next(
        offset for offset, sample in enumerate(segment) if sample is representative
    )
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
        "source_count": weight,
        "representative_source_index": representative_source_index,
        "source_gap_sum_us": _round_us(gap_sum_us),
        "gap_sum_us": _round_us(gap_sum_us),
        "approximation": "bounded_interval",
        "max_gap_error_us": _round_us(
            max(abs(as_float(sample.get("gap_us"), 0.0) - average_gap_us) for sample in segment)
        ),
        "max_skew_error_us": _round_us(
            max(abs(as_float(sample.get("arrival_skew_us"), 0.0) - representative_skew) for sample in segment)
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
        "representative_gap_error_us": _round_us(abs(as_float(representative.get("gap_us"), 0.0) - average_gap_us)),
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
            max(abs(as_float(sample.get("observed_exposed_us")) - representative_observed) for sample in segment)
        )
    record["source_segment_sha256"] = _source_segment_sha256(segment)
    record["representative_selection_method"] = (
        "joint_medoid_normalized_l1_gap_skew_offsets_compute_overlap_pressure_observed"
    )
    error_fields = (
        "max_gap_error_us",
        "max_skew_error_us",
        "max_arrival_offset_error_us",
        "max_compute_before_error_us",
        "max_overlap_error_us",
        "max_pressure_error",
        "max_observed_exposed_error_us",
        "representative_gap_error_us",
        "max_prefix_gap_error_us",
    )
    record["error_vector"] = {field: record[field] for field in error_fields if field in record}
    return record


def _joint_medoid_sample(samples: List[JsonDict], average_gap_us: float) -> JsonDict:
    gap_scale = max(1.0, max(as_float(sample.get("gap_us"), 0.0) for sample in samples))
    skew_scale = max(1.0, max(as_float(sample.get("arrival_skew_us"), 0.0) for sample in samples))
    before_scale = max(1.0, max(as_float(sample.get("compute_before_us"), 0.0) for sample in samples))
    overlap_scale = max(1.0, max(as_float(sample.get("compute_overlap_us"), 0.0) for sample in samples))
    pressure_scale = max(1.0, max(as_float(sample.get("compute_pressure"), 0.5) for sample in samples))
    observed_values = [
        as_float(sample.get("observed_exposed_us")) for sample in samples if "observed_exposed_us" in sample
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


# Named compatibility seams; the implementation details remain inside compilation.
important_timing_indices = _important_timing_indices

__all__ = ["TimingPriorityTier", "important_timing_indices"]
