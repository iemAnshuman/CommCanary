"""Bounded expansion of timing records into replay samples."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

from ..artifacts.wire import JsonDict, as_float, as_int


def _sample_offsets(step: Mapping[str, Any], timing_sample: Mapping[str, Any]) -> List[float]:
    offsets = timing_sample.get("arrival_offsets_us")
    if isinstance(offsets, list):
        return [as_float(value) for value in offsets]
    raw = step.get("arrival_offsets_us")
    if isinstance(raw, list):
        return [as_float(value) for value in raw]
    skew = as_float(step.get("arrival_skew_us"), 0.0)
    rank_count = max(1, as_int(step.get("rank_count"), len(step.get("ranks", [])) or 1))
    return [0.0 for _ in range(max(0, rank_count - 1))] + [skew]


def _fallback_timing(step: Mapping[str, Any]) -> JsonDict:
    sample: JsonDict = {
        "gap_us": step.get("gap_us", 0.0),
        "arrival_offsets_us": step.get("arrival_offsets_us", []),
        "compute_before_us": step.get("compute_before_us", 0.0),
        "compute_overlap_us": step.get("compute_overlap_us", 0.0),
        "compute_pressure": step.get("compute_pressure", 0.5),
        "weight": step.get("repeat", 1),
    }
    if "observed_exposed_us" in step:
        sample["observed_exposed_us"] = step.get("observed_exposed_us")
    if step.get("compute_fields_uncertain") is True:
        sample["compute_fields_uncertain"] = True
        sample["uncertain_weight"] = sample["weight"]
    return sample


def _iter_timing_samples(step: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    samples = step.get("timing_samples")
    if not isinstance(samples, list) or not samples:
        yield from _sample_repetitions(_fallback_timing(step))
        return
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        pattern = sample.get("timing_pattern")
        if isinstance(pattern, list) and pattern:
            yield from _pattern_repetitions(sample, pattern)
        else:
            yield from _sample_repetitions(sample)


def _pattern_repetitions(parent: Mapping[str, Any], pattern: List[Any]) -> Iterable[Mapping[str, Any]]:
    repeats = as_int(parent.get("pattern_repeats"), 1)
    parent_occurrence = _occurrence_base(parent)
    expected_gap_sum = 0.0
    child_weight = 0
    for child in pattern:
        if isinstance(child, Mapping):
            expected_gap_sum += _record_gap_sum(child)
            child_weight += as_int(child.get("weight"), 1)
    expected_gap_sum *= repeats
    parent_gap_sum = _record_gap_sum(parent)
    residual = parent_gap_sum - expected_gap_sum
    total_emitted = child_weight * repeats

    emitted = 0
    for _repeat_index in range(repeats):
        for child in pattern:
            if isinstance(child, Mapping):
                for item in _sample_repetitions(child):
                    emitted += 1
                    item = dict(item)
                    item["noise_occurrence"] = parent_occurrence + emitted - 1
                    if emitted == total_emitted and abs(residual) > 0.0:
                        item["gap_us"] = as_float(item.get("gap_us"), 0.0) + residual
                    yield item


def _sample_repetitions(sample: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    weight = as_int(sample.get("weight"), 1)
    uncertain_weight = as_int(
        sample.get(
            "uncertain_weight",
            weight if sample.get("compute_fields_uncertain") is True else 0,
        )
    )
    gap_sum_us = _record_gap_sum(sample)
    base_gap_us = gap_sum_us / weight
    occurrence_base = _occurrence_base(sample)
    consumed = 0.0
    for index in range(weight):
        item = dict(sample)
        gap_us = gap_sum_us - consumed if index == weight - 1 else base_gap_us
        if index != weight - 1:
            consumed += gap_us
        item["gap_us"] = gap_us
        item["weight"] = 1
        item["noise_occurrence"] = occurrence_base + index
        if index < uncertain_weight:
            item["compute_fields_uncertain"] = True
            item["uncertain_weight"] = 1
        else:
            item.pop("compute_fields_uncertain", None)
            item.pop("uncertain_weight", None)
        yield item


def _occurrence_base(sample: Mapping[str, Any]) -> int:
    if "source_start" in sample:
        return as_int(sample.get("source_start"))
    return as_int(sample.get("source_index"), 0)


def _record_gap_sum(sample: Mapping[str, Any]) -> float:
    if "gap_sum_us" in sample:
        return as_float(sample.get("gap_sum_us"))
    return as_float(sample.get("gap_us"), 0.0) * as_int(sample.get("weight"), 1)
