"""Canary semantic, calibration, scheduler, and provenance hashes."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from ..formats import ARTIFACT_PROVENANCE_ALGORITHM, CANARY_INTEGRITY_PROFILE
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from .canary_expansion import iter_canary_logical_events
from .json_codec import canonical_json_bytes
from .wire import JsonDict, as_float, as_int, normalize_ranks

CANARY_HASH_FIELD_NAMES = {
    "source_normalized_sha256",
    "source_trace_sha256",
    "execution_semantic_sha256",
    "scheduler_execution_sha256",
    "calibration_evaluation_sha256",
    "artifact_provenance_sha256",
    "canary_bytes",
    "byte_compression_ratio",
}


def canary_execution_sha256(
    canary: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> str:
    stable = {
        "format": canary.get("format"),
        "events": [
            _execution_event_projection(event)
            for event in iter_canary_logical_events(
                canary.get("events", []),
                limits=limits,
            )
        ],
    }
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def canary_scheduler_execution_sha256(
    canary: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> str:
    return canary_execution_sha256(canary, limits=limits)


def canary_calibration_sha256(
    canary: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> str:
    stable = {
        "format": canary.get("format"),
        "events": [
            _calibration_event_projection(event)
            for event in iter_canary_logical_events(
                canary.get("events", []),
                limits=limits,
            )
        ],
    }
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def canary_artifact_provenance_sha256(canary: Mapping[str, Any]) -> str:
    compiler = canary.get("compiler", {})
    if (
        isinstance(compiler, Mapping)
        and compiler.get("integrity_profile") == CANARY_INTEGRITY_PROFILE
        and compiler.get("artifact_provenance_algorithm") == ARTIFACT_PROVENANCE_ALGORITHM
    ):
        stable = {key: value for key, value in canary.items() if key != "created_at"}
        stable_compiler = {
            key: value
            for key, value in compiler.items()
            if key
            not in {
                "artifact_provenance_sha256",
                "canary_bytes",
                "byte_compression_ratio",
            }
        }
        stable["compiler"] = stable_compiler
    else:
        stable = _strip_canary_hash_fields(canary)
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def _strip_canary_hash_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_canary_hash_fields(child)
            for key, child in value.items()
            if key != "created_at" and key not in CANARY_HASH_FIELD_NAMES
        }
    if isinstance(value, list):
        return [_strip_canary_hash_fields(child) for child in value]
    return value


def _execution_event_projection(event: Mapping[str, Any]) -> JsonDict:
    projected: JsonDict = {}
    if "phase" in event:
        projected["phase"] = str(event.get("phase"))
    if "op" in event:
        projected["op"] = str(event.get("op"))
    if "bytes" in event:
        projected["bytes"] = as_int(event.get("bytes"))
    ranks = None
    if "ranks" in event:
        ranks = normalize_ranks(event.get("ranks"))
        projected["ranks"] = ranks
    if "group" in event:
        projected["group"] = str(event.get("group"))
    for key in ("sender_rank", "receiver_rank", "message_sequence"):
        if key in event:
            projected[key] = as_int(event.get(key))
    for key in ("tag", "channel"):
        if key in event:
            projected[key] = str(event.get(key))
    if "concurrent_groups" in event:
        projected["concurrent_groups"] = as_int(event.get("concurrent_groups"))
    if "execution_occurrence_base" in event:
        projected["execution_occurrence_base"] = as_int(event.get("execution_occurrence_base"))
    if ranks is not None:
        projected["rank_count"] = len(ranks)
    elif "rank_count" in event:
        projected["rank_count"] = as_int(event.get("rank_count"))
    samples = event.get("timing_samples")
    if isinstance(samples, list):
        projected["timing_runs"] = _execution_timing_runs(samples)
    return projected


def _calibration_event_projection(event: Mapping[str, Any]) -> JsonDict:
    projected: JsonDict = {}
    for key in ("phase", "op", "group"):
        if key in event:
            projected[key] = str(event.get(key))
    if "bytes" in event:
        projected["bytes"] = as_int(event.get("bytes"))
    if "ranks" in event:
        projected["ranks"] = normalize_ranks(event.get("ranks"))
    for key in ("sender_rank", "receiver_rank", "message_sequence"):
        if key in event:
            projected[key] = as_int(event.get(key))
    for key in ("tag", "channel"):
        if key in event:
            projected[key] = str(event.get(key))
    if "concurrent_groups" in event:
        projected["concurrent_groups"] = as_int(event.get("concurrent_groups"))
    samples = event.get("timing_samples")
    if isinstance(samples, list):
        projected["observed_runs"] = _calibration_observed_runs(samples)
    return projected


def _calibration_observed_runs(samples: Sequence[Any]) -> List[JsonDict]:
    runs: List[JsonDict] = []
    for observed in _calibration_observed_items(samples):
        if runs and runs[-1]["observed_exposed_us"] == observed:
            runs[-1]["weight"] += 1
        else:
            runs.append({"observed_exposed_us": observed, "weight": 1})
    return runs


def _calibration_observed_items(samples: Sequence[Any]) -> Iterable[Optional[float]]:
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        pattern = sample.get("timing_pattern")
        if isinstance(pattern, list) and pattern:
            emitted = list(_calibration_observed_items(pattern))
            repeats = as_int(sample.get("pattern_repeats"), 1)
            for _repeat in range(repeats):
                yield from emitted
        else:
            observed: Optional[float]
            if "observed_exposed_us" in sample:
                observed = round(as_float(sample.get("observed_exposed_us")), 9)
            else:
                observed = None
            for _ in range(as_int(sample.get("weight", 1))):
                yield observed


def _execution_timing_runs(samples: Sequence[Any]) -> List[JsonDict]:
    runs: List[JsonDict] = []
    for item in _execution_timing_items(samples):
        if runs and runs[-1]["item"] == item:
            runs[-1]["weight"] += 1
        else:
            runs.append({"item": item, "weight": 1})
    return runs


def _execution_timing_items(samples: Sequence[Any]) -> Iterable[JsonDict]:
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        pattern = sample.get("timing_pattern")
        if isinstance(pattern, list) and pattern:
            emitted = list(_execution_timing_items(pattern))
            repeats = as_int(sample.get("pattern_repeats"), 1)
            expected_gap_sum = sum(as_float(item.get("gap_us"), 0.0) for item in emitted) * repeats
            residual = _execution_record_gap_sum(sample) - expected_gap_sum
            total = len(emitted) * repeats
            index = 0
            for _repeat in range(repeats):
                for item in emitted:
                    index += 1
                    if index == total and abs(residual) > 0.0:
                        adjusted = dict(item)
                        adjusted["gap_us"] = round(as_float(adjusted.get("gap_us"), 0.0) + residual, 9)
                        yield adjusted
                    else:
                        yield item
        else:
            weight = as_int(sample.get("weight"), 1)
            gap_sum_us = _execution_record_gap_sum(sample)
            base_gap_us = gap_sum_us / weight
            consumed = 0.0
            for index in range(weight):
                gap_us = gap_sum_us - consumed if index == weight - 1 else base_gap_us
                if index != weight - 1:
                    consumed += gap_us
                yield _execution_timing_item(sample, gap_us)


def _execution_timing_item(sample: Mapping[str, Any], gap_us: float) -> JsonDict:
    item: JsonDict = {"gap_us": round(as_float(gap_us), 9)}
    offsets = sample.get("arrival_offsets_us")
    if isinstance(offsets, list):
        item["arrival_offsets_us"] = [round(as_float(value), 9) for value in offsets]
    if "compute_overlap_us" in sample:
        item["compute_overlap_us"] = round(as_float(sample.get("compute_overlap_us")), 9)
    if "compute_pressure" in sample:
        item["compute_pressure"] = round(as_float(sample.get("compute_pressure")), 9)
    return item


def _execution_record_gap_sum(sample: Mapping[str, Any]) -> float:
    if "gap_sum_us" in sample:
        return as_float(sample.get("gap_sum_us"))
    return as_float(sample.get("gap_us"), 0.0) * as_int(sample.get("weight"), 1)


__all__ = [
    "CANARY_HASH_FIELD_NAMES",
    "canary_artifact_provenance_sha256",
    "canary_calibration_sha256",
    "canary_execution_sha256",
    "canary_scheduler_execution_sha256",
]
