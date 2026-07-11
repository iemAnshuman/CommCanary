"""Trace normalization, event grouping, and sequence-motif construction."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.wire import JsonDict, arrival_skew_us, as_float, as_int, normalize_arrival_offsets, normalize_ranks
from ..errors import SchemaError
from ..operation_identity import CompressionKey, OperationIdentity
from ..statistics import median
from ._constants import DEFAULT_TIMING_SAMPLE_LIMIT, US_TOLERANCE
from .compression import _compress_timing_samples
from .metrics import (
    _recursive_timing_record_count,
    _round_us,
    _timing_record_uncertain_weight,
    _update_source_digest,
    _walk_timing_records,
)


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
            if derived_gap < -US_TOLERANCE:
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
                as_float(event.get("gap_us")) if "gap_us" in event else as_float(event.get("compute_before_us"), 0.0)
            )
            for event in result
        ]
        return result, gaps, "relative_gap_us"

    # A partial absolute clock cannot be ordered safely. It is only usable when
    # every event also carries an explicit relative gap, in which case input
    # order is the authoritative sequence.
    if not all("gap_us" in event for event in events):
        raise SchemaError("mixed timestamped and untimestamped events require explicit gap_us on every event")
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
    source_id = copy.deepcopy(event.get("id", source_index))

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
        "source": {
            "count": 1,
            "first_id": copy.deepcopy(source_id),
            "last_id": copy.deepcopy(source_id),
        },
        "_source_hasher": hasher,
        "_all_timing_samples": [timing_sample],
        "_sample_limit": sample_limit,
    }
    for integer_key in ("sender_rank", "receiver_rank", "message_sequence"):
        if integer_key in event:
            step[integer_key] = as_int(event.get(integer_key))
    for text_key in ("tag", "channel"):
        if text_key in event:
            step[text_key] = str(event.get(text_key))
    if event.get("custom_op") is True:
        step["custom_op"] = True
    return step


def _append_sample(target: Dict[str, Any], sample: Mapping[str, Any]) -> None:
    current_repeat = as_int(target.get("repeat"), 1)
    source = target["source"]
    sample_source = sample["source"]
    source["count"] = as_int(source.get("count"), 1) + 1
    source["last_id"] = copy.deepcopy(sample_source.get("last_id"))
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
    result.update(_grouped_event_summary(all_samples))
    result["source"]["digest"] = step["_source_hasher"].hexdigest()
    result["execution_occurrence_base"] = as_int(step.get("_execution_occurrence_base"), 0)
    result["source"]["sampled_timing_records"] = _recursive_timing_record_count(timing_samples)
    if any(_timing_record_uncertain_weight(record) for record in _walk_timing_records(timing_samples)):
        result["compute_fields_uncertain"] = True
    else:
        result.pop("compute_fields_uncertain", None)
    return result


def _grouped_event_summary(samples: List[JsonDict]) -> JsonDict:
    """Apply one component-wise median rule to every grouped summary field."""

    first_offsets = samples[0].get("arrival_offsets_us", [])
    offset_count = len(first_offsets) if isinstance(first_offsets, list) else 0
    offsets = [
        _round_us(median(as_float(sample.get("arrival_offsets_us", [])[offset]) for sample in samples))
        for offset in range(offset_count)
    ]
    summary: JsonDict = {
        "gap_us": _round_us(median(as_float(sample.get("gap_us"), 0.0) for sample in samples)),
        "arrival_skew_us": _round_us(median(as_float(sample.get("arrival_skew_us"), 0.0) for sample in samples)),
        "arrival_offsets_us": offsets,
        "compute_before_us": _round_us(median(as_float(sample.get("compute_before_us"), 0.0) for sample in samples)),
        "compute_overlap_us": _round_us(median(as_float(sample.get("compute_overlap_us"), 0.0) for sample in samples)),
        "compute_pressure": round(
            median(as_float(sample.get("compute_pressure"), 0.5) for sample in samples),
            6,
        ),
    }
    if "observed_exposed_us" in samples[0]:
        summary["observed_exposed_us"] = _round_us(
            median(as_float(sample.get("observed_exposed_us")) for sample in samples)
        )
    return summary


def _compress_sequence_motifs(events: List[JsonDict]) -> List[JsonDict]:
    if len(events) < 4:
        return events
    keys = [_sequence_motif_key(event) for event in events]
    output: List[JsonDict] = []
    index = 0
    motif_index = 0
    max_sequence_length = min(16, len(events) // 2)
    while index < len(events):
        best: Optional[Tuple[int, int, int]] = None
        max_here = min(max_sequence_length, (len(events) - index) // 2)
        for sequence_length in range(2, max_here + 1):
            sequence = keys[index : index + sequence_length]
            repeats = 1
            cursor = index + sequence_length
            while cursor + sequence_length <= len(events) and keys[cursor : cursor + sequence_length] == sequence:
                repeats += 1
                cursor += sequence_length
            if repeats < 2:
                continue
            saved_events = sequence_length * repeats - 1
            if best is None or saved_events > best[0] or (saved_events == best[0] and sequence_length > best[1]):
                best = (saved_events, sequence_length, repeats)
        if best is None:
            output.append(events[index])
            index += 1
            continue
        _saved, sequence_length, repeats = best
        output.append(
            _sequence_motif_record(
                events[index : index + sequence_length * repeats],
                sequence_length,
                repeats,
                motif_index,
            )
        )
        motif_index += 1
        index += sequence_length * repeats
    return output


def _sequence_motif_key(event: Mapping[str, Any]) -> str:
    template = _strip_sequence_source_fields(event)
    return hashlib.sha256(canonical_json_bytes(template)).hexdigest()


def _strip_sequence_source_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        stripped: JsonDict = {}
        for key, child in value.items():
            if key in {"source", "execution_occurrence_base", "execution_occurrence_stride"}:
                continue
            stripped[key] = _strip_sequence_source_fields(child)
        return stripped
    if isinstance(value, list):
        return [_strip_sequence_source_fields(child) for child in value]
    return value


def _sequence_motif_record(
    events: Sequence[JsonDict], sequence_length: int, repeats: int, motif_index: int
) -> JsonDict:
    template = [copy.deepcopy(event) for event in events[:sequence_length]]
    strides: Dict[CompressionKey, int] = {}
    for child in template:
        sig = OperationIdentity.from_mapping(child).compression_key()
        strides[sig] = strides.get(sig, 0) + 1
    for child in template:
        child["execution_occurrence_stride"] = strides[OperationIdentity.from_mapping(child).compression_key()]
    all_sources = [event.get("source", {}) for event in events if isinstance(event.get("source"), Mapping)]
    first_source = all_sources[0] if all_sources else {}
    last_source = all_sources[-1] if all_sources else {}
    source_count = sum(as_int(source.get("count"), 1) for source in all_sources)
    source_digest_inputs = [
        source.get("digest", [source.get("first_id"), source.get("last_id")]) for source in all_sources
    ]
    source_digest = hashlib.sha256(canonical_json_bytes({"sources": source_digest_inputs})).hexdigest()
    return {
        "program": "sequence_motif",
        "motif_id": f"sequence-{motif_index}",
        "program_repeats": repeats,
        "source": {
            "count": source_count,
            "first_id": first_source.get("first_id"),
            "last_id": last_source.get("last_id"),
            "digest": source_digest,
        },
        "events": template,
    }


# Named compatibility seam for the characterized legacy helper.
grouped_event_summary = _grouped_event_summary

__all__ = ["grouped_event_summary"]
