"""Independent source-correspondence and fidelity recomputation.

The source-commitment path intentionally owns its normalization and derived
calculations instead of importing producer compression helpers.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..artifacts.canary import canary_artifact_provenance_sha256, iter_canary_logical_events, validate_canary
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.wire import JsonDict, arrival_skew_us, as_float, as_int, normalize_arrival_offsets, normalize_ranks
from ..compilation import DEFAULT_TIMING_SAMPLE_LIMIT, compile_trace_core
from ..errors import SchemaError
from ..formats import FIDELITY_VERIFICATION_FORMAT
from ..operation_identity import OperationIdentity
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits

REFERENCE_US_TOLERANCE = 1e-6


def verify_canary_fidelity(
    trace: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Recompute compiler fidelity from source trace and compare it to a canary."""

    validate_canary(canary, limits=limits)
    compiler = canary.get("compiler", {})
    source_events = as_int(compiler.get("source_events"))
    trace_events = trace.get("events", [])
    trace_event_count = len(trace_events) if isinstance(trace_events, list) else 0
    max_events = source_events if source_events < trace_event_count else None
    timing_sample_limit = as_int(compiler.get("timing_sample_limit"), DEFAULT_TIMING_SAMPLE_LIMIT)
    timing_group_limits = _compiler_timing_group_limits(compiler, timing_sample_limit)
    expected = compile_trace_core(
        trace,
        max_events=max_events,
        timing_sample_limit=timing_sample_limit,
        timing_sample_limits_by_group=timing_group_limits,
        allow_empty=source_events == 0,
        limits=limits,
    )
    expected_compiler = expected.get("compiler", {})
    source_commitments = _verify_source_commitments(
        trace,
        canary,
        max_events=max_events,
        timing_sample_limit=timing_sample_limit,
        limits=limits,
    )
    checks = [
        _verification_check("source_format", expected.get("source_format"), canary.get("source_format")),
        _verification_check("workload", expected.get("workload"), canary.get("workload")),
        _verification_check("system", expected.get("system"), canary.get("system")),
        _verification_check(
            "source_trace_sha256",
            expected_compiler.get("source_trace_sha256"),
            compiler.get("source_trace_sha256"),
        ),
        _verification_check(
            "source_normalized_sha256",
            expected_compiler.get("source_normalized_sha256"),
            compiler.get("source_normalized_sha256"),
        ),
        _verification_check(
            "execution_semantic_sha256",
            expected_compiler.get("execution_semantic_sha256"),
            compiler.get("execution_semantic_sha256"),
        ),
        _verification_check(
            "scheduler_execution_sha256",
            expected_compiler.get("scheduler_execution_sha256"),
            compiler.get("scheduler_execution_sha256"),
        ),
        _verification_check(
            "calibration_evaluation_sha256",
            expected_compiler.get("calibration_evaluation_sha256"),
            compiler.get("calibration_evaluation_sha256"),
        ),
        _verification_check(
            "artifact_provenance_sha256",
            canary_artifact_provenance_sha256(canary),
            compiler.get("artifact_provenance_sha256"),
        ),
        _verification_check("fidelity", expected_compiler.get("fidelity"), compiler.get("fidelity")),
        _verification_check(
            "recursive_timing_records",
            expected_compiler.get("recursive_timing_records"),
            compiler.get("recursive_timing_records"),
        ),
        _verification_check(
            "approximate_timing_records",
            expected_compiler.get("approximate_timing_records"),
            compiler.get("approximate_timing_records"),
        ),
        source_commitments,
    ]
    passed = all(check["status"] == "pass" for check in checks)
    return {
        "format": FIDELITY_VERIFICATION_FORMAT,
        "status": "source_verified" if passed else "failed",
        "assurance_state": "source_corresponding" if passed else "internally_consistent",
        "source_events": source_events,
        "timing_sample_limit": timing_sample_limit,
        "checks": checks,
    }


def _verify_source_commitments(
    trace: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    max_events: Optional[int],
    timing_sample_limit: int,
    limits: ResourceLimits,
) -> JsonDict:
    ordered_events, ordered_gaps, _timing_mode = _ordered_trace_events(list(trace.get("events", [])))
    if max_events is not None:
        ordered_events = ordered_events[:max_events]
        ordered_gaps = ordered_gaps[:max_events]
    source_steps = [
        _event_to_step(
            event,
            source_index=index,
            gap_us=gap_us,
            sample_limit=timing_sample_limit,
        )
        for index, (event, gap_us) in enumerate(zip(ordered_events, ordered_gaps))
    ]
    failures: List[JsonDict] = []
    checked_intervals = 0
    pointer = 0
    for event_index, event in enumerate(iter_canary_logical_events(canary.get("events", []), limits=limits)):
        repeat = as_int(event.get("repeat"), 1)
        source_slice = source_steps[pointer : pointer + repeat]
        if len(source_slice) != repeat:
            failures.append(
                {
                    "event_index": event_index,
                    "reason": "source slice shorter than canary repeat",
                    "expected": repeat,
                    "actual": len(source_slice),
                }
            )
            break
        event_signature = OperationIdentity.from_mapping(event).compression_key()
        for local_index, source_step in enumerate(source_slice):
            source_signature = OperationIdentity.from_mapping(source_step).compression_key()
            if source_signature != event_signature:
                failures.append(
                    {
                        "event_index": event_index,
                        "source_local_index": local_index,
                        "reason": "source event signature does not match canary event",
                        "source_signature": list(source_signature),
                        "canary_signature": list(event_signature),
                    }
                )
                break
        source_samples = [step["timing_samples"][0] for step in source_slice]
        for record in _walk_timing_records(event.get("timing_samples")):
            if record.get("approximation") != "bounded_interval":
                continue
            checked_intervals += 1
            start = as_int(record.get("source_start"))
            end = as_int(record.get("source_end"))
            if end >= len(source_samples):
                failures.append(
                    {
                        "event_index": event_index,
                        "source_start": start,
                        "source_end": end,
                        "reason": "bounded interval exceeds source slice",
                    }
                )
                continue
            expected = _recompute_interval_commitment(source_samples[start : end + 1], record, source_start=start)
            mismatch = _first_commitment_mismatch(expected, record)
            if mismatch is not None:
                failures.append({"event_index": event_index, **mismatch})
        pointer += repeat
    if pointer != len(source_steps):
        failures.append(
            {
                "reason": "canary events do not consume all selected source events",
                "consumed": pointer,
                "source_events": len(source_steps),
            }
        )
    source_block_failures, checked_source_blocks = _verify_stored_source_blocks(
        source_steps,
        canary.get("events", []),
    )
    failures.extend(source_block_failures)
    return {
        "name": "source_commitments",
        "status": "pass" if not failures else "fail",
        "checked_bounded_intervals": checked_intervals,
        "checked_source_blocks": checked_source_blocks,
        "failures": failures[:20],
    }


def _verify_stored_source_blocks(
    source_steps: Sequence[Mapping[str, Any]],
    events: Any,
) -> Tuple[List[JsonDict], int]:
    """Reconstruct stored source blocks without trusting producer-provided digests."""

    failures: List[JsonDict] = []
    checked_blocks = 0
    pointer = 0
    if not isinstance(events, list):
        return ([{"reason": "canary events are not a list"}], checked_blocks)

    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        label = f"events[{event_index}].source"
        if event.get("program") != "sequence_motif":
            repeat = as_int(event.get("repeat"), 1)
            segment = source_steps[pointer : pointer + repeat]
            expected = _source_block_from_steps(segment)
            _append_source_block_mismatches(
                event.get("source"),
                expected,
                label=label,
                failures=failures,
            )
            checked_blocks += 1
            pointer += repeat
            continue

        motif_start = pointer
        children = event.get("events", [])
        repeats = as_int(event.get("program_repeats"), 1)
        occurrence_digests: List[str] = []
        if not isinstance(children, list):
            continue
        for repeat_index in range(repeats):
            for child_index, child in enumerate(children):
                if not isinstance(child, Mapping):
                    continue
                child_repeat = as_int(child.get("repeat"), 1)
                segment = source_steps[pointer : pointer + child_repeat]
                expected_child = _source_block_from_steps(segment)
                occurrence_digests.append(str(expected_child["digest"]))
                if repeat_index == 0:
                    _append_source_block_mismatches(
                        child.get("source"),
                        expected_child,
                        label=f"events[{event_index}].events[{child_index}].source",
                        failures=failures,
                    )
                    checked_blocks += 1
                pointer += child_repeat

        motif_segment = source_steps[motif_start:pointer]
        expected_motif = _source_block_from_steps(motif_segment)
        expected_motif["digest"] = hashlib.sha256(canonical_json_bytes({"sources": occurrence_digests})).hexdigest()
        _append_source_block_mismatches(
            event.get("source"),
            expected_motif,
            label=label,
            failures=failures,
        )
        checked_blocks += 1

    if pointer != len(source_steps):
        failures.append(
            {
                "reason": "stored source blocks do not consume all selected source events",
                "consumed": pointer,
                "source_events": len(source_steps),
            }
        )
    return failures, checked_blocks


def _source_block_from_steps(segment: Sequence[Mapping[str, Any]]) -> JsonDict:
    ids = [step.get("source", {}).get("first_id") for step in segment]
    hasher = hashlib.sha256()
    for source_id in ids:
        _update_source_digest(hasher, source_id)
    return {
        "count": len(ids),
        "first_id": copy.deepcopy(ids[0]) if ids else None,
        "last_id": copy.deepcopy(ids[-1]) if ids else None,
        "digest": hasher.hexdigest(),
    }


def _append_source_block_mismatches(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
    failures: List[JsonDict],
) -> None:
    actual_source = actual if isinstance(actual, Mapping) else {}
    for field in ("count", "first_id", "last_id", "digest"):
        if actual_source.get(field) == expected.get(field):
            continue
        failures.append(
            {
                "reason": "stored source block does not correspond to source trace",
                "source_block": label,
                "field": field,
                "expected": copy.deepcopy(expected.get(field)),
                "actual": copy.deepcopy(actual_source.get(field)),
            }
        )


def _recompute_interval_commitment(
    segment: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
    *,
    source_start: int,
) -> JsonDict:
    weight = len(segment)
    gap_sum_us = sum(as_float(sample.get("gap_us"), 0.0) for sample in segment)
    encoded_gap = as_float(record.get("gap_us"), 0.0)
    offsets = [_round_us(as_float(value)) for value in record.get("arrival_offsets_us", [])]
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
        prefix_encoded += encoded_gap
        max_prefix_error = max(max_prefix_error, abs(prefix_source - prefix_encoded))
    representative_source_index = as_int(record.get("representative_source_index"))
    representative_local_index = representative_source_index - source_start
    representative_gap_error = 0.0
    if 0 <= representative_local_index < len(segment):
        representative_gap_error = abs(as_float(segment[representative_local_index].get("gap_us"), 0.0) - encoded_gap)
    errors: JsonDict = {
        "max_gap_error_us": _round_us(
            max(abs(as_float(sample.get("gap_us"), 0.0) - encoded_gap) for sample in segment) if segment else 0.0
        ),
        "max_skew_error_us": _round_us(
            max(abs(as_float(sample.get("arrival_skew_us"), 0.0) - representative_skew) for sample in segment)
            if segment
            else 0.0
        ),
        "max_arrival_offset_error_us": _round_us(max_offset_error),
        "max_compute_before_error_us": _round_us(
            max(
                abs(as_float(sample.get("compute_before_us"), 0.0) - as_float(record.get("compute_before_us"), 0.0))
                for sample in segment
            )
            if segment
            else 0.0
        ),
        "max_overlap_error_us": _round_us(
            max(
                abs(as_float(sample.get("compute_overlap_us"), 0.0) - as_float(record.get("compute_overlap_us"), 0.0))
                for sample in segment
            )
            if segment
            else 0.0
        ),
        "max_pressure_error": round(
            max(
                abs(as_float(sample.get("compute_pressure"), 0.5) - as_float(record.get("compute_pressure"), 0.5))
                for sample in segment
            )
            if segment
            else 0.0,
            6,
        ),
        "representative_gap_error_us": _round_us(representative_gap_error),
        "max_prefix_gap_error_us": _round_us(max_prefix_error),
    }
    if "observed_exposed_us" in record:
        errors["max_observed_exposed_error_us"] = _round_us(
            max(
                abs(as_float(sample.get("observed_exposed_us")) - as_float(record.get("observed_exposed_us")))
                for sample in segment
            )
            if segment
            else 0.0
        )
    return {
        "source_count": weight,
        "source_gap_sum_us": _round_us(gap_sum_us),
        "source_segment_sha256": _source_segment_sha256(segment),
        "error_vector": errors,
        **errors,
    }


def _first_commitment_mismatch(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Optional[JsonDict]:
    for key in ("source_count", "source_gap_sum_us", "source_segment_sha256"):
        if actual.get(key) != expected.get(key):
            return {
                "source_start": actual.get("source_start"),
                "source_end": actual.get("source_end"),
                "field": key,
                "expected": expected.get(key),
                "actual": actual.get(key),
            }
    actual_vector = actual.get("error_vector", {})
    expected_vector = expected.get("error_vector", {})
    if not isinstance(actual_vector, Mapping):
        return {
            "source_start": actual.get("source_start"),
            "source_end": actual.get("source_end"),
            "field": "error_vector",
            "expected": expected_vector,
            "actual": actual_vector,
        }
    for key, expected_value in expected_vector.items():
        if key not in actual_vector:
            return {
                "source_start": actual.get("source_start"),
                "source_end": actual.get("source_end"),
                "field": f"error_vector.{key}",
                "expected": expected_value,
                "actual": None,
            }
        if abs(as_float(actual_vector.get(key)) - as_float(expected_value)) > 1e-6:
            return {
                "source_start": actual.get("source_start"),
                "source_end": actual.get("source_end"),
                "field": f"error_vector.{key}",
                "expected": expected_value,
                "actual": actual_vector.get(key),
            }
    return None


def _verification_check(name: str, expected: Any, actual: Any) -> JsonDict:
    return {
        "name": name,
        "status": "pass" if expected == actual else "fail",
        "expected": copy.deepcopy(expected),
        "actual": copy.deepcopy(actual),
    }


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
            if derived_gap < -REFERENCE_US_TOLERANCE:
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


def _update_source_digest(hasher: Any, source_id: Any) -> None:
    try:
        encoded = canonical_json_bytes(source_id)
    except SchemaError as exc:
        raise SchemaError(f"source id is not JSON serializable: {exc}") from exc
    hasher.update(encoded)
    hasher.update(b"\0")


def _walk_timing_records(records: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, Mapping):
            continue
        yield record
        yield from _walk_timing_records(record.get("timing_pattern"))


def _compiler_timing_group_limits(
    compiler: Mapping[str, Any],
    default_limit: int,
) -> Dict[int, int]:
    raw = compiler.get("timing_sample_limits_by_group")
    if raw is None:
        return {}
    return _normalize_timing_group_limits(raw, default_limit)


def _normalize_timing_group_limits(
    value: Optional[Mapping[Any, int]],
    default_limit: int,
) -> Dict[int, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SchemaError("timing_sample_limits_by_group must be a mapping")
    normalized: Dict[int, int] = {}
    for raw_group, raw_limit in value.items():
        group_id = as_int(raw_group)
        if group_id < 0:
            raise SchemaError("timing sample group ids must be non-negative")
        if group_id in normalized:
            raise SchemaError(f"duplicate timing sample group id {group_id}")
        limit = as_int(raw_limit)
        if limit < 2:
            raise SchemaError("timing sample limits must be at least 2")
        if limit > default_limit:
            raise SchemaError(
                f"timing sample limit for group {group_id} cannot exceed default timing_sample_limit={default_limit}"
            )
        normalized[group_id] = limit
    return normalized


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


__all__ = ["verify_canary_fidelity"]
