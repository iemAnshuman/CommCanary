from __future__ import annotations

import copy
import random
from collections import defaultdict
from statistics import median as statistics_median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .schema import (
    TRACE_FORMAT,
    JsonDict,
    SchemaError,
    arrival_skew_us,
    as_float,
    as_int,
    normalize_arrival_offsets,
    normalize_ranks,
    validate_trace,
)


_EVENT_COPY_FIELDS = (
    "phase",
    "op",
    "bytes",
    "ranks",
    "rank_count",
    "group",
    "rank_arrival_us",
    "arrival_skew_us",
    "compute_before_us",
    "compute_overlap_us",
    "compute_pressure",
    "concurrent_groups",
    "sender_rank",
    "receiver_rank",
    "tag",
    "channel",
    "message_sequence",
    "observed_exposed_us",
    "custom_op",
)


def isolated_collective_baseline_trace(trace: Mapping[str, Any]) -> JsonDict:
    """Build an isolated-collective microbenchmark-style baseline trace.

    The baseline intentionally discards workload order, skew, queue reset gaps,
    and overlap. It is useful as a negative control for ranking-inversion
    experiments, not as a source-verified canary candidate.
    """

    validate_trace(trace)
    events = list(trace.get("events", []))
    representatives: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    for event in events:
        representatives.setdefault(_operation_signature(event, include_phase=False), event)

    baseline_events: List[JsonDict] = []
    for index, event in enumerate(representatives.values()):
        ranks = normalize_ranks(event.get("ranks"))
        record = _shape_event(event, f"isolated-{index:06d}")
        record["phase"] = "isolated_collective"
        record["gap_us"] = 1.0
        record["rank_arrival_us"] = {str(rank): 0.0 for rank in ranks}
        record.pop("arrival_skew_us", None)
        record["compute_before_us"] = 0.0
        record["compute_overlap_us"] = 0.0
        record["compute_pressure"] = 0.5
        record["concurrent_groups"] = 1
        record.pop("observed_exposed_us", None)
        record["metadata"] = {"commcanary_baseline": "isolated_collective"}
        baseline_events.append(record)
    return _baseline_trace(trace, "isolated_collective", baseline_events)


def random_sampling_baseline_trace(
    trace: Mapping[str, Any],
    *,
    sample_count: int,
    seed: int = 0,
    preserve_source_event_count: bool = True,
) -> JsonDict:
    """Build a random sampling baseline trace.

    With ``preserve_source_event_count=True`` the selected samples are tiled to
    the original event count. This makes behavioral comparisons count-fair
    while still being explicitly not source-verified against the original
    workload.
    """

    validate_trace(trace)
    events = list(trace.get("events", []))
    if not events:
        raise SchemaError("cannot build a sampling baseline from an empty trace")
    parsed_sample_count = as_int(sample_count)
    if parsed_sample_count <= 0:
        raise SchemaError("sample_count must be positive")
    parsed_seed = as_int(seed)
    rng = random.Random(parsed_seed)
    selected_indices = sorted(rng.sample(range(len(events)), min(parsed_sample_count, len(events))))
    selected = [events[index] for index in selected_indices]
    output_count = len(events) if preserve_source_event_count else len(selected)
    baseline_events = []
    for index in range(output_count):
        source = selected[index % len(selected)]
        record = _shape_event(source, f"random-sample-{index:06d}")
        record["gap_us"] = _source_gap_us(source)
        record.pop("start_us", None)
        record["metadata"] = {
            "commcanary_baseline": "random_sampling",
            "sample_count": len(selected),
            "seed": parsed_seed,
            "selected_source_index": selected_indices[index % len(selected)],
        }
        baseline_events.append(record)
    return _baseline_trace(trace, "random_sampling", baseline_events)


def frequency_representative_baseline_trace(trace: Mapping[str, Any]) -> JsonDict:
    """Build a frequency/representative baseline trace.

    For every operation signature, this baseline replaces each occurrence by a
    medoid representative of that signature. It preserves operation frequency
    and order but deliberately removes within-signature tails, burst changes,
    and timing correlations.
    """

    validate_trace(trace)
    events = list(trace.get("events", []))
    if not events:
        raise SchemaError("cannot build a frequency baseline from an empty trace")
    groups: Dict[Tuple[Any, ...], List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, event in enumerate(events):
        groups[_operation_signature(event, include_phase=True)].append((index, event))
    representatives = {
        key: _representative_event([event for _index, event in grouped])
        for key, grouped in groups.items()
    }
    baseline_events: List[JsonDict] = []
    for index, event in enumerate(events):
        key = _operation_signature(event, include_phase=True)
        representative = representatives[key]
        record = _shape_event(representative, f"frequency-representative-{index:06d}")
        record["gap_us"] = _source_gap_us(representative)
        record.pop("start_us", None)
        record["metadata"] = {
            "commcanary_baseline": "frequency_representative",
            "source_signature_size": len(groups[key]),
        }
        baseline_events.append(record)
    return _baseline_trace(trace, "frequency_representative", baseline_events)


def clustering_representative_baseline_trace(
    trace: Mapping[str, Any],
    *,
    cluster_count: int = 8,
) -> JsonDict:
    """Build a deterministic clustering/medoid baseline trace.

    This is a stronger negative control than the frequency representative:
    every operation signature may keep multiple timing medoids. It preserves
    event count, operation order, and operation signatures, but intentionally
    discards exact burst/tail correlations and source commitments.
    """

    validate_trace(trace)
    events = list(trace.get("events", []))
    if not events:
        raise SchemaError("cannot build a clustering baseline from an empty trace")
    parsed_cluster_count = as_int(cluster_count)
    if parsed_cluster_count <= 0:
        raise SchemaError("cluster_count must be positive")

    groups: Dict[Tuple[Any, ...], List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, event in enumerate(events):
        groups[_operation_signature(event, include_phase=True)].append((index, event))
    representatives = {
        key: _cluster_representatives([event for _index, event in grouped], parsed_cluster_count)
        for key, grouped in groups.items()
    }

    baseline_events: List[JsonDict] = []
    for index, event in enumerate(events):
        key = _operation_signature(event, include_phase=True)
        reps = representatives[key]
        representative, representative_index = _nearest_representative(event, reps)
        record = _shape_event(representative, f"cluster-representative-{index:06d}")
        record["gap_us"] = _source_gap_us(representative)
        record.pop("start_us", None)
        record["metadata"] = {
            "commcanary_baseline": "clustering_representative",
            "cluster_count": len(reps),
            "representative_index": representative_index,
            "source_signature_size": len(groups[key]),
        }
        baseline_events.append(record)
    return _baseline_trace(trace, "clustering_representative", baseline_events)


def _baseline_trace(trace: Mapping[str, Any], method: str, events: Sequence[JsonDict]) -> JsonDict:
    workload = dict(trace.get("workload", {}))
    notes = str(workload.get("notes", ""))
    suffix = f"CommCanary research baseline: {method}. Not source-verified against the original trace."
    workload["notes"] = f"{notes} {suffix}".strip()
    workload["baseline_method"] = method
    return {
        "format": TRACE_FORMAT,
        "workload": workload,
        "system": dict(trace.get("system", {})),
        "events": list(events),
    }


def _operation_signature(event: Mapping[str, Any], *, include_phase: bool) -> Tuple[Any, ...]:
    ranks = tuple(normalize_ranks(event.get("ranks")))
    phase = str(event.get("phase", "unknown")) if include_phase else "*"
    return (
        phase,
        str(event.get("op")),
        as_int(event.get("bytes")),
        ranks,
        str(event.get("group", "default")),
        event.get("sender_rank"),
        event.get("receiver_rank"),
        event.get("tag"),
        event.get("channel"),
    )


def _shape_event(event: Mapping[str, Any], event_id: str) -> JsonDict:
    record: JsonDict = {"id": event_id}
    for key in _EVENT_COPY_FIELDS:
        if key in event:
            record[key] = copy.deepcopy(event.get(key))
    record.setdefault("phase", str(event.get("phase", "unknown")))
    record.setdefault("group", str(event.get("group", "default")))
    record.setdefault("concurrent_groups", as_int(event.get("concurrent_groups", 1)))
    return record


def _source_gap_us(event: Mapping[str, Any]) -> float:
    if "gap_us" in event:
        return max(0.0, as_float(event.get("gap_us")))
    if "compute_before_us" in event:
        return max(0.0, as_float(event.get("compute_before_us")))
    return 0.0


def _representative_event(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(events) == 1:
        return events[0]
    features = [_features(event) for event in events]
    centers = [statistics_median(values) for values in zip(*features)]
    scales = [max(1.0, max(abs(row[index]) for row in features)) for index in range(len(centers))]

    def distance(item: Tuple[int, Mapping[str, Any]]) -> Tuple[float, int]:
        index, _event = item
        row = features[index]
        value = sum(abs(row[column] - centers[column]) / scales[column] for column in range(len(centers)))
        return value, index

    return min(enumerate(events), key=distance)[1]


def _cluster_representatives(
    events: Sequence[Mapping[str, Any]],
    cluster_count: int,
) -> List[Mapping[str, Any]]:
    if len(events) <= cluster_count:
        return list(events)
    features = [_features(event) for event in events]
    order = sorted(range(len(events)), key=lambda index: features[index])
    representatives: List[Mapping[str, Any]] = []
    for cluster_index in range(cluster_count):
        start = int(cluster_index * len(order) / cluster_count)
        end = int((cluster_index + 1) * len(order) / cluster_count)
        bucket_indices = order[start:end] or [order[min(start, len(order) - 1)]]
        bucket_events = [events[index] for index in bucket_indices]
        representatives.append(_representative_event(bucket_events))
    # De-duplicate identical medoids while preserving order; repeated medoids
    # add no behavioral information and make the baseline look stronger than
    # it is.
    unique: List[Mapping[str, Any]] = []
    seen = set()
    for representative in representatives:
        key = tuple(_features(representative))
        if key not in seen:
            seen.add(key)
            unique.append(representative)
    return unique or [events[0]]


def _nearest_representative(
    event: Mapping[str, Any],
    representatives: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], int]:
    if not representatives:
        raise SchemaError("clustering baseline has no representatives")
    all_features = [_features(event), *(_features(representative) for representative in representatives)]
    scales = _feature_scales(all_features)
    event_features = all_features[0]
    best_index = min(
        range(len(representatives)),
        key=lambda index: (_feature_distance(event_features, _features(representatives[index]), scales), index),
    )
    return representatives[best_index], best_index


def _feature_scales(features: Sequence[Tuple[float, ...]]) -> Tuple[float, ...]:
    return tuple(max(1.0, max(abs(row[index]) for row in features)) for index in range(len(features[0])))


def _feature_distance(left: Tuple[float, ...], right: Tuple[float, ...], scales: Tuple[float, ...]) -> float:
    return sum(abs(left[index] - right[index]) / scales[index] for index in range(len(scales)))


def _features(event: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    ranks = normalize_ranks(event.get("ranks"))
    offsets = normalize_arrival_offsets(event, ranks)
    return (
        _source_gap_us(event),
        arrival_skew_us(offsets),
        as_float(event.get("compute_before_us"), 0.0),
        as_float(event.get("compute_overlap_us"), 0.0),
        as_float(event.get("compute_pressure"), 0.5),
    )
