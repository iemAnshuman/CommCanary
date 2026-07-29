"""Capture-shard discovery, validation, and deterministic reconciliation."""

from __future__ import annotations

import copy
import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_float, as_int, load_json, normalize_ranks
from ..errors import SchemaError
from ..formats import TRACE_FORMAT
from ..operation_identity import OperationIdentity
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    require_within,
    validate_json_mapping,
)

TraceLoader = Callable[..., JsonDict]
CaptureBucket = Union[JsonDict, List[JsonDict]]
NamedTrace = Tuple[str, Mapping[str, Any]]


def merge_trace_shards(
    shard_dir: str,
    *,
    workload_name: str,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Merge trace shards using the production bounded JSON loader."""

    return merge_trace_shards_with_loader(
        shard_dir,
        workload_name=workload_name,
        limits=limits,
        load_trace=load_json,
    )


def merge_trace_shards_with_loader(
    shard_dir: str,
    *,
    workload_name: str,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    load_trace: TraceLoader,
) -> JsonDict:
    shard_paths = _trace_shard_paths(shard_dir, limits=limits)

    def loaded_traces() -> Iterable[NamedTrace]:
        for shard_path in shard_paths:
            yield Path(shard_path).name, load_trace(shard_path, limits=limits)

    return _merge_named_traces(
        loaded_traces(),
        source_count=len(shard_paths),
        workload_name=workload_name,
        limits=limits,
    )


def merge_trace_documents(
    named_traces: Sequence[NamedTrace],
    *,
    workload_name: str,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Merge already-loaded rank contributions under capture invariants.

    ``name`` values identify independent sources in diagnostics and persisted
    ``merged_shards`` metadata. They must be unique, non-empty basenames. Input
    mappings remain caller-owned; the returned trace is detached.
    """

    seen_names: Set[str] = set()
    detached_names: List[NamedTrace] = []
    for index, item in enumerate(named_traces):
        if not isinstance(item, tuple) or len(item) != 2:
            raise SchemaError(f"named trace {index} must be a (name, trace) tuple")
        name, trace = item
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise SchemaError(f"named trace {index} name must be a non-empty basename")
        if name in seen_names:
            raise SchemaError(f"named trace names must be unique; duplicate {name!r}")
        if not isinstance(trace, Mapping):
            raise SchemaError(f"named trace {name!r} must be an object")
        seen_names.add(name)
        detached_names.append((name, trace))
    try:
        require_within(
            len(detached_names),
            limits.max_capture_shards,
            label="capture shards",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    return _merge_named_traces(
        detached_names,
        source_count=len(detached_names),
        workload_name=workload_name,
        limits=limits,
    )


def _merge_named_traces(
    named_traces: Iterable[NamedTrace],
    *,
    source_count: int,
    workload_name: str,
    limits: ResourceLimits,
) -> JsonDict:
    # Most capture directories contain one already-complete event for each
    # identity. Store that common case directly; allocate a contribution list
    # only after an actual cross-shard collision is observed.
    buckets: Dict[Tuple[Any, ...], CaptureBucket] = {}
    systems: List[JsonDict] = []
    workload: JsonDict = {"name": workload_name}
    strict_sharded_merge = source_count > 1
    seen_occurrences: Set[Tuple[str, str, str]] = set()
    session_ids: Set[str] = set()
    canonical_workload: Optional[JsonDict] = None
    aggregate_event_count = 0

    for shard_name, trace in named_traces:
        try:
            validate_json_mapping(trace, limits=limits)
        except JsonResourceError as exc:
            raise SchemaError(f"capture shard violates JSON resource constraints: {exc}") from exc
        validate_trace(trace, allow_partial_arrivals=True, limits=limits)
        trace_workload = dict(trace.get("workload", {}))
        if canonical_workload is None:
            canonical_workload = trace_workload
        elif trace_workload != canonical_workload:
            raise SchemaError("trace shards contain conflicting workload metadata")
        workload.update(trace_workload)

        system = trace.get("system", {})
        if not isinstance(system, Mapping):
            raise SchemaError("trace shard system metadata must be an object")
        system_session = system.get("capture_session_id")
        if system_session:
            session_ids.add(str(system_session))
        clock_offset_us = _clock_offset_us(system)
        clock_alignment = "explicit_offset_us" if clock_offset_us is not None else "uncalibrated"
        systems.append({**system, "clock_alignment": clock_alignment, "clock_offset_us": clock_offset_us})

        shard_events = trace.get("events", [])
        try:
            aggregate_event_count = checked_add(
                aggregate_event_count,
                len(shard_events),
                label="capture aggregate events",
            )
            require_within(
                aggregate_event_count,
                limits.max_capture_events,
                label="capture aggregate events",
            )
        except JsonResourceError as exc:
            raise SchemaError(str(exc)) from exc

        for event in shard_events:
            session = event.get("capture_session_id")
            collective_id = event.get("collective_id")
            if session:
                session_ids.add(str(session))
            if system_session and session and str(system_session) != str(session):
                raise SchemaError("event capture_session_id conflicts with shard system metadata")
            if strict_sharded_merge:
                if not session:
                    raise SchemaError("sharded capture merge requires each event to include capture_session_id")
                if not collective_id:
                    raise SchemaError("sharded capture merge requires each event to include a stable collective_id")
                occurrence_key = (str(session), str(collective_id), shard_name)
                if occurrence_key in seen_occurrences:
                    raise SchemaError(
                        f"collective_id {collective_id!r} was reused in shard {shard_name}; "
                        "use a globally unique occurrence id"
                    )
                seen_occurrences.add(occurrence_key)

            copied = dict(event)
            copied["shard"] = shard_name
            system_rank = system.get("rank")
            copied["_shard_rank"] = (
                system_rank if system_rank not in (None, "", "unknown") else event.get("recorder_rank", "unknown")
            )
            copied["_clock_calibrated"] = clock_offset_us is not None
            copied["_strict_sharded_merge"] = strict_sharded_merge
            copied["_aligned_start_us"] = as_float(event.get("start_us"), 0.0) + (clock_offset_us or 0.0)
            bucket_key = OperationIdentity.from_mapping(copied).capture_coalescing_key()
            existing = buckets.get(bucket_key)
            if existing is None:
                buckets[bucket_key] = copied
            elif isinstance(existing, list):
                existing.append(copied)
            else:
                buckets[bucket_key] = [existing, copied]

    if len(session_ids) > 1:
        raise SchemaError("trace shard directory must contain exactly one capture session")
    if strict_sharded_merge and len(session_ids) != 1:
        raise SchemaError("trace shard directory must contain exactly one capture session")

    events: List[JsonDict] = []
    for group in buckets.values():
        if isinstance(group, list):
            events.append(_coalesce_events(group))
        else:
            events.append(_coalesce_single_event(group))
    # Cross-rank groups produce new coalesced mappings. Release their rank
    # contributions and all identity keys before ordering/final validation.
    buckets.clear()
    _reject_unordered_uncalibrated_domains(events, systems)
    events.sort(key=lambda event: as_float(event.get("start_us"), 0.0))
    merged: JsonDict = {
        "format": TRACE_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": workload,
        "system": {
            "capture_mode": "sharded",
            "shards": source_count,
            "capture_session_id": next(iter(session_ids)) if len(session_ids) == 1 else None,
            "shard_systems": systems,
        },
        "events": events,
    }
    validate_trace(merged, limits=limits)
    return merged


def _trace_shard_paths(shard_dir: str, *, limits: ResourceLimits) -> List[str]:
    patterns = ("*.trace.json", "*.trace.*.json", "*.rank-*-pid-*.json")
    paths: Set[str] = set()
    for pattern in patterns:
        for path in glob.iglob(os.path.join(shard_dir, pattern)):
            paths.add(path)
            try:
                require_within(
                    len(paths),
                    limits.max_capture_shards,
                    label="capture shards",
                )
            except JsonResourceError as exc:
                raise SchemaError(str(exc)) from exc
    return sorted(paths)


def _reject_unordered_uncalibrated_domains(events: List[JsonDict], systems: List[JsonDict]) -> None:
    if len(events) < 2 or len(systems) < 2:
        return
    if all(system.get("clock_alignment") == "explicit_offset_us" for system in systems):
        return
    representative: Optional[frozenset[int]] = None
    for event in events:
        rank_domain = frozenset(normalize_ranks(event.get("ranks")))
        if not rank_domain:
            continue
        if representative is None:
            representative = rank_domain
        elif rank_domain != representative:
            raise SchemaError("cannot globally order different collective rank domains from uncalibrated rank clocks")


def _clock_offset_us(system: Mapping[str, Any]) -> Optional[float]:
    if "clock_offset_us" in system:
        return as_float(system.get("clock_offset_us"))
    calibration = system.get("clock_calibration")
    if isinstance(calibration, Mapping) and "offset_us" in calibration:
        return as_float(calibration.get("offset_us"))
    return None


def _coalesce_events(events: List[JsonDict]) -> JsonDict:
    if len(events) == 1:
        return _coalesce_single_event(events[0])

    identity_event = events[0]
    ranks = normalize_ranks(identity_event.get("ranks"))
    expected_ranks = {str(rank) for rank in ranks}
    merged_op, point_to_point = _validate_collective_identity(events, ranks)
    recorder_ranks = _recorder_ranks(events, expected_ranks)
    calibrated_values = {bool(event.get("_clock_calibrated")) for event in events}
    if len(calibrated_values) > 1:
        raise SchemaError("cannot merge mixed calibrated and uncalibrated rank clocks")
    calibrated = calibrated_values == {True}
    starts = [as_float(event.get("_aligned_start_us"), 0.0) for event in events]
    if calibrated:
        timeline_event = min(events, key=lambda event: as_float(event.get("_aligned_start_us"), 0.0))
        coalesced_start_us = min(starts)
    else:
        canonical_rank = min(as_int(rank) for rank in recorder_ranks)
        timeline_event = _event_for_recorder_rank(events, canonical_rank)
        coalesced_start_us = as_float(timeline_event.get("_aligned_start_us"), 0.0)

    coalesced: JsonDict = {
        key: value
        for key, value in timeline_event.items()
        if not key.startswith("_") and key not in {"shard", "rank_arrival_us", "recorder_rank"}
    }
    coalesced["id"] = f"collective-{timeline_event.get('collective_id', timeline_event.get('collective_seq', 0))}"
    coalesced["op"] = merged_op
    coalesced["start_us"] = round(coalesced_start_us, 9)
    if point_to_point is not None:
        coalesced["sender_rank"] = point_to_point["sender_rank"]
        coalesced["receiver_rank"] = point_to_point["receiver_rank"]
        coalesced["tag"] = str(point_to_point.get("tag", "default"))
        coalesced["channel"] = str(point_to_point.get("channel", coalesced.get("group", "default")))
        coalesced["message_sequence"] = as_int(
            point_to_point.get("message_sequence", coalesced.get("collective_seq", 0))
        )
        coalesced["send_observation"] = point_to_point.get("send_observation", {})
        coalesced["recv_observation"] = point_to_point.get("recv_observation", {})

    raw_map_events = [event for event in events if isinstance(event.get("rank_arrival_us"), Mapping)]
    if raw_map_events:
        if len(raw_map_events) != len(events):
            raise SchemaError("cannot merge mixed rank_arrival_us and scalar arrival records")
        full_map = _identical_full_arrival_map(events, expected_ranks)
        if full_map is not None:
            earliest = min(full_map.values())
            coalesced["rank_arrival_us"] = {
                rank: round(value - earliest, 9)
                for rank, value in sorted(full_map.items(), key=lambda item: int(item[0]))
            }
        elif calibrated:
            absolute_arrivals: Dict[str, float] = {}
            for event in events:
                start_us = as_float(event.get("_aligned_start_us"), 0.0)
                for raw_rank, raw_offset in _own_rank_arrival_map(event).items():
                    rank_key = str(as_int(raw_rank))
                    if rank_key in absolute_arrivals:
                        raise SchemaError(f"duplicate arrival record for rank {rank_key}")
                    absolute_arrivals[rank_key] = start_us + as_float(raw_offset)
            if set(absolute_arrivals) != expected_ranks:
                missing = ", ".join(sorted(expected_ranks - set(absolute_arrivals), key=int))
                raise SchemaError(f"missing arrival records for ranks: {missing}")
            earliest = min(absolute_arrivals.values())
            coalesced["rank_arrival_us"] = {
                rank: round(value - earliest, 9)
                for rank, value in sorted(absolute_arrivals.items(), key=lambda item: int(item[0]))
            }
        else:
            local_skews = []
            for event in events:
                values = [as_float(value) for value in _own_rank_arrival_map(event).values()]
                if values:
                    local_skews.append(max(values) - min(values))
            coalesced["arrival_skew_us"] = round(max(local_skews, default=0.0), 9)
            coalesced["arrival_skew_unknown"] = True
    else:
        scalar_skew = max(as_float(event.get("arrival_skew_us"), 0.0) for event in events)
        if calibrated:
            coalesced["arrival_skew_us"] = round(max(scalar_skew, max(starts) - min(starts)), 9)
        else:
            coalesced["arrival_skew_us"] = round(scalar_skew, 9)
            coalesced["arrival_skew_unknown"] = True

    coalesced["merged_shards"] = sorted(str(event.get("shard")) for event in events)
    coalesced["recorder_ranks"] = sorted(recorder_ranks, key=int)
    _aggregate_reduction_op(coalesced, events)
    _aggregate_compute_fields(coalesced, events)
    _aggregate_compute_recipes(coalesced, events)
    _aggregate_observed_field(coalesced, events)
    return coalesced


def _coalesce_single_event(event: JsonDict) -> JsonDict:
    """Finalize the merge-owned event copy without allocating a second dict."""

    if event.get("_strict_sharded_merge") and len(normalize_ranks(event.get("ranks"))) > 1:
        raise SchemaError("missing rank contributions for sharded collective")
    event["start_us"] = round(as_float(event.get("_aligned_start_us"), event.get("start_us", 0.0)), 9)
    for key in ("_shard_rank", "_aligned_start_us", "_clock_calibrated", "_strict_sharded_merge"):
        event.pop(key, None)
    return event


def _validate_collective_identity(events: List[JsonDict], ranks: List[int]) -> Tuple[str, Optional[JsonDict]]:
    first = events[0]
    session = first.get("capture_session_id")
    collective_id = first.get("collective_id")
    if not session:
        raise SchemaError("merged collective is missing capture_session_id")
    if not collective_id:
        raise SchemaError("merged collective is missing collective_id")
    expected = {
        "capture_session_id": session,
        "collective_id": collective_id,
        "phase": first.get("phase", "unknown"),
        "dtype": first.get("dtype"),
        "bytes": as_int(first.get("bytes")),
        "group": first.get("group", "default"),
        "ranks": tuple(ranks),
        "concurrent_groups": as_int(first.get("concurrent_groups"), 1),
        "kineto_message_shape": _kineto_message_shape_identity(first),
    }
    for optional_key in ("root_rank", "tag", "channel", "message_sequence"):
        if optional_key in first:
            expected[optional_key] = first.get(optional_key)
    ops = {str(first.get("op"))}
    for event in events[1:]:
        actual = {
            "capture_session_id": event.get("capture_session_id"),
            "collective_id": event.get("collective_id"),
            "phase": event.get("phase", "unknown"),
            "dtype": event.get("dtype"),
            "bytes": as_int(event.get("bytes")),
            "group": event.get("group", "default"),
            "ranks": tuple(normalize_ranks(event.get("ranks"))),
            "concurrent_groups": as_int(event.get("concurrent_groups"), 1),
            "kineto_message_shape": _kineto_message_shape_identity(event),
        }
        for optional_key in ("root_rank", "tag", "channel", "message_sequence"):
            if optional_key in expected or optional_key in event:
                actual[optional_key] = event.get(optional_key)
        if actual != expected:
            raise SchemaError("conflicting records share the same collective identity")
        ops.add(str(event.get("op")))
    if len(ops) == 1:
        return next(iter(ops)), None
    if ops == {"send", "recv"}:
        senders = [
            as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown")))
            for event in events
            if event.get("op") == "send"
        ]
        receivers = [
            as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown")))
            for event in events
            if event.get("op") == "recv"
        ]
        if len(senders) == 1 and len(receivers) == 1 and senders[0] != receivers[0]:
            send_event = next(event for event in events if event.get("op") == "send")
            recv_event = next(event for event in events if event.get("op") == "recv")
            return "point_to_point", {
                "sender_rank": senders[0],
                "receiver_rank": receivers[0],
                "tag": first.get("tag", "default"),
                "channel": first.get("channel", first.get("group", "default")),
                "message_sequence": first.get("message_sequence", first.get("collective_seq", 0)),
                "send_observation": _p2p_observation(send_event),
                "recv_observation": _p2p_observation(recv_event),
            }
    raise SchemaError("conflicting records share the same collective identity")


def _kineto_message_shape_identity(event: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping) or "kineto_message_shape_status" not in metadata:
        return None
    input_splits = metadata.get("kineto_in_split_sizes")
    output_splits = metadata.get("kineto_out_split_sizes")
    return (
        metadata.get("kineto_message_shape_status"),
        metadata.get("kineto_in_msg_nelems"),
        metadata.get("kineto_out_msg_nelems"),
        tuple(input_splits) if isinstance(input_splits, list) else None,
        tuple(output_splits) if isinstance(output_splits, list) else None,
    )


def _aggregate_reduction_op(coalesced: JsonDict, events: List[JsonDict]) -> None:
    """Preserve a reduction operator only when every rank derived the same one."""

    observed = [str(event["reduction_op"]) for event in events if "reduction_op" in event]
    if len(set(observed)) > 1:
        raise SchemaError("conflicting reduction operators share the same collective identity")
    if observed and len(observed) == len(events):
        coalesced["reduction_op"] = observed[0]
        return
    coalesced.pop("reduction_op", None)
    if not observed:
        return
    raw_metadata = coalesced.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    for key in (
        "kineto_reduction_kernel_count",
        "kineto_reduction_method",
        "kineto_reduction_op",
    ):
        metadata.pop(key, None)
    metadata["kineto_reduction_status"] = "unavailable_incomplete_rank_evidence"
    coalesced["metadata"] = metadata


def _p2p_observation(event: Mapping[str, Any]) -> JsonDict:
    observed: JsonDict = {
        "rank": as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown"))),
        "start_us": round(as_float(event.get("_aligned_start_us", event.get("start_us", 0.0))), 9),
    }
    if "rank_arrival_us" in event:
        observed["rank_arrival_us"] = copy.deepcopy(event.get("rank_arrival_us"))
    if "observed_exposed_us" in event:
        observed["observed_exposed_us"] = round(as_float(event.get("observed_exposed_us")), 9)
    return observed


def _recorder_ranks(events: List[JsonDict], expected_ranks: Set[str]) -> Set[str]:
    recorder_ranks: Set[str] = set()
    for event in events:
        raw_rank = event.get("_shard_rank", event.get("recorder_rank", "unknown"))
        rank_key = str(as_int(raw_rank))
        if rank_key not in expected_ranks:
            raise SchemaError(f"recorder rank {rank_key} is not part of the collective ranks")
        if rank_key in recorder_ranks:
            raise SchemaError(f"duplicate contribution from recorder rank {rank_key}")
        recorder_ranks.add(rank_key)
    if recorder_ranks != expected_ranks:
        missing = ", ".join(sorted(expected_ranks - recorder_ranks, key=int))
        raise SchemaError(f"missing recorder contributions for ranks: {missing}")
    return recorder_ranks


def _event_for_recorder_rank(events: List[JsonDict], rank: int) -> JsonDict:
    rank_key = str(rank)
    for event in events:
        raw_rank = event.get("_shard_rank", event.get("recorder_rank", "unknown"))
        if str(as_int(raw_rank)) == rank_key:
            return event
    raise SchemaError(f"missing recorder contribution for rank {rank_key}")


def _own_rank_arrival_map(event: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_arrivals = event.get("rank_arrival_us", {})
    if not isinstance(raw_arrivals, Mapping):
        raise SchemaError("rank_arrival_us must be an object")
    rank_key = str(as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown"))))
    if {str(key) for key in raw_arrivals} != {rank_key}:
        raise SchemaError("partial rank_arrival_us must contain exactly the recorder's own rank")
    return raw_arrivals


def _identical_full_arrival_map(events: List[JsonDict], expected_ranks: Set[str]) -> Optional[Dict[str, float]]:
    full_maps: List[Dict[str, float]] = []
    for event in events:
        raw_arrivals = event.get("rank_arrival_us", {})
        if not isinstance(raw_arrivals, Mapping):
            return None
        if {str(key) for key in raw_arrivals} != expected_ranks:
            return None
        full_maps.append({str(key): as_float(value) for key, value in raw_arrivals.items()})
    first = full_maps[0]
    for full_map in full_maps[1:]:
        if full_map != first:
            raise SchemaError("full rank_arrival_us maps must be identical across shards")
    return first


def _aggregate_compute_fields(coalesced: JsonDict, events: List[JsonDict]) -> None:
    by_rank: JsonDict = {}
    known_overlap_values: List[float] = []
    overlap_known_for_every_rank = True
    for event in events:
        rank_key = str(as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown"))))
        rank_fields: JsonDict = {
            "compute_before_us": round(as_float(event.get("compute_before_us"), 0.0), 9),
            "compute_pressure": round(as_float(event.get("compute_pressure"), 0.5), 6),
        }
        overlap_known = "compute_overlap_us" in event and event.get("compute_overlap_unknown") is not True
        if overlap_known:
            overlap_us = round(as_float(event.get("compute_overlap_us")), 9)
            rank_fields["compute_overlap_us"] = overlap_us
            known_overlap_values.append(overlap_us)
        else:
            rank_fields["compute_overlap_unknown"] = True
            overlap_known_for_every_rank = False
        by_rank[rank_key] = rank_fields
    ordered = {rank: by_rank[rank] for rank in sorted(by_rank, key=int)}
    coalesced["compute_by_rank"] = ordered
    before_values = [values["compute_before_us"] for values in ordered.values()]
    pressure_values = [values["compute_pressure"] for values in ordered.values()]
    coalesced["compute_before_us"] = round(max(before_values), 9) if before_values else 0.0
    coalesced.pop("compute_overlap_us", None)
    coalesced.pop("compute_overlap_unknown", None)
    if overlap_known_for_every_rank:
        coalesced["compute_overlap_us"] = round(min(known_overlap_values), 9)
    else:
        coalesced["compute_overlap_unknown"] = True
    coalesced["compute_pressure"] = round(max(pressure_values), 6) if pressure_values else 0.5
    if (
        len(set(before_values)) > 1
        or not overlap_known_for_every_rank
        or len(set(known_overlap_values)) > 1
        or len(set(pressure_values)) > 1
    ):
        coalesced["compute_fields_uncertain"] = True


def _aggregate_compute_recipes(coalesced: JsonDict, events: List[JsonDict]) -> None:
    """Retain each rank's causal compute work without inventing a scalar proxy."""

    by_rank: JsonDict = {}
    present = []
    for event in events:
        rank_key = str(as_int(event.get("_shard_rank", event.get("recorder_rank", "unknown"))))
        recipe = event.get("compute_recipe")
        present.append(isinstance(recipe, list))
        if isinstance(recipe, list):
            by_rank[rank_key] = copy.deepcopy(recipe)
    coalesced.pop("compute_recipe", None)
    coalesced.pop("compute_recipe_by_rank", None)
    if any(present) and not all(present):
        raw_metadata = coalesced.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        for key in (
            "kineto_compute_recipe_kernel_count",
            "kineto_compute_recipe_method",
            "kineto_compute_recipe_operation_count",
            "kineto_compute_recipe_duration_us",
        ):
            metadata.pop(key, None)
        metadata["kineto_compute_recipe_status"] = "unavailable_incomplete_rank_evidence"
        coalesced["metadata"] = metadata
        return
    if present and all(present):
        coalesced["compute_recipe_by_rank"] = {rank: by_rank[rank] for rank in sorted(by_rank, key=int)}


def _aggregate_observed_field(coalesced: JsonDict, events: List[JsonDict]) -> None:
    flags = ["observed_exposed_us" in event for event in events]
    if any(flags) and not all(flags):
        raise SchemaError("observed_exposed_us must be present on every rank contribution or none")
    if flags and all(flags):
        coalesced["observed_exposed_us"] = round(max(as_float(event.get("observed_exposed_us")) for event in events), 9)


__all__ = [
    "NamedTrace",
    "TraceLoader",
    "merge_trace_documents",
    "merge_trace_shards",
    "merge_trace_shards_with_loader",
]
