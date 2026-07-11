"""Bounded canary expansion and stored/logical work accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping

from ..errors import SchemaError
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    checked_multiply,
    require_within,
)
from .wire import JsonDict, as_int


@dataclass(frozen=True)
class CanaryExpansionCounts:
    """Stored and logical work counts proven before canary expansion."""

    stored_events: int
    stored_timing_records: int
    logical_events: int
    logical_timing_records: int


def preflight_canary_expansion(
    events: Any,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> CanaryExpansionCounts:
    """Count compact and logical canary work without expanding motifs or weights."""

    if not isinstance(events, list):
        raise SchemaError("canary events must be a list")
    stored_events = 0
    stored_timing_records = 0
    logical_events = 0
    logical_timing_records = 0
    try:
        for index, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise SchemaError(f"canary event {index} must be an object")
            stored_events = checked_add(stored_events, 1, label="stored canary events")
            if event.get("program") == "sequence_motif":
                label = f"canary event {index}"
                children = _sequence_motif_children(event, label)
                repeats = as_int(event.get("program_repeats"))
                if repeats <= 0:
                    raise SchemaError(f"{label} program_repeats must be positive")
                stored_events = checked_add(
                    stored_events,
                    len(children),
                    label="stored canary events",
                )
                logical_events = checked_add(
                    logical_events,
                    checked_multiply(
                        len(children),
                        repeats,
                        label="logical canary events",
                    ),
                    label="logical canary events",
                )
                motif_occurrences = 0
                for child_index, child in enumerate(children):
                    child_label = f"{label} child {child_index}"
                    stored_timing_records = checked_add(
                        stored_timing_records,
                        stored_event_timing_record_count(child),
                        label="stored timing records",
                    )
                    motif_occurrences = checked_add(
                        motif_occurrences,
                        _event_timing_occurrence_count(child, child_label),
                        label="logical timing records",
                    )
                logical_timing_records = checked_add(
                    logical_timing_records,
                    checked_multiply(
                        motif_occurrences,
                        repeats,
                        label="logical timing records",
                    ),
                    label="logical timing records",
                )
            else:
                logical_events = checked_add(
                    logical_events,
                    1,
                    label="logical canary events",
                )
                stored_timing_records = checked_add(
                    stored_timing_records,
                    stored_event_timing_record_count(event),
                    label="stored timing records",
                )
                logical_timing_records = checked_add(
                    logical_timing_records,
                    _event_timing_occurrence_count(event, f"canary event {index}"),
                    label="logical timing records",
                )
        require_within(
            stored_events,
            limits.max_stored_events,
            label="stored canary events",
        )
        require_within(
            stored_timing_records,
            limits.max_stored_timing_records,
            label="stored timing records",
        )
        require_within(
            logical_events,
            limits.max_expanded_events,
            label="logical canary events",
        )
        require_within(
            logical_timing_records,
            limits.max_expanded_timing_records,
            label="logical timing records",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    return CanaryExpansionCounts(
        stored_events=stored_events,
        stored_timing_records=stored_timing_records,
        logical_events=logical_events,
        logical_timing_records=logical_timing_records,
    )


def _event_timing_occurrence_count(event: Mapping[str, Any], label: str) -> int:
    samples = event.get("timing_samples")
    if not isinstance(samples, list) or not samples:
        repeat = as_int(event.get("repeat"), 1)
        if repeat <= 0:
            raise SchemaError(f"{label} repeat must be positive")
        return repeat
    total = 0
    try:
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise SchemaError(f"{label} timing sample {sample_index} must be an object")
            weight = as_int(sample.get("weight", 1))
            if weight <= 0:
                raise SchemaError(f"{label} timing sample {sample_index} weight must be positive")
            total = checked_add(total, weight, label="logical timing records")
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    return total


def iter_canary_logical_events(
    events: Any,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Iterable[JsonDict]:
    """Yield the ordered simulator inputs encoded by a canary event list."""

    preflight_canary_expansion(events, limits=limits)
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise SchemaError(f"canary event {index} must be an object")
        if event.get("program") == "sequence_motif":
            yield from expand_sequence_motif(event, f"canary event {index}")
        else:
            yield dict(event)


def iter_canary_stored_leaf_events(
    events: Any,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Iterable[Mapping[str, Any]]:
    """Yield leaf event templates stored in the artifact without expanding repeats."""

    preflight_canary_expansion(events, limits=limits)
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise SchemaError(f"canary event {index} must be an object")
        if event.get("program") == "sequence_motif":
            children = _sequence_motif_children(event, f"canary event {index}")
            for child in children:
                yield child
        else:
            yield event


def expand_sequence_motif(event: Mapping[str, Any], label: str) -> Iterable[JsonDict]:
    repeats = as_int(event.get("program_repeats"))
    if repeats <= 0:
        raise SchemaError(f"{label} program_repeats must be positive")
    children = _sequence_motif_children(event, label)
    for repeat_index in range(repeats):
        for child_index, child in enumerate(children):
            child_copy = dict(child)
            stride = as_int(child_copy.pop("execution_occurrence_stride", 1))
            if stride <= 0:
                raise SchemaError(f"{label} child {child_index} execution_occurrence_stride must be positive")
            base = as_int(child_copy.get("execution_occurrence_base"), 0)
            child_copy["execution_occurrence_base"] = base + repeat_index * stride
            yield child_copy


def _sequence_motif_children(event: Mapping[str, Any], label: str) -> List[Mapping[str, Any]]:
    motif_id = event.get("motif_id")
    if motif_id is not None and (not isinstance(motif_id, str) or not motif_id):
        raise SchemaError(f"{label} motif_id must be a non-empty string")
    if event.get("program") != "sequence_motif":
        raise SchemaError(f"{label} has unsupported program")
    children = event.get("events")
    if not isinstance(children, list) or len(children) < 2:
        raise SchemaError(f"{label} sequence_motif requires at least two child events")
    source = event.get("source")
    if not isinstance(source, Mapping):
        raise SchemaError(f"{label} sequence_motif requires a source object")
    repeats = as_int(event.get("program_repeats"))
    child_count = 0
    for child_index, child in enumerate(children):
        if not isinstance(child, Mapping):
            raise SchemaError(f"{label} child {child_index} must be an object")
        if child.get("program") is not None:
            raise SchemaError(f"{label} child {child_index} must not be a nested program")
        child_source = child.get("source")
        if isinstance(child_source, Mapping):
            child_count += as_int(child_source.get("count"), as_int(child.get("repeat"), 1))
        else:
            child_count += as_int(child.get("repeat"), 1)
    if as_int(source.get("count")) != child_count * repeats:
        raise SchemaError(f"{label} source.count must match expanded sequence length")
    return children


def stored_event_timing_record_count(event: Mapping[str, Any]) -> int:
    samples = event.get("timing_samples")
    if not isinstance(samples, list):
        return 0
    total = 0
    for sample in samples:
        if isinstance(sample, Mapping):
            total += 1
            pattern = sample.get("timing_pattern")
            if isinstance(pattern, list):
                for child in pattern:
                    if isinstance(child, Mapping):
                        total += 1
    return total


__all__ = [
    "CanaryExpansionCounts",
    "expand_sequence_motif",
    "iter_canary_logical_events",
    "iter_canary_stored_leaf_events",
    "preflight_canary_expansion",
    "stored_event_timing_record_count",
]
