"""Hardware-independent PARAM basic-trace materialization contract."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set, Tuple

from ..errors import SchemaError
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    checked_multiply,
    require_within,
)
from .canary import (
    iter_canary_logical_events,
    iter_canary_stored_leaf_events,
    iter_canary_timing_samples,
    validate_canary,
)
from .dtypes import dtype_size_bytes, require_param_dtype
from .wire import JsonDict, as_float, as_int, normalize_ranks, validate_reduction_metadata

PARAM_COLLECTIVE_OP_NAMES = {
    "all_reduce": "all_reduce",
    "all_gather": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "all_to_all": "all_to_all",
    "broadcast": "broadcast",
}
PARAM_POINT_TO_POINT_OPS = ("point_to_point", "send", "recv")


def param_materialization_requirements(
    canary: Mapping[str, Any],
    *,
    dtype: Optional[str] = None,
    require_event_dtype: bool = False,
    require_reduction_op: bool = False,
    require_source_bounded_overlap: bool = False,
    skip_unsupported: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Validate hardware-independent PARAM constraints without expanding output."""

    validate_canary(canary, limits=limits)
    if not isinstance(require_event_dtype, bool):
        raise SchemaError("require_event_dtype must be a boolean")
    if not isinstance(require_reduction_op, bool):
        raise SchemaError("require_reduction_op must be a boolean")
    if not isinstance(require_source_bounded_overlap, bool):
        raise SchemaError("require_source_bounded_overlap must be a boolean")
    dtype_override = require_param_dtype(dtype, label="PARAM export dtype") if dtype is not None else None
    process_groups: Dict[str, Tuple[int, ...]] = {}
    represented_ranks: Set[int] = set()
    communication_dtypes: Set[str] = set()
    communication_reduction_ops: Set[str] = set()
    exportable_leaf_count = 0
    for event in iter_canary_stored_leaf_events(canary.get("events", []), limits=limits):
        op = str(event.get("op"))
        if op not in PARAM_COLLECTIVE_OP_NAMES and op not in PARAM_POINT_TO_POINT_OPS:
            if skip_unsupported:
                continue
            raise SchemaError(
                f"op {op!r} has no PARAM comms-replay equivalent; a qualification request cannot skip source operations"
            )
        if op in PARAM_POINT_TO_POINT_OPS and ("sender_rank" not in event or "receiver_rank" not in event):
            if skip_unsupported:
                continue
            raise SchemaError(f"{op} events need sender_rank and receiver_rank for PARAM export")
        if op == "broadcast" and "root_rank" not in event:
            raise SchemaError(
                "broadcast events need source-bound root_rank for PARAM export; "
                "qualification will not guess the first process-group rank"
            )
        validate_reduction_metadata(event, f"PARAM event {exportable_leaf_count}")
        if op in {"all_reduce", "reduce_scatter"}:
            if "reduction_op" not in event:
                if require_reduction_op:
                    raise SchemaError(
                        f"{op} events need source-bound reduction_op for qualification; "
                        "qualification will not guess SUM"
                    )
            else:
                communication_reduction_ops.add(str(event["reduction_op"]))
        ranks = tuple(normalize_ranks(event.get("ranks")))
        group = str(event.get("group", "default"))
        if group in process_groups and process_groups[group] != ranks:
            raise SchemaError(
                f"communicator group {group!r} appears with two different rank "
                f"sets ({list(process_groups[group])} vs {list(ranks)}); PARAM "
                "process groups need a single membership per group"
            )
        process_groups[group] = ranks
        represented_ranks.update(ranks)
        event_dtype = param_event_dtype(
            event,
            dtype_override=dtype_override,
            require_event_dtype=require_event_dtype,
        )
        communication_dtypes.add(event_dtype)
        nelems = param_element_count(as_int(event.get("bytes")), event_dtype)
        param_message_sizes(op, nelems, len(ranks))
        exportable_leaf_count += 1
    if exportable_leaf_count == 0:
        raise SchemaError("canary produced no PARAM-exportable entries")
    dense_rank_domain = list(range(max(represented_ranks) + 1))
    if sorted(represented_ranks) != dense_rank_domain:
        raise SchemaError(
            "PARAM-derived process groups must cover a dense global rank domain "
            f"starting at zero; observed {sorted(represented_ranks)}"
        )
    entry_count = preflight_param_entry_count(
        canary,
        skip_unsupported=skip_unsupported,
        compute_fill=False,
        overlap_structure=False,
        limits=limits,
    )
    result: JsonDict = {
        "communication_dtypes": sorted(communication_dtypes),
        "communication_reduction_ops": sorted(communication_reduction_ops),
        "process_group_count": len(process_groups),
        "param_entry_count_without_compute": entry_count,
    }
    if require_source_bounded_overlap:
        result["source_bounded_overlap"] = source_bounded_overlap_requirements(
            canary,
            limits=limits,
        )
    return result


def source_bounded_overlap_requirements(
    canary: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Prove that scalar overlap can be lowered without inventing a dependency graph."""

    pending_overlap_us: Optional[float] = None
    collective_count = 0
    positive_overlap_count = 0
    for event in iter_canary_logical_events(canary.get("events", []), limits=limits):
        op = str(event.get("op"))
        for sample in iter_canary_timing_samples(event):
            gap_us = max(0.0, as_float(sample.get("gap_us"), 0.0))
            if pending_overlap_us is not None and pending_overlap_us > gap_us:
                raise SchemaError(
                    f"source collective overlap_us={pending_overlap_us!r} exceeds "
                    f"the following inter-communication start gap_us={gap_us!r}; "
                    "this trace can contain pipelined in-flight collectives and "
                    "requires an explicit dependency graph"
                )
            overlap_us = max(0.0, as_float(sample.get("compute_overlap_us"), 0.0))
            if op in PARAM_POINT_TO_POINT_OPS:
                if overlap_us > 0.0:
                    raise SchemaError(
                        "source-bounded overlap materialization does not support "
                        f"positive overlap on synchronous point-to-point op {op!r}"
                    )
                pending_overlap_us = None
                continue
            if op in PARAM_COLLECTIVE_OP_NAMES:
                collective_count = checked_add(
                    collective_count,
                    1,
                    label="source-bounded overlap collective occurrences",
                )
                if overlap_us > 0.0:
                    positive_overlap_count = checked_add(
                        positive_overlap_count,
                        1,
                        label="source-bounded positive overlap occurrences",
                    )
                pending_overlap_us = overlap_us
            else:
                pending_overlap_us = None
    return {
        "policy": "single-inflight-source-bounded",
        "collective_count": collective_count,
        "positive_overlap_count": positive_overlap_count,
        "tail_overlap_us": 0.0 if pending_overlap_us is None else round(pending_overlap_us, 9),
    }


def preflight_param_entry_count(
    canary: Mapping[str, Any],
    *,
    skip_unsupported: bool,
    compute_fill: bool,
    overlap_structure: bool,
    limits: ResourceLimits,
) -> int:
    """Reject oversized PARAM output using only the compact stored program."""

    events = canary.get("events", [])
    if not isinstance(events, list):
        raise SchemaError("canary events must be a list")
    total = 0
    used_groups = set()
    try:
        for index, stored_event in enumerate(events):
            if not isinstance(stored_event, Mapping):
                raise SchemaError(f"canary event {index} must be an object")
            if stored_event.get("program") == "sequence_motif":
                multiplier = as_int(stored_event.get("program_repeats"))
                children = stored_event.get("events")
                if not isinstance(children, list):
                    raise SchemaError(f"canary event {index} motif events must be a list")
                leaf_events = children
            else:
                multiplier = 1
                leaf_events = [stored_event]
            if multiplier <= 0:
                raise SchemaError(f"canary event {index} program_repeats must be positive")
            for child_index, child in enumerate(leaf_events):
                if not isinstance(child, Mapping):
                    raise SchemaError(f"canary event {index} child {child_index} must be an object")
                op = str(child.get("op"))
                if op in PARAM_POINT_TO_POINT_OPS:
                    communication_entries = 2
                elif op in PARAM_COLLECTIVE_OP_NAMES:
                    communication_entries = 1
                elif skip_unsupported:
                    continue
                else:
                    # The semantic preflight emits the more specific
                    # unsupported-op error before expansion.
                    continue
                occurrences = checked_multiply(
                    as_int(child.get("repeat"), 1),
                    multiplier,
                    label="PARAM logical occurrences",
                )
                per_occurrence = communication_entries
                if compute_fill:
                    # Source-bounded overlap can split one gap into overlap and
                    # serialized entries. Zero-count components are omitted.
                    per_occurrence += 2 if overlap_structure else 1
                if overlap_structure and op in PARAM_COLLECTIVE_OP_NAMES:
                    per_occurrence += 1
                total = checked_add(
                    total,
                    checked_multiply(
                        occurrences,
                        per_occurrence,
                        label="PARAM trace entries",
                    ),
                    label="PARAM trace entries",
                )
                used_groups.add(str(child.get("group", "default")))
        total = checked_add(
            total,
            len(used_groups),
            label="PARAM trace entries",
        )
        return require_within(
            total,
            limits.max_param_entries,
            label="PARAM trace entries",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc


def param_message_sizes(op: str, nelems: int, world_size: int) -> Tuple[int, int]:
    """Return PARAM per-operation input and output element counts."""

    if op in ("all_gather", "reduce_scatter", "all_to_all"):
        if world_size <= 0 or nelems % world_size:
            raise SchemaError(
                f"{op} event bytes do not divide evenly across {world_size} ranks "
                "for PARAM export; choose a dtype whose element size divides the "
                "per-rank shard"
            )
        if op == "all_to_all":
            return nelems, nelems
        shard = nelems // world_size
        if op == "all_gather":
            return shard, nelems
        return nelems, shard
    return nelems, nelems


def param_element_count(byte_count: int, dtype: str) -> int:
    """Convert bytes to elements without changing the represented volume."""

    element_bytes = dtype_size_bytes(dtype)
    if byte_count <= 0 or byte_count % element_bytes:
        raise SchemaError(
            f"PARAM event bytes={byte_count} must divide evenly by dtype {dtype!r} element width={element_bytes}"
        )
    return byte_count // element_bytes


def param_event_dtype(
    event: Mapping[str, Any],
    *,
    dtype_override: Optional[str],
    require_event_dtype: bool,
) -> str:
    """Resolve one event's PARAM dtype under an optional explicit override."""

    if dtype_override is not None:
        return dtype_override
    if "dtype" in event:
        return require_param_dtype(event.get("dtype"), label=f"PARAM dtype for op {event.get('op')!r}")
    if require_event_dtype:
        raise SchemaError(
            f"PARAM materialization requires source-bound dtype on op {event.get('op')!r}; "
            "the qualification workflow will not guess one"
        )
    return "float32"


__all__ = [
    "PARAM_COLLECTIVE_OP_NAMES",
    "PARAM_POINT_TO_POINT_OPS",
    "param_element_count",
    "param_event_dtype",
    "param_materialization_requirements",
    "param_message_sizes",
    "preflight_param_entry_count",
    "source_bounded_overlap_requirements",
]
