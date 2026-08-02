"""Exact-work program materialization for hardware qualification.

This adapter is intentionally narrower than a general execution-trace format.
It accepts only collectives whose source trace carries a complete, per-rank
recipe of contiguous GEMMs observed between asynchronous collective issue and
its explicit wait.  Target execution reproduces that work; it never converts
source elapsed time into synthetic compute.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Set, Tuple

from ..errors import SchemaError
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    checked_multiply,
    require_within,
)
from .dtypes import dtype_size_bytes, require_param_compute_dtype, require_param_dtype
from .json_codec import canonical_json_bytes
from .param import PARAM_COLLECTIVE_OP_NAMES, param_element_count, param_message_sizes
from .qualification import (
    QUALIFICATION_COMPUTE_RECIPE_METHOD,
    QUALIFICATION_COMPUTE_RECIPE_PROJECTION,
)
from .trace import validate_trace
from .wire import SUPPORTED_REDUCTION_OPS, JsonDict, as_float, as_int, normalize_ranks


def qualification_compute_recipe_audit(
    trace: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Return a canonical audit of source-bound rank-local compute work."""

    _require_qualification_trace(trace, limits=limits)
    events = _trace_events(trace)
    rank_operation_counts: Dict[int, int] = {}
    operation_count = 0
    source_kernel_count = 0
    source_kernel_duration_us = 0.0
    matmul_flop_count = 0
    rank_recipe_count = 0
    projection_events: List[JsonDict] = []

    for event_index, event in enumerate(events):
        ranks = tuple(normalize_ranks(event.get("ranks")))
        recipes = _rank_recipes(event, event_index=event_index)
        projection_by_rank: JsonDict = {}
        for rank in ranks:
            operations = recipes[str(rank)]
            rank_recipe_count = _bounded_add(
                rank_recipe_count,
                1,
                limit=limits.max_param_entries,
                label="qualification rank recipe count",
            )
            projected_operations: List[JsonDict] = []
            for operation_index, operation in enumerate(operations):
                projected = _project_operation(
                    operation,
                    label=(f"qualification source event {event_index} rank {rank} operation {operation_index}"),
                    limits=limits,
                )
                projected_operations.append(projected)
                operation_count = _bounded_add(
                    operation_count,
                    1,
                    limit=limits.max_param_compute_operations,
                    label="qualification compute operation count",
                )
                rank_operation_counts[rank] = _bounded_add(
                    rank_operation_counts.get(rank, 0),
                    1,
                    limit=limits.max_param_compute_operations,
                    label=f"qualification rank {rank} compute operation count",
                )
                source_kernel_count = _bounded_add(
                    source_kernel_count,
                    as_int(operation["source_kernel_count"]),
                    limit=limits.max_execution_compute_operations,
                    label="qualification source kernel count",
                )
                source_kernel_duration_us += as_float(operation["source_kernel_duration_us"])
                operation_flops = _checked_multiply(
                    2,
                    _checked_multiply(
                        as_int(projected["m"]),
                        _checked_multiply(
                            as_int(projected["n"]),
                            as_int(projected["k"]),
                            label="qualification GEMM n*k work",
                        ),
                        label="qualification GEMM m*n*k work",
                    ),
                    label="qualification GEMM FLOPs",
                )
                matmul_flop_count = _bounded_add(
                    matmul_flop_count,
                    operation_flops,
                    limit=(1 << 63) - 1,
                    label="qualification total GEMM FLOPs",
                )
            projection_by_rank[str(rank)] = projected_operations
        projection_events.append(
            {
                "source_event_index": event_index,
                "op": str(event["op"]),
                "group": str(event.get("group", "default")),
                "ranks": list(ranks),
                "recipe_by_rank": projection_by_rank,
            }
        )

    represented_ranks = sorted({rank for event in events for rank in normalize_ranks(event.get("ranks"))})
    if represented_ranks != list(range(max(represented_ranks) + 1)):
        raise SchemaError(
            "qualification compute recipes must cover a dense global rank domain "
            f"starting at zero; observed {represented_ranks}"
        )
    canonical_rank_counts = {str(rank): rank_operation_counts.get(rank, 0) for rank in represented_ranks}
    projection: JsonDict = {
        "format": QUALIFICATION_COMPUTE_RECIPE_PROJECTION,
        "method": QUALIFICATION_COMPUTE_RECIPE_METHOD,
        "events": projection_events,
    }
    return {
        "provenance": "source_trace_exact_rank_local_work",
        "method": QUALIFICATION_COMPUTE_RECIPE_METHOD,
        "projection_sha256": hashlib.sha256(canonical_json_bytes(projection)).hexdigest(),
        "event_count": len(events),
        "rank_recipe_count": rank_recipe_count,
        "operation_count": operation_count,
        "rank_operation_counts": canonical_rank_counts,
        "source_kernel_count": source_kernel_count,
        "source_kernel_duration_us": round(source_kernel_duration_us, 9),
        "matmul_flop_count": matmul_flop_count,
    }


def trace_to_qualification_program(
    trace: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Tuple[List[JsonDict], JsonDict]:
    """Materialize ``issue -> exact rank-local work -> wait`` for every event."""

    audit = qualification_compute_recipe_audit(trace, limits=limits)
    events = _trace_events(trace)
    process_groups: Dict[str, Tuple[int, ...]] = {}
    used_groups: Set[str] = set()
    for event_index, event in enumerate(events):
        group = str(event.get("group", "default"))
        ranks = tuple(normalize_ranks(event.get("ranks")))
        if group in process_groups and process_groups[group] != ranks:
            raise SchemaError(f"qualification source group {group!r} changes rank membership at event {event_index}")
        process_groups[group] = ranks
        used_groups.add(group)
    group_ids = {group: index for index, group in enumerate(sorted(used_groups))}
    entries: List[JsonDict] = [
        {
            "comms": "init",
            "pg_id": group_ids[group],
            "global_ranks": list(process_groups[group]),
            "world_size": len(process_groups[group]),
            "markers": [f"commcanary:qualification:pg-init:{group}"],
        }
        for group in sorted(used_groups)
    ]
    request_id = 0
    for event_index, event in enumerate(events):
        operation = str(event.get("op"))
        ranks = tuple(normalize_ranks(event.get("ranks")))
        group = str(event.get("group", "default"))
        dtype = require_param_dtype(
            event.get("dtype"),
            label=f"qualification source event {event_index} communication dtype",
        )
        in_elements, out_elements = _message_sizes(
            event,
            event_index=event_index,
            dtype=dtype,
            rank_count=len(ranks),
        )
        issue: JsonDict = {
            "comms": PARAM_COLLECTIVE_OP_NAMES[operation],
            "req": request_id,
            "source_event_index": event_index,
            "world_size": len(ranks),
            "global_ranks": list(ranks),
            "pg_id": group_ids[group],
            "in_msg_size": in_elements,
            "out_msg_size": out_elements,
            "dtype": dtype,
            "markers": [f"commcanary:qualification:issue:{group}:{operation}"],
        }
        if operation == "broadcast":
            issue["root"] = as_int(event["root_rank"])
        if operation in {"all_reduce", "reduce_scatter"}:
            issue["reduction_op"] = str(event["reduction_op"])
        recipes = _rank_recipes(event, event_index=event_index)
        projected_recipes: JsonDict = {
            str(rank): [
                _project_operation(
                    recipe,
                    label=(f"qualification source event {event_index} rank {rank} operation {recipe_index}"),
                    limits=limits,
                )
                for recipe_index, recipe in enumerate(recipes[str(rank)])
            ]
            for rank in ranks
        }
        entries.extend(
            [
                issue,
                {
                    "compute": "gemm_recipe",
                    "compute_phase": "source-bound-overlap",
                    "overlap_request": request_id,
                    "source_event_index": event_index,
                    "global_ranks": list(ranks),
                    "recipe_by_rank": projected_recipes,
                    "markers": [f"commcanary:qualification:source-work:{group}:{operation}"],
                },
                {
                    "comms": "wait",
                    "req": request_id,
                    "source_event_index": event_index,
                    "markers": [f"commcanary:qualification:complete:{group}:{operation}"],
                },
            ]
        )
        request_id += 1
    try:
        require_within(
            len(entries),
            limits.max_param_entries,
            label="qualification program entries",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    return entries, audit


def qualification_program_communication_inventory(
    entries: List[JsonDict],
) -> JsonDict:
    """Describe the communication semantics present in a generated program."""

    operations: Set[str] = set()
    dtypes: Set[str] = set()
    reduction_ops: Set[str] = set()
    message_shapes: Set[Tuple[str, str, int, int, int]] = set()
    supported_operations = set(PARAM_COLLECTIVE_OP_NAMES.values())
    for index, entry in enumerate(entries):
        operation = entry.get("comms")
        if operation in {None, "init", "wait"}:
            continue
        if not isinstance(operation, str) or operation not in supported_operations:
            raise SchemaError(f"qualification program entry {index} has an unsupported communication operation")
        dtype = require_param_dtype(
            entry.get("dtype"),
            label=f"qualification program entry {index} communication dtype",
        )
        world_size = as_int(entry.get("world_size"))
        in_msg_size = as_int(entry.get("in_msg_size"))
        out_msg_size = as_int(entry.get("out_msg_size"))
        if world_size <= 0 or in_msg_size <= 0 or out_msg_size <= 0:
            raise SchemaError(f"qualification program entry {index} has a non-positive message shape")
        reduction_op = entry.get("reduction_op")
        if operation in {"all_reduce", "reduce_scatter"}:
            if not isinstance(reduction_op, str) or reduction_op not in SUPPORTED_REDUCTION_OPS:
                raise SchemaError(f"qualification program entry {index} lacks a supported reduction operator")
            reduction_ops.add(reduction_op)
        elif reduction_op is not None:
            raise SchemaError(f"qualification program entry {index} attaches reduction semantics to {operation}")
        operations.add(operation)
        dtypes.add(dtype)
        message_shapes.add((operation, dtype, world_size, in_msg_size, out_msg_size))
    if not operations:
        raise SchemaError("qualification program contains no communication operations")
    return {
        "communication_operations": sorted(operations),
        "communication_dtypes": sorted(dtypes),
        "communication_reduction_ops": sorted(reduction_ops),
        "communication_message_shapes": [
            {
                "operation": operation,
                "dtype": dtype,
                "world_size": world_size,
                "in_msg_size": in_msg_size,
                "out_msg_size": out_msg_size,
            }
            for operation, dtype, world_size, in_msg_size, out_msg_size in sorted(message_shapes)
        ],
    }


def _require_qualification_trace(
    trace: Mapping[str, Any],
    *,
    limits: ResourceLimits,
) -> None:
    validate_trace(trace, require_known_overlap=True, limits=limits)
    workload = trace.get("workload")
    system = trace.get("system")
    if not isinstance(workload, Mapping) or workload.get("import_source") != "pytorch-kineto":
        raise SchemaError("qualification exact-work execution currently requires a PyTorch Kineto source trace")
    if not isinstance(system, Mapping) or system.get("source_format") != "pytorch-kineto":
        raise SchemaError("qualification exact-work execution requires source_format='pytorch-kineto'")
    events = _trace_events(trace)
    for event_index, event in enumerate(events):
        operation = str(event.get("op"))
        if operation not in PARAM_COLLECTIVE_OP_NAMES:
            raise SchemaError(
                f"qualification source event {event_index} operation {operation!r} "
                "is outside the exact-work collective contract"
            )
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            raise SchemaError(f"qualification source event {event_index} lacks compute-recipe provenance")
        if metadata.get("kineto_compute_recipe_status") != "derived":
            raise SchemaError(f"qualification source event {event_index} compute recipe is not derived")
        if metadata.get("kineto_compute_recipe_method") != QUALIFICATION_COMPUTE_RECIPE_METHOD:
            raise SchemaError(f"qualification source event {event_index} compute-recipe method is unsupported")
        _rank_recipes(event, event_index=event_index)


def _trace_events(trace: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw_events = trace.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise SchemaError("qualification source trace must contain events")
    if not all(isinstance(event, Mapping) for event in raw_events):
        raise SchemaError("qualification source trace events must be objects")
    return list(raw_events)


def _rank_recipes(event: Mapping[str, Any], *, event_index: int) -> Mapping[str, Any]:
    ranks = tuple(normalize_ranks(event.get("ranks")))
    recipes = event.get("compute_recipe_by_rank")
    if not isinstance(recipes, Mapping):
        raise SchemaError(f"qualification source event {event_index} requires explicit compute_recipe_by_rank")
    expected = {str(rank) for rank in ranks}
    if set(recipes) != expected:
        raise SchemaError(
            f"qualification source event {event_index} compute_recipe_by_rank must exactly match participating ranks"
        )
    if any(not isinstance(recipes[str(rank)], list) for rank in ranks):
        raise SchemaError(f"qualification source event {event_index} rank recipes must be arrays")
    return recipes


def _project_operation(
    operation: Mapping[str, Any],
    *,
    label: str,
    limits: ResourceLimits,
) -> JsonDict:
    if operation.get("op") != "gemm":
        raise SchemaError(f"{label} must be a GEMM")
    dtype = require_param_compute_dtype(
        operation.get("dtype"),
        label=f"{label} dtype",
    )
    projected: JsonDict = {"op": "gemm", "dtype": dtype}
    for dimension in ("m", "n", "k"):
        value = as_int(operation.get(dimension))
        if value <= 0 or value > limits.max_param_gemm_dim:
            raise SchemaError(f"{label} {dimension} must be in [1, {limits.max_param_gemm_dim}]")
        projected[dimension] = value
    return projected


def _message_sizes(
    event: Mapping[str, Any],
    *,
    event_index: int,
    dtype: str,
    rank_count: int,
) -> Tuple[int, int]:
    metadata = event.get("metadata")
    if (
        isinstance(metadata, Mapping)
        and metadata.get("kineto_message_shape_status") == "derived"
        and metadata.get("kineto_message_shape_method") == "record-param-comms-in-out-nelems.v1"
    ):
        in_elements = as_int(metadata.get("kineto_in_msg_nelems"))
        out_elements = as_int(metadata.get("kineto_out_msg_nelems"))
        if in_elements <= 0 or out_elements <= 0:
            raise SchemaError(f"qualification source event {event_index} has non-positive exact message shape")
        operation = str(event.get("op"))
        if operation in {"all_reduce", "broadcast"}:
            shape_matches = in_elements == out_elements
        elif operation == "all_gather":
            shape_matches = out_elements == in_elements * rank_count
        elif operation == "reduce_scatter":
            shape_matches = in_elements == out_elements * rank_count
        else:
            shape_matches = in_elements == out_elements and in_elements % rank_count == 0
        if not shape_matches:
            raise SchemaError(
                f"qualification source event {event_index} exact message shape does not match {operation} semantics"
            )
        if max(in_elements, out_elements) * dtype_size_bytes(dtype) != as_int(event.get("bytes")):
            raise SchemaError(
                f"qualification source event {event_index} bytes do not match its exact input/output element counts"
            )
        return in_elements, out_elements
    nelems = param_element_count(as_int(event.get("bytes")), dtype)
    return param_message_sizes(str(event.get("op")), nelems, rank_count)


def qualification_compute_tensor_elements(operation: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Return exact left, right, and output tensor element counts."""

    m = as_int(operation["m"])
    n = as_int(operation["n"])
    k = as_int(operation["k"])
    return (
        _checked_multiply(m, k, label="qualification GEMM left elements"),
        _checked_multiply(k, n, label="qualification GEMM right elements"),
        _checked_multiply(m, n, label="qualification GEMM output elements"),
    )


def qualification_compute_tensor_bytes(operation: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Return exact left, right, and output allocation sizes."""

    width = dtype_size_bytes(operation["dtype"])
    left_elements, right_elements, output_elements = qualification_compute_tensor_elements(operation)
    return (
        _checked_multiply(left_elements, width, label="qualification GEMM left tensor bytes"),
        _checked_multiply(right_elements, width, label="qualification GEMM right tensor bytes"),
        _checked_multiply(output_elements, width, label="qualification GEMM output tensor bytes"),
    )


def _checked_multiply(left: int, right: int, *, label: str) -> int:
    try:
        return checked_multiply(left, right, label=label)
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc


def _bounded_add(left: int, right: int, *, limit: int, label: str) -> int:
    try:
        return require_within(
            checked_add(left, right, label=label),
            limit,
            label=label,
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc


__all__ = [
    "QUALIFICATION_COMPUTE_RECIPE_METHOD",
    "QUALIFICATION_COMPUTE_RECIPE_PROJECTION",
    "qualification_compute_recipe_audit",
    "qualification_compute_tensor_bytes",
    "qualification_compute_tensor_elements",
    "qualification_program_communication_inventory",
    "trace_to_qualification_program",
]
