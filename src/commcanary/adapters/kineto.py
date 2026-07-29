"""Fail-closed PyTorch/Kineto trace import adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast

from ..artifacts.dtypes import dtype_size_bytes, normalize_dtype, require_param_compute_dtype
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.trace import validate_trace
from ..artifacts.wire import (
    JsonDict,
    as_float,
    as_int,
    load_json_document,
    load_json_document_with_identity,
    normalize_ranks,
)
from ..errors import SchemaError
from ..formats import TRACE_FORMAT
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    require_within,
    validate_json_mapping,
)
from .capture_merge import NamedTrace, merge_trace_documents

_KINETO_COLLECTIVE_EVENT_NAME = "record_param_comms"
_KINETO_CONTROL_OPS = {"wait", "barrier", "init", "broadcastuniquencclid"}
_KINETO_OVERLAP_METHOD = "linked-kernel-interval-union.v1"
_KINETO_COMPUTE_RECIPE_METHOD = "explicit-wait-linked-contiguous-gemm.v1"
_KINETO_BROADCAST_ROOT_METHOD = "c10d-concrete-input-root-rank.v1"
_KINETO_REDUCTION_METHOD = "linked-nccl-kernel-name.v1"
_KINETO_MESSAGE_SHAPE_METHOD = "record-param-comms-in-out-nelems.v1"
_REDUCTION_COLLECTIVES = frozenset({"all_reduce", "reduce_scatter"})
_EQUAL_MESSAGE_SHAPE_OPS = frozenset(
    {
        "all_reduce",
        "broadcast",
        "point_to_point",
        "send",
        "recv",
    }
)
_REDUCTION_KERNEL_RE = re.compile(
    r"(?:allreduce|reduce_?scatter)_(sum|prod(?:uct)?|min|max|avg)(?:_|[<(]|$)",
    re.IGNORECASE,
)

# Kineto collective names (normalized) to commcanary ops. Anything absent is
# imported as a custom op rather than silently dropped or mislabelled.
_KINETO_OP_ALIASES = {
    "allreduce": "all_reduce",
    "allreducecoalesced": "all_reduce",
    "allgather": "all_gather",
    "allgatherbase": "all_gather",
    "allgatherintotensorcoalesced": "all_gather",
    "reducescatter": "reduce_scatter",
    "reducescatterbase": "reduce_scatter",
    "reducescattertensorcoalesced": "reduce_scatter",
    "alltoall": "all_to_all",
    "alltoallv": "all_to_all",
    "alltoallbase": "all_to_all",
    "broadcast": "broadcast",
    "send": "send",
    "recv": "recv",
    "recvanysource": "recv",
}


@dataclass(frozen=True)
class _KernelSpan:
    index: int
    name: str
    start_us: float
    end_us: float
    device: str
    stream: str
    external_id: Optional[int]
    communication: bool


@dataclass(frozen=True)
class _OverlapDerivation:
    status: str
    overlap_us: Optional[float] = None
    communication_kernel_count: int = 0
    compute_kernel_count: int = 0
    communication_duration_us: float = 0.0


@dataclass(frozen=True)
class _ComputeRecipeDerivation:
    status: str
    operations: Tuple[JsonDict, ...] = ()


@dataclass(frozen=True)
class _ReductionDerivation:
    status: str
    reduction_op: Optional[str] = None
    communication_kernel_count: int = 0


def load_kineto_trace(
    path: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Load a torch.profiler Chrome-trace JSON file.

    Unlike ``schema.load_json`` this accepts both the object form
    (``{"traceEvents": [...]}``) and a bare event array.
    """

    return _kineto_document(load_json_document(path, limits=limits))


def load_kineto_trace_with_identity(
    path: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Tuple[JsonDict, JsonDict]:
    """Load a Kineto trace and commit the exact bounded bytes decoded."""

    data, identity = load_json_document_with_identity(path, limits=limits)
    return _kineto_document(data), identity


def _kineto_document(data: Any) -> JsonDict:
    if isinstance(data, list):
        return {"traceEvents": data}
    if isinstance(data, dict):
        return data
    raise SchemaError("kineto trace must be a JSON object or event array")


def kineto_trace_to_commcanary_trace(
    kineto: Mapping[str, Any],
    *,
    workload_name: str = "kineto-import",
    phase: Optional[str] = None,
    process_group: Optional[str] = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Convert one rank's Kineto profile without retaining its clock origin."""

    trace, _trace_start_us = _kineto_trace_to_commcanary_trace(
        kineto,
        workload_name=workload_name,
        phase=phase,
        process_group=process_group,
        limits=limits,
    )
    return trace


def _kineto_trace_to_commcanary_trace(
    kineto: Mapping[str, Any],
    *,
    workload_name: str,
    phase: Optional[str],
    process_group: Optional[str],
    limits: ResourceLimits,
) -> Tuple[JsonDict, float]:
    """Convert one rank's Kineto profiler trace into a commcanary trace.

    Selects ``cpu_op``/``record_param_comms`` events (the stable collective
    anchor since torch 2.2), maps element counts times dtype size to bytes,
    reconstructs process-group ranks, and keeps single-rank timestamps as
    ``start_us``. Control ops (wait/barrier/init) and zero-sized messages are
    skipped and counted rather than fabricated. When complete CUDA kernel
    activities link a collective to one or more NCCL kernels, the importer
    measures the union of intervals in which non-communication kernels execute
    concurrently on another stream. Missing, malformed, or ambiguous linkage
    keeps overlap explicitly unknown so canary-producing workflows refuse it.
    """

    try:
        validate_json_mapping(kineto, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"kineto trace violates JSON resource constraints: {exc}") from exc
    raw_events = kineto.get("traceEvents")
    if not isinstance(raw_events, list):
        raise SchemaError("kineto trace is missing a traceEvents list")
    nested_skip, nested_families = _nested_collective_event_families(raw_events)
    broadcast_roots = _kineto_broadcast_root_ranks(raw_events)
    selected: List[Tuple[float, int, JsonDict]] = []
    skipped_control = 0
    skipped_empty = 0
    skipped_nested = 0
    broadcast_roots_derived = 0
    broadcast_roots_unknown = 0
    message_shapes_derived = 0
    message_shapes_unknown = 0
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            continue
        if raw.get("name") != _KINETO_COLLECTIVE_EVENT_NAME:
            continue
        if raw.get("cat") not in (None, "cpu_op"):
            continue
        if index in nested_skip:
            skipped_nested += 1
            continue
        args = raw.get("args")
        if not isinstance(args, Mapping) or "Collective name" not in args:
            continue
        collective = _normalize_op_token(str(args.get("Collective name")))
        if collective in _KINETO_CONTROL_OPS:
            skipped_control += 1
            continue
        group = str(args.get("Process Group Name", "") or "default")
        if process_group is not None and group != str(process_group):
            continue
        in_nelems = as_int(args.get("In msg nelems"), 0)
        out_nelems = as_int(args.get("Out msg nelems"), 0)
        nelems = max(in_nelems, out_nelems)
        if nelems <= 0:
            skipped_empty += 1
            continue
        dtype = str(args.get("dtype", ""))
        canonical_dtype = _normalize_kineto_dtype(dtype)
        element_bytes = dtype_size_bytes(canonical_dtype)
        ranks, ranks_assumed = _kineto_group_ranks(args)
        op = _KINETO_OP_ALIASES.get(collective)
        input_split_sizes = _parse_kineto_split_sizes(args.get("In split size"))
        output_split_sizes = _parse_kineto_split_sizes(args.get("Out split size"))
        message_shape_status = _kineto_message_shape_status(
            op,
            in_nelems=in_nelems,
            out_nelems=out_nelems,
            ranks=ranks,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
        )
        event: JsonDict = {
            "id": f"kineto-{index:06d}",
            "op": op if op is not None else collective,
            "dtype": canonical_dtype,
            "bytes": nelems * element_bytes,
            "ranks": ranks,
            "group": group,
            "start_us": max(0.0, as_float(raw.get("ts"), 0.0)),
            "metadata": {
                "kineto_collective_name": str(args.get("Collective name")),
                "kineto_dtype": dtype,
                "kineto_in_msg_nelems": in_nelems,
                "kineto_out_msg_nelems": out_nelems,
                "kineto_dur_us": as_float(raw.get("dur"), 0.0),
                "kineto_message_shape_status": message_shape_status,
            },
        }
        if input_split_sizes is not None:
            event["metadata"]["kineto_in_split_sizes"] = input_split_sizes
        if output_split_sizes is not None:
            event["metadata"]["kineto_out_split_sizes"] = output_split_sizes
        if message_shape_status == "derived":
            event["metadata"]["kineto_message_shape_method"] = _KINETO_MESSAGE_SHAPE_METHOD
            message_shapes_derived += 1
        else:
            message_shapes_unknown += 1
        if op is None:
            event["custom_op"] = True
        if ranks_assumed:
            event["metadata"]["kineto_ranks_assumed"] = True
        if op == "broadcast":
            root_rank = broadcast_roots.get(index)
            if root_rank is None:
                event["metadata"]["kineto_broadcast_root_status"] = "unknown"
                broadcast_roots_unknown += 1
            else:
                if root_rank not in ranks:
                    raise SchemaError(f"kineto broadcast root rank {root_rank} is outside process-group ranks {ranks}")
                event["root_rank"] = root_rank
                event["metadata"]["kineto_broadcast_root_rank"] = root_rank
                event["metadata"]["kineto_broadcast_root_status"] = "derived"
                event["metadata"]["kineto_broadcast_root_method"] = _KINETO_BROADCAST_ROOT_METHOD
                broadcast_roots_derived += 1
        if phase is not None:
            event["phase"] = str(phase)
        if op in ("send", "recv"):
            if "Src Rank" in args:
                event["metadata"]["kineto_src_rank"] = as_int(args.get("Src Rank"))
            if "Dst Rank" in args:
                event["metadata"]["kineto_dst_rank"] = as_int(args.get("Dst Rank"))
            if "Src Rank" in args and "Dst Rank" in args:
                sender = as_int(args.get("Src Rank"))
                receiver = as_int(args.get("Dst Rank"))
                if sender != receiver and sender in ranks and receiver in ranks:
                    event["sender_rank"] = sender
                    event["receiver_rank"] = receiver
        if "Seq" in args:
            event["metadata"]["kineto_seq"] = as_int(args.get("Seq"))
        selected.append((event["start_us"], index, event))
    if not selected:
        raise SchemaError(
            "no importable record_param_comms collective events found; "
            "traces from torch < 2.2 do not carry collective metadata"
        )
    overlap_derivations = _derive_compute_overlap(
        raw_events,
        {index: nested_families.get(index, (index,)) for _ts, index, _event in selected},
    )
    compute_recipe_derivations = _derive_compute_recipes(
        raw_events,
        {index: nested_families.get(index, (index,)) for _ts, index, _event in selected},
        limits=limits,
    )
    reduction_derivations = _derive_reduction_operators(
        raw_events,
        {index: nested_families.get(index, (index,)) for _ts, index, _event in selected},
    )
    overlap_derived = 0
    overlap_unknown = 0
    compute_recipes_derived = 0
    compute_recipes_unknown = 0
    reductions_derived = 0
    reductions_unknown = 0
    for _ts, index, event in selected:
        derivation = overlap_derivations[index]
        compute_recipe = compute_recipe_derivations[index]
        metadata = event["metadata"]
        metadata["kineto_compute_recipe_status"] = compute_recipe.status
        if compute_recipe.status == "derived":
            event["compute_recipe"] = copy.deepcopy(list(compute_recipe.operations))
            metadata["kineto_compute_recipe_method"] = _KINETO_COMPUTE_RECIPE_METHOD
            metadata["kineto_compute_recipe_operation_count"] = len(compute_recipe.operations)
            metadata["kineto_compute_recipe_kernel_count"] = sum(
                as_int(operation["source_kernel_count"]) for operation in compute_recipe.operations
            )
            metadata["kineto_compute_recipe_duration_us"] = round(
                sum(as_float(operation["source_kernel_duration_us"]) for operation in compute_recipe.operations),
                9,
            )
            compute_recipes_derived += 1
        else:
            compute_recipes_unknown += 1
        if event["op"] in _REDUCTION_COLLECTIVES:
            reduction = reduction_derivations[index]
            metadata["kineto_reduction_status"] = reduction.status
            if reduction.reduction_op is None:
                reductions_unknown += 1
            else:
                event["reduction_op"] = reduction.reduction_op
                metadata["kineto_reduction_op"] = reduction.reduction_op
                metadata["kineto_reduction_method"] = _KINETO_REDUCTION_METHOD
                metadata["kineto_reduction_kernel_count"] = reduction.communication_kernel_count
                reductions_derived += 1
        metadata["kineto_overlap_status"] = derivation.status
        if derivation.overlap_us is None:
            event["compute_overlap_unknown"] = True
            overlap_unknown += 1
            continue
        event["compute_overlap_us"] = round(derivation.overlap_us, 9)
        metadata["kineto_overlap_method"] = _KINETO_OVERLAP_METHOD
        metadata["kineto_communication_kernel_count"] = derivation.communication_kernel_count
        metadata["kineto_compute_kernel_count"] = derivation.compute_kernel_count
        metadata["kineto_communication_duration_us"] = round(derivation.communication_duration_us, 9)
        overlap_derived += 1
    selected.sort(key=lambda item: (item[0], item[1]))
    # Rebase timestamps to the trace start: Kineto ts values are on a
    # monotonic/awake-time clock (epoch-scale in older producers), so raw
    # values are semantically meaningless as start_us and can exceed the
    # schema's maximum supported duration on long-uptime hosts.
    base_start_us = selected[0][0]
    events = []
    for _ts, _index, event in selected:
        event["start_us"] = round(event["start_us"] - base_start_us, 3)
        events.append(event)

    system: JsonDict = {"source_format": "pytorch-kineto"}
    distributed = kineto.get("distributedInfo")
    if isinstance(distributed, Mapping):
        for key in ("backend", "rank", "world_size", "nccl_version"):
            if key in distributed:
                system[f"kineto_{key}"] = copy.deepcopy(distributed.get(key))
    trace: JsonDict = {
        "format": TRACE_FORMAT,
        "workload": {
            "name": str(workload_name),
            "notes": (
                "Imported from a single rank's PyTorch Kineto profiler trace. "
                "Single-rank observational import: no cross-rank arrival skew, "
                "or measured exposed latency is claimed. Compute overlap is "
                f"derived for {overlap_derived} event(s) from linked CUDA kernel "
                f"intervals and remains explicitly unknown for {overlap_unknown} "
                "event(s); canary compilation refuses unless every event is known. "
                f"An exact contiguous-GEMM compute recipe was derived for "
                f"{compute_recipes_derived} event(s) and remains unavailable for "
                f"{compute_recipes_unknown} event(s). "
                f"Broadcast root rank was derived for {broadcast_roots_derived} "
                f"event(s) and remains unknown for {broadcast_roots_unknown} event(s). "
                f"Reduction operator was derived for {reductions_derived} "
                f"event(s) and remains unknown for {reductions_unknown} event(s). "
                f"Input/output message shape was derived for {message_shapes_derived} "
                f"event(s) and remains unavailable for {message_shapes_unknown} event(s). "
                f"Skipped {skipped_control} control op(s), "
                f"{skipped_empty} zero-sized message(s), and "
                f"{skipped_nested} nested duplicate record(s)."
            ),
            "import_source": "pytorch-kineto",
            "imported_events": len(events),
            "skipped_control_events": skipped_control,
            "skipped_empty_events": skipped_empty,
            "skipped_nested_events": skipped_nested,
            "overlap_derived_events": overlap_derived,
            "overlap_unknown_events": overlap_unknown,
            "compute_recipes_derived_events": compute_recipes_derived,
            "compute_recipes_unknown_events": compute_recipes_unknown,
            "broadcast_roots_derived_events": broadcast_roots_derived,
            "broadcast_roots_unknown_events": broadcast_roots_unknown,
            "reduction_ops_derived_events": reductions_derived,
            "reduction_ops_unknown_events": reductions_unknown,
            "message_shapes_derived_events": message_shapes_derived,
            "message_shapes_unknown_events": message_shapes_unknown,
        },
        "system": system,
        "events": events,
    }
    validate_trace(trace, limits=limits)
    return trace, base_start_us


def kineto_traces_to_commcanary_trace(
    kinetos: Sequence[Mapping[str, Any]],
    *,
    workload_name: str = "kineto-import",
    phase: Optional[str] = None,
    process_group: Optional[str] = None,
    clock_offsets_us: Optional[Mapping[Any, Any]] = None,
    assume_shared_clock: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Convert a complete set of rank-local Kineto profiles into one trace.

    Rank-local collective occurrences are aligned by invocation ordinal within
    an exact process-group rank domain. The capture reconciler then requires
    one compatible contribution from every participating rank.
    ``clock_offsets_us`` contains additive offsets from each rank's Kineto
    timestamp clock into one common reference clock. ``assume_shared_clock``
    is the explicit zero-offset claim for profiles recorded on one shared
    clock. Omitting both keeps arrival skew explicitly unknown, which makes
    compilation refuse the trace.
    """

    runtime_kinetos = cast(Any, kinetos)
    if isinstance(runtime_kinetos, (str, bytes)) or not isinstance(runtime_kinetos, Sequence):
        raise SchemaError("multi-rank Kineto input must be a sequence of profile objects")
    if not isinstance(assume_shared_clock, bool):
        raise SchemaError("assume_shared_clock must be a boolean")
    if len(kinetos) < 2:
        raise SchemaError("multi-rank Kineto import requires at least two rank profiles")
    try:
        require_within(
            len(kinetos),
            limits.max_capture_shards,
            label="Kineto rank profiles",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc

    imported: List[Tuple[int, int, str, Optional[str], JsonDict, float]] = []
    seen_ranks: Set[int] = set()
    expected_world_size: Optional[int] = None
    expected_backend: Optional[str] = None
    observed_nccl_versions: Set[str] = set()
    aggregate_event_count = 0
    for profile_index, kineto in enumerate(kinetos):
        if not isinstance(kineto, Mapping):
            raise SchemaError(f"Kineto profile {profile_index} must be an object")
        rank, world_size, backend, nccl_version = _kineto_distributed_identity(kineto, profile_index)
        if rank in seen_ranks:
            raise SchemaError(f"duplicate Kineto profile for distributed rank {rank}")
        seen_ranks.add(rank)
        if expected_world_size is None:
            expected_world_size = world_size
        elif world_size != expected_world_size:
            raise SchemaError("Kineto rank profiles disagree on distributed world_size")
        if expected_backend is None:
            expected_backend = backend
        elif backend != expected_backend:
            raise SchemaError("Kineto rank profiles disagree on distributed backend")
        if nccl_version is not None:
            observed_nccl_versions.add(nccl_version)

        trace, trace_start_us = _kineto_trace_to_commcanary_trace(
            kineto,
            workload_name=workload_name,
            phase=phase,
            process_group=process_group,
            limits=limits,
        )
        try:
            aggregate_event_count = checked_add(
                aggregate_event_count,
                len(trace["events"]),
                label="Kineto aggregate imported events",
            )
            require_within(
                aggregate_event_count,
                limits.max_capture_events,
                label="Kineto aggregate imported events",
            )
        except JsonResourceError as exc:
            raise SchemaError(str(exc)) from exc
        imported.append((rank, world_size, backend, nccl_version, trace, trace_start_us))
    if len(observed_nccl_versions) > 1:
        raise SchemaError("Kineto rank profiles disagree on NCCL version")

    if assume_shared_clock and clock_offsets_us is not None:
        raise SchemaError("assume_shared_clock and clock_offsets_us are mutually exclusive")
    offsets = _normalized_clock_offsets(
        {rank: 0.0 for rank in seen_ranks} if assume_shared_clock else clock_offsets_us,
        seen_ranks,
    )
    trace_starts_by_rank = {
        rank: trace_start_us for rank, _world_size, _backend, _nccl_version, _trace, trace_start_us in imported
    }
    aligned_origin_us: Optional[float] = None
    if offsets is not None:
        aligned_origin_us = min(trace_starts_by_rank[rank] + offsets[rank] for rank in seen_ranks)

    session_id = "kineto-multi-rank-v1"
    named_traces: List[NamedTrace] = []
    for rank, _world_size, _backend, _nccl_version, source_trace, _trace_start_us in sorted(imported):
        counters: Dict[Tuple[Any, ...], int] = {}
        contribution_events: List[JsonDict] = []
        for source_event in source_trace["events"]:
            event = copy.deepcopy(source_event)
            ranks = sorted(normalize_ranks(event.get("ranks")))
            op = str(event.get("op"))
            if op in ("send", "recv"):
                if "sender_rank" not in event or "receiver_rank" not in event:
                    raise SchemaError(
                        "multi-rank Kineto import requires Src Rank and Dst Rank metadata for every send/recv event"
                    )
                sender = as_int(event.get("sender_rank"))
                receiver = as_int(event.get("receiver_rank"))
                if (op == "send" and rank != sender) or (op == "recv" and rank != receiver):
                    raise SchemaError(f"Kineto rank {rank} {op} record conflicts with its source/destination metadata")
                ranks = sorted((sender, receiver))
                domain: Tuple[Any, ...] = ("point_to_point", str(event.get("group", "default")), *ranks)
            else:
                domain = ("collective", str(event.get("group", "default")), *ranks)
            if rank not in ranks:
                raise SchemaError(
                    f"Kineto rank {rank} recorded an event whose participant ranks do not include that rank"
                )
            ordinal = counters.get(domain, 0)
            counters[domain] = ordinal + 1
            event.update(
                {
                    "ranks": ranks,
                    "capture_session_id": session_id,
                    "collective_id": _kineto_collective_id(domain, ordinal),
                    "collective_seq": ordinal,
                    "recorder_rank": rank,
                    "rank_arrival_us": {str(rank): 0.0},
                    "partial_rank_arrival": True,
                }
            )
            if op in ("send", "recv"):
                metadata = event.get("metadata", {})
                if isinstance(metadata, Mapping) and "kineto_seq" in metadata:
                    event["message_sequence"] = as_int(metadata.get("kineto_seq"))
                else:
                    event["message_sequence"] = ordinal
            contribution_events.append(event)

        contribution_system = copy.deepcopy(source_trace["system"])
        contribution_system.update(
            {
                "rank": rank,
                "capture_session_id": session_id,
            }
        )
        if offsets is not None:
            assert aligned_origin_us is not None
            contribution_system["clock_offset_us"] = round(
                trace_starts_by_rank[rank] + offsets[rank] - aligned_origin_us,
                9,
            )
        contribution: JsonDict = {
            "format": TRACE_FORMAT,
            "workload": {"name": str(workload_name)},
            "system": contribution_system,
            "events": contribution_events,
        }
        named_traces.append((f"kineto-rank-{rank}.trace.json", contribution))

    merged = merge_trace_documents(
        named_traces,
        workload_name=str(workload_name),
        limits=limits,
    )
    # Capture directories record wall-clock merge time. An in-memory format
    # conversion must remain byte-deterministic for identical source profiles.
    merged.pop("created_at", None)
    overlap_derived = sum("compute_overlap_us" in event for event in merged["events"])
    overlap_unknown = len(merged["events"]) - overlap_derived
    compute_recipes_derived = sum(
        isinstance(event.get("compute_recipe_by_rank"), Mapping)
        and set(event["compute_recipe_by_rank"]) == {str(rank) for rank in normalize_ranks(event.get("ranks"))}
        for event in merged["events"]
    )
    compute_recipes_unknown = len(merged["events"]) - compute_recipes_derived
    broadcast_events = [event for event in merged["events"] if event.get("op") == "broadcast"]
    broadcast_roots_derived = sum("root_rank" in event for event in broadcast_events)
    broadcast_roots_unknown = len(broadcast_events) - broadcast_roots_derived
    reduction_events = [event for event in merged["events"] if event.get("op") in _REDUCTION_COLLECTIVES]
    reduction_ops_derived = sum("reduction_op" in event for event in reduction_events)
    reduction_ops_unknown = len(reduction_events) - reduction_ops_derived
    message_shapes_derived = sum(
        isinstance(event.get("metadata"), Mapping) and event["metadata"].get("kineto_message_shape_status") == "derived"
        for event in merged["events"]
    )
    message_shapes_unknown = len(merged["events"]) - message_shapes_derived
    skipped_keys = (
        "skipped_control_events",
        "skipped_empty_events",
        "skipped_nested_events",
    )
    skipped_totals = {
        key: sum(
            as_int(trace["workload"].get(key), 0) for _rank, _world, _backend, _nccl, trace, _trace_start in imported
        )
        for key in skipped_keys
    }
    merged["workload"] = {
        "name": str(workload_name),
        "notes": (
            f"Merged {len(imported)} rank-local PyTorch Kineto profiles into "
            f"{len(merged['events'])} logical event(s). Rank contributions are "
            "matched by invocation ordinal within an exact process-group rank "
            "domain and must agree on operation identity. "
            + (
                "Explicit additive clock offsets were applied before deriving cross-rank arrival offsets. "
                if offsets is not None
                else "No clock calibration was supplied; arrival skew remains explicitly unknown and compilation refuses it. "
            )
            + f"Compute overlap is known for {overlap_derived} event(s) and "
            f"explicitly unknown for {overlap_unknown} event(s). "
            f"Exact rank-local GEMM compute recipes are known for "
            f"{compute_recipes_derived} event(s) and unavailable for "
            f"{compute_recipes_unknown} event(s). "
            f"Broadcast root rank is known for {broadcast_roots_derived} "
            f"event(s) and explicitly unknown for {broadcast_roots_unknown} event(s). "
            f"Reduction operator is known for {reduction_ops_derived} "
            f"event(s) and explicitly unknown for {reduction_ops_unknown} event(s). "
            f"Input/output message shape is known for {message_shapes_derived} "
            f"event(s) and unavailable for {message_shapes_unknown} event(s)."
        ),
        "import_source": "pytorch-kineto",
        "import_mode": "multi-rank",
        "imported_rank_profiles": len(imported),
        "imported_ranks": sorted(seen_ranks),
        "imported_events": len(merged["events"]),
        "source_rank_events": aggregate_event_count,
        "overlap_derived_events": overlap_derived,
        "overlap_unknown_events": overlap_unknown,
        "compute_recipes_derived_events": compute_recipes_derived,
        "compute_recipes_unknown_events": compute_recipes_unknown,
        "broadcast_roots_derived_events": broadcast_roots_derived,
        "broadcast_roots_unknown_events": broadcast_roots_unknown,
        "reduction_ops_derived_events": reduction_ops_derived,
        "reduction_ops_unknown_events": reduction_ops_unknown,
        "message_shapes_derived_events": message_shapes_derived,
        "message_shapes_unknown_events": message_shapes_unknown,
        **skipped_totals,
    }
    merged_system = merged["system"]
    merged_system.update(
        {
            "capture_mode": "kineto-multi-rank",
            "source_format": "pytorch-kineto",
            "kineto_backend": expected_backend,
            "kineto_world_size": expected_world_size,
            "kineto_imported_ranks": sorted(seen_ranks),
            "clock_alignment": (
                "assumed_shared_clock"
                if assume_shared_clock
                else "explicit_offset_us"
                if offsets is not None
                else "uncalibrated"
            ),
        }
    )
    if observed_nccl_versions:
        merged_system["kineto_nccl_version"] = next(iter(observed_nccl_versions))
    if offsets is not None:
        merged_system["kineto_clock_offsets_us"] = {str(rank): offsets[rank] for rank in sorted(offsets)}
    validate_trace(merged, limits=limits)
    return merged


def _kineto_distributed_identity(
    kineto: Mapping[str, Any],
    profile_index: int,
) -> Tuple[int, int, str, Optional[str]]:
    distributed = kineto.get("distributedInfo")
    if not isinstance(distributed, Mapping):
        raise SchemaError(f"Kineto profile {profile_index} is missing distributedInfo")
    if "rank" not in distributed or "world_size" not in distributed or "backend" not in distributed:
        raise SchemaError(f"Kineto profile {profile_index} distributedInfo must include rank, world_size, and backend")
    rank = as_int(distributed.get("rank"))
    world_size = as_int(distributed.get("world_size"))
    backend = str(distributed.get("backend", "")).strip().lower()
    if world_size <= 1:
        raise SchemaError("multi-rank Kineto import requires distributed world_size greater than one")
    if rank < 0 or rank >= world_size:
        raise SchemaError(f"Kineto profile {profile_index} distributed rank is outside world_size")
    if not backend:
        raise SchemaError(f"Kineto profile {profile_index} distributed backend must be non-empty")
    nccl_version = distributed.get("nccl_version")
    return rank, world_size, backend, str(nccl_version) if nccl_version is not None else None


def _normalized_clock_offsets(
    raw_offsets: Optional[Mapping[Any, Any]],
    expected_ranks: Set[int],
) -> Optional[Dict[int, float]]:
    if raw_offsets is None:
        return None
    if not isinstance(raw_offsets, Mapping):
        raise SchemaError("clock_offsets_us must be a rank-to-offset mapping")
    normalized: Dict[int, float] = {}
    for raw_rank, raw_offset in raw_offsets.items():
        rank = as_int(raw_rank)
        if rank in normalized:
            raise SchemaError(f"duplicate clock offset for rank {rank}")
        normalized[rank] = as_float(raw_offset)
    if set(normalized) != expected_ranks:
        missing = sorted(expected_ranks - set(normalized))
        unexpected = sorted(set(normalized) - expected_ranks)
        details = []
        if missing:
            details.append(f"missing ranks {missing}")
        if unexpected:
            details.append(f"unexpected ranks {unexpected}")
        raise SchemaError("clock offsets must exactly match imported ranks: " + "; ".join(details))
    return normalized


def _kineto_collective_id(domain: Tuple[Any, ...], ordinal: int) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": list(domain),
                "ordinal": ordinal,
            }
        )
    ).hexdigest()
    return f"kineto-{digest}"


def _derive_compute_overlap(
    raw_events: Sequence[Any],
    collective_families: Mapping[int, Sequence[int]],
) -> Dict[int, _OverlapDerivation]:
    """Derive overlap only from complete, unambiguous kernel evidence.

    Kineto writes a collective CPU op's external correlation id onto its linked
    NCCL kernel. Kernel ``pid`` identifies the device and kernel ``tid`` (or
    ``args.stream``) identifies the CUDA stream. For each selected collective
    this function unions the linked communication-kernel intervals, intersects
    them with non-communication kernels on other streams of the same device,
    and unions those intersections so concurrent kernels are never
    double-counted.
    """

    spans, compute_evidence_complete, incomplete_communication_ids = _kernel_spans(raw_events)
    if not compute_evidence_complete:
        return {
            index: _OverlapDerivation(status="unavailable_incomplete_kernel_activity") for index in collective_families
        }

    collective_external_ids: Dict[int, Tuple[int, ...]] = {}
    external_id_owners: Dict[int, List[int]] = {}
    for index, family in collective_families.items():
        family_external_ids: List[int] = []
        for member_index in family:
            raw = raw_events[member_index]
            args = raw.get("args") if isinstance(raw, Mapping) else None
            external_id = _optional_kineto_id(args.get("External id")) if isinstance(args, Mapping) else None
            if external_id is not None and external_id not in family_external_ids:
                family_external_ids.append(external_id)
        collective_external_ids[index] = tuple(family_external_ids)
        for external_id in family_external_ids:
            external_id_owners.setdefault(external_id, []).append(index)

    communication_by_external_id: Dict[int, List[_KernelSpan]] = {}
    compute_by_device_stream: Dict[str, Dict[str, List[_KernelSpan]]] = {}
    for span in spans:
        if span.communication:
            if span.external_id is not None:
                communication_by_external_id.setdefault(span.external_id, []).append(span)
        else:
            compute_by_device_stream.setdefault(span.device, {}).setdefault(span.stream, []).append(span)
    for streams in compute_by_device_stream.values():
        for stream_spans in streams.values():
            stream_spans.sort(key=lambda span: (span.start_us, span.end_us, span.index))
    compute_starts_by_device_stream = {
        device: {stream: [span.start_us for span in stream_spans] for stream, stream_spans in streams.items()}
        for device, streams in compute_by_device_stream.items()
    }

    derivations: Dict[int, _OverlapDerivation] = {}
    for index in collective_families:
        external_ids = collective_external_ids[index]
        if not external_ids:
            derivations[index] = _OverlapDerivation(status="unavailable_missing_external_id")
            continue
        if any(len(external_id_owners.get(external_id, [])) != 1 for external_id in external_ids):
            derivations[index] = _OverlapDerivation(status="unavailable_nonunique_external_id")
            continue
        if any(external_id in incomplete_communication_ids for external_id in external_ids):
            derivations[index] = _OverlapDerivation(status="unavailable_incomplete_linked_communication_kernel")
            continue
        communication = [
            span for external_id in external_ids for span in communication_by_external_id.get(external_id, [])
        ]
        if not communication:
            derivations[index] = _OverlapDerivation(status="unavailable_no_linked_communication_kernel")
            continue
        devices = {span.device for span in communication}
        if len(devices) != 1:
            derivations[index] = _OverlapDerivation(status="unavailable_ambiguous_communication_device")
            continue

        communication_intervals = _merged_intervals([(span.start_us, span.end_us) for span in communication])
        communication_duration_us = sum(end - start for start, end in communication_intervals)
        overlap_intervals: List[Tuple[float, float]] = []
        overlapping_compute_indices: Set[int] = set()
        device = next(iter(devices))
        for comm in communication:
            for stream, compute_spans in compute_by_device_stream.get(device, {}).items():
                if stream == comm.stream:
                    continue
                starts = compute_starts_by_device_stream[device][stream]
                candidate_index = max(0, bisect_left(starts, comm.start_us) - 1)
                while candidate_index < len(compute_spans):
                    compute = compute_spans[candidate_index]
                    if compute.start_us >= comm.end_us:
                        break
                    candidate_index += 1
                    if compute.end_us <= comm.start_us:
                        continue
                    start = max(compute.start_us, comm.start_us)
                    end = min(compute.end_us, comm.end_us)
                    if start < end:
                        overlap_intervals.append((start, end))
                        overlapping_compute_indices.add(compute.index)
        overlap_us = sum(end - start for start, end in _merged_intervals(overlap_intervals))
        derivations[index] = _OverlapDerivation(
            status="derived",
            overlap_us=overlap_us,
            communication_kernel_count=len(communication),
            compute_kernel_count=len(overlapping_compute_indices),
            communication_duration_us=communication_duration_us,
        )
    return derivations


def _derive_compute_recipes(
    raw_events: Sequence[Any],
    collective_families: Mapping[int, Sequence[int]],
    *,
    limits: ResourceLimits,
) -> Dict[int, _ComputeRecipeDerivation]:
    """Recover exact contiguous GEMMs launched between issue and explicit wait.

    A start-to-start communication gap is not compute work: it also contains
    collective launch, exposed communication, Python overhead, and idle time.
    For the narrow schedule CommCanary can execute honestly, this derivation
    instead requires one unambiguous ``record_param_comms(wait)`` on the same
    CPU thread before the next collective. Non-communication CUDA kernels are
    attributed through their external id to CPU operators launched after the
    collective call returns and before that wait begins. Only contiguous
    two-dimensional ``aten::mm`` operations are currently materializable.

    Full linked kernel durations are retained as source observations, but the
    executable recipe is the GEMM work itself. Target execution must run that
    work rather than convert a source duration into unrelated square GEMMs.
    """

    spans, compute_evidence_complete, incomplete_communication_ids = _kernel_spans(raw_events)
    if not compute_evidence_complete:
        return {
            index: _ComputeRecipeDerivation(status="unavailable_incomplete_kernel_activity")
            for index in collective_families
        }
    compute_spans = [span for span in spans if not span.communication]
    if any(span.external_id is None for span in compute_spans):
        return {
            index: _ComputeRecipeDerivation(status="unavailable_unlinked_compute_kernel")
            for index in collective_families
        }

    communication_by_external_id: Dict[int, List[_KernelSpan]] = {}
    compute_by_external_id: Dict[int, List[_KernelSpan]] = {}
    external_id_owners: Dict[int, List[int]] = {}
    collective_external_ids: Dict[int, Tuple[int, ...]] = {}
    for index, family in collective_families.items():
        family_external_ids: List[int] = []
        for member_index in family:
            raw = raw_events[member_index]
            args = raw.get("args") if isinstance(raw, Mapping) else None
            external_id = _optional_kineto_id(args.get("External id")) if isinstance(args, Mapping) else None
            if external_id is not None and external_id not in family_external_ids:
                family_external_ids.append(external_id)
        collective_external_ids[index] = tuple(family_external_ids)
        for external_id in family_external_ids:
            external_id_owners.setdefault(external_id, []).append(index)
    for span in spans:
        if span.external_id is None:
            continue
        target = communication_by_external_id if span.communication else compute_by_external_id
        target.setdefault(span.external_id, []).append(span)

    ordered_collectives = sorted(
        collective_families,
        key=lambda index: (
            as_float(raw_events[index].get("ts"), 0.0) if isinstance(raw_events[index], Mapping) else 0.0,
            index,
        ),
    )
    derivations: Dict[int, _ComputeRecipeDerivation] = {}
    for position, index in enumerate(ordered_collectives):
        raw = raw_events[index]
        if not isinstance(raw, Mapping):
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_malformed_collective_interval")
            continue
        issue_start = _optional_non_negative_float(raw.get("ts"))
        issue_duration = _optional_non_negative_float(raw.get("dur"))
        if issue_start is None or issue_duration is None:
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_malformed_collective_interval")
            continue
        pid = raw.get("pid")
        tid = raw.get("tid")
        next_start: Optional[float] = None
        for following_index in ordered_collectives[position + 1 :]:
            following = raw_events[following_index]
            if not isinstance(following, Mapping):
                continue
            if following.get("pid") != pid or following.get("tid") != tid:
                continue
            next_start = _optional_non_negative_float(following.get("ts"))
            if next_start is not None:
                break

        waits: List[Tuple[float, int]] = []
        issue_end = issue_start + issue_duration
        for candidate_index, candidate in enumerate(raw_events):
            if not isinstance(candidate, Mapping):
                continue
            if candidate.get("name") != _KINETO_COLLECTIVE_EVENT_NAME:
                continue
            if candidate.get("cat") not in (None, "cpu_op"):
                continue
            if candidate.get("pid") != pid or candidate.get("tid") != tid:
                continue
            args = candidate.get("args")
            if not isinstance(args, Mapping):
                continue
            if _normalize_op_token(str(args.get("Collective name", ""))) != "wait":
                continue
            wait_start = _optional_non_negative_float(candidate.get("ts"))
            if wait_start is None or wait_start + 0.001 < issue_end:
                continue
            if next_start is not None and wait_start >= next_start - 0.001:
                continue
            waits.append((wait_start, candidate_index))
        if not waits:
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_missing_explicit_wait")
            continue
        if len(waits) != 1:
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_ambiguous_explicit_wait")
            continue
        wait_start, _wait_index = waits[0]

        external_ids = collective_external_ids[index]
        if not external_ids:
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_missing_external_id")
            continue
        if any(len(external_id_owners.get(external_id, [])) != 1 for external_id in external_ids):
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_nonunique_external_id")
            continue
        if any(external_id in incomplete_communication_ids for external_id in external_ids):
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_incomplete_linked_communication_kernel")
            continue
        communication = [
            span for external_id in external_ids for span in communication_by_external_id.get(external_id, [])
        ]
        if not communication:
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_no_linked_communication_kernel")
            continue
        communication_devices = {span.device for span in communication}
        if len(communication_devices) != 1:
            derivations[index] = _ComputeRecipeDerivation(status="unavailable_ambiguous_communication_device")
            continue
        communication_device = next(iter(communication_devices))

        cpu_owners: Dict[int, List[Tuple[float, int, Mapping[str, Any]]]] = {}
        for owner_index, owner in enumerate(raw_events):
            if not isinstance(owner, Mapping) or owner.get("cat") != "cpu_op":
                continue
            if owner.get("ph") not in (None, "X"):
                continue
            if owner.get("pid") != pid or owner.get("tid") != tid:
                continue
            owner_start = _optional_non_negative_float(owner.get("ts"))
            owner_duration = _optional_non_negative_float(owner.get("dur"))
            if owner_start is None or owner_duration is None:
                continue
            if owner_start + 0.001 < issue_end or owner_start + owner_duration > wait_start + 0.001:
                continue
            owner_args = owner.get("args")
            if not isinstance(owner_args, Mapping):
                continue
            external_id = _optional_kineto_id(owner_args.get("External id"))
            if external_id is None or external_id not in compute_by_external_id:
                continue
            cpu_owners.setdefault(external_id, []).append((owner_start, owner_index, owner))

        recipe_operations: List[Tuple[float, int, JsonDict]] = []
        unavailable_status: Optional[str] = None
        for external_id, owners in cpu_owners.items():
            linked_spans = compute_by_external_id[external_id]
            devices = {span.device for span in linked_spans}
            if devices != {communication_device}:
                unavailable_status = "unavailable_ambiguous_compute_device"
                break
            if len(owners) != 1:
                unavailable_status = "unavailable_nonunique_compute_external_id"
                break
            owner_start, owner_index, owner = owners[0]
            try:
                operation = _kineto_contiguous_gemm_recipe(
                    owner,
                    linked_spans,
                    limits=limits,
                )
            except SchemaError:
                unavailable_status = "unavailable_unsupported_compute_operator"
                break
            recipe_operations.append((owner_start, owner_index, operation))
        if unavailable_status is not None:
            derivations[index] = _ComputeRecipeDerivation(status=unavailable_status)
            continue
        recipe_operations.sort(key=lambda item: (item[0], item[1]))
        derivations[index] = _ComputeRecipeDerivation(
            status="derived",
            operations=tuple(operation for _start, _index, operation in recipe_operations),
        )
    return derivations


def _kineto_contiguous_gemm_recipe(
    owner: Mapping[str, Any],
    linked_spans: Sequence[_KernelSpan],
    *,
    limits: ResourceLimits,
) -> JsonDict:
    if owner.get("name") != "aten::mm":
        raise SchemaError("unsupported Kineto compute operator")
    args = owner.get("args")
    if not isinstance(args, Mapping):
        raise SchemaError("Kineto aten::mm lacks arguments")
    raw_dims = args.get("Input Dims")
    raw_strides = args.get("Input Strides")
    raw_types = args.get("Input type")
    if (
        not isinstance(raw_dims, list)
        or len(raw_dims) != 2
        or not isinstance(raw_strides, list)
        or len(raw_strides) != 2
        or not isinstance(raw_types, list)
        or len(raw_types) != 2
    ):
        raise SchemaError("Kineto aten::mm arguments are incomplete")
    if any(not isinstance(shape, list) or len(shape) != 2 for shape in raw_dims):
        raise SchemaError("Kineto aten::mm shapes must be two-dimensional")
    if any(not isinstance(strides, list) or len(strides) != 2 for strides in raw_strides):
        raise SchemaError("Kineto aten::mm strides must be two-dimensional")
    left_m, left_k = (as_int(value) for value in raw_dims[0])
    right_k, right_n = (as_int(value) for value in raw_dims[1])
    if min(left_m, left_k, right_k, right_n) <= 0 or left_k != right_k:
        raise SchemaError("Kineto aten::mm dimensions are incompatible")
    if max(left_m, left_k, right_n) > limits.max_param_gemm_dim:
        raise SchemaError("Kineto aten::mm dimension exceeds resource policy")
    left_strides = [as_int(value) for value in raw_strides[0]]
    right_strides = [as_int(value) for value in raw_strides[1]]
    if left_strides != [left_k, 1] or right_strides != [right_n, 1]:
        raise SchemaError("Kineto aten::mm inputs must be contiguous row-major matrices")
    dtypes = [_normalize_kineto_compute_dtype(value) for value in raw_types]
    if len(set(dtypes)) != 1:
        raise SchemaError("Kineto aten::mm input dtypes disagree")
    duration_us = sum(
        end - start for start, end in _merged_intervals([(span.start_us, span.end_us) for span in linked_spans])
    )
    return {
        "op": "gemm",
        "dtype": dtypes[0],
        "m": left_m,
        "n": right_n,
        "k": left_k,
        "source_kernel_count": len(linked_spans),
        "source_kernel_duration_us": round(duration_us, 9),
    }


def _normalize_kineto_compute_dtype(value: Any) -> str:
    token = str(value).strip()
    if "::" in token:
        token = token.rsplit("::", 1)[1]
    return require_param_compute_dtype(token, label="Kineto GEMM dtype")


def _derive_reduction_operators(
    raw_events: Sequence[Any],
    collective_families: Mapping[int, Sequence[int]],
) -> Dict[int, _ReductionDerivation]:
    """Bind reduction semantics only from uniquely linked NCCL kernel names."""

    spans, _compute_evidence_complete, incomplete_communication_ids = _kernel_spans(raw_events)
    collective_external_ids: Dict[int, Tuple[int, ...]] = {}
    external_id_owners: Dict[int, List[int]] = {}
    for index, family in collective_families.items():
        family_external_ids: List[int] = []
        for member_index in family:
            raw = raw_events[member_index]
            args = raw.get("args") if isinstance(raw, Mapping) else None
            external_id = _optional_kineto_id(args.get("External id")) if isinstance(args, Mapping) else None
            if external_id is not None and external_id not in family_external_ids:
                family_external_ids.append(external_id)
        collective_external_ids[index] = tuple(family_external_ids)
        for external_id in family_external_ids:
            external_id_owners.setdefault(external_id, []).append(index)

    communication_by_external_id: Dict[int, List[_KernelSpan]] = {}
    for span in spans:
        if span.communication and span.external_id is not None:
            communication_by_external_id.setdefault(span.external_id, []).append(span)

    derivations: Dict[int, _ReductionDerivation] = {}
    for index in collective_families:
        external_ids = collective_external_ids[index]
        if not external_ids:
            derivations[index] = _ReductionDerivation(status="unavailable_missing_external_id")
            continue
        if any(len(external_id_owners.get(external_id, [])) != 1 for external_id in external_ids):
            derivations[index] = _ReductionDerivation(status="unavailable_nonunique_external_id")
            continue
        if any(external_id in incomplete_communication_ids for external_id in external_ids):
            derivations[index] = _ReductionDerivation(status="unavailable_incomplete_linked_communication_kernel")
            continue
        communication = [
            span for external_id in external_ids for span in communication_by_external_id.get(external_id, [])
        ]
        if not communication:
            derivations[index] = _ReductionDerivation(status="unavailable_no_linked_communication_kernel")
            continue
        operators = [_reduction_op_from_kernel_name(span.name) for span in communication]
        if any(operator is None for operator in operators):
            derivations[index] = _ReductionDerivation(
                status="unavailable_unrecognized_kernel_name",
                communication_kernel_count=len(communication),
            )
            continue
        unique_operators = {operator for operator in operators if operator is not None}
        if len(unique_operators) != 1:
            derivations[index] = _ReductionDerivation(
                status="unavailable_conflicting_kernel_names",
                communication_kernel_count=len(communication),
            )
            continue
        derivations[index] = _ReductionDerivation(
            status="derived",
            reduction_op=next(iter(unique_operators)),
            communication_kernel_count=len(communication),
        )
    return derivations


def _reduction_op_from_kernel_name(name: str) -> Optional[str]:
    match = _REDUCTION_KERNEL_RE.search(name)
    if match is None:
        return None
    token = match.group(1).lower()
    return "product" if token in {"prod", "product"} else token


def _kernel_spans(raw_events: Sequence[Any]) -> Tuple[List[_KernelSpan], bool, Set[int]]:
    spans: List[_KernelSpan] = []
    compute_evidence_complete = True
    incomplete_communication_ids: Set[int] = set()
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping) or raw.get("cat") != "kernel":
            continue
        args = raw.get("args")
        args = args if isinstance(args, Mapping) else {}
        start_us = _optional_non_negative_float(raw.get("ts"))
        duration_us = _optional_non_negative_float(raw.get("dur"))
        device = _consistent_identity(raw.get("pid"), args.get("device"))
        stream = _consistent_identity(raw.get("tid"), args.get("stream"))
        name = raw.get("name")
        external_id = _optional_kineto_id(args.get("External id"))
        communication = _is_communication_kernel(name, args)
        external_id_present = "External id" in args
        malformed = (
            raw.get("ph") not in (None, "X")
            or start_us is None
            or duration_us is None
            or device is None
            or stream is None
            or not isinstance(name, str)
            or (external_id_present and external_id is None)
            or (communication and external_id is None)
        )
        if malformed:
            if communication:
                if external_id is not None:
                    incomplete_communication_ids.add(external_id)
            else:
                # An unplaceable non-communication kernel could overlap any
                # selected collective. No event can claim measured zero until
                # the compute activity inventory is complete.
                compute_evidence_complete = False
            continue
        if duration_us == 0.0:
            # A compute interval measured as zero at profiler resolution has
            # an empty intersection and can be ignored safely. A linked
            # communication interval measured as zero could hide part of the
            # collective duration, so that collective stays unknown.
            if communication and external_id is not None:
                incomplete_communication_ids.add(external_id)
            continue
        assert start_us is not None
        assert duration_us is not None
        assert device is not None
        assert stream is not None
        assert isinstance(name, str)
        spans.append(
            _KernelSpan(
                index=index,
                name=name,
                start_us=start_us,
                end_us=start_us + duration_us,
                device=device,
                stream=stream,
                external_id=external_id,
                communication=communication,
            )
        )
    return spans, compute_evidence_complete, incomplete_communication_ids


def _is_communication_kernel(name: Any, args: Mapping[str, Any]) -> bool:
    if "Collective name" in args:
        return True
    if not isinstance(name, str):
        return False
    token = _normalize_op_token(name)
    return "nccl" in token or "rccl" in token


def _optional_non_negative_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def _optional_kineto_id(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if isinstance(value, str) and str(result) != value.strip():
        return None
    return result if result > 0 else None


def _consistent_identity(primary: Any, metadata: Any) -> Optional[str]:
    primary_token = _identity_token(primary)
    metadata_token = _identity_token(metadata)
    if primary_token is not None and metadata_token is not None and primary_token != metadata_token:
        return None
    return metadata_token if metadata_token is not None else primary_token


def _identity_token(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
        return None
    token = str(value).strip()
    return token or None


def _merged_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _nested_collective_event_indices(raw_events: Sequence[Any]) -> Set[int]:
    """Indices of record_param_comms events nested inside another one.

    torch >= 2.4 emits a frontend/backend RecordFunction *pair* per
    collective, both carrying the named collective args; importing both would
    double every event and halve the gaps. An event whose [ts, ts+dur]
    interval lies within another record_param_comms interval on the same
    pid/tid is the inner duplicate. Traces without nesting are unaffected.
    """

    nested, _families = _nested_collective_event_families(raw_events)
    return nested


def _nested_collective_event_families(
    raw_events: Sequence[Any],
) -> Tuple[Set[int], Dict[int, Tuple[int, ...]]]:
    """Return nested duplicates and the outer record that owns each family."""

    spans: Dict[Tuple[Any, Any], List[Tuple[float, float, int]]] = {}
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            continue
        if raw.get("name") != _KINETO_COLLECTIVE_EVENT_NAME:
            continue
        if raw.get("cat") not in (None, "cpu_op"):
            continue
        start = as_float(raw.get("ts"), 0.0)
        end = start + max(0.0, as_float(raw.get("dur"), 0.0))
        spans.setdefault((raw.get("pid"), raw.get("tid")), []).append((start, end, index))

    nested: Set[int] = set()
    families: Dict[int, List[int]] = {}
    for group in spans.values():
        group.sort(key=lambda item: (item[0], -item[1]))
        active: List[Tuple[float, float, int]] = []
        for start, end, index in group:
            while active and active[-1][1] <= start:
                active.pop()
            if active and start >= active[-1][0] and end <= active[-1][1]:
                nested.add(index)
                owner = active[-1][2]
                families.setdefault(owner, [owner]).append(index)
                continue
            active.append((start, end, index))
    return nested, {owner: tuple(members) for owner, members in families.items()}


def _kineto_broadcast_root_ranks(raw_events: Sequence[Any]) -> Dict[int, int]:
    """Recover broadcast roots from the containing ``c10d::broadcast_`` call.

    ``record_param_comms`` does not copy ``BroadcastOptions.rootRank`` into its
    named metadata. The containing c10d RecordFunction does retain the concrete
    dispatcher inputs, whose third value is the root rank in the published
    ``c10d::broadcast_`` schema. Only an interval-contained event on the same
    CPU thread is accepted; otherwise the root remains unknown.
    """

    parents: Dict[Tuple[Any, Any], List[Tuple[float, float, Optional[int]]]] = {}
    records: List[Tuple[int, Tuple[Any, Any], float, float]] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping) or raw.get("cat") not in (None, "cpu_op"):
            continue
        start = as_float(raw.get("ts"), 0.0)
        end = start + max(0.0, as_float(raw.get("dur"), 0.0))
        thread = (raw.get("pid"), raw.get("tid"))
        if raw.get("name") == "c10d::broadcast_":
            parents.setdefault(thread, []).append((start, end, _kineto_broadcast_parent_root(raw.get("args"))))
            continue
        if raw.get("name") != _KINETO_COLLECTIVE_EVENT_NAME:
            continue
        args = raw.get("args")
        if not isinstance(args, Mapping):
            continue
        if _normalize_op_token(str(args.get("Collective name", ""))) == "broadcast":
            records.append((index, thread, start, end))

    parent_starts: Dict[Tuple[Any, Any], List[float]] = {}
    for thread, spans in parents.items():
        spans.sort(key=lambda item: (item[0], item[1]))
        parent_starts[thread] = [span[0] for span in spans]

    result: Dict[int, int] = {}
    for index, thread, start, end in records:
        candidate_spans = parents.get(thread)
        starts = parent_starts.get(thread)
        if not candidate_spans or not starts:
            continue
        parent_index = bisect_right(starts, start) - 1
        if parent_index < 0:
            continue
        parent_start, parent_end, root_rank = candidate_spans[parent_index]
        if parent_start <= start and end <= parent_end and root_rank is not None:
            result[index] = root_rank
    return result


def _kineto_broadcast_parent_root(raw_args: Any) -> Optional[int]:
    if not isinstance(raw_args, Mapping):
        return None
    concrete_inputs = raw_args.get("Concrete Inputs")
    input_types = raw_args.get("Input type")
    if not isinstance(concrete_inputs, list) or len(concrete_inputs) < 3:
        return None
    if isinstance(input_types, list) and len(input_types) >= 3 and input_types[2] != "Scalar":
        return None
    try:
        root_rank = as_int(concrete_inputs[2])
    except SchemaError:
        return None
    return root_rank if root_rank >= 0 else None


def _normalize_op_token(name: str) -> str:
    return "".join(char for char in name.lower() if char.isalpha())


def _normalize_kineto_dtype(dtype: str) -> str:
    try:
        return normalize_dtype(dtype, label="kineto dtype")
    except SchemaError as exc:
        raise SchemaError(f"unknown kineto dtype {dtype!r}; cannot derive message bytes") from exc


def _kineto_message_shape_status(
    op: Optional[str],
    *,
    in_nelems: int,
    out_nelems: int,
    ranks: Sequence[int],
    input_split_sizes: Optional[Sequence[int]],
    output_split_sizes: Optional[Sequence[int]],
) -> str:
    """Classify whether exact Kineto message sizes fit the supported semantics."""

    if input_split_sizes is None or output_split_sizes is None:
        return "unavailable_malformed_split_sizes"
    rank_count = len(ranks)
    if rank_count <= 0:
        return "unavailable_empty_process_group"
    if op in _EQUAL_MESSAGE_SHAPE_OPS:
        return "derived" if in_nelems == out_nelems else "unavailable_in_out_mismatch"
    if op == "all_gather":
        return "derived" if out_nelems == in_nelems * rank_count else "unavailable_in_out_mismatch"
    if op == "reduce_scatter":
        return "derived" if in_nelems == out_nelems * rank_count else "unavailable_in_out_mismatch"
    if op == "all_to_all":
        if input_split_sizes or output_split_sizes:
            return "unsupported_uneven_all_to_all"
        if in_nelems != out_nelems or in_nelems % rank_count:
            return "unavailable_in_out_mismatch"
        return "derived"
    return "unavailable_unsupported_operation"


def _parse_kineto_split_sizes(value: Any) -> Optional[List[int]]:
    """Return an exact non-negative Kineto split vector, including an empty one."""

    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not (text.startswith("[") and text.endswith("]")) or "..." in text:
            return None
        try:
            parsed = json.loads(text)
        except ValueError:
            return None
    if not isinstance(parsed, list):
        return None
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in parsed):
        return None
    return list(parsed)


def _kineto_group_ranks(args: Mapping[str, Any]) -> Tuple[List[int], bool]:
    """Reconstruct process-group ranks; returns (ranks, assumed_world_group).

    Fails closed instead of fabricating membership: a truncated or otherwise
    unparseable rank list is only reconstructed from an explicit positive
    Global rank start/stride pair (PyTorch omits both for non-uniform
    groups). The contiguous [0..N-1] fallback applies only when no rank list
    was recorded at all, and is flagged so consumers can see the assumption.
    """

    raw = args.get("Process Group Ranks")
    parsed = _parse_int_list(raw)
    if parsed:
        return parsed, False
    group_size = as_int(args.get("Group size"), 0)
    if group_size <= 0:
        raise SchemaError("kineto collective event carries no usable group information")
    if "Global rank start" in args and "Global rank stride" in args:
        start = as_int(args.get("Global rank start"))
        stride = as_int(args.get("Global rank stride"))
        if start >= 0 and stride > 0:
            return [start + stride * index for index in range(group_size)], False
    if raw is None or (isinstance(raw, str) and raw.strip() in ("", "[]")):
        return list(range(group_size)), True
    raise SchemaError(
        "kineto process-group ranks are truncated or non-uniform and cannot be "
        "reconstructed from a global rank start/stride; refusing to fabricate "
        "group membership"
    )


def _parse_int_list(value: Any) -> Optional[List[int]]:
    if not isinstance(value, str) or "..." in value:
        return None
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        return None
    return parsed if parsed else None


__all__ = [
    "kineto_trace_to_commcanary_trace",
    "kineto_traces_to_commcanary_trace",
    "load_kineto_trace",
    "load_kineto_trace_with_identity",
]
