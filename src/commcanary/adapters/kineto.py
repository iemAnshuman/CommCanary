"""Fail-closed PyTorch/Kineto trace import adapter."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_float, as_int, load_json_document
from ..errors import SchemaError
from ..formats import TRACE_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits

_KINETO_COLLECTIVE_EVENT_NAME = "record_param_comms"
_KINETO_CONTROL_OPS = {"wait", "barrier", "init", "broadcastuniquencclid"}

# c10::ScalarType names (case-insensitive) to element byte sizes.
_DTYPE_BYTES = {
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
    "half": 2,
    "float16": 2,
    "bfloat16": 2,
    "int": 4,
    "int32": 4,
    "long": 8,
    "int64": 8,
    "short": 2,
    "int16": 2,
    "byte": 1,
    "uint8": 1,
    "char": 1,
    "int8": 1,
    "bool": 1,
    "complexhalf": 4,
    "complexfloat": 8,
    "complexdouble": 16,
}

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


def load_kineto_trace(
    path: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Load a torch.profiler Chrome-trace JSON file.

    Unlike ``schema.load_json`` this accepts both the object form
    (``{"traceEvents": [...]}``) and a bare event array.
    """

    data = load_json_document(path, limits=limits)
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
) -> JsonDict:
    """Convert one rank's Kineto profiler trace into a commcanary trace.

    Selects ``cpu_op``/``record_param_comms`` events (the stable collective
    anchor since torch 2.2), maps element counts times dtype size to bytes,
    reconstructs process-group ranks, and keeps single-rank timestamps as
    ``start_us``. Control ops (wait/barrier/init) and zero-sized messages are
    skipped and counted rather than fabricated.
    """

    raw_events = kineto.get("traceEvents")
    if not isinstance(raw_events, list):
        raise SchemaError("kineto trace is missing a traceEvents list")
    nested_skip = _nested_collective_event_indices(raw_events)
    selected: List[Tuple[float, int, JsonDict]] = []
    skipped_control = 0
    skipped_empty = 0
    skipped_nested = 0
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
        nelems = max(as_int(args.get("In msg nelems"), 0), as_int(args.get("Out msg nelems"), 0))
        if nelems <= 0:
            skipped_empty += 1
            continue
        dtype = str(args.get("dtype", ""))
        element_bytes = _dtype_element_bytes(dtype)
        ranks, ranks_assumed = _kineto_group_ranks(args)
        op = _KINETO_OP_ALIASES.get(collective)
        event: JsonDict = {
            "id": f"kineto-{index:06d}",
            "op": op if op is not None else collective,
            "bytes": nelems * element_bytes,
            "ranks": ranks,
            "group": group,
            "start_us": max(0.0, as_float(raw.get("ts"), 0.0)),
            "metadata": {
                "kineto_collective_name": str(args.get("Collective name")),
                "kineto_dtype": dtype,
                "kineto_in_msg_nelems": as_int(args.get("In msg nelems"), 0),
                "kineto_out_msg_nelems": as_int(args.get("Out msg nelems"), 0),
                "kineto_dur_us": as_float(raw.get("dur"), 0.0),
            },
        }
        if op is None:
            event["custom_op"] = True
        if ranks_assumed:
            event["metadata"]["kineto_ranks_assumed"] = True
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
    if "baseTimeNanoseconds" in kineto:
        system["kineto_base_time_ns"] = as_int(kineto.get("baseTimeNanoseconds"))
    distributed = kineto.get("distributedInfo")
    if isinstance(distributed, Mapping):
        for key in ("backend", "rank", "world_size", "nccl_version"):
            if key in distributed:
                system[f"kineto_{key}"] = copy.deepcopy(distributed.get(key))
    trace: JsonDict = {
        "format": TRACE_FORMAT,
        "workload": {
            "name": str(workload_name),
            "kineto_trace_start_us": base_start_us,
            "notes": (
                "Imported from a single rank's PyTorch Kineto profiler trace. "
                "Single-rank observational import: no cross-rank arrival skew, "
                "compute overlap, or measured exposed latency is claimed. "
                f"Skipped {skipped_control} control op(s), "
                f"{skipped_empty} zero-sized message(s), and "
                f"{skipped_nested} nested duplicate record(s)."
            ),
            "import_source": "pytorch-kineto",
            "imported_events": len(events),
            "skipped_control_events": skipped_control,
            "skipped_empty_events": skipped_empty,
            "skipped_nested_events": skipped_nested,
        },
        "system": system,
        "events": events,
    }
    validate_trace(trace)
    return trace


def _nested_collective_event_indices(raw_events: Sequence[Any]) -> Set[int]:
    """Indices of record_param_comms events nested inside another one.

    torch >= 2.4 emits a frontend/backend RecordFunction *pair* per
    collective, both carrying the named collective args; importing both would
    double every event and halve the gaps. An event whose [ts, ts+dur]
    interval lies within another record_param_comms interval on the same
    pid/tid is the inner duplicate. Traces without nesting are unaffected.
    """

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
    for group in spans.values():
        group.sort(key=lambda item: (item[0], -item[1]))
        active: List[Tuple[float, float, int]] = []
        for start, end, index in group:
            while active and active[-1][1] <= start:
                active.pop()
            if active and start >= active[-1][0] and end <= active[-1][1]:
                nested.add(index)
                continue
            active.append((start, end, index))
    return nested


def _normalize_op_token(name: str) -> str:
    return "".join(char for char in name.lower() if char.isalpha())


def _dtype_element_bytes(dtype: str) -> int:
    key = dtype.lower()
    if key.startswith("float8"):
        return 1
    if key in _DTYPE_BYTES:
        return _DTYPE_BYTES[key]
    raise SchemaError(f"unknown kineto dtype {dtype!r}; cannot derive message bytes")


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


__all__ = ["kineto_trace_to_commcanary_trace", "load_kineto_trace"]
