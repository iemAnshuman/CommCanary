"""Interop with the PyTorch/Kineto and PARAM comms-replay ecosystems.

Two conversions, both fail-closed:

- ``kineto_trace_to_commcanary_trace``: import the ``record_param_comms``
  collective metadata from a ``torch.profiler`` Chrome-trace JSON (one rank's
  view) into a ``commcanary.trace.v1`` document. The import is observational:
  it carries operation identity, sizes, groups, and single-rank timestamps.
  It does not invent cross-rank arrival skew, compute overlap, or measured
  exposed latency, and it says so in the workload notes.
- ``canary_to_param_comms_trace``: export a compiled canary's expanded event
  program as a PARAM comms-replay "basic" JSON trace (the legacy
  ``--trace-type basic`` format of facebookresearch/param), which gives the
  minimized artifact a physical NCCL execution path via PARAM's replayer.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import (
    TRACE_FORMAT,
    JsonDict,
    SchemaError,
    as_float,
    as_int,
    iter_canary_logical_events,
    normalize_ranks,
    validate_canary,
    validate_trace,
)

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

# commcanary collective ops to PARAM basic-trace names. PARAM normalizes via
# paramToCommName, so these canonical snake_case names are accepted. p2p ops
# (send/recv/point_to_point) are handled separately because PARAM's parser
# unconditionally requires src_rank/dst_rank/use_batch on them.
_PARAM_OP_NAMES = {
    "all_reduce": "all_reduce",
    "all_gather": "all_gather",
    "reduce_scatter": "reduce_scatter",
    "all_to_all": "all_to_all",
    "broadcast": "broadcast",
}

_P2P_OPS = ("point_to_point", "send", "recv")

_PARAM_EXPORT_DTYPES = {
    "float": 4,
    "float32": 4,
    "float16": 2,
    "half": 2,
    "float64": 8,
    "double": 8,
    "bfloat16": 2,
    "int": 4,
    "int32": 4,
    "long": 8,
    "byte": 1,
    "uint8": 1,
    "int8": 1,
    "short": 2,
    "bool": 1,
}


def load_kineto_trace(path: str) -> JsonDict:
    """Load a torch.profiler Chrome-trace JSON file.

    Unlike ``schema.load_json`` this accepts both the object form
    (``{"traceEvents": [...]}``) and a bare event array.
    """

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise SchemaError(f"could not load kineto trace {path!r}: {exc}") from exc
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
                system[f"kineto_{key}"] = distributed.get(key)
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


def canary_to_param_comms_trace(
    canary: Mapping[str, Any],
    *,
    dtype: str = "float32",
    skip_unsupported: bool = False,
    compute_fill_us_per_gemm: Optional[float] = None,
    compute_fill_gemm_dim: int = 1024,
    overlap_structure: bool = False,
) -> List[JsonDict]:
    """Export a canary's expanded event program as PARAM basic trace entries.

    Every logical occurrence becomes one entry with element counts derived
    from the canary's byte sizes and the requested dtype, plus cumulative
    ``startTime_ns`` timestamps so PARAM's ``--use-timestamp`` mode reproduces
    inter-op gaps. Point-to-point events need sender and receiver ranks;
    unsupported or custom ops fail closed unless ``skip_unsupported`` is set.

    With ``compute_fill_us_per_gemm`` set, inter-collective gaps are exported
    as PARAM ``{"compute": "gemm", "mm_dim": D, "count": N}`` entries instead
    of idle time: N is the gap divided by the calibrated per-GEMM duration.
    Communication-only replay fills gaps with silence, which cannot reproduce
    compute/communication interference; compute fill occupies the gaps with
    real matrix multiplies. Replay compute-filled traces WITHOUT
    ``--use-timestamp`` so pacing comes from the compute, not wall-clock
    sleeps. The per-GEMM duration is hardware- and dim-specific and must be
    calibrated on the target device.

    With ``overlap_structure`` additionally set, collectives are emitted for
    asynchronous issue with an explicit ``{"comms": "wait", "req": r}`` entry
    placed AFTER the next gap's GEMM entries, reconstructing the workload's
    compute/communication concurrency: collective k executes while the GEMMs
    of gap k+1 occupy the SMs. Serialized replayers (PARAM's is hardwired
    blocking) degrade this gracefully back to sequential execution; an
    overlap-aware replayer expresses the concurrency. Issue entries carry an
    ``issue`` marker so latency parsers can distinguish issue lines from the
    completion-bearing wait lines. Requires compute fill.
    """

    validate_canary(canary)
    dtype_key = str(dtype).lower()
    if dtype_key not in _PARAM_EXPORT_DTYPES:
        supported = ", ".join(sorted(_PARAM_EXPORT_DTYPES))
        raise SchemaError(f"unsupported PARAM export dtype {dtype!r}; supported: {supported}")
    element_bytes = _PARAM_EXPORT_DTYPES[dtype_key]
    fill_us = None
    if compute_fill_us_per_gemm is not None:
        fill_us = as_float(compute_fill_us_per_gemm)
        if fill_us <= 0.0:
            raise SchemaError("compute_fill_us_per_gemm must be positive")
    gemm_dim = as_int(compute_fill_gemm_dim)
    if gemm_dim <= 0:
        raise SchemaError("compute_fill_gemm_dim must be positive")
    if overlap_structure and fill_us is None:
        raise SchemaError("overlap_structure requires compute_fill_us_per_gemm")

    entries: List[JsonDict] = []
    skipped = 0
    pg_ids: Dict[str, int] = {}
    pg_ranks: Dict[int, List[int]] = {}
    pg_used: set = set()
    clock_ns = 0
    request_id = 0
    pending_wait: Optional[JsonDict] = None
    for event in iter_canary_logical_events(canary.get("events", [])):
        op = str(event.get("op"))
        ranks = normalize_ranks(event.get("ranks"))
        group = str(event.get("group", "default"))
        pg_id = pg_ids.setdefault(group, len(pg_ids))
        if pg_id not in pg_ranks:
            pg_ranks[pg_id] = list(ranks)
        elif pg_ranks[pg_id] != list(ranks):
            raise SchemaError(
                f"communicator group {group!r} appears with two different rank "
                f"sets ({pg_ranks[pg_id]} vs {list(ranks)}); PARAM process "
                "groups need a single membership per group"
            )
        nelems = max(1, -(-as_int(event.get("bytes")) // element_bytes))
        in_size, out_size = _param_message_sizes(op, nelems, len(ranks))
        for gap_us in _expanded_gaps_us(event):
            clock_ns += max(0, round(gap_us * 1000.0))
            if fill_us is not None:
                gemm_count = int(round(max(0.0, gap_us) / fill_us))
                if gemm_count > 0:
                    entries.append(
                        {
                            "compute": "gemm",
                            "mm_dim": gemm_dim,
                            "count": gemm_count,
                            "dtype": dtype_key,
                            "req": request_id,
                            "startTime_ns": clock_ns,
                            "markers": [f"commcanary:compute-fill:{group}"],
                        }
                    )
                    request_id += 1
            if overlap_structure and pending_wait is not None:
                # completion of the previous collective, placed AFTER this
                # gap's GEMMs: collective k overlaps the compute of gap k+1.
                # Its req intentionally EQUALS the issuing collective's req --
                # that is how the stored async handle is looked up.
                entries.append(pending_wait)
                pending_wait = None
            if op in _P2P_OPS:
                sender = event.get("sender_rank")
                receiver = event.get("receiver_rank")
                if sender is None or receiver is None:
                    if skip_unsupported:
                        skipped += 1
                        continue
                    raise SchemaError(
                        f"{op} events need sender_rank and receiver_rank for PARAM export; "
                        "re-run with skip_unsupported to drop such events"
                    )
                # PARAM executes a send entry only on src_rank and a recv
                # entry only on dst_rank, so one transfer needs a matched
                # pair of entries or physical replay deadlocks.
                occurrence_entries: List[JsonDict] = [
                    {
                        "comms": "send",
                        "src_rank": as_int(sender),
                        "dst_rank": as_int(receiver),
                        "use_batch": False,
                    },
                    {
                        "comms": "recv",
                        "src_rank": as_int(sender),
                        "dst_rank": as_int(receiver),
                        "use_batch": False,
                    },
                ]
            elif op in _PARAM_OP_NAMES:
                occurrence_entries = [{"comms": _PARAM_OP_NAMES[op]}]
            else:
                if skip_unsupported:
                    skipped += 1
                    continue
                raise SchemaError(
                    f"op {op!r} has no PARAM comms-replay equivalent; "
                    "re-run with skip_unsupported to drop such events"
                )
            pg_used.add(pg_id)
            for entry in occurrence_entries:
                entry.update(
                    {
                        "req": request_id,
                        "startTime_ns": clock_ns,
                        "world_size": len(ranks),
                        "global_ranks": list(ranks),
                        "pg_id": pg_id,
                        "in_msg_size": in_size,
                        "out_msg_size": out_size,
                        "dtype": dtype_key,
                        "markers": [f"commcanary:{group}:{op}"],
                    }
                )
                entries.append(entry)
                request_id += 1
            if overlap_structure and op in _PARAM_OP_NAMES:
                # async issue: the completion wait is emitted after the NEXT
                # gap's GEMMs (see pending_wait placement above). p2p pairs
                # stay synchronous.
                comm_entry = entries[-1]
                comm_entry["markers"] = [f"commcanary:issue:{group}:{op}"]
                pending_wait = {
                    "comms": "wait",
                    "req": comm_entry["req"],
                    "startTime_ns": clock_ns,
                    "markers": [f"commcanary:{group}:{op}"],
                }
    if overlap_structure and pending_wait is not None:
        entries.append(pending_wait)
        pending_wait = None
    if not entries:
        raise SchemaError("canary produced no PARAM-exportable entries")
    if skipped:
        entries[0].setdefault("markers", []).append(f"commcanary:skipped_unsupported={skipped}")
    # PARAM only registers a process group from an explicit init entry at the
    # head of the trace (commsTraceReplay: `if curComm.comms == "init":
    # groupRanks[pgId] = groupRanks`); collectives that reference an
    # unregistered pg_id crash the replay with a KeyError.
    group_names = {pg_id: name for name, pg_id in pg_ids.items()}
    init_entries: List[JsonDict] = [
        {
            "comms": "init",
            "pg_id": pg_id,
            "global_ranks": list(pg_ranks[pg_id]),
            "world_size": len(pg_ranks[pg_id]),
            "req": index,
            "startTime_ns": 0,
            "markers": [f"commcanary:pg-init:{group_names[pg_id]}"],
        }
        for index, pg_id in enumerate(sorted(pg_used))
    ]
    offset = len(init_entries)
    for entry in entries:
        entry["req"] = as_int(entry["req"]) + offset
    return init_entries + entries


def _param_message_sizes(op: str, nelems: int, world_size: int) -> Tuple[int, int]:
    """Per-op in/out element counts matching PARAM's replay conventions.

    CommCanary ``bytes`` is the total/largest buffer of the collective (the
    Kineto importer takes max(in, out) nelems). PARAM allocates tensors from
    the trace sizes as-is: all_gather gathers world_size shards of
    in_msg_size elements, and reduce_scatter scatters an in_msg_size input
    into out_msg_size shards, so asymmetric collectives must not export
    in == out or the physical replay allocates the wrong volume or crashes.
    """

    if op in ("all_gather", "reduce_scatter"):
        if world_size <= 0 or nelems % world_size:
            raise SchemaError(
                f"{op} event bytes do not divide evenly across {world_size} ranks "
                "for PARAM export; choose a dtype whose element size divides the "
                "per-rank shard"
            )
        shard = nelems // world_size
        if op == "all_gather":
            return shard, nelems
        return nelems, shard
    return nelems, nelems


def write_param_comms_trace(path: str, entries: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write PARAM basic trace entries (a JSON array)."""

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(prefix=".commcanary-", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(list(entries), stream, indent=1, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _expanded_gaps_us(event: Mapping[str, Any]) -> Iterable[float]:
    """Yield one readiness gap per logical occurrence of a canary event.

    Mirrors the replay expansion semantics: a record of weight N contributes
    N occurrences at gap_sum/N each with the final occurrence absorbing the
    rounding residual, and one nesting level of timing_pattern repetition.
    """

    for sample in event.get("timing_samples", []):
        pattern = sample.get("timing_pattern")
        if isinstance(pattern, list) and pattern:
            repeats = as_int(sample.get("pattern_repeats", 1))
            for _repeat in range(repeats):
                for child in pattern:
                    yield from _record_gaps_us(child)
        else:
            yield from _record_gaps_us(sample)


def _record_gaps_us(sample: Mapping[str, Any]) -> Iterable[float]:
    weight = as_int(sample.get("weight", 1))
    if weight <= 0:
        raise SchemaError("timing record weight must be positive")
    gap_sum = as_float(sample.get("gap_sum_us", as_float(sample.get("gap_us"), 0.0) * weight))
    average = gap_sum / weight
    emitted = 0.0
    for index in range(weight - 1):
        yield average
        emitted += average
    yield gap_sum - emitted


def _nested_collective_event_indices(raw_events: Sequence[Any]) -> set:
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

    nested: set = set()
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


def _reject_constant(value: str) -> None:
    raise SchemaError(f"kineto trace contains unsupported JSON constant {value!r}")
