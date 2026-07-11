"""Fail-closed PARAM comms-replay export adapter."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..artifacts.canary import iter_canary_logical_events, validate_canary
from ..artifacts.io import PARAM_TRACE_POLICY, atomic_write_json
from ..artifacts.wire import JsonDict, as_float, as_int, normalize_ranks
from ..errors import SchemaError
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    checked_multiply,
    require_within,
)

LogicalEventIterator = Callable[..., Iterable[JsonDict]]
RankCache = Dict[int, Tuple[Any, Tuple[int, ...]]]
GapCache = Dict[int, Tuple[Any, Tuple[float, ...]]]

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


def canary_to_param_comms_trace(
    canary: Mapping[str, Any],
    *,
    dtype: str = "float32",
    skip_unsupported: bool = False,
    compute_fill_us_per_gemm: Optional[float] = None,
    compute_fill_gemm_dim: int = 1024,
    overlap_structure: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> List[JsonDict]:
    """Export a canary through the production logical-event iterator."""

    return export_param_comms_trace(
        canary,
        dtype=dtype,
        skip_unsupported=skip_unsupported,
        compute_fill_us_per_gemm=compute_fill_us_per_gemm,
        compute_fill_gemm_dim=compute_fill_gemm_dim,
        overlap_structure=overlap_structure,
        limits=limits,
        logical_event_iterator=iter_canary_logical_events,
    )


def export_param_comms_trace(
    canary: Mapping[str, Any],
    *,
    dtype: str = "float32",
    skip_unsupported: bool = False,
    compute_fill_us_per_gemm: Optional[float] = None,
    compute_fill_gemm_dim: int = 1024,
    overlap_structure: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
    logical_event_iterator: LogicalEventIterator,
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

    validate_canary(canary, limits=limits)
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
    _preflight_param_entry_count(
        canary,
        skip_unsupported=skip_unsupported,
        compute_fill=fill_us is not None,
        overlap_structure=overlap_structure,
        limits=limits,
    )

    entries: List[JsonDict] = []
    skipped = 0
    pg_ids: Dict[str, int] = {}
    pg_ranks: Dict[int, Tuple[int, ...]] = {}
    pg_used: Set[int] = set()
    clock_ns = 0
    request_id = 0
    pending_wait: Optional[JsonDict] = None
    # Production motif expansion shallow-copies each stored child, so its
    # immutable-in-practice ranks and timing-sample objects retain identity
    # across repetitions. Cache only for that production iterator; injected
    # iterators remain fully dynamic for the historical test/adapter seam.
    cache_templates = logical_event_iterator is iter_canary_logical_events
    rank_cache: Optional[RankCache] = {} if cache_templates else None
    gap_cache: Optional[GapCache] = {} if cache_templates else None
    marker_cache: Dict[Tuple[str, str, str], str] = {}
    for event in logical_event_iterator(
        canary.get("events", []),
        limits=limits,
    ):
        op = str(event.get("op"))
        ranks = _cached_ranks(event, rank_cache)
        group = str(event.get("group", "default"))
        pg_id = pg_ids.setdefault(group, len(pg_ids))
        if pg_id not in pg_ranks:
            pg_ranks[pg_id] = ranks
        elif pg_ranks[pg_id] != ranks:
            raise SchemaError(
                f"communicator group {group!r} appears with two different rank "
                f"sets ({list(pg_ranks[pg_id])} vs {list(ranks)}); PARAM process "
                "groups need a single membership per group"
            )
        nelems = max(1, -(-as_int(event.get("bytes")) // element_bytes))
        in_size, out_size = _param_message_sizes(op, nelems, len(ranks))
        for gap_us in _cached_expanded_gaps_us(event, gap_cache):
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
                            "markers": [_cached_marker(marker_cache, "compute-fill", group, op)],
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
                    f"op {op!r} has no PARAM comms-replay equivalent; re-run with skip_unsupported to drop such events"
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
                        "markers": [_cached_marker(marker_cache, "complete", group, op)],
                    }
                )
                entries.append(entry)
                request_id += 1
            if overlap_structure and op in _PARAM_OP_NAMES:
                # async issue: the completion wait is emitted after the NEXT
                # gap's GEMMs (see pending_wait placement above). p2p pairs
                # stay synchronous.
                comm_entry = entries[-1]
                comm_entry["markers"] = [_cached_marker(marker_cache, "issue", group, op)]
                pending_wait = {
                    "comms": "wait",
                    "req": comm_entry["req"],
                    "startTime_ns": clock_ns,
                    "markers": [_cached_marker(marker_cache, "complete", group, op)],
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


def _preflight_param_entry_count(
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
                if op in _P2P_OPS:
                    communication_entries = 2
                elif op in _PARAM_OP_NAMES:
                    communication_entries = 1
                elif skip_unsupported:
                    continue
                else:
                    # The exporter will emit the more specific unsupported-op
                    # error before producing output. It contributes no count.
                    continue
                occurrences = checked_multiply(
                    as_int(child.get("repeat"), 1),
                    multiplier,
                    label="PARAM logical occurrences",
                )
                per_occurrence = communication_entries
                if compute_fill:
                    # Zero-duration gaps may omit this entry; counting one is a
                    # safe preflight upper bound that never underestimates work.
                    per_occurrence += 1
                if overlap_structure and op in _PARAM_OP_NAMES:
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

    atomic_write_json(
        path,
        list(entries),
        indent=1,
        policy=PARAM_TRACE_POLICY,
    )


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


def _cached_ranks(event: Mapping[str, Any], cache: Optional[RankCache]) -> Tuple[int, ...]:
    raw_ranks = event.get("ranks")
    if cache is not None:
        cache_key = id(raw_ranks)
        cached = cache.get(cache_key)
        if cached is not None and cached[0] is raw_ranks:
            return cached[1]
    ranks = tuple(normalize_ranks(raw_ranks))
    if cache is not None:
        cache[cache_key] = (raw_ranks, ranks)
    return ranks


def _cached_expanded_gaps_us(event: Mapping[str, Any], cache: Optional[GapCache]) -> Iterable[float]:
    raw_samples = event.get("timing_samples")
    if cache is None or not isinstance(raw_samples, list):
        return _expanded_gaps_us(event)
    cache_key = id(raw_samples)
    cached = cache.get(cache_key)
    if cached is not None and cached[0] is raw_samples:
        return cached[1]
    gaps = tuple(_expanded_gaps_us(event))
    cache[cache_key] = (raw_samples, gaps)
    return gaps


def _cached_marker(cache: Dict[Tuple[str, str, str], str], kind: str, group: str, op: str) -> str:
    key = (kind, group, op)
    marker = cache.get(key)
    if marker is not None:
        return marker
    if kind == "compute-fill":
        marker = f"commcanary:compute-fill:{group}"
    elif kind == "issue":
        marker = f"commcanary:issue:{group}:{op}"
    else:
        marker = f"commcanary:{group}:{op}"
    cache[key] = marker
    return marker


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


__all__ = [
    "LogicalEventIterator",
    "canary_to_param_comms_trace",
    "export_param_comms_trace",
    "write_param_comms_trace",
]
