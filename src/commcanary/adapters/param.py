"""Fail-closed PARAM-basic-derived, rank-aware program adapter.

Current upstream PARAM accepts Chakra host execution traces rather than this
extended JSON encoding. Rank-specific compute-fill fields require a conforming
CommCanary adapter. Callers must not infer upstream executability from a
successful export.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..artifacts.canary import (
    iter_canary_logical_events,
    iter_canary_timing_samples,
    timing_sample_offsets,
)
from ..artifacts.dtypes import require_param_compute_dtype, require_param_dtype
from ..artifacts.io import PARAM_TRACE_POLICY, atomic_write_json
from ..artifacts.param import (
    PARAM_COLLECTIVE_OP_NAMES,
    PARAM_POINT_TO_POINT_OPS,
    param_element_count,
    param_event_dtype,
    param_materialization_requirements,
    param_message_sizes,
    preflight_param_entry_count,
)
from ..artifacts.wire import JsonDict, as_float, as_int, normalize_ranks
from ..errors import SchemaError
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    MAX_CHECKED_COUNT,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    checked_multiply,
    require_within,
)

LogicalEventIterator = Callable[..., Iterable[JsonDict]]
RankCache = Dict[int, Tuple[Any, Tuple[int, ...]]]
TimingOccurrence = Tuple[float, Tuple[float, ...], float]
TimingCache = Dict[int, Tuple[Any, Tuple[TimingOccurrence, ...]]]
PendingOverlap = Tuple[JsonDict, float, Tuple[int, ...], str, str, str]


def audit_param_program_compute_operations(
    entries: Sequence[Mapping[str, Any]],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Recount physical rank-local GEMMs from an encoded replay program."""

    represented_ranks: Set[int] = set()
    rank_counts: Dict[int, int] = {}
    gemm_entry_count = 0
    base_gemm_operation_count = 0
    rank_extra_gemm_operation_count = 0
    total_rank_gemm_operation_count = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise SchemaError(f"PARAM replay program entry {index} must be an object")
        if entry.get("comms") == "init":
            represented_ranks.update(normalize_ranks(entry.get("global_ranks")))
        if entry.get("compute") is None:
            continue
        if entry.get("compute") != "gemm" or entry.get("comms") is not None:
            raise SchemaError(f"PARAM replay program entry {index} has unsupported compute work")
        ranks = tuple(normalize_ranks(entry.get("global_ranks")))
        represented_ranks.update(ranks)
        base_count = as_int(entry.get("count"))
        if base_count < 0:
            raise SchemaError(f"PARAM replay program entry {index} count must be non-negative")
        raw_extra_counts = entry.get("rank_extra_counts")
        if not isinstance(raw_extra_counts, Mapping):
            raise SchemaError(f"PARAM replay program entry {index} rank_extra_counts must be an object")
        expected_keys = {str(rank) for rank in ranks}
        if set(raw_extra_counts) != expected_keys:
            raise SchemaError(f"PARAM replay program entry {index} rank_extra_counts must match global_ranks")
        any_compute = False
        for rank in ranks:
            extra_count = as_int(raw_extra_counts[str(rank)])
            if extra_count < 0:
                raise SchemaError(f"PARAM replay program entry {index} rank_extra_counts[{rank}] must be non-negative")
            rank_count = _bounded_compute_count(
                base_count,
                extra_count,
                limit=limits.max_param_compute_operations,
                label=f"PARAM replay program entry {index} rank {rank} compute operations",
            )
            rank_counts[rank] = _bounded_compute_count(
                rank_counts.get(rank, 0),
                rank_count,
                limit=limits.max_param_compute_operations,
                label=f"PARAM replay program rank {rank} compute operations",
            )
            rank_extra_gemm_operation_count = _bounded_compute_count(
                rank_extra_gemm_operation_count,
                extra_count,
                limit=limits.max_param_compute_operations,
                label="PARAM replay program rank-extra compute operations",
            )
            total_rank_gemm_operation_count = _bounded_compute_count(
                total_rank_gemm_operation_count,
                rank_count,
                limit=limits.max_param_compute_operations,
                label="PARAM replay program total rank compute operations",
            )
            any_compute = any_compute or rank_count > 0
        if not any_compute:
            raise SchemaError(f"PARAM replay program entry {index} has no rank-local GEMM work")
        gemm_entry_count = _bounded_compute_count(
            gemm_entry_count,
            1,
            limit=limits.max_param_entries,
            label="PARAM replay program GEMM entries",
        )
        base_gemm_operation_count = _bounded_compute_count(
            base_gemm_operation_count,
            base_count,
            limit=limits.max_param_compute_operations,
            label="PARAM replay program base compute operations",
        )

    if not represented_ranks:
        raise SchemaError("PARAM replay program represents no ranks")
    dense_rank_domain = list(range(max(represented_ranks) + 1))
    if sorted(represented_ranks) != dense_rank_domain:
        raise SchemaError(
            "PARAM replay program must cover a dense global rank domain "
            f"starting at zero; observed {sorted(represented_ranks)}"
        )
    canonical_rank_counts = {str(rank): rank_counts.get(rank, 0) for rank in dense_rank_domain}
    return {
        "gemm_entry_count": gemm_entry_count,
        "base_gemm_operation_count": base_gemm_operation_count,
        "rank_extra_gemm_operation_count": rank_extra_gemm_operation_count,
        "total_rank_gemm_operation_count": total_rank_gemm_operation_count,
        "max_rank_gemm_operation_count": max(canonical_rank_counts.values(), default=0),
        "rank_gemm_operation_counts": canonical_rank_counts,
    }


def audit_param_compute_fill_quantization(
    canary: Mapping[str, Any],
    *,
    compute_fill_us_per_gemm: float,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Audit source-bounded overlap, gap, and arrival lowering to target GEMMs."""

    fill_us = _require_compute_fill_us(compute_fill_us_per_gemm)
    try:
        max_arrival_offsets = checked_multiply(
            limits.max_expanded_timing_records,
            limits.max_ranks,
            label="PARAM compute-fill arrival-offset ceiling",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    gap_count = 0
    positive_gap_count = 0
    zero_gemm_positive_gap_count = 0
    overlap_count = 0
    positive_overlap_count = 0
    zero_gemm_positive_overlap_count = 0
    arrival_offset_count = 0
    positive_arrival_offset_count = 0
    zero_gemm_positive_arrival_offset_count = 0
    gemm_entry_count = 0
    base_gemm_operation_count = 0
    rank_extra_gemm_operation_count = 0
    total_rank_gemm_operation_count = 0
    rank_gemm_operation_counts: Dict[int, int] = {}
    source_gap_sum = _StableSum()
    materialized_gap_sum = _StableSum()
    gap_absolute_error_sum = _StableSum()
    source_arrival_sum = _StableSum()
    materialized_arrival_sum = _StableSum()
    arrival_absolute_error_sum = _StableSum()
    source_overlap_sum = _StableSum()
    materialized_overlap_sum = _StableSum()
    overlap_absolute_error_sum = _StableSum()
    gap_max_absolute_error = 0.0
    arrival_max_absolute_error = 0.0
    overlap_max_absolute_error = 0.0
    pending_overlap: Optional[Tuple[float, Tuple[int, ...]]] = None

    for event in iter_canary_logical_events(canary.get("events", []), limits=limits):
        op = str(event.get("op"))
        ranks = tuple(normalize_ranks(event.get("ranks")))
        for raw_gap_us, raw_offsets, raw_overlap_us in _expanded_timing_occurrences(event):
            if len(raw_offsets) != len(ranks):
                raise SchemaError("PARAM compute-fill arrival offsets must match event ranks")
            gap_us = max(0.0, as_float(raw_gap_us))
            base_count = _compute_fill_gemm_count(gap_us, fill_us)
            overlap_base_count = 0
            serialized_base_count = base_count
            if pending_overlap is not None:
                pending_overlap_us, _pending_ranks = pending_overlap
                _require_overlap_fits_following_gap(pending_overlap_us, gap_us)
                overlap_base_count = _compute_fill_gemm_count(pending_overlap_us, fill_us)
                if overlap_base_count > base_count:
                    raise SchemaError("source-overlap GEMM count exceeds the following gap's total GEMM count")
                serialized_base_count = base_count - overlap_base_count
            gap_count = _bounded_compute_count(
                gap_count,
                1,
                limit=limits.max_expanded_timing_records,
                label="PARAM compute-fill gaps",
            )
            if gap_us > 0.0:
                positive_gap_count += 1
                if base_count == 0:
                    zero_gemm_positive_gap_count += 1
            materialized_gap_us = _materialized_component_us(
                base_count,
                fill_us,
                label="materialized compute-fill gap duration",
            )
            gap_error_us = _quantization_error(
                materialized_gap_us,
                gap_us,
                label="compute-fill gap quantization error",
            )
            source_gap_sum.add(gap_us, label="source compute-fill gap total")
            materialized_gap_sum.add(
                materialized_gap_us,
                label="materialized compute-fill gap total",
            )
            gap_absolute_error_sum.add(
                gap_error_us,
                label="compute-fill gap absolute error total",
            )
            gap_max_absolute_error = max(gap_max_absolute_error, gap_error_us)
            base_gemm_operation_count = _bounded_compute_count(
                base_gemm_operation_count,
                base_count,
                limit=limits.max_param_compute_operations,
                label="PARAM base compute-fill operations",
            )

            any_arrival_compute = False
            for rank, raw_offset_us in zip(ranks, raw_offsets):
                offset_us = max(0.0, as_float(raw_offset_us))
                extra_count = _compute_fill_gemm_count(offset_us, fill_us)
                arrival_offset_count = _bounded_compute_count(
                    arrival_offset_count,
                    1,
                    limit=max_arrival_offsets,
                    label="PARAM compute-fill arrival offsets",
                )
                if offset_us > 0.0:
                    positive_arrival_offset_count += 1
                    if extra_count == 0:
                        zero_gemm_positive_arrival_offset_count += 1
                materialized_offset_us = _materialized_component_us(
                    extra_count,
                    fill_us,
                    label="materialized compute-fill arrival offset",
                )
                arrival_error_us = _quantization_error(
                    materialized_offset_us,
                    offset_us,
                    label="compute-fill arrival quantization error",
                )
                source_arrival_sum.add(
                    offset_us,
                    label="source compute-fill arrival total",
                )
                materialized_arrival_sum.add(
                    materialized_offset_us,
                    label="materialized compute-fill arrival total",
                )
                arrival_absolute_error_sum.add(
                    arrival_error_us,
                    label="compute-fill arrival absolute error total",
                )
                arrival_max_absolute_error = max(
                    arrival_max_absolute_error,
                    arrival_error_us,
                )
                rank_extra_gemm_operation_count = _bounded_compute_count(
                    rank_extra_gemm_operation_count,
                    extra_count,
                    limit=limits.max_param_compute_operations,
                    label="PARAM rank-extra compute-fill operations",
                )
                rank_count = base_count + extra_count
                rank_gemm_operation_counts[rank] = _bounded_compute_count(
                    rank_gemm_operation_counts.get(rank, 0),
                    rank_count,
                    limit=limits.max_param_compute_operations,
                    label=f"PARAM rank {rank} compute-fill operations",
                )
                total_rank_gemm_operation_count = _bounded_compute_count(
                    total_rank_gemm_operation_count,
                    rank_count,
                    limit=limits.max_param_compute_operations,
                    label="PARAM total rank compute-fill operations",
                )
                any_arrival_compute = any_arrival_compute or extra_count > 0
            if overlap_base_count > 0:
                gemm_entry_count = _bounded_compute_count(
                    gemm_entry_count,
                    1,
                    limit=limits.max_param_entries,
                    label="PARAM compute-fill GEMM entries",
                )
            if serialized_base_count > 0 or any_arrival_compute:
                gemm_entry_count = _bounded_compute_count(
                    gemm_entry_count,
                    1,
                    limit=limits.max_param_entries,
                    label="PARAM compute-fill GEMM entries",
                )

            overlap_us = max(0.0, as_float(raw_overlap_us))
            if op in PARAM_POINT_TO_POINT_OPS and overlap_us > 0.0:
                raise SchemaError(
                    "source-bounded overlap materialization does not support "
                    f"positive overlap on synchronous point-to-point op {op!r}"
                )
            if op in PARAM_COLLECTIVE_OP_NAMES:
                materialized_overlap_count = _compute_fill_gemm_count(overlap_us, fill_us)
                overlap_count = _bounded_compute_count(
                    overlap_count,
                    1,
                    limit=limits.max_expanded_timing_records,
                    label="PARAM compute-fill overlap components",
                )
                if overlap_us > 0.0:
                    positive_overlap_count += 1
                    if materialized_overlap_count == 0:
                        zero_gemm_positive_overlap_count += 1
                materialized_overlap_us = _materialized_component_us(
                    materialized_overlap_count,
                    fill_us,
                    label="materialized compute-fill overlap duration",
                )
                overlap_error_us = _quantization_error(
                    materialized_overlap_us,
                    overlap_us,
                    label="compute-fill overlap quantization error",
                )
                source_overlap_sum.add(
                    overlap_us,
                    label="source compute-fill overlap total",
                )
                materialized_overlap_sum.add(
                    materialized_overlap_us,
                    label="materialized compute-fill overlap total",
                )
                overlap_absolute_error_sum.add(
                    overlap_error_us,
                    label="compute-fill overlap absolute error total",
                )
                overlap_max_absolute_error = max(
                    overlap_max_absolute_error,
                    overlap_error_us,
                )
                pending_overlap = (overlap_us, ranks)
            else:
                pending_overlap = None

    if pending_overlap is not None:
        tail_overlap_us, tail_ranks = pending_overlap
        tail_overlap_count = _compute_fill_gemm_count(tail_overlap_us, fill_us)
        if tail_overlap_count > 0:
            gemm_entry_count = _bounded_compute_count(
                gemm_entry_count,
                1,
                limit=limits.max_param_entries,
                label="PARAM compute-fill GEMM entries",
            )
            base_gemm_operation_count = _bounded_compute_count(
                base_gemm_operation_count,
                tail_overlap_count,
                limit=limits.max_param_compute_operations,
                label="PARAM base compute-fill operations",
            )
            for rank in tail_ranks:
                rank_gemm_operation_counts[rank] = _bounded_compute_count(
                    rank_gemm_operation_counts.get(rank, 0),
                    tail_overlap_count,
                    limit=limits.max_param_compute_operations,
                    label=f"PARAM rank {rank} compute-fill operations",
                )
                total_rank_gemm_operation_count = _bounded_compute_count(
                    total_rank_gemm_operation_count,
                    tail_overlap_count,
                    limit=limits.max_param_compute_operations,
                    label="PARAM total rank compute-fill operations",
                )

    source_gap_total = source_gap_sum.value
    materialized_gap_total = materialized_gap_sum.value
    source_arrival_total = source_arrival_sum.value
    materialized_arrival_total = materialized_arrival_sum.value
    source_overlap_total = source_overlap_sum.value
    materialized_overlap_total = materialized_overlap_sum.value
    gap_signed_error = materialized_gap_total - source_gap_total
    arrival_signed_error = materialized_arrival_total - source_arrival_total
    overlap_signed_error = materialized_overlap_total - source_overlap_total
    total_absolute_error = (
        gap_absolute_error_sum.value + arrival_absolute_error_sum.value + overlap_absolute_error_sum.value
    )
    max_absolute_error = max(
        gap_max_absolute_error,
        arrival_max_absolute_error,
        overlap_max_absolute_error,
    )
    if not all(
        math.isfinite(value)
        for value in (
            gap_signed_error,
            arrival_signed_error,
            overlap_signed_error,
            total_absolute_error,
            max_absolute_error,
        )
    ):
        raise SchemaError("compute-fill aggregate quantization audit is not finite")
    rank_counts = {str(rank): rank_gemm_operation_counts[rank] for rank in sorted(rank_gemm_operation_counts)}
    return {
        "gap_count": gap_count,
        "positive_gap_count": positive_gap_count,
        "zero_gemm_positive_gap_count": zero_gemm_positive_gap_count,
        "source_gap_sum_us": _rounded_audit_us(source_gap_total),
        "materialized_gap_sum_us": _rounded_audit_us(materialized_gap_total),
        "gap_signed_error_us": _rounded_audit_us(gap_signed_error),
        "gap_total_absolute_error_us": _rounded_audit_us(gap_absolute_error_sum.value),
        "gap_max_absolute_error_us": _rounded_audit_us(gap_max_absolute_error),
        "overlap_count": overlap_count,
        "positive_overlap_count": positive_overlap_count,
        "zero_gemm_positive_overlap_count": zero_gemm_positive_overlap_count,
        "source_overlap_sum_us": _rounded_audit_us(source_overlap_total),
        "materialized_overlap_sum_us": _rounded_audit_us(materialized_overlap_total),
        "overlap_signed_error_us": _rounded_audit_us(overlap_signed_error),
        "overlap_total_absolute_error_us": _rounded_audit_us(overlap_absolute_error_sum.value),
        "overlap_max_absolute_error_us": _rounded_audit_us(overlap_max_absolute_error),
        "arrival_offset_count": arrival_offset_count,
        "positive_arrival_offset_count": positive_arrival_offset_count,
        "zero_gemm_positive_arrival_offset_count": (zero_gemm_positive_arrival_offset_count),
        "source_arrival_sum_us": _rounded_audit_us(source_arrival_total),
        "materialized_arrival_sum_us": _rounded_audit_us(materialized_arrival_total),
        "arrival_signed_error_us": _rounded_audit_us(arrival_signed_error),
        "arrival_total_absolute_error_us": _rounded_audit_us(arrival_absolute_error_sum.value),
        "arrival_max_absolute_error_us": _rounded_audit_us(arrival_max_absolute_error),
        "total_absolute_error_us": _rounded_audit_us(total_absolute_error),
        "max_absolute_error_us": _rounded_audit_us(max_absolute_error),
        "gemm_entry_count": gemm_entry_count,
        "base_gemm_operation_count": base_gemm_operation_count,
        "rank_extra_gemm_operation_count": rank_extra_gemm_operation_count,
        "total_rank_gemm_operation_count": total_rank_gemm_operation_count,
        "max_rank_gemm_operation_count": max(rank_counts.values(), default=0),
        "rank_gemm_operation_counts": rank_counts,
    }


def canary_to_param_comms_trace(
    canary: Mapping[str, Any],
    *,
    dtype: Optional[str] = None,
    skip_unsupported: bool = False,
    compute_fill_us_per_gemm: Optional[float] = None,
    compute_fill_gemm_dim: int = 1024,
    compute_fill_dtype: Optional[str] = None,
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
        compute_fill_dtype=compute_fill_dtype,
        overlap_structure=overlap_structure,
        limits=limits,
        logical_event_iterator=iter_canary_logical_events,
    )


def export_param_comms_trace(
    canary: Mapping[str, Any],
    *,
    dtype: Optional[str] = None,
    skip_unsupported: bool = False,
    compute_fill_us_per_gemm: Optional[float] = None,
    compute_fill_gemm_dim: int = 1024,
    compute_fill_dtype: Optional[str] = None,
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

    With ``compute_fill_us_per_gemm`` set, inter-collective gaps and per-rank
    arrival offsets are exported as rank-aware GEMM entries instead of idle
    time. ``count`` is the common gap fill and ``rank_extra_counts`` binds each
    participating rank's additional readiness delay, both quantized by the
    calibrated per-GEMM duration. Communication-only replay fills gaps with
    silence and discards arrival skew, so it cannot reproduce workload-shaped
    compute/communication interference. Replay compute-filled traces WITHOUT
    ``--use-timestamp`` so pacing comes from compute rather than wall-clock
    sleeps. The per-GEMM duration is hardware- and dim-specific and must be
    calibrated on the target device.

    With ``overlap_structure`` additionally set, collectives are emitted for
    asynchronous issue and each source ``compute_overlap_us`` component bounds
    how many of the following gap's GEMMs may run before the explicit
    ``{"comms": "wait", "req": r}``. The remainder of that gap and all
    rank-arrival fill run after the wait, so a zero-overlap source cannot become
    an overlapping replay. A positive final overlap component is emitted as
    tail compute before the final wait. Traces whose overlap crosses the next
    communication start require a dependency graph and fail closed instead of
    being serialized into a different program. The pinned historical PARAM
    replayer is hardwired blocking and degrades this back to sequential
    execution; a conforming overlap-aware adapter is required. Issue entries carry an
    ``issue`` marker so latency parsers can distinguish issue lines from the
    completion-bearing wait lines. Requires compute fill.
    """

    dtype_override = require_param_dtype(dtype, label="PARAM export dtype") if dtype is not None else None
    fill_dtype = (
        require_param_compute_dtype(compute_fill_dtype, label="PARAM compute-fill dtype")
        if compute_fill_dtype is not None
        else None
    )
    fill_us = None
    if compute_fill_us_per_gemm is not None:
        fill_us = _require_compute_fill_us(compute_fill_us_per_gemm)
    gemm_dim = as_int(compute_fill_gemm_dim)
    if gemm_dim <= 0:
        raise SchemaError("compute_fill_gemm_dim must be positive")
    if gemm_dim > limits.max_param_gemm_dim:
        raise SchemaError(f"compute_fill_gemm_dim={gemm_dim} exceeds max_param_gemm_dim={limits.max_param_gemm_dim}")
    if overlap_structure and fill_us is None:
        raise SchemaError("overlap_structure requires compute_fill_us_per_gemm")
    if fill_dtype is not None and fill_us is None:
        raise SchemaError("compute_fill_dtype requires compute_fill_us_per_gemm")
    param_materialization_requirements(
        canary,
        dtype=dtype_override,
        require_event_dtype=False,
        skip_unsupported=skip_unsupported,
        limits=limits,
    )
    preflight_param_entry_count(
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
    compute_operation_count = 0
    pending_overlap: Optional[PendingOverlap] = None
    # Production motif expansion shallow-copies each stored child, so its
    # immutable-in-practice ranks and timing-sample objects retain identity
    # across repetitions. Cache only for that production iterator; injected
    # iterators remain fully dynamic for the historical test/adapter seam.
    cache_templates = logical_event_iterator is iter_canary_logical_events
    rank_cache: Optional[RankCache] = {} if cache_templates else None
    timing_cache: Optional[TimingCache] = {} if cache_templates else None
    marker_cache: Dict[Tuple[str, str, str], str] = {}
    for event in logical_event_iterator(
        canary.get("events", []),
        limits=limits,
    ):
        op = str(event.get("op"))
        ranks = _cached_ranks(event, rank_cache)
        event_dtype = param_event_dtype(
            event,
            dtype_override=dtype_override,
            require_event_dtype=False,
        )
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
        nelems = param_element_count(as_int(event.get("bytes")), event_dtype)
        in_size, out_size = param_message_sizes(op, nelems, len(ranks))
        for gap_us, arrival_offsets_us, overlap_us in _cached_expanded_timing_occurrences(
            event,
            timing_cache,
        ):
            if len(arrival_offsets_us) != len(ranks):
                raise SchemaError(
                    f"PARAM compute-fill arrival offsets for group {group!r} must match its process-group ranks"
                )
            clock_ns += max(0, round(gap_us * 1000.0))
            if fill_us is not None:
                base_count = _compute_fill_gemm_count(max(0.0, gap_us), fill_us)
                overlap_base_count = 0
                if overlap_structure and pending_overlap is not None:
                    (
                        pending_wait,
                        pending_overlap_us,
                        pending_ranks,
                        pending_group,
                        pending_op,
                        pending_dtype,
                    ) = pending_overlap
                    _require_overlap_fits_following_gap(
                        pending_overlap_us,
                        max(0.0, gap_us),
                    )
                    overlap_base_count = _compute_fill_gemm_count(
                        pending_overlap_us,
                        fill_us,
                    )
                    if overlap_base_count > base_count:
                        raise SchemaError("source-overlap GEMM count exceeds the following gap's total GEMM count")
                    if overlap_base_count > 0:
                        compute_operation_count = _bounded_compute_count(
                            compute_operation_count,
                            overlap_base_count * len(pending_ranks),
                            limit=limits.max_param_compute_operations,
                            label="PARAM compute-fill operations",
                        )
                        entries.append(
                            _compute_fill_entry(
                                count=overlap_base_count,
                                rank_extra_counts={str(rank): 0 for rank in pending_ranks},
                                ranks=pending_ranks,
                                dtype=fill_dtype or pending_dtype,
                                gemm_dim=gemm_dim,
                                request_id=request_id,
                                clock_ns=clock_ns,
                                marker=_cached_marker(
                                    marker_cache,
                                    "compute-fill-overlap",
                                    pending_group,
                                    pending_op,
                                ),
                                phase="source-overlap",
                                overlap_request=as_int(pending_wait["req"]),
                            )
                        )
                        request_id += 1
                    entries.append(pending_wait)
                    pending_overlap = None
                rank_extra_counts = {
                    str(rank): _compute_fill_gemm_count(
                        max(0.0, as_float(arrival_offset_us)),
                        fill_us,
                    )
                    for rank, arrival_offset_us in zip(ranks, arrival_offsets_us)
                }
                serialized_base_count = base_count - overlap_base_count
                rank_counts = [serialized_base_count + rank_extra_counts[str(rank)] for rank in ranks]
                physical_gemm_count = sum(rank_counts)
                if physical_gemm_count > 0:
                    compute_operation_count = _bounded_compute_count(
                        compute_operation_count,
                        physical_gemm_count,
                        limit=limits.max_param_compute_operations,
                        label="PARAM compute-fill operations",
                    )
                    entries.append(
                        _compute_fill_entry(
                            count=serialized_base_count,
                            rank_extra_counts=rank_extra_counts,
                            ranks=ranks,
                            dtype=fill_dtype or event_dtype,
                            gemm_dim=gemm_dim,
                            request_id=request_id,
                            clock_ns=clock_ns,
                            marker=_cached_marker(
                                marker_cache,
                                "compute-fill-serialized",
                                group,
                                op,
                            ),
                            phase="serialized-readiness",
                        )
                    )
                    request_id += 1
            elif overlap_structure and pending_overlap is not None:
                raise SchemaError("overlap_structure requires compute fill")
            if overlap_structure and op in PARAM_POINT_TO_POINT_OPS and overlap_us > 0.0:
                raise SchemaError(
                    "source-bounded overlap materialization does not support "
                    f"positive overlap on synchronous point-to-point op {op!r}"
                )
            if op in PARAM_POINT_TO_POINT_OPS:
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
            elif op in PARAM_COLLECTIVE_OP_NAMES:
                occurrence_entry: JsonDict = {"comms": PARAM_COLLECTIVE_OP_NAMES[op]}
                if op == "broadcast":
                    occurrence_entry["root"] = as_int(event.get("root_rank"))
                if op in {"all_reduce", "reduce_scatter"} and "reduction_op" in event:
                    occurrence_entry["reduction_op"] = str(event["reduction_op"])
                occurrence_entries = [occurrence_entry]
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
                        "dtype": event_dtype,
                        "markers": [_cached_marker(marker_cache, "complete", group, op)],
                    }
                )
                entries.append(entry)
                request_id += 1
            if overlap_structure and op in PARAM_COLLECTIVE_OP_NAMES:
                # The source overlap component, not the whole next gap,
                # determines how much compute may precede this wait.
                comm_entry = entries[-1]
                comm_entry["markers"] = [_cached_marker(marker_cache, "issue", group, op)]
                pending_overlap = (
                    {
                        "comms": "wait",
                        "req": comm_entry["req"],
                        "startTime_ns": clock_ns,
                        "markers": [_cached_marker(marker_cache, "complete", group, op)],
                    },
                    max(0.0, overlap_us),
                    ranks,
                    group,
                    op,
                    event_dtype,
                )
    if overlap_structure and pending_overlap is not None:
        (
            pending_wait,
            pending_overlap_us,
            pending_ranks,
            pending_group,
            pending_op,
            pending_dtype,
        ) = pending_overlap
        if fill_us is None:
            raise SchemaError("overlap_structure requires compute fill")
        tail_overlap_count = _compute_fill_gemm_count(pending_overlap_us, fill_us)
        if tail_overlap_count > 0:
            compute_operation_count = _bounded_compute_count(
                compute_operation_count,
                tail_overlap_count * len(pending_ranks),
                limit=limits.max_param_compute_operations,
                label="PARAM compute-fill operations",
            )
            entries.append(
                _compute_fill_entry(
                    count=tail_overlap_count,
                    rank_extra_counts={str(rank): 0 for rank in pending_ranks},
                    ranks=pending_ranks,
                    dtype=fill_dtype or pending_dtype,
                    gemm_dim=gemm_dim,
                    request_id=request_id,
                    clock_ns=clock_ns,
                    marker=_cached_marker(
                        marker_cache,
                        "compute-fill-overlap",
                        pending_group,
                        pending_op,
                    ),
                    phase="source-overlap",
                    overlap_request=as_int(pending_wait["req"]),
                )
            )
            request_id += 1
        entries.append(pending_wait)
        pending_overlap = None
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
        if "overlap_request" in entry:
            entry["overlap_request"] = as_int(entry["overlap_request"]) + offset
    return init_entries + entries


def write_param_comms_trace(path: str, entries: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write PARAM basic trace entries (a JSON array)."""

    atomic_write_json(
        path,
        list(entries),
        indent=1,
        policy=PARAM_TRACE_POLICY,
    )


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


def _expanded_timing_occurrences(
    event: Mapping[str, Any],
) -> Iterable[TimingOccurrence]:
    for sample in iter_canary_timing_samples(event):
        yield (
            as_float(sample.get("gap_us"), 0.0),
            tuple(timing_sample_offsets(event, sample)),
            as_float(sample.get("compute_overlap_us"), 0.0),
        )


def _cached_expanded_timing_occurrences(
    event: Mapping[str, Any],
    cache: Optional[TimingCache],
) -> Iterable[TimingOccurrence]:
    raw_samples = event.get("timing_samples")
    if cache is None or not isinstance(raw_samples, list):
        return _expanded_timing_occurrences(event)
    cache_key = id(raw_samples)
    cached = cache.get(cache_key)
    if cached is not None and cached[0] is raw_samples:
        return cached[1]
    occurrences = tuple(_expanded_timing_occurrences(event))
    cache[cache_key] = (raw_samples, occurrences)
    return occurrences


def _cached_marker(cache: Dict[Tuple[str, str, str], str], kind: str, group: str, op: str) -> str:
    key = (kind, group, op)
    marker = cache.get(key)
    if marker is not None:
        return marker
    if kind == "compute-fill-overlap":
        marker = f"commcanary:compute-fill:source-overlap:{group}"
    elif kind == "compute-fill-serialized":
        marker = f"commcanary:compute-fill:serialized-readiness:{group}"
    elif kind == "issue":
        marker = f"commcanary:issue:{group}:{op}"
    else:
        marker = f"commcanary:{group}:{op}"
    cache[key] = marker
    return marker


def _compute_fill_entry(
    *,
    count: int,
    rank_extra_counts: Mapping[str, int],
    ranks: Tuple[int, ...],
    dtype: str,
    gemm_dim: int,
    request_id: int,
    clock_ns: int,
    marker: str,
    phase: str,
    overlap_request: Optional[int] = None,
) -> JsonDict:
    entry: JsonDict = {
        "compute": "gemm",
        "compute_phase": phase,
        "mm_dim": gemm_dim,
        "count": count,
        "rank_extra_counts": dict(rank_extra_counts),
        "global_ranks": list(ranks),
        "dtype": dtype,
        "req": request_id,
        "startTime_ns": clock_ns,
        "markers": [marker],
    }
    if overlap_request is not None:
        entry["overlap_request"] = overlap_request
    return entry


class _StableSum:
    """Small compensated sum that remains streaming under large artifacts."""

    def __init__(self) -> None:
        self._total = 0.0
        self._compensation = 0.0

    def add(self, value: float, *, label: str) -> None:
        adjusted = value - self._compensation
        updated = self._total + adjusted
        self._compensation = (updated - self._total) - adjusted
        if not math.isfinite(updated):
            raise SchemaError(f"{label} is not finite")
        self._total = updated

    @property
    def value(self) -> float:
        return self._total


def _require_compute_fill_us(value: float) -> float:
    fill_us = as_float(value)
    if fill_us <= 0.0:
        raise SchemaError("compute_fill_us_per_gemm must be positive")
    return fill_us


def _compute_fill_gemm_count(gap_us: float, fill_us: float) -> int:
    ratio = max(0.0, as_float(gap_us)) / fill_us
    if not math.isfinite(ratio):
        raise SchemaError("compute-fill GEMM count is not finite")
    try:
        count = int(round(ratio))
    except (OverflowError, ValueError) as exc:
        raise SchemaError("compute-fill GEMM count exceeds the supported count range") from exc
    if count > MAX_CHECKED_COUNT:
        raise SchemaError("compute-fill GEMM count exceeds the supported count range")
    return count


def _require_overlap_fits_following_gap(overlap_us: float, following_gap_us: float) -> None:
    overlap = max(0.0, as_float(overlap_us))
    gap = max(0.0, as_float(following_gap_us))
    if overlap > gap:
        raise SchemaError(
            f"source collective overlap_us={overlap!r} exceeds the following "
            f"inter-communication start gap_us={gap!r}; this trace can contain "
            "pipelined in-flight collectives and requires an explicit dependency graph"
        )


def _materialized_component_us(count: int, fill_us: float, *, label: str) -> float:
    value = count * fill_us
    if not math.isfinite(value):
        raise SchemaError(f"{label} is not finite")
    return value


def _quantization_error(materialized_us: float, source_us: float, *, label: str) -> float:
    value = abs(materialized_us - source_us)
    if not math.isfinite(value):
        raise SchemaError(f"{label} is not finite")
    return value


def _bounded_compute_count(total: int, increment: int, *, limit: int, label: str) -> int:
    try:
        return require_within(
            checked_add(total, increment, label=label),
            limit,
            label=label,
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc


def _rounded_audit_us(value: float) -> float:
    rounded = round(value, 9)
    return 0.0 if rounded == 0.0 else rounded


__all__ = [
    "LogicalEventIterator",
    "audit_param_compute_fill_quantization",
    "audit_param_program_compute_operations",
    "canary_to_param_comms_trace",
    "export_param_comms_trace",
    "param_materialization_requirements",
    "write_param_comms_trace",
]
