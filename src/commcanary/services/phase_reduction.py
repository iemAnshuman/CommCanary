"""Phase-representative repetition reduction application service.

Structural reduction (``services.reduction``) removes events from *within* a
program.  This module removes *repetitions* of a program.  A sustained serving
or training workload is a long sequence drawn from a small number of recurring
structural phases; replaying a weighted representative of each phase is where
large physical speedups come from.

The reduction is deliberately conservative about the property the August 2026
Rostam campaign identified as load bearing: an iteration's signature includes
its compute/communication concurrency, so two iterations that issue identical
collectives under different overlap never collapse into one cluster.  Overlap is
required, not defaulted; an event that does not declare it is refused.

The output is a decision-preserving *candidate*.  Like ddmin reduction it is not
source verified and cannot receive a strong behavioral claim on its own.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_float, as_int
from ..errors import SchemaError
from ..formats import TRACE_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits

REDUCTION_METHOD = "phase_representative"

#: Continuous concurrency features are bucketed before they enter a signature so
#: that physically equivalent iterations cluster.  The bucket count is part of
#: the declared method: changing it changes which iterations are considered the
#: same phase, so it is recorded in the emitted reduction record.
DEFAULT_OVERLAP_BUCKETS = 16

#: A cluster is represented by its medoid plus the member sitting at this
#: quantile of iteration duration. Pure medoid selection preserves the median
#: and destroys the tail by construction -- measured at 58-80% under on p95/p99
#: -- which is disqualifying for a gate whose decision metric is p99 latency.
DEFAULT_TAIL_QUANTILE = 0.95

#: Selection order for a cluster's representative and for cluster ranking.  Both
#: are total and deterministic; no tie is resolved by dictionary order.
_REPRESENTATIVE_SELECTION = "medoid_by_bucketed_feature_distance_then_first_occurrence"
_CLUSTER_RANKING = "weight_then_first_occurrence"

_OVERLAP_FIELDS: Tuple[str, ...] = (
    "compute_overlap_us",
    "compute_before_us",
    "compute_pressure",
)


class Iteration(NamedTuple):
    """One repetition unit: a contiguous, ordered run of trace events."""

    ordinal: int
    start: int
    stop: int
    events: Tuple[Mapping[str, Any], ...]


class PhaseCluster(NamedTuple):
    """A set of iterations sharing one bucketed structural signature."""

    signature: str
    members: Tuple[int, ...]
    representative: int
    first_occurrence: int


class Selection(NamedTuple):
    """One retained iteration and how many source iterations it stands for."""

    iteration: int
    weight: int
    role: str


def _require_overlap(event: Mapping[str, Any], index: int) -> None:
    """Refuse an event that does not declare its concurrency.

    Compilation already refuses to turn unknown skew into zero because zero skew
    is a strong physical claim.  Zero overlap is the same kind of claim, and the
    Rostam campaign measured it as the load-bearing one, so it gets the same
    treatment here rather than a silent default.
    """

    for field in _OVERLAP_FIELDS:
        if event.get(field) is None:
            raise SchemaError(
                f"trace event {index} does not declare {field}; phase reduction refuses to "
                "assume zero concurrency (see services.phase_reduction)"
            )


def _structural_token(event: Mapping[str, Any]) -> str:
    """Return the exact-match part of an event's identity."""

    ranks = event.get("ranks")
    rank_count = len(ranks) if isinstance(ranks, Sequence) else as_int(event.get("rank_count", 0))
    parts = (
        str(event.get("op")),
        str(as_int(event.get("bytes"))),
        str(event.get("dtype", "")),
        str(event.get("group", "")),
        str(event.get("phase", "")),
        str(event.get("reduction_op", "")),
        str(rank_count),
    )
    return "|".join(parts)


def _bucket(value: float, low: float, high: float, buckets: int) -> int:
    if not (high > low):
        return 0
    position = (value - low) / (high - low)
    if position <= 0.0:
        return 0
    if position >= 1.0:
        return buckets - 1
    return int(position * buckets)


def _arrival_span_us(event: Mapping[str, Any]) -> float:
    arrivals = event.get("rank_arrival_us")
    if not isinstance(arrivals, Mapping) or not arrivals:
        return 0.0
    values = [as_float(entry) for entry in arrivals.values()]
    return max(values) - min(values)


def _iteration_features(iteration: Iteration) -> Tuple[float, ...]:
    """Continuous concurrency features that survive into the signature."""

    overlap = 0.0
    before = 0.0
    pressure = 0.0
    groups = 0.0
    skew = 0.0
    for event in iteration.events:
        overlap += as_float(event.get("compute_overlap_us"))
        before += as_float(event.get("compute_before_us"))
        pressure += as_float(event.get("compute_pressure"))
        groups += float(as_int(event.get("concurrent_groups", 1)))
        skew = max(skew, _arrival_span_us(event))
    return (overlap, before, pressure, groups, skew)


def declared_boundaries(events: Sequence[Mapping[str, Any]]) -> Optional[List[int]]:
    """Return iteration start offsets declared by the capture, if any.

    The repetition unit is a property of the workload, not of the trace, and the
    producer is the only party that knows it.  A capture that emits
    ``iteration_index`` gets segmented exactly; everything else has to fall back
    to inference, which is strictly worse.
    """

    if not any("iteration_index" in event for event in events):
        return None
    boundaries: List[int] = []
    previous: Optional[int] = None
    for index, event in enumerate(events):
        if "iteration_index" not in event:
            raise SchemaError(
                f"trace event {index} is missing 'iteration_index' while other events declare it; "
                "phase reduction refuses a partially declared segmentation"
            )
        current = as_int(event.get("iteration_index"))
        if previous is None or current != previous:
            boundaries.append(index)
            previous = current
    return boundaries


def inferred_boundaries(
    events: Sequence[Mapping[str, Any]],
    *,
    gap_ratio: float,
) -> List[int]:
    """Infer iteration starts from pre-collective compute gaps.

    An engine step ends with work no layer performs -- sampling, scheduling,
    detokenisation -- so the compute preceding the first collective of a step is
    materially larger than the compute preceding an interior layer.  A boundary
    is declared where ``compute_before_us`` exceeds ``gap_ratio`` times the
    median.  This is a heuristic and is recorded as one in the emitted record.
    """

    gaps = [as_float(event.get("compute_before_us")) for event in events]
    ordered = sorted(gaps)
    median = ordered[len(ordered) // 2]
    if median <= 0.0:
        raise SchemaError("cannot infer iteration boundaries: median compute_before_us is not positive")
    threshold = median * gap_ratio
    boundaries = [0]
    boundaries.extend(index for index in range(1, len(events)) if gaps[index] > threshold)
    return boundaries


def segment_iterations(
    events: Sequence[Mapping[str, Any]],
    *,
    period: Optional[int] = None,
    infer_boundaries: bool = False,
    gap_ratio: float = 4.0,
) -> Tuple[List[Iteration], str]:
    """Split an ordered event list into contiguous repetition units.

    Resolution order is declared segmentation, then an explicit period, then --
    only on request -- gap inference.  There is deliberately no automatic
    structural fallback.  Every compression objective over the collective stream
    alone is maximised by shredding the stream into single events, because a
    ranking is a projection that survives almost any subset; that is the
    mechanism behind the campaign's tau=0.204 result.  Guessing the repetition
    unit from the data would therefore optimise for exactly the artifact this
    module exists to avoid, so an undeclared segmentation is refused.
    """

    if not events:
        raise SchemaError("cannot segment an empty trace")

    declared = declared_boundaries(events)
    if declared is not None:
        starts, source = declared, "declared_iteration_index"
    elif period is not None:
        resolved = as_int(period)
        if resolved <= 0:
            raise SchemaError("iteration period must be positive")
        if resolved > len(events):
            raise SchemaError("iteration period cannot exceed the event count")
        starts, source = list(range(0, len(events), resolved)), "explicit_period"
    elif infer_boundaries:
        starts, source = inferred_boundaries(events, gap_ratio=gap_ratio), "inferred_compute_gap"
    else:
        raise SchemaError(
            "phase reduction cannot determine the repetition unit: the trace declares no "
            "'iteration_index', no period was supplied, and boundary inference was not requested. "
            "Refusing to infer it from event structure, which is maximised by discarding structure."
        )

    iterations: List[Iteration] = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(events)
        iterations.append(
            Iteration(
                ordinal=index,
                start=start,
                stop=stop,
                events=tuple(events[start:stop]),
            )
        )
    return iterations, source


def _signatures(
    iterations: Sequence[Iteration],
    *,
    buckets: int,
) -> Tuple[List[str], List[Tuple[int, ...]]]:
    """Return one bucketed signature and bucket vector per iteration."""

    features = [_iteration_features(iteration) for iteration in iterations]
    dimensions = len(features[0]) if features else 0
    ranges: List[Tuple[float, float]] = []
    for dimension in range(dimensions):
        column = [row[dimension] for row in features]
        ranges.append((min(column), max(column)))
    signatures: List[str] = []
    vectors: List[Tuple[int, ...]] = []
    for iteration, row in zip(iterations, features):
        vector = tuple(
            _bucket(row[dimension], ranges[dimension][0], ranges[dimension][1], buckets)
            for dimension in range(dimensions)
        )
        structure = ",".join(_structural_token(event) for event in iteration.events)
        signatures.append(structure + "#" + ",".join(str(value) for value in vector))
        vectors.append(vector)
    return signatures, vectors


def _iteration_span_us(iteration: Iteration) -> float:
    """Wall-clock span of one iteration on the source timeline."""

    starts = [as_float(event.get("start_us")) for event in iteration.events]
    return max(starts) - min(starts) if starts else 0.0


def rebase_timeline(
    iterations: Sequence[Iteration],
    selected: Sequence[int],
    *,
    cadence_us: Union[float, Mapping[int, float]],
) -> List[JsonDict]:
    """Re-anchor retained iterations onto a contiguous timeline.

    Dropping iterations leaves holes in absolute ``start_us``.  Replay models
    queueing from inter-arrival time, so a holed timeline presents fictitious
    idle between surviving iterations and the artifact stops being comparable to
    its source.  Each retained iteration therefore keeps its exact *internal*
    offsets and is re-anchored at the source's median iteration cadence.
    """

    rebased: List[JsonDict] = []
    cursor = 0.0
    for iteration_index in selected:
        step = cadence_us if isinstance(cadence_us, float) else float(cadence_us[iteration_index])
        iteration = iterations[iteration_index]
        origin = min(as_float(event.get("start_us")) for event in iteration.events)
        for event in iteration.events:
            copied = copy.deepcopy(dict(event))
            copied["start_us"] = cursor + (as_float(event.get("start_us")) - origin)
            rebased.append(copied)
        cursor += step
    return rebased


def allocate_iterations(
    clusters: Sequence[PhaseCluster],
    iterations: Sequence[Iteration],
    *,
    budget: int,
    tail_quantile: float,
) -> List[Selection]:
    """Sample ``budget`` iterations while preserving each phase's proportion.

    Percentiles are proportions, so an artifact only reproduces them if its
    phase mix matches the source's. Two earlier heuristics both failed on the
    local probe for the same underlying reason -- they optimised for typicality
    or for extremity instead of for the mix:

    * medoid-only selection held the median to within 17% and put p95/p99
      58-80% *under*, because a medoid is by construction not a tail; and
    * reserving budget for the longest-iteration clusters recovered p99 to
      within 17% but pushed p95 80% and the median 237% *over*, because prefill
      then occupied a quarter of the artifact instead of its true 3.2%.

    Slots are therefore allocated across clusters by largest remainder, with one
    slot floored for any cluster holding an iteration at or beyond
    ``tail_quantile`` of the global span distribution so a rare phase is not
    rounded away. Within a cluster the chosen members are spread across that
    cluster's own duration order rather than all taken from the middle.

    This bounds what reduction can achieve, and the bound is a property of the
    workload rather than of the method: preserving a q-quantile needs enough
    retained iterations that the tail phase still appears in it, so a decision
    metric of p99 over a phase occupying 3% of the workload cannot be estimated
    from a handful of iterations however they are chosen.
    """

    if budget <= 0:
        raise SchemaError("max_representatives must be positive")
    if not 0.0 < tail_quantile < 1.0:
        raise SchemaError("tail_quantile must be strictly between 0 and 1")

    spans = [_iteration_span_us(iteration) for iteration in iterations]
    ordered_spans = sorted(spans)
    cut = ordered_spans[min(len(ordered_spans) - 1, int(len(ordered_spans) * tail_quantile))]
    total = sum(len(cluster.members) for cluster in clusters)

    exact = [(cluster, len(cluster.members) * budget / total) for cluster in clusters]
    allocation: Dict[str, int] = {}
    for cluster, share in exact:
        carries_tail = any(spans[member] >= cut for member in cluster.members)
        allocation[cluster.signature] = max(1 if carries_tail else 0, int(share))

    # Largest remainder, then trim from the largest clusters if the floors
    # pushed the total over budget. A tail-carrying cluster never drops to zero.
    while sum(allocation.values()) < budget:
        cluster = max(
            exact,
            key=lambda row: (row[1] - allocation[row[0].signature], -row[0].first_occurrence),
        )[0]
        allocation[cluster.signature] += 1
    while sum(allocation.values()) > budget:
        candidates = [
            cluster
            for cluster, _share in exact
            if allocation[cluster.signature] > 1
            or (allocation[cluster.signature] == 1 and not any(spans[m] >= cut for m in cluster.members))
        ]
        if not candidates:
            break
        cluster = max(candidates, key=lambda item: (allocation[item.signature], -item.first_occurrence))
        allocation[cluster.signature] -= 1

    selections: List[Selection] = []
    for cluster in clusters:
        slots = allocation.get(cluster.signature, 0)
        if slots <= 0:
            continue
        members = sorted(cluster.members, key=lambda index: (spans[index], index))
        picks = [
            members[min(len(members) - 1, int((position + 0.5) * len(members) / slots))] for position in range(slots)
        ]
        picks = sorted(dict.fromkeys(picks))
        base, extra = divmod(len(cluster.members), len(picks))
        for position, member in enumerate(picks):
            weight = base + (1 if position < extra else 0)
            role = "tail" if spans[member] >= cut else "typical"
            selections.append(Selection(iteration=member, weight=max(1, weight), role=role))
    return sorted(selections, key=lambda selection: selection.iteration)


def cluster_cadences(
    iterations: Sequence[Iteration],
    clusters: Sequence[PhaseCluster],
) -> Dict[int, float]:
    """Median start-to-start delta for the iterations of each cluster.

    A single global cadence packs long iterations at short-iteration spacing;
    on the local probe that inflated replayed latency by 827% once enough
    representatives were retained. Each cluster carries its own instead.
    """

    origins = [min(as_float(event.get("start_us")) for event in iteration.events) for iteration in iterations]
    fallback = _median([origins[index + 1] - origins[index] for index in range(len(origins) - 1)])
    cadences: Dict[int, float] = {}
    for cluster in clusters:
        deltas = [origins[member + 1] - origins[member] for member in cluster.members if member + 1 < len(origins)]
        cadence = _median(deltas) if deltas else fallback
        for member in cluster.members:
            cadences[member] = cadence if cadence > 0.0 else fallback
    return cadences


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _select_representative(members: Sequence[int], vectors: Sequence[Tuple[int, ...]]) -> int:
    """Return the medoid member, breaking ties by first occurrence."""

    best_member = members[0]
    best_distance: Optional[int] = None
    for candidate in members:
        distance = 0
        for other in members:
            distance += sum(abs(left - right) for left, right in zip(vectors[candidate], vectors[other]))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_member = candidate
    return best_member


def cluster_iterations(
    iterations: Sequence[Iteration],
    *,
    overlap_buckets: int = DEFAULT_OVERLAP_BUCKETS,
) -> List[PhaseCluster]:
    """Group iterations into phases by exact bucketed signature."""

    if overlap_buckets < 2:
        raise SchemaError("overlap_buckets must be at least 2")
    signatures, vectors = _signatures(iterations, buckets=overlap_buckets)
    grouped: Dict[str, List[int]] = {}
    for index, signature in enumerate(signatures):
        grouped.setdefault(signature, []).append(index)
    clusters = [
        PhaseCluster(
            signature=signature,
            members=tuple(members),
            representative=_select_representative(members, vectors),
            first_occurrence=members[0],
        )
        for signature, members in grouped.items()
    ]
    clusters.sort(key=lambda cluster: (-len(cluster.members), cluster.first_occurrence))
    return clusters


def phase_representative_reduction(
    trace: Mapping[str, Any],
    *,
    period: Optional[int] = None,
    infer_boundaries: bool = False,
    gap_ratio: float = 4.0,
    max_representatives: Optional[int] = None,
    overlap_buckets: int = DEFAULT_OVERLAP_BUCKETS,
    tail_quantile: float = DEFAULT_TAIL_QUANTILE,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Reduce a trace to weighted representatives of its recurring phases.

    Each retained iteration is a contiguous run of the original events, so
    intra-iteration ordering, per-rank arrival skew, and compute/communication
    overlap are carried through unmodified.  Weights record how many source
    iterations each representative stands for, which is what lets a caller
    extrapolate a full-run cost estimate from a reduced physical measurement.
    """

    validate_trace(trace, require_known_overlap=True, limits=limits)
    events = list(trace.get("events", []))
    if not events:
        raise SchemaError("cannot reduce an empty trace")
    for index, event in enumerate(events):
        _require_overlap(event, index)

    iterations, segmentation_source = segment_iterations(
        events,
        period=period,
        infer_boundaries=infer_boundaries,
        gap_ratio=gap_ratio,
    )
    clusters = cluster_iterations(iterations, overlap_buckets=overlap_buckets)

    budget = len(clusters) if max_representatives is None else as_int(max_representatives)
    if budget <= 0:
        raise SchemaError("max_representatives must be positive")
    selections = allocate_iterations(clusters, iterations, budget=budget, tail_quantile=tail_quantile)

    weights: Dict[int, int] = {}
    roles: Dict[int, str] = {}
    for selection in selections:
        weights[selection.iteration] = weights.get(selection.iteration, 0) + selection.weight
        roles[selection.iteration] = selection.role
    folds: List[JsonDict] = []

    selected = sorted(weights)
    representatives: List[JsonDict] = []
    for iteration_index in selected:
        iteration = iterations[iteration_index]
        representatives.append(
            {
                "iteration_index": iteration.ordinal,
                "source_event_start": iteration.start,
                "source_event_stop": iteration.stop,
                "events": len(iteration.events),
                "weight_iterations": weights[iteration_index],
                "role": roles.get(iteration_index, "typical"),
                "span_us": _iteration_span_us(iteration),
            }
        )

    cadences = cluster_cadences(iterations, clusters)
    reduced_events = rebase_timeline(iterations, selected, cadence_us=cadences)

    workload = copy.deepcopy(dict(trace.get("workload", {})))
    notes = str(workload.get("notes", ""))
    suffix = (
        "CommCanary research reduction: phase_representative. Weighted phase "
        "representatives; not source-verified against the original trace."
    )
    workload["notes"] = f"{notes} {suffix}".strip()
    workload["reduction_method"] = REDUCTION_METHOD
    workload["reduction"] = {
        "method": REDUCTION_METHOD,
        "original_events": len(events),
        "reduced_events": len(reduced_events),
        "original_iterations": len(iterations),
        "reduced_iterations": len(selected),
        "segmentation_source": segmentation_source,
        "segmentation_gap_ratio": gap_ratio if segmentation_source == "inferred_compute_gap" else None,
        "timeline": "rebased_to_per_cluster_median_cadence",
        "tail_quantile": tail_quantile,
        "overlap_buckets": overlap_buckets,
        "max_representatives": budget,
        "representative_selection": _REPRESENTATIVE_SELECTION,
        "cluster_ranking": _CLUSTER_RANKING,
        "signature_fields": [
            "op",
            "bytes",
            "dtype",
            "group",
            "phase",
            "reduction_op",
            "rank_count",
            "compute_overlap_us",
            "compute_before_us",
            "compute_pressure",
            "concurrent_groups",
            "rank_arrival_span_us",
        ],
        "representatives": representatives,
        "folded_clusters": folds,
        "weight_total_iterations": sum(weights.values()),
    }
    reduced_trace: JsonDict = {
        "format": TRACE_FORMAT,
        "workload": workload,
        "system": copy.deepcopy(dict(trace.get("system", {}))),
        "events": reduced_events,
    }
    validate_trace(reduced_trace, require_known_overlap=True, limits=limits)
    return reduced_trace


__all__ = [
    "DEFAULT_OVERLAP_BUCKETS",
    "DEFAULT_TAIL_QUANTILE",
    "Selection",
    "REDUCTION_METHOD",
    "Iteration",
    "PhaseCluster",
    "cluster_cadences",
    "cluster_iterations",
    "allocate_iterations",
    "rebase_timeline",
    "declared_boundaries",
    "inferred_boundaries",
    "phase_representative_reduction",
    "segment_iterations",
]
