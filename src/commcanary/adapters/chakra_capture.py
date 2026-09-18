"""Automatic projection from verified CommCanary/Kineto traces to Chakra ET.

The adapter emits dependency-closed overlap windows, not a claim that every
opaque application operator was reconstructed. Each collective window is an
independent executable unit: the collective is issued first, contiguous GEMMs
that ran before its explicit wait remain independent of that collective node,
and their source order is retained. The physical runner flushes outstanding
work at region boundaries.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from ..artifacts.chakra import ChakraNodeRecord, decode_chakra_execution_trace, encode_chakra_execution_trace
from ..artifacts.dtypes import dtype_size_bytes
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.physical_canary import SUPPORTED_PHYSICAL_DOMAIN, with_content_identity
from ..artifacts.trace import validate_trace
from ..artifacts.wire import JsonDict, as_float, as_int, normalize_ranks
from ..errors import SchemaError
from ..formats import CHAKRA_PROJECTION_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits

CAPTURE_ADAPTER = "commcanary-kineto-to-chakra-overlap-windows.v1"
CHAKRA_METADATA_VERSION = "0.0.4"


@dataclass(frozen=True)
class ChakraCaptureResult:
    """Exact Chakra bytes and their executable semantic projection."""

    chakra_et: bytes
    projection: JsonDict


def commcanary_trace_to_chakra(
    trace: Mapping[str, Any],
    *,
    opaque_attributes_reviewed: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> ChakraCaptureResult:
    """Convert a complete all-rank trace into executable Chakra windows."""

    validate_trace(trace, limits=limits)
    raw_events = trace.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise SchemaError("physical Chakra capture requires at least one trace event")

    source_sha256 = hashlib.sha256(canonical_json_bytes(trace)).hexdigest()
    event_features = _event_features(raw_events)
    records: List[ChakraNodeRecord] = []
    projected_nodes: List[JsonDict] = []
    regions: List[JsonDict] = []
    next_node_id = 1
    reviewed_disclosures: Set[str] = set()

    for event_index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise SchemaError(f"trace event {event_index} must be an object")
        if raw_event.get("op") != "all_reduce":
            operation = str(raw_event.get("op", "unknown")).replace(" ", "_")
            raise SchemaError(f"unsupported_for_physical_canary: reason: {operation}_not_qualified")
        ranks = tuple(normalize_ranks(raw_event.get("ranks")))
        if len(ranks) < 2:
            raise SchemaError(f"trace event {event_index} all-reduce must contain at least two ranks")
        dtype = str(raw_event.get("dtype"))
        tensor_bytes = as_int(raw_event.get("bytes"))
        if tensor_bytes < 1:
            raise SchemaError(f"trace event {event_index} all-reduce tensor bytes must be positive")
        reduction = raw_event.get("reduction_op")
        if reduction != "sum":
            raise SchemaError(f"unsupported_for_physical_canary: reason: all_reduce_{str(reduction)}_not_qualified")

        region_node_ids: List[int] = []
        features = event_features[event_index]
        phase = str(raw_event.get("phase", "inference"))
        collective_id = next_node_id
        next_node_id += 1
        records.append(
            ChakraNodeRecord(
                node_id=collective_id,
                name=f"event-{event_index:06d}-all-reduce",
                node_type=7,
                int64_attributes=(("comm_type", 0), ("comm_size", tensor_bytes)),
            )
        )
        collective_disclosures = ("parallelism_degree",)
        reviewed_disclosures.update(collective_disclosures)
        projected_nodes.append(
            {
                "node_id": collective_id,
                "chakra_node_type": 7,
                "operation": "all_reduce",
                "executed_flops": 0,
                "tensor_bytes": tensor_bytes,
                "phase": phase,
                "feature_tags": list(features),
                "disclosures": list(collective_disclosures),
                "execution": {
                    "dtype": dtype,
                    "ranks": list(ranks),
                    "reduction": "sum",
                },
            }
        )
        region_node_ids.append(collective_id)

        recipes = _rank_recipes(raw_event, ranks=ranks, event_index=event_index)
        recipe_count = len(next(iter(recipes.values())))
        previous_gemm_id = 0
        for recipe_index in range(recipe_count):
            rank_recipes = []
            executed_flops = 0
            touched_bytes = 0
            for rank in ranks:
                recipe = recipes[rank][recipe_index]
                m = as_int(recipe.get("m"))
                n = as_int(recipe.get("n"))
                k = as_int(recipe.get("k"))
                recipe_dtype = str(recipe.get("dtype"))
                if min(m, n, k) < 1 or recipe_dtype != dtype:
                    raise SchemaError(
                        f"trace event {event_index} rank {rank} GEMM recipe {recipe_index} is incompatible"
                    )
                element_bytes = dtype_size_bytes(recipe_dtype)
                executed_flops += 2 * m * n * k
                touched_bytes += element_bytes * (m * k + k * n + m * n)
                rank_recipes.append({"rank": rank, "m": m, "n": n, "k": k})
            gemm_id = next_node_id
            next_node_id += 1
            dependencies = () if previous_gemm_id == 0 else (previous_gemm_id,)
            records.append(
                ChakraNodeRecord(
                    node_id=gemm_id,
                    name=f"event-{event_index:06d}-gemm-{recipe_index:04d}",
                    node_type=4,
                    control_dependencies=dependencies,
                )
            )
            gemm_disclosures = ("batch_token_geometry", "hidden_dimension")
            reviewed_disclosures.update(gemm_disclosures)
            projected_nodes.append(
                {
                    "node_id": gemm_id,
                    "chakra_node_type": 4,
                    "operation": "gemm",
                    "executed_flops": executed_flops,
                    "tensor_bytes": touched_bytes,
                    "phase": phase,
                    "feature_tags": sorted(set(features).union({"overlap"})),
                    "disclosures": list(gemm_disclosures),
                    "execution": {
                        "dtype": dtype,
                        "rank_recipes": rank_recipes,
                    },
                }
            )
            region_node_ids.append(gemm_id)
            previous_gemm_id = gemm_id

        region_features = sorted(
            {tag for node in projected_nodes if int(node["node_id"]) in region_node_ids for tag in node["feature_tags"]}
        )
        region_disclosures = sorted(
            {
                disclosure
                for node in projected_nodes
                if int(node["node_id"]) in region_node_ids
                for disclosure in node["disclosures"]
            }
        )
        regions.append(
            {
                "region_id": f"event-{event_index:06d}",
                "node_ids": region_node_ids,
                "feature_tags": region_features,
                "disclosures": region_disclosures,
            }
        )

    chakra_et = encode_chakra_execution_trace(CHAKRA_METADATA_VERSION, records)
    decoded = decode_chakra_execution_trace(chakra_et, limits=limits)
    raw_projection: JsonDict = {
        "format": CHAKRA_PROJECTION_FORMAT,
        "source_et": {
            "sha256": decoded.source_sha256,
            "bytes": decoded.source_bytes,
            "metadata_version": decoded.metadata_version,
        },
        "capture_provenance": {
            "adapter": CAPTURE_ADAPTER,
            "source_format": str(trace.get("format")),
            "source_sha256": source_sha256,
        },
        "supported_domain": SUPPORTED_PHYSICAL_DOMAIN,
        "nodes": projected_nodes,
        "regions": regions,
        "privacy_review": {
            "opaque_chakra_attributes_reviewed": opaque_attributes_reviewed,
            "reviewed_disclosures": sorted(reviewed_disclosures),
        },
    }
    projection = with_content_identity(raw_projection, "projection_id")
    return ChakraCaptureResult(chakra_et=chakra_et, projection=projection)


def _rank_recipes(
    event: Mapping[str, Any],
    *,
    ranks: Tuple[int, ...],
    event_index: int,
) -> Dict[int, Sequence[Mapping[str, Any]]]:
    raw_by_rank = event.get("compute_recipe_by_rank")
    if raw_by_rank is None:
        shared = event.get("compute_recipe")
        if not isinstance(shared, list):
            raise SchemaError(
                f"unsupported_for_physical_canary: reason: event_{event_index}_missing_complete_gemm_recipe"
            )
        raw_by_rank = {str(rank): shared for rank in ranks}
    if not isinstance(raw_by_rank, Mapping) or set(raw_by_rank) != {str(rank) for rank in ranks}:
        raise SchemaError(f"trace event {event_index} compute recipes must cover every participating rank")
    result: Dict[int, Sequence[Mapping[str, Any]]] = {}
    counts = set()
    for rank in ranks:
        raw_recipes = raw_by_rank[str(rank)]
        if not isinstance(raw_recipes, list) or not raw_recipes:
            raise SchemaError(f"trace event {event_index} rank {rank} has no executable GEMM recipe")
        recipes: List[Mapping[str, Any]] = []
        for recipe_index, raw_recipe in enumerate(raw_recipes):
            if not isinstance(raw_recipe, Mapping) or raw_recipe.get("op") != "gemm":
                raise SchemaError(f"trace event {event_index} rank {rank} recipe {recipe_index} is not a GEMM")
            recipes.append(raw_recipe)
        result[rank] = recipes
        counts.add(len(recipes))
    if len(counts) != 1:
        raise SchemaError(f"trace event {event_index} ranks have different GEMM recipe counts")
    return result


def _event_features(events: Sequence[Any]) -> Tuple[Tuple[str, ...], ...]:
    sizes = [as_int(event.get("bytes")) if isinstance(event, Mapping) else 0 for event in events]
    starts = [as_float(event.get("start_us")) if isinstance(event, Mapping) else 0.0 for event in events]
    gaps = [max(0.0, starts[index] - starts[index - 1]) for index in range(1, len(starts))]
    rare_tail_boundary = _quantile(sizes, 0.90)
    burst_boundary = _quantile(gaps, 0.50) if gaps else 0.0
    reset_boundary = _quantile(gaps, 0.90) if gaps else math.inf
    rows: List[Tuple[str, ...]] = []
    for index, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping):
            rows.append(())
            continue
        tags = {"collective-window"}
        phase = str(raw_event.get("phase", "inference")).strip().lower().replace(" ", "-")
        if phase:
            tags.add(f"phase-{phase}")
        # An event that never declared its overlap is not an event measured at
        # zero overlap. Collapsing the two tags a real workload as genuinely
        # non-overlapping, which is the configuration the Rostam campaign
        # measured agreeing with the full workload on only 16 of 28 pairs.
        overlap_raw = raw_event.get("compute_overlap_us")
        has_recipe = bool(raw_event.get("compute_recipe_by_rank"))
        if has_recipe or (overlap_raw is not None and as_float(overlap_raw) > 0.0):
            tags.add("overlap")
        elif overlap_raw is None:
            tags.add("overlap-unknown")
        arrivals = raw_event.get("rank_arrival_us")
        if isinstance(arrivals, Mapping) and arrivals:
            values = [as_float(value) for value in arrivals.values()]
            if max(values) - min(values) > 0.0:
                tags.add("rank-skew")
        elif as_float(raw_event.get("arrival_skew_us"), 0.0) > 0.0:
            tags.add("rank-skew")
        if sizes[index] >= rare_tail_boundary:
            tags.add("rare-tail")
        if index > 0 and gaps[index - 1] <= burst_boundary:
            tags.add("burst")
        if index > 0 and gaps[index - 1] >= reset_boundary and gaps[index - 1] > 0.0:
            tags.add("queue-reset")
        rows.append(tuple(sorted(tags)))
    return tuple(rows)


def _quantile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = ["CAPTURE_ADAPTER", "CHAKRA_METADATA_VERSION", "ChakraCaptureResult", "commcanary_trace_to_chakra"]
