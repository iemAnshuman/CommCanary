"""Deterministic, on-demand fixtures for scale benchmarks.

Large fixtures are generated rather than checked into the repository.  The
manifest contains both raw file hashes and semantic counts so a benchmark run
cannot silently consume a stale or partially regenerated fixture set.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from commcanary.compiler import compile_trace
from commcanary.formats import TRACE_FORMAT
from commcanary.schema import (
    JsonDict,
    as_int,
    canary_artifact_provenance_sha256,
    canonical_json_bytes,
    preflight_canary_expansion,
    validate_canary,
    validate_trace,
    write_json,
)

FIXTURE_MANIFEST_FORMAT = "commcanary.benchmark-fixtures.v1"
FIXTURE_GENERATOR_VERSION = "1"
STANDARD_STORED_EVENT_COUNTS = (1_000, 10_000, 100_000)
STANDARD_COMPRESSED_LOGICAL_COUNTS = (10_000, 100_000)
FIXED_CREATED_AT = "2000-01-01T00:00:00+00:00"


def generate_trace(stored_events: int, *, motif_pattern_length: int = 0) -> JsonDict:
    """Return a deterministic valid trace with exactly ``stored_events`` events.

    ``motif_pattern_length`` selects a short repeating sequence of distinct
    operation signatures.  Values from 2 through 16 are suitable inputs for
    the compiler's sequence-motif encoding.  Zero produces the general scale
    fixture, whose fields vary deterministically across a wider period.
    """

    count = _positive_count(stored_events, "stored_events")
    if motif_pattern_length != 0 and not 2 <= motif_pattern_length <= 16:
        raise ValueError("motif_pattern_length must be zero or between 2 and 16")

    events = []
    for index in range(count):
        pattern_index = index % motif_pattern_length if motif_pattern_length else index
        if motif_pattern_length:
            size_multiplier = pattern_index + 1
            gap_us = float(8 + pattern_index)
            skew_us = float(2 + pattern_index)
            overlap_us = float(3 + pattern_index)
            group = "motif-tp"
            phase = "prefill" if pattern_index == 0 else "decode"
        else:
            size_multiplier = 1 + (pattern_index % 8)
            gap_us = float(8 + (pattern_index * 7) % 29)
            skew_us = float((pattern_index * 11) % 23)
            overlap_us = float((pattern_index * 5) % 17)
            group = f"tp{pattern_index % 4}"
            phase = "decode" if index % 5 else "prefill"
        events.append(
            {
                "id": f"event-{index:06d}",
                "phase": phase,
                "op": "all_reduce",
                "bytes": 16_384 * size_multiplier,
                "ranks": 4,
                "group": group,
                "gap_us": gap_us,
                "arrival_skew_us": skew_us,
                "compute_overlap_us": overlap_us,
            }
        )
    trace: JsonDict = {
        "format": TRACE_FORMAT,
        "workload": {
            "name": "commcanary-deterministic-benchmark",
            "generator_version": FIXTURE_GENERATOR_VERSION,
            "stored_events": count,
            "motif_pattern_length": motif_pattern_length,
        },
        "system": {"backend": "deterministic-simulation"},
        "events": events,
    }
    validate_trace(trace)
    return trace


def generate_compressed_canary(logical_events: int, *, motif_pattern_length: int = 2) -> JsonDict:
    """Compile a deterministic canary with a large compact logical expansion."""

    count = _positive_count(logical_events, "logical_events")
    if count < motif_pattern_length * 2:
        raise ValueError("logical_events must contain at least two complete motif repetitions")
    trace = generate_trace(count, motif_pattern_length=motif_pattern_length)
    canary = compile_trace(trace, timing_sample_limit=2, enable_sequence_motifs=True)

    # ``created_at`` is descriptive rather than semantic.  Pinning it makes the
    # generated artifact byte-for-byte reproducible; all protected commitments
    # and self-referential size fields are refreshed before validation.
    canary["created_at"] = FIXED_CREATED_AT
    compiler = canary.get("compiler")
    if not isinstance(compiler, dict):
        raise RuntimeError("compiler produced a canary without compiler metadata")
    compiler["artifact_provenance_sha256"] = canary_artifact_provenance_sha256(canary)
    _refresh_size_metrics(canary, compiler)
    validate_canary(canary)

    expansion = preflight_canary_expansion(canary.get("events", []))
    if expansion.logical_events != count:
        raise RuntimeError(
            f"compressed fixture expanded to {expansion.logical_events} events instead of requested {count}"
        )
    if expansion.stored_events >= expansion.logical_events:
        raise RuntimeError("compressed fixture did not reduce stored event records")
    return canary


def generate_behavior_search_trace(stored_events: int) -> JsonDict:
    """Return a compressible trace dedicated to behavior-search benchmarks.

    Fixture generation is intentionally outside the measured region.  The
    two-record motif gives the search real compile/replay/verification work at
    1K through 100K scale without making fixture construction or random input
    generation part of the timing.
    """

    count = _positive_count(stored_events, "stored_events")
    if count < 4:
        raise ValueError("behavior-search fixtures require at least four events")
    return generate_trace(count, motif_pattern_length=2)


def materialize_capture_shards(
    trace: Mapping[str, Any],
    output_dir: Path,
    *,
    shard_count: int = 2,
) -> Tuple[Path, ...]:
    """Prepare deterministic, valid rank-local shards for merge benchmarks.

    Events are partitioned rather than duplicated, so the total number of
    captured records remains equal to the source fixture size.  Each prepared
    event is a complete single-rank collective occurrence with a stable session
    and collective identity.  This exercises multi-file loading, validation,
    bucketing, clock alignment, ordering, and final trace validation.
    """

    validate_trace(trace)
    count = _positive_count(shard_count, "shard_count")
    raw_events = trace.get("events")
    if not isinstance(raw_events, list) or len(raw_events) < count:
        raise ValueError("capture shard preparation needs at least one event per shard")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    session_id = "commcanary-benchmark-capture-v1"
    buckets: List[List[JsonDict]] = [[] for _ in range(count)]
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise ValueError(f"trace event {index} must be an object")
        rank = index % count
        event = copy.deepcopy(dict(raw_event))
        event["id"] = f"capture-{index:06d}-rank-{rank}"
        event["capture_session_id"] = session_id
        event["collective_id"] = f"capture-{index:06d}"
        event["collective_seq"] = index
        event["recorder_rank"] = str(rank)
        event["ranks"] = [rank]
        event["start_us"] = float(index)
        event["rank_arrival_us"] = {str(rank): 0.0}
        event.pop("arrival_skew_us", None)
        buckets[rank].append(event)

    paths = []
    workload = copy.deepcopy(dict(trace.get("workload", {})))
    for rank, events in enumerate(buckets):
        shard: JsonDict = {
            "format": TRACE_FORMAT,
            "workload": copy.deepcopy(workload),
            "system": {
                "capture_mode": "sharded",
                "capture_session_id": session_id,
                "clock_offset_us": 0.0,
                "rank": str(rank),
            },
            "events": events,
        }
        validate_trace(shard, allow_partial_arrivals=True)
        path = root / f"rank-{rank}.trace.json"
        write_json(str(path), shard)
        paths.append(path)
    return tuple(paths)


def materialize_fixture_set(
    output_dir: Path,
    *,
    stored_event_counts: Sequence[int] = STANDARD_STORED_EVENT_COUNTS,
    compressed_logical_counts: Sequence[int] = STANDARD_COMPRESSED_LOGICAL_COUNTS,
) -> Path:
    """Write a deterministic fixture set and return its manifest path."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fixtures = []

    for raw_count in stored_event_counts:
        count = _positive_count(raw_count, "stored event fixture size")
        relative_path = f"trace-{count:06d}.json"
        path = root / relative_path
        trace = generate_trace(count)
        write_json(str(path), trace)
        fixtures.append(
            _fixture_record(
                case_id=f"trace-{count}",
                kind="trace",
                relative_path=relative_path,
                path=path,
                stored_events=count,
                logical_events=count,
            )
        )

    for raw_count in compressed_logical_counts:
        count = _positive_count(raw_count, "compressed logical fixture size")
        relative_path = f"canary-compressed-{count:06d}.json"
        path = root / relative_path
        canary = generate_compressed_canary(count)
        write_json(str(path), canary)
        expansion = preflight_canary_expansion(canary.get("events", []))
        fixtures.append(
            _fixture_record(
                case_id=f"canary-compressed-{count}",
                kind="canary",
                relative_path=relative_path,
                path=path,
                stored_events=expansion.stored_events,
                logical_events=expansion.logical_events,
            )
        )

    manifest: JsonDict = {
        "format": FIXTURE_MANIFEST_FORMAT,
        "generator_version": FIXTURE_GENERATOR_VERSION,
        "fixtures": fixtures,
    }
    manifest_path = root / "manifest.json"
    write_json(str(manifest_path), manifest)
    return manifest_path


def _fixture_record(
    *,
    case_id: str,
    kind: str,
    relative_path: str,
    path: Path,
    stored_events: int,
    logical_events: int,
) -> JsonDict:
    return {
        "case_id": case_id,
        "kind": kind,
        "path": relative_path,
        "stored_events": stored_events,
        "logical_events": logical_events,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _refresh_size_metrics(canary: JsonDict, compiler: Dict[str, Any]) -> None:
    last_size = -1
    for _ in range(12):
        compiler["canary_bytes"] = max(0, last_size)
        source_bytes = as_int(compiler.get("source_bytes"), 0)
        compiler["byte_compression_ratio"] = (
            round(source_bytes / last_size, 3) if source_bytes and last_size > 0 else 0.0
        )
        current_size = len(canonical_json_bytes(canary))
        if current_size == last_size:
            break
        last_size = current_size
    compiler["canary_bytes"] = len(canonical_json_bytes(canary))
    source_bytes = as_int(compiler.get("source_bytes"), 0)
    compiler["byte_compression_ratio"] = (
        round(source_bytes / as_int(compiler["canary_bytes"]), 3) if compiler["canary_bytes"] else 0.0
    )


def _positive_count(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def canonical_fixture_sha256(document: Mapping[str, Any]) -> str:
    """Return the stable semantic byte hash used by fixture tests and tooling."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()
