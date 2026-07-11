"""Shard-reconciliation behavior of merge_trace_shards / capture_merge.

Complements tests/adapters/test_capture.py's merge coverage with scenarios
that are not otherwise exercised: cross-shard metadata conflicts, three-way
bucket collisions, the ``clock_calibration.offset_us`` calibration path, and
calibrated scalar-skew aggregation.
"""

from __future__ import annotations

import os

import pytest

from commcanary.capture import merge_trace_shards
from commcanary.schema import TRACE_FORMAT, SchemaError, write_json


def _write_shard(tmp_path, name, *, workload, system, events):
    write_json(
        os.path.join(tmp_path, name),
        {"format": TRACE_FORMAT, "workload": workload, "system": system, "events": events},
    )


def test_merge_rejects_shards_with_conflicting_workload_metadata(tmp_path) -> None:
    _write_shard(
        tmp_path,
        "rank-0.trace.json",
        workload={"name": "alpha"},
        system={"rank": "0"},
        events=[
            {
                "id": "e0",
                "capture_session_id": "s",
                "collective_id": "c0",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0],
                "rank_arrival_us": {"0": 0.0},
            }
        ],
    )
    _write_shard(
        tmp_path,
        "rank-1.trace.json",
        workload={"name": "beta"},
        system={"rank": "1"},
        events=[
            {
                "id": "e1",
                "capture_session_id": "s",
                "collective_id": "c1",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0],
                "rank_arrival_us": {"0": 0.0},
            }
        ],
    )
    with pytest.raises(SchemaError, match="conflicting workload metadata"):
        merge_trace_shards(str(tmp_path), workload_name="ranked")


def test_strict_merge_requires_collective_id_on_every_event(tmp_path) -> None:
    _write_shard(
        tmp_path,
        "rank-0.trace.json",
        workload={"name": "ranked"},
        system={"rank": "0"},
        events=[
            {
                "id": "e0",
                "capture_session_id": "s",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0],
                "rank_arrival_us": {"0": 0.0},
            }
        ],
    )
    _write_shard(
        tmp_path,
        "rank-1.trace.json",
        workload={"name": "ranked"},
        system={"rank": "1"},
        events=[
            {
                "id": "e1",
                "capture_session_id": "s",
                "collective_id": "c0",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0],
                "rank_arrival_us": {"0": 0.0},
            }
        ],
    )
    with pytest.raises(SchemaError, match="stable collective_id"):
        merge_trace_shards(str(tmp_path), workload_name="ranked")


def test_event_capture_session_id_must_match_shard_system_metadata(tmp_path) -> None:
    _write_shard(
        tmp_path,
        "solo.trace.json",
        workload={"name": "ranked"},
        system={"rank": "0", "capture_session_id": "system-session"},
        events=[
            {
                "id": "e0",
                "capture_session_id": "event-session",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0],
                "rank_arrival_us": {"0": 0.0},
            }
        ],
    )
    with pytest.raises(SchemaError, match="conflicts with shard system metadata"):
        merge_trace_shards(str(tmp_path), workload_name="ranked")


def test_three_way_rank_contribution_coalesces_into_one_event(tmp_path) -> None:
    for rank in (0, 1, 2):
        _write_shard(
            tmp_path,
            f"rank-{rank}.trace.json",
            workload={"name": "three-way"},
            system={"rank": str(rank), "capture_session_id": "s"},
            events=[
                {
                    "id": f"c0-{rank}",
                    "capture_session_id": "s",
                    "collective_id": "c0",
                    "recorder_rank": str(rank),
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1, 2],
                    "start_us": 0.0,
                    "rank_arrival_us": {str(rank): 0.0},
                    "partial_rank_arrival": True,
                }
            ],
        )
    merged = merge_trace_shards(str(tmp_path), workload_name="three-way")
    assert len(merged["events"]) == 1
    event = merged["events"][0]
    assert event["merged_shards"] == ["rank-0.trace.json", "rank-1.trace.json", "rank-2.trace.json"]
    assert event["recorder_ranks"] == ["0", "1", "2"]
    assert event["arrival_skew_unknown"] is True


def test_clock_calibration_offset_field_is_used_when_clock_offset_us_is_absent(tmp_path) -> None:
    _write_shard(
        tmp_path,
        "rank-0.trace.json",
        workload={"name": "calibrated"},
        system={"rank": "0", "capture_session_id": "s", "clock_calibration": {"offset_us": 0.0}},
        events=[
            {
                "id": "c0-0",
                "capture_session_id": "s",
                "collective_id": "c0",
                "recorder_rank": "0",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 100.0,
                "rank_arrival_us": {"0": 0.0},
                "partial_rank_arrival": True,
            }
        ],
    )
    _write_shard(
        tmp_path,
        "rank-1.trace.json",
        workload={"name": "calibrated"},
        system={"rank": "1", "capture_session_id": "s", "clock_calibration": {"offset_us": 50.0}},
        events=[
            {
                "id": "c0-1",
                "capture_session_id": "s",
                "collective_id": "c0",
                "recorder_rank": "1",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 0.0,
                "rank_arrival_us": {"1": 0.0},
                "partial_rank_arrival": True,
            }
        ],
    )
    merged = merge_trace_shards(str(tmp_path), workload_name="calibrated")
    event = merged["events"][0]
    # rank 0's aligned start is 100+0=100, rank 1's is 0+50=50; the earliest
    # (50) becomes the zero point for both the coalesced start and offsets.
    assert event["start_us"] == 50.0
    assert event["rank_arrival_us"] == {"0": 50.0, "1": 0.0}
    shard_systems = merged["system"]["shard_systems"]
    assert all(system["clock_alignment"] == "explicit_offset_us" for system in shard_systems)


def test_merge_rejects_mixed_rank_arrival_map_and_scalar_records(tmp_path) -> None:
    _write_shard(
        tmp_path,
        "rank-0.trace.json",
        workload={"name": "mixed"},
        system={"rank": "0", "capture_session_id": "s"},
        events=[
            {
                "id": "c0-0",
                "capture_session_id": "s",
                "collective_id": "c0",
                "recorder_rank": "0",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 0.0,
                "rank_arrival_us": {"0": 0.0},
                "partial_rank_arrival": True,
            }
        ],
    )
    _write_shard(
        tmp_path,
        "rank-1.trace.json",
        workload={"name": "mixed"},
        system={"rank": "1", "capture_session_id": "s"},
        events=[
            {
                "id": "c0-1",
                "capture_session_id": "s",
                "collective_id": "c0",
                "recorder_rank": "1",
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 16,
                "ranks": [0, 1],
                "start_us": 1.0,
            }
        ],
    )
    with pytest.raises(SchemaError, match="mixed rank_arrival_us and scalar arrival records"):
        merge_trace_shards(str(tmp_path), workload_name="mixed")


def test_calibrated_scalar_skew_uses_the_larger_of_reported_and_aligned_spread(tmp_path) -> None:
    for rank, start_us in ((0, 10.0), (1, 11.0)):
        _write_shard(
            tmp_path,
            f"rank-{rank}.trace.json",
            workload={"name": "calibrated-skew"},
            system={"rank": str(rank), "capture_session_id": "s", "clock_offset_us": 0.0},
            events=[
                {
                    "id": f"c0-{rank}",
                    "capture_session_id": "s",
                    "collective_id": "c0",
                    "recorder_rank": str(rank),
                    "phase": "decode",
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "start_us": start_us,
                    "arrival_skew_us": 7.0,
                }
            ],
        )
    merged = merge_trace_shards(str(tmp_path), workload_name="calibrated-skew")
    event = merged["events"][0]
    assert event["arrival_skew_us"] == 7.0
    assert "arrival_skew_unknown" not in event
