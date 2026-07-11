"""Field-validation and optional-field assembly behavior of TraceRecorder.record_collective.

These tests exercise the per-argument guards in
:meth:`commcanary.adapters.capture.TraceRecorder.record_collective` directly,
independent of the higher-level trace/schema validation exercised elsewhere.
"""

from __future__ import annotations

import pytest

from commcanary.capture import TraceRecorder
from commcanary.schema import SchemaError


def _recorder(tmp_path) -> TraceRecorder:
    return TraceRecorder(str(tmp_path / "trace.json"))


def test_op_must_be_a_non_empty_string(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="op must be a non-empty string"):
        recorder.record_collective(op="   ", bytes=16, ranks=[0])
    assert recorder.events == []


def test_bytes_must_be_positive(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="bytes must be positive"):
        recorder.record_collective(op="all_reduce", bytes=0, ranks=[0])


def test_start_us_must_be_non_negative(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="start_us must be non-negative"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], start_us=-1.0)


def test_negative_compute_timing_is_rejected(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="compute timing and pressure values must be non-negative"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], compute_before_us=-1.0)


def test_concurrent_groups_must_be_positive(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="concurrent_groups must be positive"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], concurrent_groups=0)


def test_sender_rank_must_be_one_of_ranks(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="sender_rank must be one of ranks"):
        recorder.record_collective(op="send", bytes=16, ranks=[0, 1], sender_rank=5)


def test_receiver_rank_must_be_one_of_ranks(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="receiver_rank must be one of ranks"):
        recorder.record_collective(op="recv", bytes=16, ranks=[0, 1], receiver_rank=5)


def test_sender_and_receiver_rank_must_differ(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="sender_rank and receiver_rank must differ"):
        recorder.record_collective(op="send", bytes=16, ranks=[0, 1], sender_rank=0, receiver_rank=0)


def test_message_sequence_must_be_non_negative(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="message_sequence must be non-negative"):
        recorder.record_collective(op="send", bytes=16, ranks=[0, 1], message_sequence=-1)


def test_tag_must_be_non_empty_when_provided(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="tag and channel must be non-empty when provided"):
        recorder.record_collective(op="send", bytes=16, ranks=[0, 1], tag="")


def test_channel_must_be_non_empty_when_provided(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="tag and channel must be non-empty when provided"):
        recorder.record_collective(op="send", bytes=16, ranks=[0, 1], channel="")


def test_rank_arrival_us_must_be_a_non_empty_mapping(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="rank_arrival_us must be a non-empty mapping"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0, 1], rank_arrival_us={})
    with pytest.raises(SchemaError, match="rank_arrival_us must be a non-empty mapping"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0, 1], rank_arrival_us=[0.0, 1.0])


def test_rank_arrival_us_rejects_rank_outside_ranks(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="rank_arrival_us contains a rank outside ranks"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0, 1], rank_arrival_us={"5": 0.0})


def test_rank_arrival_us_values_must_be_non_negative(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="rank_arrival_us values must be non-negative"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0, 1], rank_arrival_us={"0": -1.0})


def test_rank_arrival_us_rejects_int_and_string_key_aliasing_as_duplicate(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    # 0 (int) and "0" (str) are distinct dict keys but normalize to the same
    # recorded rank, so the second occurrence must be rejected as a duplicate.
    with pytest.raises(SchemaError, match="rank_arrival_us contains duplicate rank keys"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0, 1], rank_arrival_us={0: 0.0, "0": 1.0})


def test_arrival_skew_us_must_be_non_negative(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="arrival_skew_us must be non-negative"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0, 1], arrival_skew_us=-1.0)


def test_single_rank_collective_cannot_have_positive_arrival_skew(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="a one-rank collective cannot have positive arrival skew"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], arrival_skew_us=5.0)


def test_positive_arrival_skew_is_accepted_for_multi_rank_collectives(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_collective(op="all_reduce", bytes=16, ranks=[0, 1], arrival_skew_us=5.0)
    event = recorder.events[0]
    assert event["arrival_skew_us"] == 5.0
    assert "rank_arrival_us" not in event


def test_collective_id_must_not_be_empty_string(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    with pytest.raises(SchemaError, match="collective_id must not be empty"):
        recorder.record_collective(op="all_reduce", bytes=16, ranks=[0], rank_arrival_us={"0": 0.0}, collective_id="")


def test_absent_rank_arrival_and_skew_leaves_event_without_arrival_fields(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_collective(
        op="all_reduce",
        bytes=16,
        ranks=[0, 1],
        observed_exposed_us=5.0,
    )
    event = recorder.events[0]
    assert "rank_arrival_us" not in event
    assert "arrival_skew_us" not in event
    assert event["observed_exposed_us"] == 5.0


def test_full_optional_metadata_is_recorded_with_partial_rank_arrival_flagged(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_collective(
        op="send",
        bytes=16,
        ranks=[0, 1],
        collective_id="cid-1",
        sender_rank=0,
        receiver_rank=1,
        tag="tag-1",
        channel="chan-1",
        message_sequence=3,
        rank_arrival_us={"0": 0.0},
    )
    event = recorder.events[0]
    assert event["collective_id"] == "cid-1"
    assert event["sender_rank"] == 0
    assert event["receiver_rank"] == 1
    assert event["tag"] == "tag-1"
    assert event["channel"] == "chan-1"
    assert event["message_sequence"] == 3
    assert event["rank_arrival_us"] == {"0": 0.0}
    assert event["partial_rank_arrival"] is True
