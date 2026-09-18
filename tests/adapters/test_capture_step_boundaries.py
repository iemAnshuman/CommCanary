from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from commcanary.adapters.capture import NullRecorder, TraceRecorder
from commcanary.artifacts.trace import validate_trace
from commcanary.errors import SchemaError
from commcanary.services.phase_reduction import segment_iterations


def _record(recorder: TraceRecorder, count: int = 3) -> None:
    for _ in range(count):
        recorder.record_collective(
            op="all_reduce",
            bytes=65_536,
            ranks=[0, 1, 2, 3],
            dtype="bfloat16",
            reduction_op="sum",
            compute_before_us=10.0,
            compute_overlap_us=5.0,
        )


def _trace(recorder: TraceRecorder) -> Dict[str, Any]:
    recorder.save()
    recorder.close()
    return json.loads(Path(recorder.output_path).read_text(encoding="utf-8"))


def test_declared_steps_reach_phase_reduction_from_a_real_capture(tmp_path: Path) -> None:
    """The producer half of segmentation, which previously did not exist.

    ``phase_reduction`` has always been able to read ``iteration_index``; until
    now nothing wrote it, so the ``declared_iteration_index`` path was
    unreachable from a capture and every trace fell back to inference or
    refusal.
    """

    recorder = TraceRecorder(str(tmp_path / "trace.json"), workload={"name": "serving"})
    for _ in range(4):
        recorder.begin_iteration()
        _record(recorder)
    trace = _trace(recorder)

    assert [event["iteration_index"] for event in trace["events"]] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    validate_trace(trace)
    iterations, source = segment_iterations(trace["events"])
    assert source == "declared_iteration_index"
    assert len(iterations) == 4


def test_an_unsegmented_capture_declares_nothing_rather_than_zero(tmp_path: Path) -> None:
    # A defaulted zero would assert one long iteration that never happened.
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    _record(recorder, count=2)
    trace = _trace(recorder)

    assert all("iteration_index" not in event for event in trace["events"])
    validate_trace(trace)


def test_explicit_indices_bind_to_the_workloads_own_step_numbers(tmp_path: Path) -> None:
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    assert recorder.current_iteration is None
    assert recorder.begin_iteration(7) == 7
    assert recorder.current_iteration == 7
    _record(recorder, count=1)
    assert recorder.begin_iteration(9) == 9
    _record(recorder, count=1)
    trace = _trace(recorder)

    assert [event["iteration_index"] for event in trace["events"]] == [7, 9]
    validate_trace(trace)


def test_a_per_event_override_wins_over_the_recorder_counter(tmp_path: Path) -> None:
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    recorder.begin_iteration(3)
    _record(recorder, count=1)
    recorder.record_collective(op="all_reduce", bytes=1024, ranks=[0, 1], iteration_index=5)
    trace = _trace(recorder)

    assert [event["iteration_index"] for event in trace["events"]] == [3, 5]


def test_the_recorder_refuses_an_override_that_could_never_validate(tmp_path: Path) -> None:
    # Caught here rather than at validation, which happens after the run has
    # finished and the trace is the only thing left.
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    recorder.begin_iteration(4)
    _record(recorder, count=1)
    with pytest.raises(SchemaError, match="moves backwards from 4"):
        recorder.record_collective(op="all_reduce", bytes=1024, ranks=[0, 1], iteration_index=2)
    recorder.close()


def test_step_numbers_may_not_move_backwards(tmp_path: Path) -> None:
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    recorder.begin_iteration(4)
    with pytest.raises(SchemaError, match="must not move backwards"):
        recorder.begin_iteration(1)
    with pytest.raises(SchemaError, match="must be non-negative"):
        recorder.begin_iteration(-1)
    recorder.close()


def test_a_partially_declared_segmentation_is_refused(tmp_path: Path) -> None:
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    recorder.begin_iteration()
    _record(recorder, count=3)
    trace = _trace(recorder)

    partial = json.loads(json.dumps(trace))
    del partial["events"][1]["iteration_index"]
    with pytest.raises(SchemaError, match="partially declared segmentation is refused"):
        validate_trace(partial)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-1, "must be a non-negative integer"),
        ("2", "must be a non-negative integer"),
        (True, "must be a non-negative integer"),
    ],
)
def test_declared_indices_must_be_non_negative_integers(tmp_path: Path, value: Any, message: str) -> None:
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    recorder.begin_iteration()
    _record(recorder, count=2)
    trace = _trace(recorder)
    trace["events"][1]["iteration_index"] = value
    with pytest.raises(SchemaError, match=message):
        validate_trace(trace)


def test_declared_indices_may_not_move_backwards_inside_a_trace(tmp_path: Path) -> None:
    recorder = TraceRecorder(str(tmp_path / "trace.json"))
    for _ in range(3):
        recorder.begin_iteration()
        _record(recorder, count=1)
    trace = _trace(recorder)
    trace["events"][2]["iteration_index"] = 0
    with pytest.raises(SchemaError, match="moves backwards"):
        validate_trace(trace)


def test_a_disabled_recorder_still_answers_the_segmentation_surface() -> None:
    # Capture being off must not make a segmented workload fail to run.
    recorder = NullRecorder()
    assert recorder.begin_iteration() == 0
    assert recorder.begin_iteration(11) == 11
    assert recorder.current_iteration is None
    recorder.record_collective(op="all_reduce", bytes=1024, ranks=[0, 1])


def test_shards_from_separate_ranks_agree_on_step_numbers(tmp_path: Path) -> None:
    """Ranks segment independently and must still line up.

    They call ``begin_iteration`` in lockstep once per engine step, so the
    counters agree without any communication between them. If they did not,
    a merged trace would carry contradictory segmentation.
    """

    per_rank: List[List[int]] = []
    for rank in range(2):
        recorder = TraceRecorder(str(tmp_path / f"rank{rank}.json"))
        for _ in range(3):
            recorder.begin_iteration()
            _record(recorder, count=2)
        trace = _trace(recorder)
        per_rank.append([event["iteration_index"] for event in trace["events"]])
    assert per_rank[0] == per_rank[1] == [0, 0, 1, 1, 2, 2]
