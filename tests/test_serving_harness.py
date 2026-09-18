from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import pytest

from commcanary.errors import SchemaError
from commcanary.product.serving_harness import run_traffic
from commcanary.services.traffic_synthesis import synthesize_traffic_trace

INPUT_DISTRIBUTION = {"median_tokens": 200, "sigma": 0.4, "minimum_tokens": 8, "maximum_tokens": 1024}
OUTPUT_DISTRIBUTION = {"median_tokens": 100, "sigma": 0.4, "minimum_tokens": 2, "maximum_tokens": 512}


class _FakeClock:
    """A clock the test advances by hand, so no test depends on wall time."""

    def __init__(self) -> None:
        self.now_us = 0

    def __call__(self) -> int:
        return self.now_us

    def sleep(self, seconds: float) -> None:
        self.now_us += max(1, int(seconds * 1_000_000))


class _FakeEngine:
    """A serving engine with fixed prefill and per-token decode cost."""

    def __init__(self, clock: _FakeClock, *, prefill_us: int = 30_000, per_token_us: int = 1_000) -> None:
        self._clock = clock
        self._prefill_us = prefill_us
        self._per_token_us = per_token_us
        self._scheduled: List[Dict[str, Any]] = []
        self.submitted: List[str] = []

    def submit(self, request_id: str, *, input_length: int, output_length: int) -> None:
        self.submitted.append(request_id)
        now = self._clock.now_us
        self._scheduled.append({"request_id": request_id, "at": now + self._prefill_us, "kind": "first_token"})
        self._scheduled.append(
            {
                "request_id": request_id,
                "at": now + self._prefill_us + output_length * self._per_token_us,
                "kind": "finished",
            }
        )

    def poll(self) -> Sequence[Mapping[str, Any]]:
        now = self._clock.now_us
        due = [event for event in self._scheduled if event["at"] <= now]
        self._scheduled = [event for event in self._scheduled if event["at"] > now]
        return [{"request_id": event["request_id"], "kind": event["kind"]} for event in due]


class _FailingEngine(_FakeEngine):
    def poll(self) -> Sequence[Mapping[str, Any]]:
        return [
            {"request_id": event["request_id"], "kind": "failed"}
            for event in super().poll()
            if event["kind"] == "finished"
        ]


def _trace(duration_us: int = 3_000_000, interval_us: int = 200_000) -> Dict[str, Any]:
    return synthesize_traffic_trace(
        name="harness",
        duration_us=duration_us,
        mean_interarrival_us=interval_us,
        input_distribution=INPUT_DISTRIBUTION,
        output_distribution=OUTPUT_DISTRIBUTION,
        seed=5,
        poisson=False,
    )


def _run(engine_class: Any = _FakeEngine, **overrides: Any) -> Any:
    clock = _FakeClock()
    trace = overrides.pop("trace", None) or _trace()
    engine = engine_class(clock, **overrides.pop("engine_options", {}))
    parameters: Dict[str, Any] = {
        "configuration_id": "nccl-2.20.5-tree-ll",
        "window_start_us": 500_000,
        "window_end_us": 2_500_000,
        "clock": clock,
        "sleep": clock.sleep,
        "poll_interval_us": 10_000,
    }
    parameters.update(overrides)
    return run_traffic(trace, engine, **parameters), engine, trace


def test_requests_are_issued_on_the_traces_clock_not_all_at_once() -> None:
    """The property that separates a load test from a batch benchmark.

    Submitting everything up front would measure how fast the engine can chew
    through a fixed queue, which is not what a serving team gates on.
    """

    result, engine, trace = _run()
    assert len(engine.submitted) == trace["summary"]["request_count"]
    measurement = result.measurement
    arrivals = [record["arrival_offset_us"] for record in measurement["requests"]]
    assert arrivals == sorted(arrivals)
    # Every request was submitted, and none before its arrival offset.
    assert all(record["outcome"] == "completed" for record in measurement["requests"])


def test_only_in_window_requests_reach_the_decision_metric() -> None:
    result, _, _ = _run()
    summary = result.measurement["summary"]
    assert summary["requests_excluded_by_window"] > 0
    assert summary["requests_in_window"] + summary["requests_excluded_by_window"] == len(result.measurement["requests"])
    assert summary["sustained_output_tokens_per_second"] > 0


def test_time_to_first_token_reflects_the_engines_prefill_cost() -> None:
    fast, _, _ = _run(engine_options={"prefill_us": 10_000})
    slow, _, _ = _run(engine_options={"prefill_us": 90_000})
    assert (
        slow.measurement["summary"]["time_to_first_token_us"]["p50"]
        > fast.measurement["summary"]["time_to_first_token_us"]["p50"]
    )


def test_harness_lag_is_reported_rather_than_absorbed() -> None:
    """Lag describes the instrument, so it is returned beside the measurement.

    A coarse poll interval makes the harness late; the latency it reports must
    grow rather than the lateness disappearing into the numbers.
    """

    prompt, _, _ = _run(poll_interval_us=1_000)
    coarse, _, _ = _run(poll_interval_us=250_000)
    assert coarse.submission_lag_us_max > prompt.submission_lag_us_max
    assert (
        coarse.measurement["summary"]["time_to_first_token_us"]["p50"]
        >= prompt.measurement["summary"]["time_to_first_token_us"]["p50"]
    )


def test_step_boundaries_are_emitted_once_per_step() -> None:
    steps: List[int] = []
    result, _, _ = _run(on_step=steps.append)
    assert steps == list(range(len(steps)))
    assert result.steps == len(steps)
    assert result.steps > 1


def test_a_failing_engine_produces_failed_records_not_missing_ones() -> None:
    result, _, trace = _run(engine_class=_FailingEngine)
    records = result.measurement["requests"]
    assert len(records) == trace["summary"]["request_count"]
    assert all(record["outcome"] == "failed" for record in records)
    assert all(record["first_token_offset_us"] is None for record in records)
    assert all(record["output_length"] == 0 for record in records)


def test_an_undrained_run_is_recorded_as_undrained() -> None:
    """Requests still in flight when the grace expires are failures, not gaps.

    Dropping them would quietly improve the throughput number of exactly the
    run that could not keep up.
    """

    clock = _FakeClock()
    trace = _trace()

    class _StuckEngine(_FakeEngine):
        def poll(self) -> Sequence[Mapping[str, Any]]:
            return []

    result = run_traffic(
        trace,
        _StuckEngine(clock),
        configuration_id="c",
        window_start_us=0,
        window_end_us=3_000_000,
        clock=clock,
        sleep=clock.sleep,
        poll_interval_us=100_000,
        drain_grace_us=1_000_000,
    )
    assert result.drained is False
    assert all(record["outcome"] == "failed" for record in result.measurement["requests"])


def test_an_event_for_an_unknown_request_is_refused() -> None:
    clock = _FakeClock()

    class _LyingEngine(_FakeEngine):
        def poll(self) -> Sequence[Mapping[str, Any]]:
            return [{"request_id": "r-999999", "kind": "finished"}]

    with pytest.raises(SchemaError, match="unknown request"):
        run_traffic(
            _trace(),
            _LyingEngine(clock),
            configuration_id="c",
            window_start_us=0,
            window_end_us=3_000_000,
            clock=clock,
            sleep=clock.sleep,
        )


def test_an_unsupported_event_kind_is_refused() -> None:
    clock = _FakeClock()

    class _OddEngine(_FakeEngine):
        def poll(self) -> Sequence[Mapping[str, Any]]:
            events = list(super().poll())
            return [{**event, "kind": "queued"} for event in events]

    with pytest.raises(SchemaError, match="unsupported event kind"):
        run_traffic(
            _trace(),
            _OddEngine(clock),
            configuration_id="c",
            window_start_us=0,
            window_end_us=3_000_000,
            clock=clock,
            sleep=clock.sleep,
            poll_interval_us=10_000,
        )


def test_invalid_windows_and_poll_intervals_are_refused() -> None:
    clock = _FakeClock()
    with pytest.raises(SchemaError, match="window must be a positive interval"):
        run_traffic(
            _trace(), _FakeEngine(clock), configuration_id="c", window_start_us=10, window_end_us=10, clock=clock
        )
    with pytest.raises(SchemaError, match="poll interval must be positive"):
        run_traffic(
            _trace(),
            _FakeEngine(clock),
            configuration_id="c",
            window_start_us=0,
            window_end_us=10,
            clock=clock,
            poll_interval_us=0,
        )
