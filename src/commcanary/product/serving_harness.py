"""Drive a frozen traffic trace against a serving engine and time each request.

This is the engine-agnostic half of the sustained reference workload. It owns
the parts that decide whether the measurement is honest -- when a request is
issued, which requests count, what a step boundary is -- and knows nothing
about vLLM. The engine binding is a thin adapter behind :class:`ServingEngine`,
which is also what makes the scheduling logic testable without a GPU.

Two things here are easy to get wrong and are therefore not left to the caller.

**Requests are issued on the trace's clock, not as fast as the engine will take
them.** A load test that submits everything up front measures batch throughput
and calls it serving throughput. The harness submits a request when its arrival
offset has elapsed and not before, so a saturated engine shows up as growing
queueing delay -- which is the signal a serving team actually gates on.

**Late submission is recorded, not hidden.** If the harness itself falls behind
-- the host is busy, a poll took too long -- the request's own arrival offset
still defines its latency, so harness lag inflates the measured latency instead
of vanishing into it. ``submission_lag_us`` is reported so a run whose lag is
large enough to distort the result can be thrown out rather than believed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..artifacts.serving_measurement import build_serving_measurement
from ..artifacts.traffic_trace import validate_traffic_trace
from ..errors import SchemaError

#: How long the harness waits between polls when nothing is due. Short enough
#: that first-token observations are not quantised into uselessness, long
#: enough that the poll loop does not become the workload.
DEFAULT_POLL_INTERVAL_US = 200


@dataclass(frozen=True)
class ServingRunResult:
    """A serving measurement plus what the harness itself contributed to it.

    The lag figures are the harness's own error bar. If the loop could not
    submit requests on time, every latency in the measurement is inflated by
    that amount, and the run should be discarded rather than compared against
    one that kept up. Keeping them outside the measurement is deliberate: they
    describe the instrument, not the system under test.
    """

    measurement: Dict[str, Any]
    steps: int
    drained: bool
    submission_lag_us_max: int
    submission_lag_us_median: int


class ServingEngine:
    """The engine surface the harness needs, and nothing more.

    Implementations are expected to be non-blocking: ``submit`` hands a request
    over and returns, ``poll`` reports whatever has happened since the last
    call. A blocking generate call cannot express arrival-driven load, which is
    why this interface exists rather than the engine's own.
    """

    def submit(self, request_id: str, *, input_length: int, output_length: int) -> None:
        raise NotImplementedError

    def poll(self) -> Sequence[Mapping[str, Any]]:
        """Return events since the last call.

        Each event is ``{"request_id": str, "kind": "first_token"|"finished"|"failed"}``.
        """

        raise NotImplementedError


class _RequestState:
    __slots__ = (
        "arrival_offset_us",
        "completion_offset_us",
        "first_token_offset_us",
        "input_length",
        "outcome",
        "output_length",
        "submission_lag_us",
    )

    def __init__(self, *, arrival_offset_us: int, input_length: int, output_length: int) -> None:
        self.arrival_offset_us = arrival_offset_us
        self.input_length = input_length
        self.output_length = output_length
        self.first_token_offset_us: Optional[int] = None
        self.completion_offset_us: Optional[int] = None
        self.outcome: Optional[str] = None
        self.submission_lag_us = 0


def _monotonic_us() -> int:
    return time.perf_counter_ns() // 1_000


def run_traffic(
    trace: Mapping[str, Any],
    engine: ServingEngine,
    *,
    configuration_id: str,
    window_start_us: int,
    window_end_us: int,
    on_step: Optional[Callable[[int], None]] = None,
    clock: Callable[[], int] = _monotonic_us,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_us: int = DEFAULT_POLL_INTERVAL_US,
    drain_grace_us: int = 60_000_000,
) -> ServingRunResult:
    """Issue a traffic trace against ``engine`` and return a serving measurement.

    ``on_step`` is called once per engine step with the step index, which is how
    the capture learns where a step began. Segmentation cannot be recovered from
    the collective stream afterwards, so if this is not wired up the resulting
    trace carries none.

    ``drain_grace_us`` bounds how long the harness waits after the last arrival
    for outstanding requests to finish. Requests still in flight when it expires
    are recorded as failed rather than silently dropped: a run that could not
    drain is a run whose throughput number is not trustworthy, and deleting the
    evidence of that would be the bug.
    """

    validate_traffic_trace(trace)
    if window_end_us <= window_start_us:
        raise SchemaError("serving window must be a positive interval")
    if poll_interval_us < 1:
        raise SchemaError("poll interval must be positive")

    pending = list(trace["requests"])
    states: Dict[str, _RequestState] = {}
    outstanding = 0
    step_index = 0
    drained = True
    started = clock()
    last_arrival_us = int(trace["summary"]["last_arrival_offset_us"])

    def elapsed() -> int:
        return clock() - started

    while True:
        now = elapsed()

        # Submit everything whose arrival has come due. A backlog is submitted
        # in one pass so the harness catches up rather than pacing itself at
        # one request per poll, which would silently reshape the arrival
        # process into whatever the poll loop could sustain.
        while pending and int(pending[0]["arrival_offset_us"]) <= now:
            request = pending.pop(0)
            request_id = str(request["request_id"])
            state = _RequestState(
                arrival_offset_us=int(request["arrival_offset_us"]),
                input_length=int(request["input_length"]),
                output_length=int(request["output_length"]),
            )
            state.submission_lag_us = max(0, now - state.arrival_offset_us)
            states[request_id] = state
            engine.submit(
                request_id,
                input_length=state.input_length,
                output_length=state.output_length,
            )
            outstanding += 1

        if on_step is not None:
            on_step(step_index)
        step_index += 1

        for event in engine.poll():
            event_request_id = str(event["request_id"])
            observed_state = states.get(event_request_id)
            if observed_state is None:
                raise SchemaError(f"engine reported an event for unknown request {event_request_id!r}")
            kind = event.get("kind")
            observed = elapsed()
            if kind == "first_token":
                if observed_state.first_token_offset_us is None:
                    observed_state.first_token_offset_us = observed
            elif kind == "finished":
                if observed_state.first_token_offset_us is None:
                    # A completion with no observed first token cannot support a
                    # TTFT claim, so it is not counted as one.
                    observed_state.outcome = "failed"
                else:
                    observed_state.completion_offset_us = observed
                    observed_state.outcome = "completed"
                outstanding -= 1
            elif kind == "failed":
                observed_state.outcome = "failed"
                outstanding -= 1
            else:
                raise SchemaError(f"engine reported an unsupported event kind {kind!r}")

        if not pending and outstanding == 0:
            break
        if not pending and elapsed() > last_arrival_us + drain_grace_us:
            drained = False
            for state in states.values():
                if state.outcome is None:
                    state.outcome = "failed"
            break
        sleep(poll_interval_us / 1_000_000.0)

    lags = sorted(state.submission_lag_us for state in states.values())
    measurement = build_serving_measurement(
        traffic_trace_id=str(trace["trace_id"]),
        configuration_id=configuration_id,
        window_start_us=window_start_us,
        window_end_us=window_end_us,
        records=_records(states),
    )
    return ServingRunResult(
        measurement=measurement,
        steps=step_index,
        drained=drained,
        submission_lag_us_max=lags[-1] if lags else 0,
        submission_lag_us_median=lags[len(lags) // 2] if lags else 0,
    )


def _records(states: Mapping[str, _RequestState]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for request_id, state in states.items():
        completed = state.outcome == "completed"
        records.append(
            {
                "request_id": request_id,
                "outcome": "completed" if completed else "failed",
                "arrival_offset_us": state.arrival_offset_us,
                "first_token_offset_us": state.first_token_offset_us if completed else None,
                "completion_offset_us": state.completion_offset_us if completed else None,
                "input_length": state.input_length,
                "output_length": state.output_length if completed else 0,
            }
        )
    return records


__all__ = ["DEFAULT_POLL_INTERVAL_US", "ServingEngine", "ServingRunResult", "run_traffic"]
