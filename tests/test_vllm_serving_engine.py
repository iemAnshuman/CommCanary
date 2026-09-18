from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from commcanary.errors import SchemaError
from commcanary.product.vllm_serving_engine import VllmServingEngine


class _FakeStreamingEngine:
    """A stand-in with vLLM's streaming shape: an async generator per request.

    This is the same substitution the existing application drivers use. It
    exercises the binding's own logic -- event ordering, failure isolation,
    first-token detection -- and proves nothing about the real vLLM API.
    """

    def __init__(self, *, steps: int = 3, fail_for: str = "") -> None:
        self._steps = steps
        self._fail_for = fail_for

    def generate(self, prompt: Dict[str, Any], sampling_params: Any, request_id: str) -> Any:
        steps, fail_for = self._steps, self._fail_for

        async def _stream() -> Any:
            if request_id == fail_for:
                raise RuntimeError("engine refused the request")
            # First step yields no token, as a real prefill step can.
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[])], finished=False)
            for step in range(steps):
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=list(range(step + 1)))],
                    finished=step == steps - 1,
                )

        return _stream()


def _engine(**options: Any) -> VllmServingEngine:
    return VllmServingEngine(
        _FakeStreamingEngine(**options),
        sampling_params_factory=lambda output_length: SimpleNamespace(max_tokens=output_length),
        prompt_token_ids=lambda request_id, input_length: [7] * input_length,
    )


def _drain(engine: VllmServingEngine, *, expected: int, timeout_s: float = 5.0) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    while len(collected) < expected and time.monotonic() < deadline:
        collected.extend(dict(event) for event in engine.poll())
        time.sleep(0.005)
    return collected


def test_a_request_reports_first_token_then_finished_in_order() -> None:
    engine = _engine()
    try:
        engine.submit("r-000000", input_length=8, output_length=3)
        events = _drain(engine, expected=2)
    finally:
        engine.close()

    assert [event["kind"] for event in events] == ["first_token", "finished"]
    assert {event["request_id"] for event in events} == {"r-000000"}


def test_a_prefill_step_without_tokens_does_not_count_as_a_first_token() -> None:
    """The first streamed output carries no token; TTFT must not start there.

    Counting it would report a first-token latency for a step that produced no
    token, which is the difference between measuring prefill and measuring the
    scheduler.
    """

    engine = _engine()
    try:
        engine.submit("r-000000", input_length=8, output_length=3)
        events = _drain(engine, expected=2)
    finally:
        engine.close()

    assert [event["kind"] for event in events].count("first_token") == 1


def test_one_failing_request_does_not_stop_the_others() -> None:
    engine = _engine(fail_for="r-000001")
    try:
        for index in range(3):
            engine.submit(f"r-{index:06d}", input_length=8, output_length=2)
        events = _drain(engine, expected=5)
    finally:
        engine.close()

    by_request: Dict[str, List[str]] = {}
    for event in events:
        by_request.setdefault(event["request_id"], []).append(event["kind"])
    assert by_request["r-000001"] == ["failed"]
    assert by_request["r-000000"] == ["first_token", "finished"]
    assert by_request["r-000002"] == ["first_token", "finished"]


def test_polling_an_idle_engine_returns_nothing() -> None:
    engine = _engine()
    try:
        assert list(engine.poll()) == []
    finally:
        engine.close()


def test_submitting_after_close_is_refused() -> None:
    engine = _engine()
    engine.close()
    with pytest.raises(SchemaError, match="closed serving engine"):
        engine.submit("r-000000", input_length=8, output_length=2)


def test_close_is_idempotent() -> None:
    engine = _engine()
    engine.close()
    engine.close()


def test_an_unrecognised_output_shape_yields_no_first_token() -> None:
    """A vLLM API change must surface as a missing observation, not a fake one.

    A request with no observed first token is recorded as failed by the
    measurement, which is visible. A fabricated zero-latency first token would
    not be.
    """

    class _ShapeChanged(_FakeStreamingEngine):
        def generate(self, prompt: Dict[str, Any], sampling_params: Any, request_id: str) -> Any:
            async def _stream() -> Any:
                yield SimpleNamespace(completions=["surprise"], finished=True)

            return _stream()

    engine = VllmServingEngine(
        _ShapeChanged(),
        sampling_params_factory=lambda output_length: SimpleNamespace(max_tokens=output_length),
        prompt_token_ids=lambda request_id, input_length: [7] * input_length,
    )
    try:
        engine.submit("r-000000", input_length=8, output_length=2)
        events = _drain(engine, expected=1)
    finally:
        engine.close()

    assert [event["kind"] for event in events] == ["finished"]
