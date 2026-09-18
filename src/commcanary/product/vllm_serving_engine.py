"""Bind vLLM's streaming engine to the harness's :class:`ServingEngine` surface.

Everything that decides whether the measurement is honest lives in
``serving_harness``. This module is deliberately the thinnest possible layer
over vLLM, for two reasons: the harness stays testable without a GPU, and when
vLLM's API moves -- it does, between minor versions -- exactly one small file
has to change.

**This binding has never executed against a real engine.** vLLM is not
installed in the development environment and there is no GPU to run it on, so
its tests substitute a fake with the same shape. That makes it implementation
evidence, not working evidence, and its first genuine exercise is the
sensitivity precheck on the cluster. Treat a failure there as expected cost,
not as a surprise.

The blocking ``LLM.generate`` API cannot express this workload at all: it takes
a fixed list of prompts and returns when they are all done, which is a batch
benchmark. Time to first token and arrival-driven load both require the
streaming interface, so that is what this uses.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..errors import SchemaError
from .serving_harness import ServingEngine


class VllmServingEngine(ServingEngine):
    """Submit requests to a streaming vLLM engine and report token events.

    The engine's async interface runs on its own event loop in a background
    thread. Events are pushed onto a thread-safe queue and drained by ``poll``
    from the harness's thread, so the harness never awaits and its submission
    schedule cannot be distorted by engine backpressure.
    """

    def __init__(
        self,
        engine: Any,
        *,
        sampling_params_factory: Callable[[int], Any],
        prompt_token_ids: Callable[[str, int], List[int]],
        loop_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        import asyncio

        self._engine = engine
        self._sampling_params_factory = sampling_params_factory
        self._prompt_token_ids = prompt_token_ids
        self._events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._loop = (loop_factory or asyncio.new_event_loop)()
        self._thread = threading.Thread(target=self._run_loop, name="commcanary-vllm-engine", daemon=True)
        self._thread.start()
        self._closed = False

    def _run_loop(self) -> None:
        self._loop.run_forever()

    def submit(self, request_id: str, *, input_length: int, output_length: int) -> None:
        if self._closed:
            raise SchemaError("cannot submit to a closed serving engine")
        import asyncio

        coroutine = self._stream(request_id, input_length=input_length, output_length=output_length)
        asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def _stream(self, request_id: str, *, input_length: int, output_length: int) -> None:
        first_seen = False
        try:
            generator = self._engine.generate(
                {"prompt_token_ids": self._prompt_token_ids(request_id, input_length)},
                self._sampling_params_factory(output_length),
                request_id,
            )
            async for output in generator:
                if not first_seen and _produced_a_token(output):
                    first_seen = True
                    self._events.put({"request_id": request_id, "kind": "first_token"})
                if getattr(output, "finished", False):
                    break
        except Exception:  # noqa: BLE001 - a failed request is data, not a crash
            # One request failing must not take the run down: the measurement
            # records it as a failure and the remaining traffic keeps flowing.
            self._events.put({"request_id": request_id, "kind": "failed"})
            return
        self._events.put({"request_id": request_id, "kind": "finished"})

    def poll(self) -> Sequence[Mapping[str, Any]]:
        drained: List[Dict[str, Any]] = []
        while True:
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                return drained

    def close(self) -> None:
        """Stop the event loop and its thread; safe to call more than once."""

        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30.0)
        self._loop.close()


def _produced_a_token(output: Any) -> bool:
    """Report whether a streamed output carries at least one generated token.

    vLLM yields an output object per step; the first one carrying a token is
    what time-to-first-token measures. Structure is probed rather than assumed
    so a shape change surfaces as False -- and therefore as a request with no
    observed first token, which the measurement records as failed -- instead of
    as a fabricated zero-latency first token.
    """

    completions = getattr(output, "outputs", None)
    if not completions:
        return False
    for completion in completions:
        token_ids = getattr(completion, "token_ids", None)
        if token_ids:
            return True
    return False


__all__ = ["VllmServingEngine"]
