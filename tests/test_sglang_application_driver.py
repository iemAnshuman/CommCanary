from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import pytest

from commcanary.product import sglang_application_driver
from commcanary.product.sglang_application_driver import _normalized_outputs, _run_generation


def _outputs(*, batch_size: int, input_length: int, output_length: int) -> list[Dict[str, Any]]:
    return [
        {
            "output_ids": list(range(output_length)),
            "meta_info": {
                "prompt_tokens": input_length,
                "completion_tokens": output_length,
                "cached_tokens": 0,
                "finish_reason": {"type": "length", "length": output_length},
            },
        }
        for _ in range(batch_size)
    ]


class _FakeEngine:
    def __init__(self) -> None:
        self.kwargs: Dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> list[Dict[str, Any]]:
        self.kwargs = kwargs
        input_ids = kwargs["input_ids"]
        output_length = kwargs["sampling_params"]["max_new_tokens"]
        return _outputs(
            batch_size=len(input_ids),
            input_length=len(input_ids[0]),
            output_length=output_length,
        )


def test_sglang_generation_uses_real_engine_batch_api_and_checks_tokens() -> None:
    engine = _FakeEngine()

    result = _run_generation(
        engine,
        batch_size=4,
        input_length=32,
        output_length=8,
        seed=123,
        iteration=2,
    )

    assert len(engine.kwargs["input_ids"]) == 4
    assert engine.kwargs["sampling_params"]["ignore_eos"] is True
    assert engine.kwargs["sampling_params"]["temperature"] == 0.0
    assert engine.kwargs["stream"] is False
    assert result["batch_latency_ms"] > 0.0
    assert result["output_token_throughput_per_second"] > 0.0


def test_sglang_output_validation_rejects_cache_or_wrong_peer_output() -> None:
    outputs = _outputs(batch_size=2, input_length=16, output_length=4)
    outputs[1]["meta_info"]["cached_tokens"] = 3

    with pytest.raises(RuntimeError, match="prefix-cached"):
        _normalized_outputs(outputs, batch_size=2, input_length=16, output_length=4)

    outputs = _outputs(batch_size=2, input_length=16, output_length=4)
    outputs[1]["output_ids"].pop()
    with pytest.raises(RuntimeError, match="exactly 4 output token IDs"):
        _normalized_outputs(outputs, batch_size=2, input_length=16, output_length=4)


def test_sglang_output_validation_rejects_malformed_engine_contracts() -> None:
    single = _outputs(batch_size=1, input_length=16, output_length=4)[0]
    assert _normalized_outputs(single, batch_size=1, input_length=16, output_length=4) == 4
    with pytest.raises(RuntimeError, match="neither one output object"):
        _normalized_outputs("wrong", batch_size=1, input_length=16, output_length=4)
    with pytest.raises(RuntimeError, match="returned 1 outputs"):
        _normalized_outputs([single], batch_size=2, input_length=16, output_length=4)
    with pytest.raises(RuntimeError, match="output 0 is not an object"):
        _normalized_outputs([object()], batch_size=1, input_length=16, output_length=4)

    missing_meta = dict(single)
    missing_meta.pop("meta_info")
    with pytest.raises(RuntimeError, match="lacks meta_info"):
        _normalized_outputs([missing_meta], batch_size=1, input_length=16, output_length=4)

    wrong_prompt = _outputs(batch_size=1, input_length=16, output_length=4)
    wrong_prompt[0]["meta_info"]["prompt_tokens"] = 15
    with pytest.raises(RuntimeError, match="reports 15 prompt tokens"):
        _normalized_outputs(wrong_prompt, batch_size=1, input_length=16, output_length=4)

    wrong_completion = _outputs(batch_size=1, input_length=16, output_length=4)
    wrong_completion[0]["meta_info"]["completion_tokens"] = 3
    with pytest.raises(RuntimeError, match="reports 3 completion tokens"):
        _normalized_outputs(wrong_completion, batch_size=1, input_length=16, output_length=4)

    wrong_finish = _outputs(batch_size=1, input_length=16, output_length=4)
    wrong_finish[0]["meta_info"]["finish_reason"] = {"type": "stop"}
    with pytest.raises(RuntimeError, match="did not terminate at the declared length"):
        _normalized_outputs(wrong_finish, batch_size=1, input_length=16, output_length=4)


def _main_argv(tmp_path: Path, **updates: Any) -> list[str]:
    values = {
        "model": str(tmp_path),
        "output": str(tmp_path / "out.json"),
        "runner-oci-digest": f"sha256:{hashlib.sha256(b'runner').hexdigest()}",
        "subject-sha256": hashlib.sha256(b"subject").hexdigest(),
        "perturbation-id": "candidate",
        "tensor-parallel-size": "2",
        "batch-size": "2",
        "input-length": "4",
        "output-length": "2",
        "warmups": "0",
        "iterations": "2",
        "seed": "1",
        "mem-fraction-static": "0.5",
        "max-running-requests": "2",
        "max-total-tokens": "8",
        "chunked-prefill-size": "8",
        "max-prefill-tokens": "8",
    }
    values.update({key: str(value) for key, value in updates.items()})
    return [item for key, value in values.items() for item in (f"--{key}", value)]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"warmups": -1}, "warmups must be non-negative"),
        ({"seed": -1}, "seed must be non-negative"),
        ({"mem-fraction-static": 0.01}, "mem-fraction-static"),
        ({"max-running-requests": 1}, "max-running-requests"),
        ({"max-total-tokens": 7}, "max-total-tokens"),
        ({"max-prefill-tokens": 7}, "max-prefill-tokens"),
        ({"runner-oci-digest": "wrong"}, "must use sha256"),
        ({"subject-sha256": "wrong"}, "lowercase SHA-256"),
        ({"perturbation-id": ""}, "perturbation-id must be non-empty"),
    ],
)
def test_sglang_main_rejects_invalid_study_inputs_before_engine_import(
    tmp_path: Path,
    updates: Dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sglang_application_driver.main(_main_argv(tmp_path, **updates))


def test_sglang_main_rejects_runner_environment_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMMCANARY_RUNNER_OCI_DIGEST", f"sha256:{'0' * 64}")
    with pytest.raises(ValueError, match="disagrees with the container environment"):
        sglang_application_driver.main(_main_argv(tmp_path))
