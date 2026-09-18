from __future__ import annotations

import pytest

from commcanary.errors import SchemaError
from commcanary.services.traffic_synthesis import synthesize_traffic_trace

INPUT_DISTRIBUTION = {"median_tokens": 240, "sigma": 0.85, "minimum_tokens": 8, "maximum_tokens": 4096}
OUTPUT_DISTRIBUTION = {"median_tokens": 180, "sigma": 0.95, "minimum_tokens": 1, "maximum_tokens": 2048}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"duration_us": 0}, "duration_us must be a positive integer"),
        ({"mean_interarrival_us": 0}, "mean_interarrival_us must be a positive integer"),
        ({"seed": -1}, "seed must be a non-negative integer"),
        ({"input_distribution": {**INPUT_DISTRIBUTION, "sigma": -1.0}}, "sigma must be finite and positive"),
    ],
)
def test_synthesis_refuses_unusable_parameters(overrides: dict, message: str) -> None:
    parameters = {
        "name": "precheck",
        "duration_us": 30_000_000,
        "mean_interarrival_us": 250_000,
        "input_distribution": INPUT_DISTRIBUTION,
        "output_distribution": OUTPUT_DISTRIBUTION,
        "seed": 1,
    }
    parameters.update(overrides)
    with pytest.raises(SchemaError, match=message):
        synthesize_traffic_trace(**parameters)


def test_lognormal_draws_are_clamped_to_the_declared_bounds() -> None:
    trace = synthesize_traffic_trace(
        name="clamped",
        duration_us=60_000_000,
        mean_interarrival_us=100_000,
        # A very wide sigma forces the tails past both bounds, so the clamp is
        # what is being observed here rather than the distribution.
        input_distribution={"median_tokens": 100, "sigma": 4.0, "minimum_tokens": 90, "maximum_tokens": 110},
        output_distribution=OUTPUT_DISTRIBUTION,
        seed=11,
    )
    lengths = [request["input_length"] for request in trace["requests"]]
    assert min(lengths) == 90
    assert max(lengths) == 110
