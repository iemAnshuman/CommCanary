"""Deterministic synthetic traffic for a sustained serving reference workload.

This is the producer half of ``artifacts.traffic_trace``. It lives in
``services`` because it draws random variates, and the artifact layer is a
validation surface with a deliberately narrow standard-library allowance.

Everything here is reproducible from the seed and the declared parameters,
both of which are carried in the artifact, so a reader can rederive the exact
request sequence from the trace alone. What it produces is still marked
``synthetic``: it may drive a sensitivity precheck, and it may not qualify a
canary.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping

from ..artifacts.traffic_trace import (
    request_identifier,
    summarize_requests,
    traffic_trace_id,
    validate_length_distribution,
    validate_traffic_trace,
)
from ..errors import SchemaError
from ..formats import TRAFFIC_TRACE_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaError(f"{label} must be a positive integer")
    return int(value)


def _non_negative_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{label} must be a non-negative integer")
    return int(value)


def sample_token_length(generator: random.Random, distribution: Mapping[str, Any]) -> int:
    """Draw one clamped lognormal token length.

    ``median_tokens`` is the distribution's median, so ``mu`` is its natural
    log: for a lognormal the median is ``exp(mu)``. Declaring the median rather
    than ``mu`` means the parameter in the artifact is the one a reader can
    interpret without redoing the algebra.
    """

    mu = math.log(float(distribution["median_tokens"]))
    sigma = float(distribution["sigma"])
    minimum = int(distribution["minimum_tokens"])
    maximum = int(distribution["maximum_tokens"])
    drawn = int(round(generator.lognormvariate(mu, sigma)))
    return max(minimum, min(maximum, drawn))


def synthesize_traffic_trace(
    *,
    name: str,
    duration_us: int,
    mean_interarrival_us: int,
    input_distribution: Mapping[str, Any],
    output_distribution: Mapping[str, Any],
    seed: int,
    poisson: bool = True,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Dict[str, Any]:
    """Build a deterministic synthetic traffic trace.

    Arrivals are a Poisson process by default, drawn as exponential gaps with
    the declared mean. ``poisson=False`` produces a fixed-cadence trace, which
    is useful as a control: it isolates whether a result depends on burstiness
    or only on the offered load.
    """

    _positive_int(duration_us, "duration_us")
    _positive_int(mean_interarrival_us, "mean_interarrival_us")
    _non_negative_int(seed, "seed")
    validate_length_distribution(input_distribution, "input distribution")
    validate_length_distribution(output_distribution, "output distribution")

    # One generator per axis, each seeded distinctly, so changing the arrival
    # rate does not reshuffle the length sequence. Without this a rate sweep
    # would silently vary two things at once.
    arrivals = random.Random(seed)
    lengths = random.Random(seed + 1)

    requests: List[Dict[str, Any]] = []
    offset = 0
    while True:
        gap = int(round(arrivals.expovariate(1.0 / mean_interarrival_us))) if poisson else mean_interarrival_us
        offset += max(0, gap)
        if offset > duration_us:
            break
        if len(requests) >= limits.max_traffic_requests:
            raise SchemaError(f"synthetic traffic exceeds limit={limits.max_traffic_requests} requests")
        requests.append(
            {
                "request_id": request_identifier(len(requests)),
                "arrival_offset_us": offset,
                "input_length": sample_token_length(lengths, input_distribution),
                "output_length": sample_token_length(lengths, output_distribution),
            }
        )

    if not requests:
        raise SchemaError("synthetic traffic produced no requests; duration is shorter than one arrival gap")

    trace: Dict[str, Any] = {
        "format": TRAFFIC_TRACE_FORMAT,
        "name": name,
        "source": "synthetic",
        "arrival_model": {
            "kind": "poisson" if poisson else "deterministic",
            "seed": seed if poisson else None,
            "mean_interarrival_us": mean_interarrival_us,
        },
        "length_model": {
            "kind": "lognormal",
            "seed": seed + 1,
            "input": dict(input_distribution),
            "output": dict(output_distribution),
        },
        "requests": requests,
        "summary": summarize_requests(requests),
    }
    trace["trace_id"] = traffic_trace_id(trace)
    validate_traffic_trace(trace, limits=limits)
    return trace


__all__ = ["sample_token_length", "synthesize_traffic_trace"]
