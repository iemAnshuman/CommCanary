"""Deterministic collective scheduler model."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Tuple

from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.wire import JsonDict, as_float, as_int, average_wait_us
from ..operation_identity import OperationIdentity
from .expansion import _sample_offsets

_MASK64 = (1 << 64) - 1


def _simulate_step(
    step: Mapping[str, Any],
    *,
    timing_sample: Mapping[str, Any],
    sequence_index: int,
    iteration: int,
    seed: int,
    bandwidth_gbps: float,
    latency_floor_us: float,
    compute_pressure: float,
    ablations: Iterable[str] = (),
) -> JsonDict:
    ablation_set = set(ablations)
    offsets = _sample_offsets(step, timing_sample)
    if "arrival_skew" in ablation_set:
        offsets = [0.0 for _ in offsets]
    skew_us = arrival_skew = (
        max(0.0, max(offsets) - min(offsets)) if offsets else as_float(step.get("arrival_skew_us"), 0.0)
    )
    wait_us = average_wait_us(offsets) if offsets else skew_us / 2.0
    bytes_ = as_int(step.get("bytes"))
    operation_identity = OperationIdentity.from_mapping(step)
    ranks = operation_identity.ranks
    rank_count = max(1, as_int(step.get("rank_count"), len(ranks) or 1))
    op = str(step.get("op", "unknown"))
    group = str(step.get("group", "default"))
    concurrent_groups = max(1, as_int(step.get("concurrent_groups"), 1))
    timing_pressure = 0.55 if "pressure" in ablation_set else as_float(timing_sample.get("compute_pressure"), 0.5)
    pressure = min(
        1.5,
        max(0.0, compute_pressure * timing_pressure / 0.55),
    )
    overlap_us = 0.0 if "compute_overlap" in ablation_set else as_float(timing_sample.get("compute_overlap_us"), 0.0)
    gap_us = 0.0 if "queue_reset_gaps" in ablation_set else as_float(timing_sample.get("gap_us"), 0.0)

    collective_us = _collective_duration_us(
        op=op,
        bytes_=bytes_,
        rank_count=rank_count,
        bandwidth_gbps=bandwidth_gbps,
        latency_floor_us=latency_floor_us,
        pressure=pressure,
        concurrent_groups=concurrent_groups,
    )
    noise_occurrence = as_int(step.get("execution_occurrence_base"), 0) + as_int(
        timing_sample.get("noise_occurrence"), 0
    )
    uniforms = _counter_uniforms(
        seed,
        iteration,
        operation_identity.noise_identity(offsets, occurrence=noise_occurrence).to_wire(),
    )
    collective_us += _tail_noise_us(
        uniforms,
        skew_us=skew_us,
        pressure=pressure,
        concurrent_groups=concurrent_groups,
        disable_tail="rare_tail_windows" in ablation_set,
    )
    collective_us = max(0.5, collective_us)

    sample: JsonDict = {
        "index": sequence_index,
        "iteration": iteration,
        "phase": step.get("phase", "unknown"),
        "op": op,
        "bytes": bytes_,
        "rank_count": rank_count,
        "group": group,
        "scheduler_resource": operation_identity.scheduler_resource_label(),
        "gap_us": gap_us,
        "compute_before_us": as_float(timing_sample.get("compute_before_us"), 0.0),
        "arrival_skew_us": arrival_skew,
        "avg_rank_wait_us": wait_us,
        "compute_overlap_us": overlap_us,
        "collective_us": collective_us,
        "concurrent_groups": concurrent_groups,
    }
    if "observed_exposed_us" in timing_sample and "observed_exposed_us" not in ablation_set:
        sample["observed_exposed_us"] = as_float(timing_sample.get("observed_exposed_us"))
    if timing_sample.get("compute_fields_uncertain") is True:
        sample["compute_fields_uncertain"] = True
    return sample


def _collective_duration_us(
    *,
    op: str,
    bytes_: int,
    rank_count: int,
    bandwidth_gbps: float,
    latency_floor_us: float,
    pressure: float,
    concurrent_groups: int,
) -> float:
    log_ranks = math.log(max(2, rank_count), 2.0)
    # Gbit/s -> bits/us: 1 Gbit/s == 1000 bits/us.
    transfer_us = (bytes_ * 8.0) / (bandwidth_gbps * 1000.0)
    op_factor = {
        "all_reduce": 2.0 * (rank_count - 1) / max(1, rank_count),
        "reduce_scatter": (rank_count - 1) / max(1, rank_count),
        "all_gather": (rank_count - 1) / max(1, rank_count),
        "all_to_all": 1.6,
        "broadcast": 0.8,
        "send": 1.0,
        "recv": 1.0,
        "point_to_point": 1.0,
    }.get(op, 1.0)
    contention = 1.0 + max(0, concurrent_groups - 1) * 0.22
    pressure_penalty = 1.0 + pressure * 0.18
    # A constant startup term keeps the model continuous and monotonic across
    # message sizes; thresholded penalties caused larger messages to be faster.
    startup = latency_floor_us + log_ranks * 1.15 + 4.0
    return startup + transfer_us * op_factor * contention * pressure_penalty


def _tail_noise_us(
    uniforms: Tuple[float, float, float],
    *,
    skew_us: float,
    pressure: float,
    concurrent_groups: int,
    disable_tail: bool = False,
) -> float:
    baseline_u, branch_u, tail_u = uniforms
    tail_probability = min(
        0.35,
        0.015 + skew_us / 180.0 + pressure * 0.035 + max(0, concurrent_groups - 1) * 0.04,
    )
    baseline_jitter = -0.45 + baseline_u * 1.35
    if disable_tail or branch_u > tail_probability:
        return baseline_jitter
    mean = 3.5 + pressure * 8.0 + skew_us * 0.08
    tail = -math.log(max(1e-15, 1.0 - tail_u)) * mean
    return baseline_jitter + tail


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _counter_uniforms(seed: int, iteration: int, identity: Mapping[str, Any]) -> Tuple[float, float, float]:
    identity_hash = _stable_identity_hash(identity)
    counter = ((seed & _MASK64) ^ ((iteration + 1) * 0xD6E8FEB86659FD93) ^ identity_hash) & _MASK64
    values = []
    for lane in range(3):
        bits = _splitmix64(counter ^ (lane * 0x9E3779B97F4A7C15))
        values.append(((bits >> 11) & ((1 << 53) - 1)) / float(1 << 53))
    return values[0], values[1], values[2]


def _stable_identity_hash(identity: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(canonical_json_bytes(identity)).digest()
    return int.from_bytes(digest[:8], "big") & _MASK64
