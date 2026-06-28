from __future__ import annotations

import hashlib
import math
import platform
from array import array
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .schema import (
    MAX_TIME_US,
    REPORT_FORMAT,
    JsonDict,
    SchemaError,
    as_float,
    as_int,
    average_wait_us,
    canary_execution_sha256,
    canonical_json_bytes,
    percentile_from_sorted,
    replay_protocol_sha256,
    summarize_latencies,
    validate_canary,
    validate_report,
)

SIMULATION_MODEL_VERSION = "deterministic-scheduler-v4"
QUANTILE_METHOD = "linear-interpolated-sorted"
DEFAULT_MAX_REPLAY_EVENTS = 1_000_000
_MASK64 = (1 << 64) - 1


def replay_canary(
    canary: Mapping[str, Any],
    *,
    backend_label: str = "simulated-nccl",
    bandwidth_gbps: float = 55.0,
    latency_floor_us: float = 7.5,
    compute_pressure: float = 0.55,
    overlap_efficiency: float = 0.72,
    iterations: int = 1,
    seed: int = 7,
    include_samples: bool = False,
    max_replay_events: int = DEFAULT_MAX_REPLAY_EVENTS,
) -> JsonDict:
    """Replay a canary through a deterministic, queue-aware model.

    This is a simulator, not a physical NCCL executor. Reports include model
    calibration error when the source trace supplies ``observed_exposed_us``.
    """

    validate_canary(canary)
    if not isinstance(backend_label, str) or not backend_label.strip():
        raise SchemaError("backend_label must be a non-empty string")
    if not isinstance(include_samples, bool):
        raise SchemaError("include_samples must be a boolean")
    bandwidth_gbps = _positive_float(bandwidth_gbps, "bandwidth_gbps")
    latency_floor_us = _non_negative_float(latency_floor_us, "latency_floor_us")
    compute_pressure = _non_negative_float(compute_pressure, "compute_pressure")
    overlap_efficiency = as_float(overlap_efficiency)
    if not 0.0 <= overlap_efficiency <= 1.0:
        raise SchemaError("overlap_efficiency must be between 0 and 1")
    iterations = as_int(iterations)
    if iterations < 1:
        raise SchemaError("iterations must be at least 1")
    seed = as_int(seed)
    max_replay_events = as_int(max_replay_events)
    if max_replay_events < 1:
        raise SchemaError("max_replay_events must be at least 1")

    logical_events = _logical_event_count(canary) * iterations
    if logical_events > max_replay_events:
        raise SchemaError(
            f"replay would execute {logical_events} events, above max_replay_events={max_replay_events}"
        )

    accumulator = ReplayAccumulator(include_samples=include_samples)
    sequence_index = 0
    for iteration in range(iterations):
        logical_clock_us = 0.0
        group_available_us: Dict[str, float] = {}
        for step in canary.get("events", []):
            if not isinstance(step, Mapping):
                continue
            for timing_sample in _iter_timing_samples(step):
                core = _simulate_step(
                    step,
                    timing_sample=timing_sample,
                    sequence_index=sequence_index,
                    iteration=iteration,
                    seed=seed,
                    bandwidth_gbps=bandwidth_gbps,
                    latency_floor_us=latency_floor_us,
                    compute_pressure=compute_pressure,
                )
                logical_clock_us += core["gap_us"]
                if logical_clock_us > MAX_TIME_US:
                    raise SchemaError("replay logical clock exceeds maximum supported duration")
                group = str(core["scheduler_resource"])
                first_arrival_us = logical_clock_us
                last_arrival_us = first_arrival_us + core["arrival_skew_us"]
                collective_start_us = max(last_arrival_us, group_available_us.get(group, 0.0))
                queue_wait_us = collective_start_us - last_arrival_us
                completion_us = collective_start_us + core["collective_us"]
                if completion_us > MAX_TIME_US:
                    raise SchemaError("replay completion time exceeds maximum supported duration")
                total_us = completion_us - first_arrival_us
                hidden_us = min(total_us, max(0.0, core["compute_overlap_us"]) * overlap_efficiency)
                exposed_us = total_us - hidden_us
                group_available_us[group] = completion_us

                core.update(
                    {
                        "ready_us": first_arrival_us,
                        "first_arrival_us": first_arrival_us,
                        "last_arrival_us": last_arrival_us,
                        "collective_start_us": collective_start_us,
                        "queue_wait_us": queue_wait_us,
                        "completion_us": completion_us,
                        "total_us": total_us,
                        "hidden_us": hidden_us,
                        "exposed_us": exposed_us,
                    }
                )
                accumulator.add(core)
                sequence_index += 1

    protocol: JsonDict = {
        "model_name": "commcanary.replay",
        "model_version": SIMULATION_MODEL_VERSION,
        "seed": seed,
        "iterations": iterations,
        "quantile_method": QUANTILE_METHOD,
        "bandwidth_unit": "Gbit/s",
        "max_replay_events": max_replay_events,
    }
    protocol["sha256"] = replay_protocol_sha256(protocol)

    compiler = canary.get("compiler", {})
    report: JsonDict = {
        "format": REPORT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canary": {
            "sha256": _canary_sha256(canary),
            "execution_semantic_sha256": canary_execution_sha256(canary),
            "format": canary.get("format"),
            "source_events": compiler.get("source_events"),
        },
        "simulation_model": {"name": "commcanary.replay", "version": SIMULATION_MODEL_VERSION},
        "replay_protocol": protocol,
        "backend": {
            "label": backend_label,
            "mode": "deterministic-simulation",
            "bandwidth_gbps": bandwidth_gbps,
            "latency_floor_us": latency_floor_us,
            "compute_pressure": compute_pressure,
            "overlap_efficiency": overlap_efficiency,
            "seed": seed,
            "iterations": iterations,
            "bandwidth_unit": "Gbit/s",
        },
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "workload": canary.get("workload", {}),
        "canary_summary": compiler,
        "metrics": accumulator.metrics(),
        "by_phase": accumulator.breakdown("phase"),
        "by_op": accumulator.breakdown("op"),
    }
    calibration = accumulator.calibration()
    if calibration is not None:
        report["calibration"] = calibration
    if include_samples:
        report["samples"] = accumulator.samples
    validate_report(report)
    return report


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
) -> JsonDict:
    offsets = _sample_offsets(step, timing_sample)
    skew_us = arrival_skew = (
        max(0.0, max(offsets) - min(offsets))
        if offsets
        else as_float(step.get("arrival_skew_us"), 0.0)
    )
    wait_us = average_wait_us(offsets) if offsets else skew_us / 2.0
    bytes_ = as_int(step.get("bytes"))
    ranks = _sample_ranks(step)
    rank_count = max(1, as_int(step.get("rank_count"), len(ranks) or 1))
    op = str(step.get("op", "unknown"))
    group = str(step.get("group", "default"))
    concurrent_groups = max(1, as_int(step.get("concurrent_groups"), 1))
    pressure = min(
        1.5,
        max(0.0, compute_pressure * as_float(timing_sample.get("compute_pressure"), 0.5) / 0.55),
    )
    overlap_us = as_float(timing_sample.get("compute_overlap_us"), 0.0)

    collective_us = _collective_duration_us(
        op=op,
        bytes_=bytes_,
        rank_count=rank_count,
        bandwidth_gbps=bandwidth_gbps,
        latency_floor_us=latency_floor_us,
        pressure=pressure,
        concurrent_groups=concurrent_groups,
    )
    uniforms = _counter_uniforms(seed, iteration, _noise_identity(step, timing_sample))
    collective_us += _tail_noise_us(
        uniforms,
        skew_us=skew_us,
        pressure=pressure,
        concurrent_groups=concurrent_groups,
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
        "scheduler_resource": _scheduler_resource_label(group, ranks),
        "gap_us": as_float(timing_sample.get("gap_us"), 0.0),
        "compute_before_us": as_float(timing_sample.get("compute_before_us"), 0.0),
        "arrival_skew_us": arrival_skew,
        "avg_rank_wait_us": wait_us,
        "compute_overlap_us": overlap_us,
        "collective_us": collective_us,
        "concurrent_groups": concurrent_groups,
    }
    if "observed_exposed_us" in timing_sample:
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
) -> float:
    baseline_u, branch_u, tail_u = uniforms
    tail_probability = min(
        0.35,
        0.015 + skew_us / 180.0 + pressure * 0.035 + max(0, concurrent_groups - 1) * 0.04,
    )
    baseline_jitter = -0.45 + baseline_u * 1.35
    if branch_u > tail_probability:
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
    counter = (
        (seed & _MASK64)
        ^ ((iteration + 1) * 0xD6E8FEB86659FD93)
        ^ identity_hash
    ) & _MASK64
    values = []
    for lane in range(3):
        bits = _splitmix64(counter ^ (lane * 0x9E3779B97F4A7C15))
        values.append(((bits >> 11) & ((1 << 53) - 1)) / float(1 << 53))
    return values[0], values[1], values[2]


def _noise_identity(step: Mapping[str, Any], timing_sample: Mapping[str, Any]) -> JsonDict:
    return {
        "phase": str(step.get("phase", "unknown")),
        "op": str(step.get("op", "unknown")),
        "ranks": list(_sample_ranks(step)),
        "group": str(step.get("group", "default")),
        "arrival_offsets_us": [round(value, 9) for value in _sample_offsets(step, timing_sample)],
        "occurrence": as_int(step.get("execution_occurrence_base"), 0)
        + as_int(timing_sample.get("noise_occurrence"), 0),
    }


def _stable_identity_hash(identity: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(canonical_json_bytes(identity)).digest()
    return int.from_bytes(digest[:8], "big") & _MASK64


def _sample_offsets(step: Mapping[str, Any], timing_sample: Mapping[str, Any]) -> List[float]:
    offsets = timing_sample.get("arrival_offsets_us")
    if isinstance(offsets, list):
        return [as_float(value) for value in offsets]
    raw = step.get("arrival_offsets_us")
    if isinstance(raw, list):
        return [as_float(value) for value in raw]
    skew = as_float(step.get("arrival_skew_us"), 0.0)
    rank_count = max(1, as_int(step.get("rank_count"), len(step.get("ranks", [])) or 1))
    return [0.0 for _ in range(max(0, rank_count - 1))] + [skew]


def _sample_ranks(step: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = step.get("ranks", [])
    if not isinstance(raw, list):
        return ()
    return tuple(as_int(rank) for rank in raw)


def _scheduler_resource_label(group: str, ranks: Tuple[int, ...]) -> str:
    if group and group != "default":
        return group
    if ranks:
        return "default:ranks=" + ",".join(str(rank) for rank in ranks)
    return group or "default"


def _fallback_timing(step: Mapping[str, Any]) -> JsonDict:
    sample: JsonDict = {
        "gap_us": step.get("gap_us", 0.0),
        "arrival_offsets_us": step.get("arrival_offsets_us", []),
        "compute_before_us": step.get("compute_before_us", 0.0),
        "compute_overlap_us": step.get("compute_overlap_us", 0.0),
        "compute_pressure": step.get("compute_pressure", 0.5),
        "weight": step.get("repeat", 1),
    }
    if "observed_exposed_us" in step:
        sample["observed_exposed_us"] = step.get("observed_exposed_us")
    if step.get("compute_fields_uncertain") is True:
        sample["compute_fields_uncertain"] = True
        sample["uncertain_weight"] = sample["weight"]
    return sample


def _iter_timing_samples(step: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    samples = step.get("timing_samples")
    if not isinstance(samples, list) or not samples:
        yield from _sample_repetitions(_fallback_timing(step))
        return
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        pattern = sample.get("timing_pattern")
        if isinstance(pattern, list) and pattern:
            yield from _pattern_repetitions(sample, pattern)
        else:
            yield from _sample_repetitions(sample)


def _pattern_repetitions(parent: Mapping[str, Any], pattern: List[Any]) -> Iterable[Mapping[str, Any]]:
    repeats = as_int(parent.get("pattern_repeats"), 1)
    parent_occurrence = _occurrence_base(parent)
    expected_gap_sum = 0.0
    child_weight = 0
    for child in pattern:
        if isinstance(child, Mapping):
            expected_gap_sum += _record_gap_sum(child)
            child_weight += as_int(child.get("weight"), 1)
    expected_gap_sum *= repeats
    parent_gap_sum = _record_gap_sum(parent)
    residual = parent_gap_sum - expected_gap_sum
    total_emitted = child_weight * repeats

    emitted = 0
    for _repeat_index in range(repeats):
        for child in pattern:
            if isinstance(child, Mapping):
                for item in _sample_repetitions(child):
                    emitted += 1
                    item = dict(item)
                    item["noise_occurrence"] = parent_occurrence + emitted - 1
                    if emitted == total_emitted and abs(residual) > 0.0:
                        item["gap_us"] = as_float(item.get("gap_us"), 0.0) + residual
                    yield item


def _sample_repetitions(sample: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    weight = as_int(sample.get("weight"), 1)
    uncertain_weight = as_int(
        sample.get(
            "uncertain_weight",
            weight if sample.get("compute_fields_uncertain") is True else 0,
        )
    )
    gap_sum_us = _record_gap_sum(sample)
    base_gap_us = gap_sum_us / weight
    occurrence_base = _occurrence_base(sample)
    consumed = 0.0
    for index in range(weight):
        item = dict(sample)
        gap_us = gap_sum_us - consumed if index == weight - 1 else base_gap_us
        if index != weight - 1:
            consumed += gap_us
        item["gap_us"] = gap_us
        item["weight"] = 1
        item["noise_occurrence"] = occurrence_base + index
        if index < uncertain_weight:
            item["compute_fields_uncertain"] = True
            item["uncertain_weight"] = 1
        else:
            item.pop("compute_fields_uncertain", None)
            item.pop("uncertain_weight", None)
        yield item


def _occurrence_base(sample: Mapping[str, Any]) -> int:
    if "source_start" in sample:
        return as_int(sample.get("source_start"))
    return as_int(sample.get("source_index"), 0)


def _record_gap_sum(sample: Mapping[str, Any]) -> float:
    if "gap_sum_us" in sample:
        return as_float(sample.get("gap_sum_us"))
    return as_float(sample.get("gap_us"), 0.0) * as_int(sample.get("weight"), 1)


def _logical_event_count(canary: Mapping[str, Any]) -> int:
    total = 0
    for step in canary.get("events", []):
        if not isinstance(step, Mapping):
            continue
        samples = step.get("timing_samples")
        if isinstance(samples, list) and samples:
            total += sum(as_int(sample.get("weight"), 1) for sample in samples if isinstance(sample, Mapping))
        else:
            total += as_int(step.get("repeat"), 1)
    return total


class ReplayAccumulator:
    def __init__(self, *, include_samples: bool) -> None:
        self.include_samples = include_samples
        self.samples: List[JsonDict] = []
        self.exposed = array("d")
        self.skew = array("d")
        self.wait = array("d")
        self.hidden_total = 0.0
        self.total = 0.0
        self.phase_values: Dict[str, array] = {}
        self.op_values: Dict[str, array] = {}
        self.observed = array("d")
        self.modeled_for_observed = array("d")

    def add(self, sample: Mapping[str, Any]) -> None:
        exposed_us = as_float(sample.get("exposed_us"))
        self.exposed.append(exposed_us)
        self.skew.append(as_float(sample.get("arrival_skew_us")))
        self.wait.append(as_float(sample.get("avg_rank_wait_us")))
        self.hidden_total += as_float(sample.get("hidden_us"))
        self.total += as_float(sample.get("total_us"))
        phase = str(sample.get("phase", "unknown"))
        op = str(sample.get("op", "unknown"))
        self.phase_values.setdefault(phase, array("d")).append(exposed_us)
        self.op_values.setdefault(op, array("d")).append(exposed_us)
        if "observed_exposed_us" in sample:
            self.observed.append(as_float(sample.get("observed_exposed_us")))
            self.modeled_for_observed.append(exposed_us)
        if self.include_samples:
            self.samples.append(_round_sample(sample))

    def metrics(self) -> JsonDict:
        skew = sorted(self.skew)
        wait = sorted(self.wait)
        hidden_pct = (self.hidden_total / self.total * 100.0) if self.total else 0.0
        result = summarize_latencies(self.exposed)
        result.update(
            {
                "arrival_skew_median_us": round(percentile_from_sorted(skew, 50.0), 3),
                "arrival_skew_p95_us": round(percentile_from_sorted(skew, 95.0), 3),
                "arrival_skew_max_us": round(skew[-1], 3) if skew else 0.0,
                "avg_rank_wait_median_us": round(percentile_from_sorted(wait, 50.0), 3),
                "communication_hidden_pct": round(hidden_pct, 2),
            }
        )
        return result

    def breakdown(self, key: str) -> List[JsonDict]:
        buckets = self.phase_values if key == "phase" else self.op_values
        rows: List[JsonDict] = []
        for label, values in sorted(buckets.items()):
            row: JsonDict = {"name": label}
            row.update(summarize_latencies(values))
            rows.append(row)
        return rows

    def calibration(self) -> Optional[JsonDict]:
        if not self.observed:
            return None
        errors = [model - observed for model, observed in zip(self.modeled_for_observed, self.observed)]
        absolute = sorted(abs(value) for value in errors)
        percentage = [
            abs(model - observed) / observed * 100.0
            for model, observed in zip(self.modeled_for_observed, self.observed)
            if observed > 0.0
        ]
        return {
            "signal": "observed_exposed_us",
            "count": len(errors),
            "mean_absolute_error_us": round(sum(absolute) / len(absolute), 3),
            "median_absolute_error_us": round(percentile_from_sorted(absolute, 50.0), 3),
            "p95_absolute_error_us": round(percentile_from_sorted(absolute, 95.0), 3),
            "max_absolute_error_us": round(absolute[-1], 3),
            "mean_bias_us": round(sum(errors) / len(errors), 3),
            "mean_absolute_percentage_error_pct": round(sum(percentage) / len(percentage), 3)
            if percentage
            else 0.0,
            "percentage_count": len(percentage),
        }


def _round_sample(sample: Mapping[str, Any]) -> JsonDict:
    total_us = round(as_float(sample.get("total_us")), 3)
    hidden_us = min(total_us, round(as_float(sample.get("hidden_us")), 3))
    rounded: JsonDict = {}
    for key, value in sample.items():
        if isinstance(value, float):
            rounded[key] = round(value, 9) if key == "gap_us" else round(value, 3)
        else:
            rounded[key] = value
    rounded["total_us"] = total_us
    rounded["hidden_us"] = hidden_us
    rounded["exposed_us"] = round(total_us - hidden_us, 3)
    return rounded


def _positive_float(value: Any, name: str) -> float:
    parsed = as_float(value)
    if parsed <= 0.0:
        raise SchemaError(f"{name} must be positive")
    return parsed


def _non_negative_float(value: Any, name: str) -> float:
    parsed = as_float(value)
    if parsed < 0.0:
        raise SchemaError(f"{name} must be non-negative")
    return parsed


def _canary_sha256(canary: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in canary.items() if key != "created_at"}
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()
