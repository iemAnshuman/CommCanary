"""Replay application core composed from expansion, scheduler, and accumulation."""

from __future__ import annotations

import copy
import hashlib
import platform
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

from ..artifacts.canary import (
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
    iter_canary_logical_events,
    preflight_canary_expansion,
    validate_canary,
)
from ..artifacts.json_codec import canonical_json_bytes
from ..artifacts.report import validate_report
from ..artifacts.wire import MAX_TIME_US, JsonDict, as_float, as_int, replay_protocol_sha256
from ..errors import SchemaError
from ..formats import REPORT_FORMAT
from ..operation_identity import OperationIdentity
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, checked_multiply
from .accumulator import ReplayAccumulator
from .expansion import _iter_timing_samples
from .scheduler import _simulate_step

SIMULATION_MODEL_VERSION = "deterministic-scheduler-v4"
QUANTILE_METHOD = "linear-interpolated-sorted"
DEFAULT_MAX_REPLAY_EVENTS = 1_000_000

SUPPORTED_ABLATIONS = {
    "arrival_skew",
    "compute_overlap",
    "operation_ordering",
    "rare_tail_windows",
    "queue_reset_gaps",
    "pressure",
    "observed_exposed_us",
}


def _normalize_ablations(ablations: Optional[Iterable[str]]) -> frozenset[str]:
    if ablations is None:
        return frozenset()
    if isinstance(ablations, str):
        raw_items = [item.strip() for item in ablations.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in ablations if str(item).strip()]
    unknown = sorted(set(raw_items) - SUPPORTED_ABLATIONS)
    if unknown:
        raise SchemaError(f"unsupported replay ablation {unknown[0]!r}")
    return frozenset(raw_items)


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
    ablations: Optional[Iterable[str]] = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Replay a canary through a deterministic, queue-aware model.

    This is a simulator, not a physical NCCL executor. Reports include model
    calibration error when the source trace supplies ``observed_exposed_us``.
    """

    validate_canary(canary, limits=limits)
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
    if max_replay_events > limits.max_replay_events:
        raise SchemaError(f"max_replay_events cannot exceed resource policy limit={limits.max_replay_events}")
    ablation_set = _normalize_ablations(ablations)

    expansion = preflight_canary_expansion(canary.get("events", []), limits=limits)
    try:
        logical_events = checked_multiply(
            expansion.logical_timing_records,
            iterations,
            label="replay events",
        )
    except JsonResourceError as exc:
        raise SchemaError(str(exc)) from exc
    if logical_events > max_replay_events:
        raise SchemaError(f"replay would execute {logical_events} events, above max_replay_events={max_replay_events}")
    logical_steps = list(iter_canary_logical_events(canary.get("events", []), limits=limits))
    if "operation_ordering" in ablation_set:
        logical_steps = sorted(
            logical_steps,
            key=lambda step: OperationIdentity.from_mapping(step).scheduler_ordering_key(),
        )

    accumulator = ReplayAccumulator(include_samples=include_samples)
    sequence_index = 0
    for iteration in range(iterations):
        logical_clock_us = 0.0
        group_available_us: Dict[str, float] = {}
        for step in logical_steps:
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
                    ablations=ablation_set,
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
        "ablations": sorted(ablation_set),
    }
    protocol["sha256"] = replay_protocol_sha256(protocol, limits=limits)

    compiler = canary.get("compiler", {})
    report: JsonDict = {
        "format": REPORT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canary": {
            "sha256": _canary_sha256(canary),
            "execution_semantic_sha256": canary_execution_sha256(
                canary,
                limits=limits,
            ),
            "scheduler_execution_sha256": canary_scheduler_execution_sha256(
                canary,
                limits=limits,
            ),
            "calibration_evaluation_sha256": canary_calibration_sha256(
                canary,
                limits=limits,
            ),
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
            "ablations": sorted(ablation_set),
        },
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "workload": copy.deepcopy(canary.get("workload", {})),
        "canary_summary": copy.deepcopy(compiler),
        "metrics": accumulator.metrics(),
        "by_phase": accumulator.breakdown("phase"),
        "by_op": accumulator.breakdown("op"),
    }
    calibration = accumulator.calibration()
    if calibration is not None:
        report["calibration"] = calibration
    if include_samples:
        report["samples"] = accumulator.samples
    validate_report(report, limits=limits)
    return report


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


__all__ = [
    "DEFAULT_MAX_REPLAY_EVENTS",
    "QUANTILE_METHOD",
    "SIMULATION_MODEL_VERSION",
    "SUPPORTED_ABLATIONS",
    "replay_canary",
]
