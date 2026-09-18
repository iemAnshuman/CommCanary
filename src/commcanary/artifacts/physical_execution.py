"""Independent validation of physical runner measurements."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from ..errors import SchemaError
from ..formats import PHYSICAL_EXECUTION_EVIDENCE_SET_FORMAT, PHYSICAL_EXECUTION_MEASUREMENT_FORMAT
from ..statistics import median
from .json_codec import canonical_json_bytes

_RUNNER_PROTOCOL = "chakra-et-collective-graph.v1"

#: Setup phases in execution order. Named rather than totalled so a consumer
#: can attribute where a run's fixed cost went; the runner produces exactly
#: these keys and the validator refuses any other set.
SETUP_COST_PHASES = (
    "program_preparation",
    "runtime_initialization",
    "allocation_and_correctness",
    "warmups",
)
_EVIDENCE_METHODS = frozenset(
    {
        "reduced_decision_canary",
        "independent_exact_replay",
        "random_sampling",
        "stratified_sampling",
        "ddmin",
        "communication_only_microbenchmark",
    }
)


def validate_physical_execution_measurement(measurement: Mapping[str, Any]) -> None:
    """Recompute the internal commitments and measured summary of one run."""

    _closed_fields(
        measurement,
        {
            "format",
            "measurement_id",
            "role",
            "runner",
            "subject_sha256",
            "perturbation_id",
            "source_et_sha256",
            "executable_sha256",
            "projection_id",
            "selected_region_ids",
            "selected_node_ids",
            "execution",
            "correctness",
            "cost",
            "samples",
            "physical_metrics",
            "environment",
            "environment_sha256",
            "telemetry",
            "telemetry_assessment",
        },
        "physical execution measurement",
    )
    if measurement.get("format") != PHYSICAL_EXECUTION_MEASUREMENT_FORMAT:
        raise SchemaError("physical execution measurement format is unsupported")
    expected_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in measurement.items() if key != "measurement_id"})
    ).hexdigest()
    if measurement.get("measurement_id") != expected_id:
        raise SchemaError("physical execution measurement_id does not match canonical content")
    if measurement.get("role") not in {"trace_derived_reference", "reduced_decision_canary"}:
        raise SchemaError("physical execution measurement role is unsupported")
    runner = _as_mapping(measurement.get("runner"), "physical execution runner")
    _closed_fields(runner, {"oci_digest", "execution_protocol"}, "physical execution runner")
    _oci_digest(runner.get("oci_digest"))
    if runner.get("execution_protocol") != _RUNNER_PROTOCOL:
        raise SchemaError("physical execution runner protocol is unsupported")
    for field in ("subject_sha256", "source_et_sha256", "executable_sha256", "projection_id"):
        _sha256(measurement.get(field), f"physical execution {field}")
    if not isinstance(measurement.get("perturbation_id"), str) or not measurement["perturbation_id"]:
        raise SchemaError("physical execution perturbation_id must be non-empty")
    selected_regions = _unique_strings(measurement.get("selected_region_ids"), "physical execution selected regions")
    if not selected_regions:
        raise SchemaError("physical execution selected regions must not be empty")
    selected_nodes = _positive_unique_ints(measurement.get("selected_node_ids"), "physical execution selected nodes")

    execution = _as_mapping(measurement.get("execution"), "physical execution configuration")
    _closed_fields(
        execution,
        {
            "world_size",
            "warmups",
            "iterations",
            "disable_overlap",
            "communication_only",
            "rank_skew_us",
            "rank_skew_mechanism",
            "collective_buffer_slots",
            "workspace_bytes_max_rank",
        },
        "physical execution configuration",
    )
    world_size = _bounded_int(execution.get("world_size"), "physical execution world_size", 2, 65_536)
    _bounded_int(execution.get("warmups"), "physical execution warmups", 0, 1_000_000)
    iterations = _bounded_int(execution.get("iterations"), "physical execution iterations", 2, 1_000_000)
    if not isinstance(execution.get("disable_overlap"), bool):
        raise SchemaError("physical execution disable_overlap must be boolean")
    if not isinstance(execution.get("communication_only"), bool):
        raise SchemaError("physical execution communication_only must be boolean")
    _finite_nonnegative(execution.get("rank_skew_us"), "physical execution rank_skew_us")
    # Recorded so a reader can tell what the injected skew physically was: the
    # runner delays collective *issue* on the host, which is not a GPU arrival
    # guarantee, and evidence must not be read as if it were.
    if execution.get("rank_skew_mechanism") != "host_issue_monotonic_spin":
        raise SchemaError("physical execution rank_skew_mechanism is unsupported")
    _bounded_int(
        execution.get("collective_buffer_slots"),
        "physical execution collective_buffer_slots",
        1,
        1_024,
    )
    _bounded_int(
        execution.get("workspace_bytes_max_rank"),
        "physical execution workspace bytes",
        0,
        (1 << 63) - 1,
    )

    correctness = measurement.get("correctness")
    if not isinstance(correctness, list) or len(correctness) != world_size:
        raise SchemaError("physical execution correctness must contain exactly one row per rank")
    correctness_ranks: List[int] = []
    for index, raw_row in enumerate(correctness):
        row = _as_mapping(raw_row, f"physical execution correctness[{index}]")
        _closed_fields(
            row,
            {"rank", "passed", "validated_node_ids", "commitment_sha256"},
            f"physical execution correctness[{index}]",
        )
        correctness_ranks.append(_bounded_int(row.get("rank"), "physical execution correctness rank", 0, 65_535))
        if row.get("passed") is not True:
            raise SchemaError("physical execution correctness must pass on every rank")
        validated = _positive_unique_ints(row.get("validated_node_ids"), "physical execution validated node IDs")
        if set(validated) != set(selected_nodes):
            raise SchemaError("physical execution correctness does not validate every selected node")
        _sha256(row.get("commitment_sha256"), "physical execution correctness commitment")
    if correctness_ranks != list(range(world_size)):
        raise SchemaError("physical execution correctness rows must use complete rank order")

    samples = measurement.get("samples")
    if not isinstance(samples, list) or len(samples) != iterations:
        raise SchemaError("physical execution sample count does not match iterations")
    runtimes: List[float] = []
    peak_memory_values: List[int] = []
    for index, raw_sample in enumerate(samples):
        sample = _as_mapping(raw_sample, f"physical execution samples[{index}]")
        _closed_fields(
            sample,
            {
                "iteration",
                "cuda_seconds_by_rank",
                "host_seconds_by_rank",
                "physical_runtime_seconds",
                "peak_memory_bytes_by_rank",
            },
            f"physical execution samples[{index}]",
        )
        if sample.get("iteration") != index:
            raise SchemaError("physical execution iterations must be contiguous and zero-based")
        cuda = _positive_float_array(sample.get("cuda_seconds_by_rank"), world_size, "CUDA rank samples")
        _positive_float_array(sample.get("host_seconds_by_rank"), world_size, "host rank samples")
        runtime = _finite_positive(sample.get("physical_runtime_seconds"), "physical execution runtime")
        if not math.isclose(runtime, max(cuda), rel_tol=1e-12, abs_tol=1e-12):
            raise SchemaError("physical execution runtime must equal the slowest CUDA rank")
        peak = _nonnegative_int_array(sample.get("peak_memory_bytes_by_rank"), world_size, "peak-memory ranks")
        runtimes.append(runtime)
        peak_memory_values.extend(peak)

    metrics = _as_mapping(measurement.get("physical_metrics"), "physical execution metrics")
    _closed_fields(
        metrics,
        {
            "physical_runtime_seconds",
            "gpu_seconds",
            "gpu_count",
            "executed_collectives",
            "executed_flops",
            "peak_memory_bytes",
        },
        "physical execution metrics",
    )
    median_runtime = median(runtimes)
    metric_runtime = _finite_positive(metrics.get("physical_runtime_seconds"), "physical execution metric runtime")
    if not math.isclose(metric_runtime, median_runtime, rel_tol=1e-12, abs_tol=1e-12):
        raise SchemaError("physical execution metric runtime does not equal the sample median")
    if metrics.get("gpu_count") != world_size:
        raise SchemaError("physical execution metric GPU count does not equal world_size")
    gpu_seconds = _finite_positive(metrics.get("gpu_seconds"), "physical execution GPU seconds")
    if not math.isclose(gpu_seconds, median_runtime * world_size, rel_tol=1e-12, abs_tol=1e-12):
        raise SchemaError("physical execution GPU seconds do not recompute")
    for field in ("executed_collectives", "executed_flops"):
        _bounded_int(metrics.get(field), f"physical execution {field}", 0, (1 << 63) - 1)
    if metrics.get("peak_memory_bytes") != max(peak_memory_values):
        raise SchemaError("physical execution peak memory does not recompute")

    cost = _as_mapping(measurement.get("cost"), "physical execution cost")
    _closed_fields(
        cost,
        {
            "setup_seconds",
            "measured_seconds",
            "instrumentation_seconds",
            "total_seconds",
            "steady_state_seconds_per_iteration",
        },
        "physical execution cost",
    )
    setup = _as_mapping(cost.get("setup_seconds"), "physical execution setup cost")
    _closed_fields(setup, set(SETUP_COST_PHASES) | {"total"}, "physical execution setup cost")
    phase_values = [
        _finite_non_negative(setup.get(name), f"physical execution setup {name}") for name in SETUP_COST_PHASES
    ]
    setup_total = _finite_non_negative(setup.get("total"), "physical execution setup total")
    if not math.isclose(setup_total, math.fsum(phase_values), rel_tol=1e-9, abs_tol=1e-12):
        raise SchemaError("physical execution setup total does not equal the sum of its phases")
    measured_seconds = _finite_positive(cost.get("measured_seconds"), "physical execution measured seconds")
    _finite_non_negative(cost.get("instrumentation_seconds"), "physical execution instrumentation seconds")
    total_seconds = _finite_positive(cost.get("total_seconds"), "physical execution total seconds")
    if not math.isclose(total_seconds, setup_total + measured_seconds, rel_tol=1e-9, abs_tol=1e-12):
        raise SchemaError("physical execution total seconds do not recompute from setup and measured seconds")
    steady_state = _finite_positive(
        cost.get("steady_state_seconds_per_iteration"), "physical execution steady-state seconds"
    )
    if not math.isclose(steady_state, measured_seconds / iterations, rel_tol=1e-9, abs_tol=1e-12):
        raise SchemaError("physical execution steady-state seconds do not recompute from measured seconds")
    # Wall time across the measured loop must cover the CUDA time it contains.
    # A run reporting less has either mistimed the loop or excluded work from
    # it, and either way the reduction ratio computed from it is wrong.
    if measured_seconds + 1e-9 < math.fsum(runtimes):
        raise SchemaError("physical execution measured seconds are shorter than the samples they contain")

    environment = _as_mapping(measurement.get("environment"), "physical execution environment")
    expected_environment_id = hashlib.sha256(canonical_json_bytes(environment)).hexdigest()
    if measurement.get("environment_sha256") != expected_environment_id:
        raise SchemaError("physical execution environment_sha256 does not match canonical content")
    if environment.get("world_size") != world_size:
        raise SchemaError("physical execution environment world_size mismatch")

    telemetry = measurement.get("telemetry")
    if not isinstance(telemetry, list):
        raise SchemaError("physical execution telemetry must be an array")
    expected_phases = [
        "before_correctness",
        "before_measured_cycle_1",
        *(f"after_measured_cycle_{index + 1}" for index in range(iterations)),
        "final",
    ]
    if [row.get("phase") if isinstance(row, Mapping) else None for row in telemetry] != expected_phases:
        raise SchemaError("physical execution telemetry phases are incomplete or out of order")
    monotonic_values: List[int] = []
    for index, raw_snapshot in enumerate(telemetry):
        snapshot = _as_mapping(raw_snapshot, f"physical execution telemetry[{index}]")
        monotonic_values.append(
            _bounded_int(snapshot.get("monotonic_ns"), "physical execution telemetry monotonic_ns", 0, (1 << 63) - 1)
        )
        gpu_observation = _as_mapping(snapshot.get("gpu_observation"), "physical execution GPU telemetry")
        if gpu_observation.get("status") == "complete":
            if gpu_observation.get("returncode") != 0:
                raise SchemaError("complete physical execution GPU telemetry must have returncode zero")
            gpus = gpu_observation.get("gpus")
            if not isinstance(gpus, list) or len(gpus) != world_size:
                raise SchemaError("complete physical execution GPU telemetry must cover every rank device")
            if [gpu.get("index") if isinstance(gpu, Mapping) else None for gpu in gpus] != list(range(world_size)):
                raise SchemaError("physical execution GPU telemetry indices must use complete order")
        xid = _as_mapping(snapshot.get("xid_observation"), "physical execution Xid telemetry")
        lines = xid.get("lines")
        if not isinstance(lines, list) or any(not isinstance(line, str) for line in lines):
            raise SchemaError("physical execution Xid lines must be strings")
        if xid.get("sha256") != hashlib.sha256(canonical_json_bytes(lines)).hexdigest():
            raise SchemaError("physical execution Xid observation digest does not recompute")
    if any(right <= left for left, right in zip(monotonic_values, monotonic_values[1:])):
        raise SchemaError("physical execution telemetry monotonic timestamps must increase")
    expected_assessment = telemetry_assessment(telemetry, world_size=world_size)
    if measurement.get("telemetry_assessment") != expected_assessment:
        raise SchemaError("physical execution telemetry assessment does not recompute")


def build_physical_execution_evidence_set(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the complete raw candidate-measurement inventory for one study."""

    if not rows:
        raise SchemaError("physical execution evidence set requires measurements")
    prepared = [
        {
            "candidate_id": row.get("candidate_id"),
            "method": row.get("method"),
            "measurement": dict(_as_mapping(row.get("measurement"), "physical execution evidence measurement")),
        }
        for row in rows
    ]
    prepared.sort(
        key=lambda row: (
            str(row["candidate_id"]),
            str(_as_mapping(row["measurement"], "physical evidence sort row").get("perturbation_id")),
            str(_as_mapping(row["measurement"], "physical evidence sort row").get("subject_sha256")),
        )
    )
    first = _as_mapping(prepared[0]["measurement"], "physical execution evidence first measurement")
    runner = _as_mapping(first.get("runner"), "physical execution evidence runner")
    raw: Dict[str, Any] = {
        "format": PHYSICAL_EXECUTION_EVIDENCE_SET_FORMAT,
        "runner_oci_digest": runner.get("oci_digest"),
        "measurements": prepared,
    }
    raw["evidence_set_id"] = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    validate_physical_execution_evidence_set(raw)
    return raw


def validate_physical_execution_evidence_set(evidence_set: Mapping[str, Any]) -> None:
    """Recompute every raw candidate measurement and its candidate identity."""

    _closed_fields(
        evidence_set,
        {"format", "evidence_set_id", "runner_oci_digest", "measurements"},
        "physical execution evidence set",
    )
    if evidence_set.get("format") != PHYSICAL_EXECUTION_EVIDENCE_SET_FORMAT:
        raise SchemaError("physical execution evidence set format is unsupported")
    expected_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in evidence_set.items() if key != "evidence_set_id"})
    ).hexdigest()
    if evidence_set.get("evidence_set_id") != expected_id:
        raise SchemaError("physical execution evidence_set_id does not match canonical content")
    runner_oci_digest = _oci_digest(evidence_set.get("runner_oci_digest"))
    rows = evidence_set.get("measurements")
    if not isinstance(rows, list) or not rows:
        raise SchemaError("physical execution evidence measurements must be a non-empty array")
    ordering: List[Tuple[str, str, str]] = []
    measurement_ids: Set[str] = set()
    request_keys: Set[Tuple[str, str]] = set()
    for index, raw_row in enumerate(rows):
        row = _as_mapping(raw_row, f"physical execution evidence measurements[{index}]")
        _closed_fields(row, {"candidate_id", "method", "measurement"}, "physical execution evidence row")
        candidate_id = _sha256(row.get("candidate_id"), "physical execution evidence candidate_id")
        method = row.get("method")
        if method not in _EVIDENCE_METHODS:
            raise SchemaError("physical execution evidence method is unsupported")
        measurement = _as_mapping(row.get("measurement"), "physical execution evidence measurement")
        validate_physical_execution_measurement(measurement)
        runner = _as_mapping(measurement.get("runner"), "physical execution evidence measurement runner")
        if runner.get("oci_digest") != runner_oci_digest:
            raise SchemaError("physical execution evidence measurement runner does not match the set")
        expected_candidate_id = hashlib.sha256(
            canonical_json_bytes(
                {
                    "method": method,
                    "selected_region_ids": measurement["selected_region_ids"],
                    "executable_sha256": measurement["executable_sha256"],
                }
            )
        ).hexdigest()
        if candidate_id != expected_candidate_id:
            raise SchemaError("physical execution evidence candidate_id does not recompute")
        measurement_id = str(measurement["measurement_id"])
        request_key = (candidate_id, str(measurement["subject_sha256"]))
        if measurement_id in measurement_ids or request_key in request_keys:
            raise SchemaError("physical execution evidence measurements must be unique")
        measurement_ids.add(measurement_id)
        request_keys.add(request_key)
        ordering.append(
            (
                candidate_id,
                str(measurement["perturbation_id"]),
                str(measurement["subject_sha256"]),
            )
        )
    if ordering != sorted(ordering):
        raise SchemaError("physical execution evidence measurements are not in canonical order")


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _closed_fields(value: Mapping[str, Any], fields: Set[str], label: str) -> None:
    if set(value) != fields:
        raise SchemaError(f"{label} fields are not closed")


def _bounded_int(value: Any, label: str, lower: int, upper: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
        raise SchemaError(f"{label} must be an integer in [{lower}, {upper}]")
    return value


def _finite_non_negative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise SchemaError(f"{label} must be finite and non-negative")
    return number


def _finite_positive(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SchemaError(f"{label} must be finite and positive")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SchemaError(f"{label} must be finite and non-negative")
    return result


def _unique_strings(value: Any, label: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise SchemaError(f"{label} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise SchemaError(f"{label} must be unique")
    return tuple(value)


def _positive_unique_ints(value: Any, label: str) -> Tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in value)
        or len(value) != len(set(value))
    ):
        raise SchemaError(f"{label} must contain unique positive integers")
    return tuple(value)


def _positive_float_array(value: Any, length: int, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaError(f"{label} must contain exactly {length} values")
    return tuple(_finite_positive(item, label) for item in value)


def _nonnegative_int_array(value: Any, length: int, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaError(f"{label} must contain exactly {length} values")
    return tuple(_bounded_int(item, label, 0, (1 << 63) - 1) for item in value)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(f"{label} must be a lowercase SHA-256")
    return value


def _oci_digest(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SchemaError("physical runner OCI digest must use sha256:<64 lowercase hex>")
    _sha256(value.partition(":")[2], "physical runner OCI digest")
    return value


def telemetry_assessment(rows: Sequence[Mapping[str, Any]], *, world_size: int) -> Dict[str, Any]:
    issues: Set[str] = set()
    gpu_snapshots = [row["gpu_observation"] for row in rows]
    complete_gpu = all(snapshot["status"] == "complete" for snapshot in gpu_snapshots)
    if not complete_gpu:
        issues.add("gpu_telemetry_unavailable")
    uuid_orders = [tuple(gpu["uuid"] for gpu in snapshot["gpus"]) for snapshot in gpu_snapshots]
    stable_uuid_order = bool(uuid_orders) and all(order == uuid_orders[0] for order in uuid_orders)
    if not stable_uuid_order or (uuid_orders and len(uuid_orders[0]) != world_size):
        issues.add("gpu_uuid_order_changed_or_incomplete")

    xid_snapshots = [row["xid_observation"] for row in rows]
    xid_available = all(snapshot["status"] == "complete" for snapshot in xid_snapshots)
    if not xid_available:
        issues.add("xid_observation_unavailable")
    xid_changed = bool(xid_snapshots) and xid_snapshots[0]["sha256"] != xid_snapshots[-1]["sha256"]
    if xid_changed:
        issues.add("xid_observation_changed")

    slurm_snapshots = [row["slurm_node_observation"] for row in rows]
    slurm_available = all(snapshot["status"] == "complete" for snapshot in slurm_snapshots)
    if not slurm_available:
        issues.add("slurm_node_observation_unavailable")

    corrected_ecc_delta: List[int] = []
    uncorrected_ecc_delta: List[int] = []
    if complete_gpu and stable_uuid_order and uuid_orders and len(uuid_orders[0]) == world_size:
        for gpu_index in range(world_size):
            corrected_ecc_delta.append(
                int(gpu_snapshots[-1]["gpus"][gpu_index]["corrected_volatile_ecc"])
                - int(gpu_snapshots[0]["gpus"][gpu_index]["corrected_volatile_ecc"])
            )
            uncorrected_ecc_delta.append(
                int(gpu_snapshots[-1]["gpus"][gpu_index]["uncorrected_volatile_ecc"])
                - int(gpu_snapshots[0]["gpus"][gpu_index]["uncorrected_volatile_ecc"])
            )
        if any(value != 0 for value in corrected_ecc_delta):
            issues.add("corrected_ecc_changed")
        if any(value != 0 for value in uncorrected_ecc_delta):
            issues.add("uncorrected_ecc_changed")

    gpu_rows = [gpu for snapshot in gpu_snapshots for gpu in snapshot["gpus"]]
    return {
        "comparable": not issues,
        "issues": sorted(issues),
        "gpu_snapshot_count": len(gpu_snapshots),
        "stable_gpu_uuid_order": stable_uuid_order,
        "gpu_uuid_order": list(uuid_orders[0]) if stable_uuid_order and uuid_orders else [],
        "xid_observation_available": xid_available,
        "xid_observation_changed": xid_changed,
        "slurm_node_observation_available": slurm_available,
        "corrected_volatile_ecc_delta": corrected_ecc_delta,
        "uncorrected_volatile_ecc_delta": uncorrected_ecc_delta,
        "observed_pstates": sorted({str(gpu["pstate"]) for gpu in gpu_rows}),
        "sm_clock_mhz_min": min((float(gpu["sm_clock_mhz"]) for gpu in gpu_rows), default=None),
        "sm_clock_mhz_max": max((float(gpu["sm_clock_mhz"]) for gpu in gpu_rows), default=None),
        "memory_clock_mhz_min": min((float(gpu["memory_clock_mhz"]) for gpu in gpu_rows), default=None),
        "memory_clock_mhz_max": max((float(gpu["memory_clock_mhz"]) for gpu in gpu_rows), default=None),
        "temperature_c_min": min((float(gpu["temperature_c"]) for gpu in gpu_rows), default=None),
        "temperature_c_max": max((float(gpu["temperature_c"]) for gpu in gpu_rows), default=None),
        "power_w_min": min((float(gpu["power_w"]) for gpu in gpu_rows), default=None),
        "power_w_max": max((float(gpu["power_w"]) for gpu in gpu_rows), default=None),
    }


__all__ = [
    "build_physical_execution_evidence_set",
    "telemetry_assessment",
    "validate_physical_execution_evidence_set",
    "validate_physical_execution_measurement",
]
