"""Hierarchical, simultaneous evaluation for replicated exact-work gates."""

from __future__ import annotations

import math
import random
import statistics
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast

from ..decision_gate_schedule import (
    CONFIGURATION_ORDER_METHOD,
    REPLICATED_ORDER_METHOD,
    WILLIAMS_CYCLE_LENGTH,
    configuration_order_by_repetition,
    frozen_schedule_inventory,
)
from ..harness import JSONResourceLimits, canonical_sha256, sha256_hex, strict_json_loads
from ..lib.executor_artifact import ExecutorArtifact
from .decision_fidelity import (
    REPORTED_METRIC_TO_VERDICT_FIELD,
    DecisionFidelityError,
    _frozen_evaluator_record,
    _integer,
    _median,
    _number,
    _object,
    _pair_label,
    _pair_threshold,
    _percentile,
    _policy_input_binding,
    _representation_metrics,
    _sha256,
    decision_fidelity_policy_sha256,
)
from .pipeline import ANALYSIS_SCHEMA
from .schemas import (
    DECISION_FIDELITY_POLICY_SCHEMA_V2,
    DECISION_FIDELITY_VERDICT_SCHEMA_V2,
    PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2,
)

_POLICY_LIMITS = JSONResourceLimits(max_document_bytes=1024 * 1024, max_items=10_000)
_REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")
_EVALUATED_REPRESENTATIONS = ("exact_work", "stratified", "isolated", "no_overlap", "no_rank_skew")
_UNCERTAINTY_REPRESENTATIONS = ("source", "exact_work")
_STABILITY_REPRESENTATIONS = ("source", "exact_work", "stratified", "isolated")
_BINDING_ENVIRONMENT_FIELDS = {
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_GPUS",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_PROCID",
    "SLURM_STEP_GPUS",
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")

RepetitionSamples = Dict[int, Dict[str, Dict[str, Tuple[float, ...]]]]
MedianVector = Dict[str, Dict[str, float]]


def _environment_identity(
    raw: Any,
    field: str,
    *,
    comparability: Mapping[str, Any],
) -> Tuple[str, str]:
    environment = _object(
        raw,
        field,
        {
            "schema",
            "invariants",
            "telemetry",
            "probe_policy",
            "platform_sha256",
            "observation_sha256",
        },
    )
    if environment["schema"] != "commcanary.rostam.runtime-observation.v3":
        raise DecisionFidelityError(f"{field}.schema is unsupported")
    declared_observation_sha256 = _sha256(
        environment["observation_sha256"],
        f"{field}.observation_sha256",
    )
    platform_sha256 = _sha256(environment["platform_sha256"], f"{field}.platform_sha256")
    invariants = _object(
        environment["invariants"],
        f"{field}.invariants",
        {"driver_version", "nccl_library_sha256", "gpu_count", "gpus", "topology", "binding"},
    )
    _sha256(invariants["nccl_library_sha256"], f"{field}.invariants.nccl_library_sha256")
    if not isinstance(invariants["driver_version"], str) or not invariants["driver_version"]:
        raise DecisionFidelityError(f"{field}.invariants.driver_version is invalid")
    gpus = invariants["gpus"]
    required_gpu_fields = {
        "index",
        "uuid",
        "name",
        "driver_version",
        "pci_bus_id",
        "persistence_mode",
        "power_limit_w",
    }
    if not isinstance(gpus, list) or len(gpus) != 4 or invariants["gpu_count"] != len(gpus):
        raise DecisionFidelityError(f"{field}.invariants.gpus must contain the four-GPU inventory")
    power_limits: Dict[int, float] = {}
    for index, raw_gpu in enumerate(gpus):
        gpu = _object(raw_gpu, f"{field}.invariants.gpus[{index}]", required_gpu_fields)
        if gpu["index"] != index or gpu["driver_version"] != invariants["driver_version"]:
            raise DecisionFidelityError(f"{field}.invariants.gpus[{index}] identity is invalid")
        for text_field in ("uuid", "name", "pci_bus_id", "persistence_mode"):
            if not isinstance(gpu[text_field], str) or not gpu[text_field]:
                raise DecisionFidelityError(f"{field}.invariants.gpus[{index}].{text_field} is invalid")
        power_limits[index] = _number(
            gpu["power_limit_w"],
            f"{field}.invariants.gpus[{index}].power_limit_w",
            minimum=0.000001,
        )
    topology = _object(invariants["topology"], f"{field}.invariants.topology", {"method", "text"})
    binding = _object(
        invariants["binding"],
        f"{field}.invariants.binding",
        {"environment", "cpu_affinity", "cpu_affinity_method"},
    )
    if (
        topology["method"] != "nvidia-smi topo -m"
        or not isinstance(topology["text"], str)
        or not topology["text"]
        or binding["cpu_affinity_method"] != "sched_getaffinity"
        or not isinstance(binding["cpu_affinity"], list)
        or not binding["cpu_affinity"]
    ):
        raise DecisionFidelityError(f"{field} lacks required topology or affinity evidence")
    binding_environment = binding["environment"]
    if (
        not isinstance(binding_environment, Mapping)
        or set(binding_environment) != _BINDING_ENVIRONMENT_FIELDS
        or any(
            value is not None and (not isinstance(value, str) or not value) for value in binding_environment.values()
        )
        or binding["cpu_affinity"] != sorted(set(binding["cpu_affinity"]))
    ):
        raise DecisionFidelityError(f"{field}.invariants.binding is invalid")
    for cpu in binding["cpu_affinity"]:
        _integer(cpu, f"{field}.binding.cpu_affinity[]", maximum=1_000_000)
    probe_policy = _object(
        environment["probe_policy"],
        f"{field}.probe_policy",
        {"timeout_seconds", "max_output_bytes_per_stream"},
    )
    _integer(probe_policy["timeout_seconds"], f"{field}.probe_policy.timeout_seconds", minimum=1)
    _integer(
        probe_policy["max_output_bytes_per_stream"],
        f"{field}.probe_policy.max_output_bytes_per_stream",
        minimum=1,
    )
    platform = {
        "driver_version": invariants["driver_version"],
        "gpu_count": invariants["gpu_count"],
        "gpus": list(gpus),
        "topology": dict(topology),
        "binding": dict(binding),
    }
    if canonical_sha256(platform) != platform_sha256:
        raise DecisionFidelityError(f"{field}.platform_sha256 does not recompute")

    telemetry = _object(environment["telemetry"], f"{field}.telemetry", {"method", "pre", "post"})
    if telemetry["method"] != "bounded-pre-post-nvidia-smi.v1":
        raise DecisionFidelityError(f"{field}.telemetry method is unsupported")
    temperatures: Dict[str, List[int]] = {}
    captured_at: Dict[str, datetime] = {}
    for phase in ("pre", "post"):
        snapshot = _object(
            telemetry[phase],
            f"{field}.telemetry.{phase}",
            {"captured_at", "gpus", "node_state"},
        )
        if not isinstance(snapshot["captured_at"], str):
            raise DecisionFidelityError(f"{field}.telemetry.{phase}.captured_at is invalid")
        try:
            captured_at[phase] = datetime.strptime(snapshot["captured_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as exc:
            raise DecisionFidelityError(f"{field}.telemetry.{phase}.captured_at is invalid") from exc
        node_state = _object(
            snapshot["node_state"],
            f"{field}.telemetry.{phase}.node_state",
            {"method", "text"},
        )
        if (
            node_state["method"] != "scontrol show node --oneliner HOSTNAME"
            or not isinstance(node_state["text"], str)
            or not node_state["text"]
        ):
            raise DecisionFidelityError(f"{field}.telemetry.{phase}.node_state is invalid")
        telemetry_gpus = snapshot["gpus"]
        if not isinstance(telemetry_gpus, list) or len(telemetry_gpus) != len(gpus):
            raise DecisionFidelityError(f"{field}.telemetry.{phase}.gpus is incomplete")
        temperatures[phase] = []
        for index, raw_gpu in enumerate(telemetry_gpus):
            gpu = _object(
                raw_gpu,
                f"{field}.telemetry.{phase}.gpus[{index}]",
                {
                    "index",
                    "performance_state",
                    "temperature_c",
                    "power_draw_w",
                    "sm_clock_mhz",
                    "memory_clock_mhz",
                },
            )
            if gpu["index"] != index or not isinstance(gpu["performance_state"], str) or not gpu["performance_state"]:
                raise DecisionFidelityError(f"{field}.telemetry.{phase}.gpus[{index}] identity is invalid")
            temperature = _integer(
                gpu["temperature_c"],
                f"{field}.telemetry.{phase}.gpus[{index}].temperature_c",
                minimum=int(comparability["minimum_temperature_c"]),
                maximum=int(comparability["maximum_temperature_c"]),
            )
            temperatures[phase].append(temperature)
            power_draw = _number(
                gpu["power_draw_w"],
                f"{field}.telemetry.{phase}.gpus[{index}].power_draw_w",
                minimum=0.0,
            )
            if power_draw / power_limits[index] > float(comparability["maximum_power_draw_to_limit_ratio"]):
                raise DecisionFidelityError(f"{field}.telemetry.{phase}.gpus[{index}] exceeds its power range")
            _integer(
                gpu["sm_clock_mhz"],
                f"{field}.telemetry.{phase}.gpus[{index}].sm_clock_mhz",
                maximum=int(comparability["maximum_sm_clock_mhz"]),
            )
            _integer(
                gpu["memory_clock_mhz"],
                f"{field}.telemetry.{phase}.gpus[{index}].memory_clock_mhz",
                maximum=int(comparability["maximum_memory_clock_mhz"]),
            )
    if any(
        abs(pre - post) > int(comparability["maximum_pre_post_temperature_delta_c"])
        for pre, post in zip(temperatures["pre"], temperatures["post"])
    ):
        raise DecisionFidelityError(f"{field}.telemetry temperature delta exceeds policy")
    if captured_at["post"] <= captured_at["pre"]:
        raise DecisionFidelityError(f"{field}.telemetry timestamps are not ordered")
    normalized_observation_sha256 = canonical_sha256(
        {
            "invariants": dict(invariants),
            "telemetry": dict(telemetry),
            "probe_policy": dict(probe_policy),
        }
    )
    if declared_observation_sha256 != normalized_observation_sha256:
        raise DecisionFidelityError(f"{field}.observation_sha256 does not recompute")
    return normalized_observation_sha256, platform_sha256


def _cycle_telemetry_summary(
    raw: Any,
    field: str,
    *,
    node: str,
    expected_gpus: Sequence[Mapping[str, Any]],
    comparability: Mapping[str, Any],
) -> Dict[str, Any]:
    telemetry = _object(raw, field, {"schema", "method", "snapshots"})
    if (
        telemetry["schema"] != "commcanary.rostam.decision-gate-cycle-telemetry.v1"
        or telemetry["method"] != "bounded-between-six-row-cycles.v1"
    ):
        raise DecisionFidelityError(f"{field} contract is unsupported")
    expected_labels = [
        "before_warmup",
        "before_measured_cycle_1",
        "after_measured_cycle_1",
        "after_measured_cycle_2",
        "after_measured_cycle_3",
        "after_measured_cycle_4",
        "final",
    ]
    snapshots = telemetry["snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != len(expected_labels):
        raise DecisionFidelityError(f"{field}.snapshots inventory is incomplete")
    expected_uuids = [str(gpu["uuid"]) for gpu in expected_gpus]
    power_limits = [float(gpu["power_limit_w"]) for gpu in expected_gpus]
    allowed_performance_states = set(comparability["allowed_performance_states"])
    allowed_throttle_states = set(comparability["allowed_throttle_reasons_active"])
    allowed_node_states = set(comparability["allowed_node_states"])
    timestamps: List[datetime] = []
    temperatures: List[int] = []
    power_draws: List[float] = []
    sm_clocks: List[int] = []
    memory_clocks: List[int] = []
    performance_states: Set[str] = set()
    first_ecc: Optional[List[Tuple[int, int]]] = None
    previous_ecc: Optional[List[Tuple[int, int]]] = None
    first_xid: Optional[int] = None
    previous_xid: Optional[int] = None
    first_xid_sha256: Optional[str] = None
    final_xid_sha256: Optional[str] = None
    for snapshot_index, (raw_snapshot, expected_label) in enumerate(zip(snapshots, expected_labels)):
        snapshot_field = f"{field}.snapshots[{snapshot_index}]"
        snapshot = _object(
            raw_snapshot,
            snapshot_field,
            {"label", "captured_at", "gpus", "node_state", "xid"},
        )
        captured_at = snapshot["captured_at"]
        if snapshot["label"] != expected_label or not isinstance(captured_at, str):
            raise DecisionFidelityError(f"{snapshot_field} identity is invalid")
        try:
            timestamps.append(datetime.strptime(captured_at, "%Y-%m-%dT%H:%M:%S.%fZ"))
        except ValueError as exc:
            raise DecisionFidelityError(f"{snapshot_field}.captured_at is invalid") from exc
        node_state = _object(
            snapshot["node_state"],
            f"{snapshot_field}.node_state",
            {"method", "node", "state"},
        )
        if (
            node_state["method"] != "scontrol show node --oneliner HOSTNAME"
            or node_state["node"] != node
            or node_state["state"] not in allowed_node_states
        ):
            raise DecisionFidelityError(f"{snapshot_field}.node_state is outside policy")
        raw_gpus = snapshot["gpus"]
        if not isinstance(raw_gpus, list) or len(raw_gpus) != len(expected_gpus):
            raise DecisionFidelityError(f"{snapshot_field}.gpus is incomplete")
        uuids: List[str] = []
        ecc: List[Tuple[int, int]] = []
        for gpu_index, raw_gpu in enumerate(raw_gpus):
            gpu_field = f"{snapshot_field}.gpus[{gpu_index}]"
            gpu = _object(
                raw_gpu,
                gpu_field,
                {
                    "index",
                    "uuid",
                    "performance_state",
                    "temperature_c",
                    "power_draw_w",
                    "sm_clock_mhz",
                    "memory_clock_mhz",
                    "throttle_reasons_active",
                    "ecc_corrected_volatile_total",
                    "ecc_uncorrected_volatile_total",
                },
            )
            uuid = gpu["uuid"]
            performance_state = gpu["performance_state"]
            throttle_state = gpu["throttle_reasons_active"]
            if (
                gpu["index"] != gpu_index
                or not isinstance(uuid, str)
                or uuid != expected_uuids[gpu_index]
                or not isinstance(performance_state, str)
                or performance_state not in allowed_performance_states
                or not isinstance(throttle_state, str)
                or throttle_state not in allowed_throttle_states
            ):
                raise DecisionFidelityError(f"{gpu_field} identity or state is outside policy")
            temperature = _integer(
                gpu["temperature_c"],
                f"{gpu_field}.temperature_c",
                minimum=int(comparability["minimum_temperature_c"]),
                maximum=int(comparability["maximum_temperature_c"]),
            )
            power_draw = _number(gpu["power_draw_w"], f"{gpu_field}.power_draw_w", minimum=0.0)
            if power_draw / power_limits[gpu_index] > float(comparability["maximum_power_draw_to_limit_ratio"]):
                raise DecisionFidelityError(f"{gpu_field}.power_draw_w is outside policy")
            sm_clock = _integer(
                gpu["sm_clock_mhz"],
                f"{gpu_field}.sm_clock_mhz",
                minimum=int(comparability["minimum_sm_clock_mhz"]),
                maximum=int(comparability["maximum_sm_clock_mhz"]),
            )
            memory_clock = _integer(
                gpu["memory_clock_mhz"],
                f"{gpu_field}.memory_clock_mhz",
                minimum=int(comparability["minimum_memory_clock_mhz"]),
                maximum=int(comparability["maximum_memory_clock_mhz"]),
            )
            corrected = _integer(
                gpu["ecc_corrected_volatile_total"],
                f"{gpu_field}.ecc_corrected_volatile_total",
                maximum=2**63 - 1,
            )
            uncorrected = _integer(
                gpu["ecc_uncorrected_volatile_total"],
                f"{gpu_field}.ecc_uncorrected_volatile_total",
                maximum=2**63 - 1,
            )
            uuids.append(uuid)
            ecc.append((corrected, uncorrected))
            temperatures.append(temperature)
            power_draws.append(power_draw)
            sm_clocks.append(sm_clock)
            memory_clocks.append(memory_clock)
            performance_states.add(performance_state)
        if uuids != expected_uuids:
            raise DecisionFidelityError(f"{snapshot_field}.gpus changed UUID order")
        if first_ecc is None:
            first_ecc = ecc
        if previous_ecc is not None and any(
            corrected < old_corrected or uncorrected < old_uncorrected
            for (corrected, uncorrected), (old_corrected, old_uncorrected) in zip(ecc, previous_ecc)
        ):
            raise DecisionFidelityError(f"{snapshot_field}.gpus ECC counters regressed")
        previous_ecc = ecc
        xid = _object(snapshot["xid"], f"{snapshot_field}.xid", {"method", "event_count", "window_sha256"})
        xid_count = _integer(xid["event_count"], f"{snapshot_field}.xid.event_count", maximum=1_000_000)
        xid_sha256 = _sha256(xid["window_sha256"], f"{snapshot_field}.xid.window_sha256")
        if xid["method"] != "journalctl --dmesg --boot --no-pager --grep NVRM.*Xid":
            raise DecisionFidelityError(f"{snapshot_field}.xid method is unsupported")
        if previous_xid is not None and xid_count < previous_xid:
            raise DecisionFidelityError(f"{snapshot_field}.xid event counter regressed")
        if first_xid is None:
            first_xid = xid_count
            first_xid_sha256 = xid_sha256
        previous_xid = xid_count
        final_xid_sha256 = xid_sha256
    if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
        raise DecisionFidelityError(f"{field}.snapshots timestamps are not strictly ordered")
    if first_ecc is None or previous_ecc is None or first_xid is None or previous_xid is None:
        raise DecisionFidelityError(f"{field}.snapshots inventory is incomplete")
    corrected_delta = max(current[0] - first[0] for first, current in zip(first_ecc, previous_ecc))
    uncorrected_delta = max(current[1] - first[1] for first, current in zip(first_ecc, previous_ecc))
    xid_delta = previous_xid - first_xid
    if (
        corrected_delta > int(comparability["maximum_ecc_corrected_delta"])
        or uncorrected_delta > int(comparability["maximum_ecc_uncorrected_delta"])
        or xid_delta > int(comparability["maximum_xid_event_delta"])
        or (xid_delta == 0 and first_xid_sha256 != final_xid_sha256)
    ):
        raise DecisionFidelityError(f"{field} records a disallowed ECC or Xid delta")
    return {
        "snapshot_count": len(snapshots),
        "minimum_temperature_c": min(temperatures),
        "maximum_temperature_c": max(temperatures),
        "minimum_power_draw_w": min(power_draws),
        "maximum_power_draw_w": max(power_draws),
        "minimum_sm_clock_mhz": min(sm_clocks),
        "maximum_sm_clock_mhz": max(sm_clocks),
        "minimum_memory_clock_mhz": min(memory_clocks),
        "maximum_memory_clock_mhz": max(memory_clocks),
        "performance_states": sorted(performance_states),
        "maximum_ecc_corrected_delta": corrected_delta,
        "maximum_ecc_uncorrected_delta": uncorrected_delta,
        "xid_event_delta": xid_delta,
    }


def validate_decision_fidelity_policy_v2(raw: Any) -> Dict[str, Any]:
    """Validate the v2 policy without relying on optional JSON Schema tooling."""

    policy = _object(
        raw,
        "decision fidelity policy v2",
        {
            "schema",
            "policy_id",
            "scope",
            "measurement",
            "comparison",
            "pass_criteria",
            "outcomes",
            "outcome_precedence",
            "claim_boundary",
        },
    )
    if policy["schema"] != DECISION_FIDELITY_POLICY_SCHEMA_V2:
        raise DecisionFidelityError("decision fidelity policy v2 schema is unsupported")
    policy_id = _sha256(policy["policy_id"], "decision fidelity policy v2.policy_id")
    if policy_id != decision_fidelity_policy_sha256(policy):
        raise DecisionFidelityError("decision fidelity policy v2 ID does not recompute")

    scope = _object(
        policy["scope"],
        "decision fidelity policy v2.scope",
        {"workload_id", "supported_domain", "configuration_ids", "representations"},
    )
    configurations = scope["configuration_ids"]
    if (
        scope["workload_id"] != "decision-gate-exact-replicated"
        or not isinstance(scope["supported_domain"], str)
        or not scope["supported_domain"]
        or not isinstance(configurations, list)
        or len(configurations) < 2
        or configurations != sorted(set(configurations))
    ):
        raise DecisionFidelityError("decision fidelity policy v2 scope is invalid")
    representations = _object(scope["representations"], "decision fidelity policy v2.scope.representations")
    if dict(representations) != {
        "source": "trace_derived_reference",
        "exact_work": "exact_materialization_control",
        "stratified": "sampling_baseline",
        "isolated": "incumbent_baseline",
        "no_overlap": "causal_ablation",
        "no_rank_skew": "causal_ablation",
    }:
        raise DecisionFidelityError("decision fidelity policy v2 representation roles are invalid")

    measurement = _object(
        policy["measurement"],
        "decision fidelity policy v2.measurement",
        {
            "allocation_policy",
            "configuration_repetitions",
            "cross_configuration_pairing",
            "configuration_order_method",
            "configuration_order_by_repetition",
            "maximum_repetition_span_seconds",
            "required_scheduler_evidence",
            "order_method",
            "representation_schedule",
            "timing_semantics",
            "warmup",
            "measured_repetitions",
            "required_correctness",
            "required_environment_evidence",
            "environment_comparability",
            "max_relative_iqr_pct",
            "require_distinct_job_ids",
            "retry_policy",
        },
    )
    expected_measurement = {
        "allocation_policy": "one-fresh-exclusive-allocation-per-configuration-cell",
        "cross_configuration_pairing": "none-independent-scheduler-allocations",
        "configuration_order_method": CONFIGURATION_ORDER_METHOD,
        "order_method": REPLICATED_ORDER_METHOD,
        "timing_semantics": "maximum-rank-cuda-event-whole-program-duration",
        "measured_repetitions": 24,
        "required_correctness": "passed",
        "required_environment_evidence": [
            "cpu_affinity",
            "gpu_identity",
            "gpu_persistence_mode",
            "gpu_power_limit",
            "gpu_topology",
            "nccl_library_digest",
            "between_cycle_gpu_node_xid_ecc_telemetry",
            "pre_post_gpu_telemetry",
            "pre_post_node_state",
        ],
        "require_distinct_job_ids": True,
        "retry_policy": "infrastructure-failure-only-never-retry-for-noise",
        "required_scheduler_evidence": [
            "planned_position",
            "scheduler_start_time",
            "node",
            "elapsed_from_repetition_start_seconds",
            "chunk_identifier",
        ],
    }
    if any(measurement.get(field) != expected for field, expected in expected_measurement.items()):
        raise DecisionFidelityError("decision fidelity policy v2 measurement semantics are unsupported")
    configuration_repetitions = _integer(
        measurement["configuration_repetitions"],
        "configuration_repetitions",
        minimum=5,
        maximum=10,
    )
    warmup = _integer(measurement["warmup"], "warmup", minimum=1, maximum=100)
    if warmup % WILLIAMS_CYCLE_LENGTH:
        raise DecisionFidelityError("decision fidelity policy v2 warmup must contain complete Williams cycles")
    maximum_repetition_span_seconds = _integer(
        measurement["maximum_repetition_span_seconds"],
        "maximum_repetition_span_seconds",
        minimum=1,
        maximum=86_400,
    )
    if maximum_repetition_span_seconds != 3600:
        raise DecisionFidelityError("decision fidelity policy v2 repetition span is unsupported")
    expected_configuration_order = [
        list(row) for row in configuration_order_by_repetition(cast(Sequence[str], configurations))
    ][:configuration_repetitions]
    if measurement["configuration_order_by_repetition"] != expected_configuration_order:
        raise DecisionFidelityError("decision fidelity policy v2 configuration order is not the frozen design")
    _number(measurement["max_relative_iqr_pct"], "max_relative_iqr_pct", minimum=0.0, maximum=1000.0)
    environment_comparability = _object(
        measurement["environment_comparability"],
        "decision fidelity policy v2.measurement.environment_comparability",
        {
            "allowed_node_states",
            "allowed_performance_states",
            "allowed_throttle_reasons_active",
            "maximum_ecc_corrected_delta",
            "maximum_ecc_uncorrected_delta",
            "require_identical_platform_fingerprint",
            "minimum_temperature_c",
            "maximum_temperature_c",
            "maximum_pre_post_temperature_delta_c",
            "maximum_power_draw_to_limit_ratio",
            "minimum_sm_clock_mhz",
            "maximum_sm_clock_mhz",
            "minimum_memory_clock_mhz",
            "maximum_memory_clock_mhz",
            "maximum_xid_event_delta",
        },
    )
    if environment_comparability["require_identical_platform_fingerprint"] is not True:
        raise DecisionFidelityError("decision fidelity policy v2 must require one platform fingerprint")
    minimum_temperature = _integer(
        environment_comparability["minimum_temperature_c"],
        "minimum_temperature_c",
        minimum=-100,
        maximum=200,
    )
    maximum_temperature = _integer(
        environment_comparability["maximum_temperature_c"],
        "maximum_temperature_c",
        minimum=-100,
        maximum=200,
    )
    if minimum_temperature > maximum_temperature:
        raise DecisionFidelityError("decision fidelity policy v2 temperature range is inverted")
    _integer(
        environment_comparability["maximum_pre_post_temperature_delta_c"],
        "maximum_pre_post_temperature_delta_c",
        maximum=200,
    )
    _number(
        environment_comparability["maximum_power_draw_to_limit_ratio"],
        "maximum_power_draw_to_limit_ratio",
        minimum=0.000001,
        maximum=10.0,
    )
    if environment_comparability["allowed_node_states"] != ["ALLOCATED"]:
        raise DecisionFidelityError("decision fidelity policy v2 node states are unsupported")
    if environment_comparability["allowed_performance_states"] != ["P0", "P2", "P8"]:
        raise DecisionFidelityError("decision fidelity policy v2 performance states are unsupported")
    if environment_comparability["allowed_throttle_reasons_active"] != ["0x0000000000000000"]:
        raise DecisionFidelityError("decision fidelity policy v2 throttle states are unsupported")
    minimum_sm_clock = _integer(
        environment_comparability["minimum_sm_clock_mhz"],
        "minimum_sm_clock_mhz",
        minimum=1,
    )
    maximum_sm_clock = _integer(
        environment_comparability["maximum_sm_clock_mhz"],
        "maximum_sm_clock_mhz",
        minimum=1,
    )
    minimum_memory_clock = _integer(
        environment_comparability["minimum_memory_clock_mhz"],
        "minimum_memory_clock_mhz",
        minimum=1,
    )
    maximum_memory_clock = _integer(
        environment_comparability["maximum_memory_clock_mhz"],
        "maximum_memory_clock_mhz",
        minimum=1,
    )
    if minimum_sm_clock > maximum_sm_clock or minimum_memory_clock > maximum_memory_clock:
        raise DecisionFidelityError("decision fidelity policy v2 clock range is inverted")
    for field in (
        "maximum_ecc_corrected_delta",
        "maximum_ecc_uncorrected_delta",
        "maximum_xid_event_delta",
    ):
        if _integer(environment_comparability[field], field) != 0:
            raise DecisionFidelityError(f"decision fidelity policy v2 {field} is unsupported")
    expected_schedule = frozen_schedule_inventory(
        configuration_repetitions=configuration_repetitions,
        iterations=int(measurement["measured_repetitions"]),
        warmup=warmup,
    )
    if measurement["representation_schedule"] != expected_schedule:
        raise DecisionFidelityError("decision fidelity policy v2 representation schedule is not the frozen design")

    comparison = _object(
        policy["comparison"],
        "decision fidelity policy v2.comparison",
        {
            "primary_metric",
            "configuration_pair_order",
            "pair_label_method",
            "absolute_tie_threshold_us",
            "relative_tie_threshold_pct",
            "relative_threshold_reference",
            "classification_error_counting",
            "uncertainty",
            "reported_metrics",
        },
    )
    if (
        comparison.get("primary_metric") != "median-of-configuration-repetition-medians-us"
        or comparison.get("configuration_pair_order") != "lexicographic-unordered-pairs"
        or comparison.get("pair_label_method") != "larger-of-absolute-or-relative-tie-band.v1"
        or comparison.get("relative_threshold_reference") != "smaller_pair_median"
    ):
        raise DecisionFidelityError("decision fidelity policy v2 comparison method is unsupported")
    _number(comparison["absolute_tie_threshold_us"], "absolute tie threshold", minimum=0.0)
    _number(comparison["relative_tie_threshold_pct"], "relative tie threshold", minimum=0.0)
    classification = _object(
        comparison["classification_error_counting"],
        "decision fidelity policy v2 classification",
        {"source_direction_candidate_tie", "source_tie_candidate_direction", "opposite_direction"},
    )
    if dict(classification) != {
        "source_direction_candidate_tie": "false_negative",
        "source_tie_candidate_direction": "false_positive",
        "opposite_direction": "one_false_negative_and_one_false_positive",
    }:
        raise DecisionFidelityError("decision fidelity policy v2 classification semantics are unsupported")
    uncertainty = _object(
        comparison["uncertainty"],
        "decision fidelity policy v2 uncertainty",
        {
            "method",
            "confidence",
            "resamples",
            "seed",
            "outer_unit",
            "inner_unit",
            "simultaneous_scope",
            "boundary_crossing_outcome",
        },
    )
    expected_uncertainty = {
        "method": "hierarchical-unpaired-repetition-bootstrap-policy-margin-standardized-max.v3",
        "outer_unit": "configuration-repetition-resampled-independently-per-configuration",
        "inner_unit": "complete-six-iteration-williams-cycle-within-cell",
        "simultaneous_scope": "source-and-exact-work-pair-margins-exact-work-metrics-and-criteria",
        "boundary_crossing_outcome": "inconclusive",
    }
    if any(uncertainty.get(field) != expected for field, expected in expected_uncertainty.items()):
        raise DecisionFidelityError("decision fidelity policy v2 uncertainty semantics are unsupported")
    confidence = _number(uncertainty["confidence"], "bootstrap confidence")
    if not 0.5 < confidence < 1.0:
        raise DecisionFidelityError("bootstrap confidence must be greater than 0.5 and less than 1")
    _integer(uncertainty["resamples"], "bootstrap resamples", minimum=100, maximum=100_000)
    _integer(uncertainty["seed"], "bootstrap seed", maximum=2**63 - 1)
    if comparison["reported_metrics"] != list(REPORTED_METRIC_TO_VERDICT_FIELD.values()):
        raise DecisionFidelityError("decision fidelity policy v2 reported metric inventory is unsupported")

    criteria = _object(policy["pass_criteria"], "decision fidelity policy v2.pass_criteria")
    expected_criteria = {
        "exact_work_min_pairwise_agreement",
        "exact_work_min_kendall_tau_b",
        "exact_work_max_false_negative_count",
        "exact_work_max_false_positive_count",
        "exact_work_max_median_absolute_relative_error_pct",
        "exact_work_max_p95_absolute_relative_error_pct",
        "exact_work_min_agreement_advantage_pairs_over_isolated",
        "exact_work_min_agreement_advantage_pairs_over_stratified",
    }
    if set(criteria) != expected_criteria:
        raise DecisionFidelityError("decision fidelity policy v2 pass criteria are incomplete")
    _number(criteria["exact_work_min_pairwise_agreement"], "minimum pairwise agreement", minimum=0, maximum=1)
    _number(criteria["exact_work_min_kendall_tau_b"], "minimum Kendall tau-b", minimum=-1, maximum=1)
    for field in (
        "exact_work_max_false_negative_count",
        "exact_work_max_false_positive_count",
        "exact_work_min_agreement_advantage_pairs_over_isolated",
        "exact_work_min_agreement_advantage_pairs_over_stratified",
    ):
        _integer(criteria[field], field)
    for field in (
        "exact_work_max_median_absolute_relative_error_pct",
        "exact_work_max_p95_absolute_relative_error_pct",
    ):
        _number(criteria[field], field, minimum=0)

    outcomes = _object(policy["outcomes"], "decision fidelity policy v2.outcomes")
    if set(outcomes) != {"pass", "fail", "inconclusive", "incomparable"} or any(
        not isinstance(value, str) or not value for value in outcomes.values()
    ):
        raise DecisionFidelityError("decision fidelity policy v2 outcomes are invalid")
    if policy["outcome_precedence"] != ["incomparable", "fail", "inconclusive", "pass"]:
        raise DecisionFidelityError("decision fidelity policy v2 outcome precedence is unsupported")
    boundary = _object(
        policy["claim_boundary"],
        "decision fidelity policy v2.claim_boundary",
        {
            "mode",
            "exact_work_claim",
            "reduced_canary_claim",
            "cost_claim",
            "generality_claim",
            "independent_operator_claim",
        },
    )
    if dict(boundary) != {
        "mode": "exact_materialization_conformance_control",
        "exact_work_claim": "identical_compiled_instruction_path_measurement_floor",
        "reduced_canary_claim": "not_evaluated",
        "cost_claim": "not_evaluated_by_exact_materialization_control",
        "generality_claim": "not_evaluated_beyond_the_declared_supported_domain",
        "independent_operator_claim": "not_evaluated_by_this_campaign",
    }:
        raise DecisionFidelityError("decision fidelity policy v2 claim boundary is unsupported")
    return cast(Dict[str, Any], dict(policy))


def _nested_median_vector(
    samples: RepetitionSamples,
    *,
    configuration_repetitions: Sequence[int],
    configurations: Sequence[str],
) -> MedianVector:
    return {
        configuration: {
            representation: _median(
                [
                    _median(samples[repetition][configuration][representation])
                    for repetition in configuration_repetitions
                ]
            )
            for representation in _REPRESENTATIONS
        }
        for configuration in configurations
    }


def _policy_margin(
    first: float,
    second: float,
    *,
    target_label: str,
    comparison: Mapping[str, Any],
) -> float:
    difference = second - first
    threshold = _pair_threshold(first, second, comparison)
    if target_label == "first_faster":
        return difference - threshold
    if target_label == "second_faster":
        return -difference - threshold
    if target_label == "tie":
        return threshold - abs(difference)
    raise DecisionFidelityError(f"unknown pair label {target_label!r}")


def _pair_rows(
    vector: MedianVector,
    *,
    configurations: Sequence[str],
    comparison: Mapping[str, Any],
    targets: Optional[Mapping[Tuple[str, str, str], str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for first, second in combinations(configurations, 2):
        representations: Dict[str, Any] = {}
        for representation in _REPRESENTATIONS:
            first_median = vector[first][representation]
            second_median = vector[second][representation]
            observed_label = _pair_label(first_median, second_median, comparison)
            target = observed_label if targets is None else targets[(first, second, representation)]
            representations[representation] = {
                "observed_label": observed_label,
                "difference_us": second_median - first_median,
                "tie_threshold_us": _pair_threshold(first_median, second_median, comparison),
                "policy_margin_us": _policy_margin(
                    first_median,
                    second_median,
                    target_label=target,
                    comparison=comparison,
                ),
            }
        rows.append(
            {
                "first_configuration_id": first,
                "second_configuration_id": second,
                "representations": representations,
            }
        )
    return rows


def _metrics(
    vector: MedianVector,
    *,
    configurations: Sequence[str],
    pair_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    singleton_samples = {
        configuration: {representation: (vector[configuration][representation],) for representation in _REPRESENTATIONS}
        for configuration in configurations
    }
    return {
        representation: _representation_metrics(
            representation,
            configurations=configurations,
            samples=singleton_samples,
            pair_rows=pair_rows,
        )
        for representation in _EVALUATED_REPRESENTATIONS
    }


def _criterion_values(metrics: Mapping[str, Any], criteria: Mapping[str, Any]) -> Dict[str, Tuple[float, str, float]]:
    exact = cast(Mapping[str, Any], metrics["exact_work"])
    isolated = cast(Mapping[str, Any], metrics["isolated"])
    stratified = cast(Mapping[str, Any], metrics["stratified"])
    return {
        "exact_work_min_pairwise_agreement": (
            float(exact["pairwise_ranking_agreement"]),
            ">=",
            float(criteria["exact_work_min_pairwise_agreement"]),
        ),
        "exact_work_min_kendall_tau_b": (
            float(exact["kendall_tau_b"]),
            ">=",
            float(criteria["exact_work_min_kendall_tau_b"]),
        ),
        "exact_work_max_false_negative_count": (
            float(exact["false_negative_count"]),
            "<=",
            float(criteria["exact_work_max_false_negative_count"]),
        ),
        "exact_work_max_false_positive_count": (
            float(exact["false_positive_count"]),
            "<=",
            float(criteria["exact_work_max_false_positive_count"]),
        ),
        "exact_work_max_median_absolute_relative_error_pct": (
            float(exact["median_absolute_relative_error_pct"]),
            "<=",
            float(criteria["exact_work_max_median_absolute_relative_error_pct"]),
        ),
        "exact_work_max_p95_absolute_relative_error_pct": (
            float(exact["p95_absolute_relative_error_pct"]),
            "<=",
            float(criteria["exact_work_max_p95_absolute_relative_error_pct"]),
        ),
        "exact_work_min_agreement_advantage_pairs_over_isolated": (
            float(exact["pairwise_agreement_count"] - isolated["pairwise_agreement_count"]),
            ">=",
            float(criteria["exact_work_min_agreement_advantage_pairs_over_isolated"]),
        ),
        "exact_work_min_agreement_advantage_pairs_over_stratified": (
            float(exact["pairwise_agreement_count"] - stratified["pairwise_agreement_count"]),
            ">=",
            float(criteria["exact_work_min_agreement_advantage_pairs_over_stratified"]),
        ),
    }


def _criterion_margin(value: float, operator: str, required: float) -> float:
    return value - required if operator == ">=" else required - value


def _statistics(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for row in pair_rows:
        first = str(row["first_configuration_id"])
        second = str(row["second_configuration_id"])
        representations = cast(Mapping[str, Any], row["representations"])
        for representation in _UNCERTAINTY_REPRESENTATIONS:
            result = cast(Mapping[str, Any], representations[representation])
            values[f"pair|{first}|{second}|{representation}"] = float(result["policy_margin_us"])
    exact_metrics = cast(Mapping[str, Any], metrics["exact_work"])
    for metric in REPORTED_METRIC_TO_VERDICT_FIELD.values():
        values[f"metric|exact_work|{metric}"] = float(exact_metrics[metric])
    for criterion_id, (observed, operator, required) in _criterion_values(metrics, criteria).items():
        values[f"criterion|{criterion_id}"] = _criterion_margin(observed, operator, required)
    return values


def _bootstrap_vector(
    samples: RepetitionSamples,
    *,
    configuration_repetitions: Sequence[int],
    configurations: Sequence[str],
    measured_repetitions: int,
    rng: random.Random,
) -> MedianVector:
    if measured_repetitions % WILLIAMS_CYCLE_LENGTH:
        raise DecisionFidelityError("measured repetitions do not form complete Williams cycles")
    cycle_count = measured_repetitions // WILLIAMS_CYCLE_LENGTH
    repetition_medians: Dict[str, Dict[str, List[float]]] = {
        configuration: {representation: [] for representation in _REPRESENTATIONS} for configuration in configurations
    }
    for configuration in configurations:
        selected_repetitions = [
            configuration_repetitions[rng.randrange(len(configuration_repetitions))] for _ in configuration_repetitions
        ]
        for repetition in selected_repetitions:
            selected_cycles = [rng.randrange(cycle_count) for _ in range(cycle_count)]
            indices = [
                cycle * WILLIAMS_CYCLE_LENGTH + offset
                for cycle in selected_cycles
                for offset in range(WILLIAMS_CYCLE_LENGTH)
            ]
            for representation in _REPRESENTATIONS:
                values = samples[repetition][configuration][representation]
                repetition_medians[configuration][representation].append(_median([values[index] for index in indices]))
    return {
        configuration: {
            representation: _median(repetition_medians[configuration][representation])
            for representation in _REPRESENTATIONS
        }
        for configuration in configurations
    }


def _simultaneous_intervals(
    observed: Mapping[str, float],
    bootstrap: Sequence[Mapping[str, float]],
    *,
    confidence: float,
    pair_count: int,
) -> Tuple[Dict[str, Tuple[float, float]], float]:
    if not bootstrap:
        raise DecisionFidelityError("simultaneous bootstrap requires at least one resample")
    scales: Dict[str, float] = {}
    constant_keys: Set[str] = set()
    for key in observed:
        values = [row[key] for row in bootstrap]
        if all(value == observed[key] for value in values):
            constant_keys.add(key)
            continue
        scale = statistics.stdev(values) if len(values) > 1 else 0.0
        if not scale > 0.0 or not math.isfinite(scale):
            raise DecisionFidelityError(
                f"bootstrap statistic {key!r} has zero variance but disagrees with its observation"
            )
        scales[key] = scale
    maxima = (
        [max(abs((row[key] - observed[key]) / scales[key]) for key in scales) for row in bootstrap] if scales else [0.0]
    )
    critical_value = _percentile(maxima, confidence)
    intervals: Dict[str, Tuple[float, float]] = {}
    for key, value in observed.items():
        if key in constant_keys:
            intervals[key] = (value, value)
            continue
        lower = value - critical_value * scales[key]
        upper = value + critical_value * scales[key]
        if key == "metric|exact_work|pairwise_ranking_agreement":
            lower, upper = max(0.0, lower), min(1.0, upper)
        elif key == "metric|exact_work|kendall_tau_b":
            lower, upper = max(-1.0, lower), min(1.0, upper)
        elif key in {
            "metric|exact_work|false_negative_count",
            "metric|exact_work|false_positive_count",
        }:
            lower, upper = max(0.0, lower), min(float(pair_count), upper)
        intervals[key] = (lower, upper)
    return intervals, critical_value


def _relative_iqr(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[midpoint:] if len(ordered) % 2 == 0 else ordered[midpoint + 1 :]
    iqr = 0.0 if not lower or not upper else _median(upper) - _median(lower)
    median = _median(values)
    return math.inf if median == 0.0 and iqr > 0.0 else (0.0 if median == 0.0 else iqr / median * 100.0)


def _outcome_for_states(
    *,
    issues: Sequence[Mapping[str, Any]],
    criterion_statuses: Sequence[str],
    unstable: bool,
    inconclusive_pairs: bool,
) -> str:
    """Apply the policy-frozen mandatory-failure precedence."""

    if issues:
        return "incomparable"
    if "fail" in criterion_statuses:
        return "fail"
    if unstable or inconclusive_pairs or "inconclusive" in criterion_statuses:
        return "inconclusive"
    return "pass"


def evaluate_decision_fidelity_v2(
    aggregate: Mapping[str, Any],
    policy_bytes: bytes,
    *,
    executor_artifact: Optional[ExecutorArtifact] = None,
) -> Dict[str, Any]:
    """Evaluate independent per-configuration repetitions under the v2 policy."""

    aggregate = _object(aggregate, "aggregate")
    if not isinstance(policy_bytes, bytes) or not 0 < len(policy_bytes) <= _POLICY_LIMITS.max_document_bytes:
        raise DecisionFidelityError("decision fidelity policy v2 bytes are outside the supported limit")
    policy = validate_decision_fidelity_policy_v2(strict_json_loads(policy_bytes, limits=_POLICY_LIMITS))
    policy_sha256 = sha256_hex(policy_bytes)
    if aggregate.get("schema") != ANALYSIS_SCHEMA:
        raise DecisionFidelityError("decision fidelity v2 requires a trusted validated aggregate")
    campaign = _policy_input_binding(aggregate, policy_sha256=policy_sha256, policy_size_bytes=len(policy_bytes))
    analyzer_record = _frozen_evaluator_record(
        aggregate,
        campaign,
        executor_artifact=executor_artifact,
        policy_sha256=policy_sha256,
    )
    completeness = _object(aggregate.get("completeness"), "aggregate.completeness")
    configurations = list(policy["scope"]["configuration_ids"])
    repetition_count = int(policy["measurement"]["configuration_repetitions"])
    configuration_repetitions = list(range(repetition_count))
    workload_id = str(policy["scope"]["workload_id"])
    selected = aggregate.get("selected_cells")
    if not isinstance(selected, list):
        raise DecisionFidelityError("aggregate selected cell inventory is invalid")
    issues: List[Dict[str, Any]] = []
    if completeness.get("complete") is not True or completeness.get("issue_codes") != []:
        issues.append({"code": "incomplete_evidence", "detail": "persisted completeness is not complete"})

    rows: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    for index, raw_row in enumerate(selected):
        row = _object(raw_row, f"aggregate.selected_cells[{index}]")
        configuration = row.get("configuration_id")
        repetition = row.get("repetition")
        if (
            row.get("workload_id") != workload_id
            or row.get("measurement_schema") != PHYSICAL_DECISION_GATE_MEASUREMENT_SCHEMA_V2
            or not isinstance(configuration, str)
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or not 0 <= repetition < repetition_count
            or (repetition, configuration) in rows
        ):
            raise DecisionFidelityError("aggregate selected cell is outside the replicated decision gate")
        rows[(repetition, configuration)] = row
    expected_cells = {
        (repetition, configuration) for repetition in configuration_repetitions for configuration in configurations
    }
    if set(rows) != expected_cells:
        issues.append(
            {
                "code": "incomplete_configuration_repetition_inventory",
                "detail": "selected cells do not cover every configuration repetition",
            }
        )

    samples: RepetitionSamples = {}
    identities: Set[Tuple[str, str, str]] = set()
    environment_observations: Set[str] = set()
    platform_fingerprints: Set[str] = set()
    jobs: List[str] = []
    nodes: Set[str] = set()
    chunk_identifiers: Set[str] = set()
    scheduler_starts: Dict[int, List[Tuple[str, datetime, float]]] = {}
    scheduler_spans: Dict[int, float] = {}
    cycle_telemetry_summaries: List[Dict[str, Any]] = []
    measured_repetitions = int(policy["measurement"]["measured_repetitions"])
    schedule_inventory = cast(Mapping[str, Any], policy["measurement"]["representation_schedule"])
    schedule_rows = cast(Sequence[Sequence[str]], schedule_inventory["rows"])
    schedule_indices = cast(
        Sequence[Sequence[int]],
        schedule_inventory["row_index_by_configuration_repetition"],
    )
    if not issues:
        for repetition in configuration_repetitions:
            samples[repetition] = {}
            for configuration in configurations:
                row = rows[(repetition, configuration)]
                gate = _object(
                    row.get("decision_gate"),
                    f"configuration repetition {repetition}, {configuration}.decision_gate",
                )
                execution = _object(
                    gate.get("execution"),
                    f"configuration repetition {repetition}, {configuration}.execution",
                )
                expected_orders = [list(schedule_rows[index]) for index in schedule_indices[repetition]]
                if (
                    execution.get("configuration_repetition") != repetition
                    or execution.get("iterations") != measured_repetitions
                    or execution.get("warmup") != policy["measurement"]["warmup"]
                    or execution.get("order_method") != policy["measurement"]["order_method"]
                    or execution.get("timing_semantics") != policy["measurement"]["timing_semantics"]
                    or execution.get("representation_order_by_iteration") != expected_orders
                ):
                    raise DecisionFidelityError("replicated decision-gate execution disagrees with policy")
                representations = _object(gate.get("representations"), "replicated decision-gate representations")
                samples[repetition][configuration] = {}
                for representation in _REPRESENTATIONS:
                    value = _object(representations.get(representation), f"representation {representation}")
                    timings = value.get("timings_us")
                    if not isinstance(timings, list):
                        raise DecisionFidelityError("replicated decision-gate timings are invalid")
                    parsed = tuple(_number(item, "replicated decision-gate timing", minimum=0.0) for item in timings)
                    if len(parsed) != measured_repetitions:
                        raise DecisionFidelityError("replicated decision-gate timing inventory is incomplete")
                    samples[repetition][configuration][representation] = parsed
                if _median(samples[repetition][configuration]["source"]) <= 0.0:
                    issues.append(
                        {
                            "code": "nonpositive_source_timing",
                            "detail": (
                                "source timing is not positive for configuration repetition "
                                f"{repetition}, {configuration}"
                            ),
                        }
                    )
                runtime = _object(row.get("decision_gate_runtime"), "replicated decision-gate runtime")
                job_id = runtime.get("job_id")
                hostname = runtime.get("hostname")
                if not isinstance(job_id, str) or not job_id or not isinstance(hostname, str) or not hostname:
                    raise DecisionFidelityError("replicated decision-gate job and hostname are required")
                jobs.append(job_id)
                node = hostname.split(".", 1)[0]
                nodes.add(node)
                scheduler = _object(
                    row.get("decision_gate_scheduler"),
                    f"configuration repetition {repetition}, {configuration}.decision_gate_scheduler",
                    {
                        "schema",
                        "method",
                        "planned_position",
                        "scheduler_start_time",
                        "node",
                        "elapsed_from_repetition_start_seconds",
                        "chunk_identifier",
                    },
                )
                expected_configuration_row = policy["measurement"]["configuration_order_by_repetition"][repetition]
                chunk_identifier = scheduler["chunk_identifier"]
                start_time = scheduler["scheduler_start_time"]
                elapsed = _number(
                    scheduler["elapsed_from_repetition_start_seconds"],
                    "replicated scheduler elapsed time",
                    minimum=0.0,
                )
                if (
                    scheduler["schema"] != "commcanary.rostam.scheduler-evidence.v1"
                    or scheduler["method"] != "scontrol show job --oneliner JOBID"
                    or scheduler["planned_position"] != expected_configuration_row.index(configuration)
                    or scheduler["node"] != node
                    or not isinstance(chunk_identifier, str)
                    or not chunk_identifier.startswith("p-")
                    or len(chunk_identifier) != 26
                    or any(character not in _SHA256_CHARACTERS for character in chunk_identifier[2:])
                    or not isinstance(start_time, str)
                ):
                    raise DecisionFidelityError("replicated scheduler evidence disagrees with policy or runtime")
                try:
                    parsed_start = datetime.fromisoformat(start_time)
                except ValueError as exc:
                    raise DecisionFidelityError("replicated scheduler start time is invalid") from exc
                scheduler_starts.setdefault(repetition, []).append((configuration, parsed_start, elapsed))
                chunk_identifiers.add(chunk_identifier)
                raw_environment = row.get("decision_gate_environment")
                observation_sha256, platform_sha256 = _environment_identity(
                    raw_environment,
                    f"configuration repetition {repetition}, {configuration}.decision_gate_environment",
                    comparability=cast(
                        Mapping[str, Any],
                        policy["measurement"]["environment_comparability"],
                    ),
                )
                environment = _object(raw_environment, "replicated decision-gate environment")
                invariants = _object(environment["invariants"], "replicated decision-gate invariants")
                expected_gpus = invariants["gpus"]
                if not isinstance(expected_gpus, list):  # pragma: no cover - validated above
                    raise DecisionFidelityError("replicated decision-gate GPU inventory is invalid")
                cycle_telemetry_summaries.append(
                    _cycle_telemetry_summary(
                        gate.get("telemetry_checkpoints"),
                        f"configuration repetition {repetition}, {configuration}.telemetry_checkpoints",
                        node=node,
                        expected_gpus=cast(Sequence[Mapping[str, Any]], expected_gpus),
                        comparability=cast(
                            Mapping[str, Any],
                            policy["measurement"]["environment_comparability"],
                        ),
                    )
                )
                environment_observations.add(observation_sha256)
                platform_fingerprints.add(platform_sha256)
                identities.add(
                    (
                        canonical_sha256(gate["request"]),
                        canonical_sha256(gate["materialization"]),
                        canonical_sha256(gate["policy"]),
                    )
                )
        if len(set(jobs)) != len(jobs):
            issues.append(
                {
                    "code": "allocation_job_reuse",
                    "detail": "every configuration repetition must have a distinct scheduler job ID",
                }
            )
        if len(identities) != 1:
            issues.append({"code": "artifact_identity_mismatch", "detail": "cells used different immutable inputs"})
        if len(environment_observations) != len(rows):
            issues.append(
                {
                    "code": "environment_observation_reuse",
                    "detail": "every configuration repetition must bind a distinct runtime observation",
                }
            )
        if len(platform_fingerprints) != 1:
            issues.append(
                {
                    "code": "environment_invariant_mismatch",
                    "detail": "configuration repetitions do not share one platform fingerprint",
                }
            )
        for repetition, starts in scheduler_starts.items():
            if len(starts) != len(configurations):
                raise DecisionFidelityError("replicated scheduler evidence is incomplete")
            timestamps = [started for _configuration, started, _elapsed in starts]
            if any((value.tzinfo is None) != (timestamps[0].tzinfo is None) for value in timestamps):
                raise DecisionFidelityError("replicated scheduler timestamps mix timezone forms")
            beginning = min(timestamps)
            for configuration, started, elapsed in starts:
                expected_elapsed = (started - beginning).total_seconds()
                if abs(elapsed - expected_elapsed) > 1e-9:
                    raise DecisionFidelityError(
                        f"replicated scheduler elapsed time is stale for repetition {repetition}, {configuration}"
                    )
            span = (max(timestamps) - beginning).total_seconds()
            scheduler_spans[repetition] = span
            if span > float(policy["measurement"]["maximum_repetition_span_seconds"]):
                issues.append(
                    {
                        "code": "repetition_time_window_exceeded",
                        "detail": (f"configuration repetition {repetition} scheduler starts span {span} seconds"),
                    }
                )

    stability_issues: List[str] = []
    observed_pair_rows: List[Dict[str, Any]] = []
    observed_metrics: Dict[str, Any] = {}
    criteria_rows: List[Dict[str, Any]] = []
    metric_intervals: Dict[str, Any] = {}
    inconclusive_pairs: List[str] = []
    critical_value: Optional[float] = None
    if not issues:
        maximum_iqr = float(policy["measurement"]["max_relative_iqr_pct"])
        for repetition in configuration_repetitions:
            for configuration in configurations:
                for representation in _STABILITY_REPRESENTATIONS:
                    relative_iqr = _relative_iqr(samples[repetition][configuration][representation])
                    if relative_iqr > maximum_iqr:
                        stability_issues.append(f"{repetition}|{configuration}|{representation}|{relative_iqr}")

        comparison = cast(Mapping[str, Any], policy["comparison"])
        observed_vector = _nested_median_vector(
            samples,
            configuration_repetitions=configuration_repetitions,
            configurations=configurations,
        )
        observed_pair_rows = _pair_rows(
            observed_vector,
            configurations=configurations,
            comparison=comparison,
        )
        observed_targets = {
            (
                str(row["first_configuration_id"]),
                str(row["second_configuration_id"]),
                representation,
            ): str(cast(Mapping[str, Any], row["representations"])[representation]["observed_label"])
            for row in observed_pair_rows
            for representation in _REPRESENTATIONS
        }
        observed_metrics = _metrics(
            observed_vector,
            configurations=configurations,
            pair_rows=observed_pair_rows,
        )
        observed_statistics = _statistics(
            pair_rows=observed_pair_rows,
            metrics=observed_metrics,
            criteria=policy["pass_criteria"],
        )
        uncertainty = cast(Mapping[str, Any], comparison["uncertainty"])
        rng = random.Random(int(uncertainty["seed"]))
        bootstrap_statistics: List[Dict[str, float]] = []
        for _ in range(int(uncertainty["resamples"])):
            vector = _bootstrap_vector(
                samples,
                configuration_repetitions=configuration_repetitions,
                configurations=configurations,
                measured_repetitions=measured_repetitions,
                rng=rng,
            )
            pairs = _pair_rows(
                vector,
                configurations=configurations,
                comparison=comparison,
                targets=observed_targets,
            )
            metrics = _metrics(vector, configurations=configurations, pair_rows=pairs)
            bootstrap_statistics.append(_statistics(pair_rows=pairs, metrics=metrics, criteria=policy["pass_criteria"]))
        intervals, critical_value = _simultaneous_intervals(
            observed_statistics,
            bootstrap_statistics,
            confidence=float(uncertainty["confidence"]),
            pair_count=len(observed_pair_rows),
        )

        for row in observed_pair_rows:
            first = str(row["first_configuration_id"])
            second = str(row["second_configuration_id"])
            representation_rows = cast(Dict[str, Any], row["representations"])
            for representation, pair_result in representation_rows.items():
                if representation not in _UNCERTAINTY_REPRESENTATIONS:
                    pair_result["uncertainty_label"] = "not_evaluated"
                    pair_result["simultaneous_margin_interval_us"] = []
                    continue
                key = f"pair|{first}|{second}|{representation}"
                interval = intervals[key]
                pair_result["simultaneous_margin_interval_us"] = list(interval)
                if interval[0] > 0.0:
                    pair_result["uncertainty_label"] = pair_result["observed_label"]
                else:
                    pair_result["uncertainty_label"] = "inconclusive"
                    inconclusive_pairs.append(
                        f"{first}|{second}|{representation}|observed={pair_result['observed_label']}"
                    )

        metric_intervals["exact_work"] = {
            metric: list(intervals[f"metric|exact_work|{metric}"])
            for metric in REPORTED_METRIC_TO_VERDICT_FIELD.values()
        }
        for criterion_id, (observed, operator, required) in _criterion_values(
            observed_metrics,
            policy["pass_criteria"],
        ).items():
            margin_interval = intervals[f"criterion|{criterion_id}"]
            criteria_rows.append(
                {
                    "criterion_id": criterion_id,
                    "observed": observed,
                    "operator": operator,
                    "required": required,
                    "observed_margin": _criterion_margin(observed, operator, required),
                    "simultaneous_margin_interval": list(margin_interval),
                    "status": (
                        "pass"
                        if margin_interval[0] >= 0.0
                        else ("fail" if margin_interval[1] < 0.0 else "inconclusive")
                    ),
                }
            )

    criterion_statuses = [str(row["status"]) for row in criteria_rows]
    criteria_inconclusive = "inconclusive" in criterion_statuses
    outcome = _outcome_for_states(
        issues=issues,
        criterion_statuses=criterion_statuses,
        unstable=bool(stability_issues),
        inconclusive_pairs=bool(inconclusive_pairs),
    )
    positioning = (
        "exact_capsule_positive_control_supported"
        if outcome == "pass"
        else "exact_capsule_positive_control_unvalidated"
    )
    cycle_telemetry_summary = None
    if cycle_telemetry_summaries:
        cycle_telemetry_summary = {
            "cell_count": len(cycle_telemetry_summaries),
            "snapshots_per_cell": sorted({int(summary["snapshot_count"]) for summary in cycle_telemetry_summaries}),
            "minimum_temperature_c": min(
                int(summary["minimum_temperature_c"]) for summary in cycle_telemetry_summaries
            ),
            "maximum_temperature_c": max(
                int(summary["maximum_temperature_c"]) for summary in cycle_telemetry_summaries
            ),
            "minimum_power_draw_w": min(
                float(summary["minimum_power_draw_w"]) for summary in cycle_telemetry_summaries
            ),
            "maximum_power_draw_w": max(
                float(summary["maximum_power_draw_w"]) for summary in cycle_telemetry_summaries
            ),
            "minimum_sm_clock_mhz": min(int(summary["minimum_sm_clock_mhz"]) for summary in cycle_telemetry_summaries),
            "maximum_sm_clock_mhz": max(int(summary["maximum_sm_clock_mhz"]) for summary in cycle_telemetry_summaries),
            "minimum_memory_clock_mhz": min(
                int(summary["minimum_memory_clock_mhz"]) for summary in cycle_telemetry_summaries
            ),
            "maximum_memory_clock_mhz": max(
                int(summary["maximum_memory_clock_mhz"]) for summary in cycle_telemetry_summaries
            ),
            "performance_states": sorted(
                {
                    str(state)
                    for summary in cycle_telemetry_summaries
                    for state in cast(Sequence[str], summary["performance_states"])
                }
            ),
            "maximum_ecc_corrected_delta": max(
                int(summary["maximum_ecc_corrected_delta"]) for summary in cycle_telemetry_summaries
            ),
            "maximum_ecc_uncorrected_delta": max(
                int(summary["maximum_ecc_uncorrected_delta"]) for summary in cycle_telemetry_summaries
            ),
            "maximum_xid_event_delta": max(int(summary["xid_event_delta"]) for summary in cycle_telemetry_summaries),
        }
    result: Dict[str, Any] = {
        "schema": DECISION_FIDELITY_VERDICT_SCHEMA_V2,
        "outcome": outcome,
        "policy": {
            "schema": policy["schema"],
            "policy_id": policy["policy_id"],
            "sha256": policy_sha256,
            "size_bytes": len(policy_bytes),
        },
        "evidence": {
            "aggregate_sha256": canonical_sha256(aggregate),
            "run_id": campaign.get("run_id"),
            "manifest_sha256": campaign.get("manifest_sha256"),
            "selection_sha256": campaign.get("selection_sha256"),
            "completeness_verdict_sha256": campaign.get("verdict_sha256"),
            "configuration_repetition_count": repetition_count,
            "configuration_count": len(configurations),
            "configuration_pair_count": len(observed_pair_rows),
            "distinct_job_count": len(set(jobs)),
            "environment_observation_count": len(environment_observations),
            "cycle_telemetry_summary": cycle_telemetry_summary,
            "platform_sha256": next(iter(platform_fingerprints)) if len(platform_fingerprints) == 1 else None,
            "nodes": sorted(nodes),
            "submission_chunks": sorted(chunk_identifiers),
            "scheduler_start_span_seconds_by_repetition": {
                str(repetition): scheduler_spans[repetition] for repetition in sorted(scheduler_spans)
            },
        },
        "issues": issues,
        "uncertainty": {
            "status": (
                "not_evaluated"
                if issues
                else ("inconclusive" if stability_issues or inconclusive_pairs or criteria_inconclusive else "decisive")
            ),
            "method": policy["comparison"]["uncertainty"]["method"],
            "confidence": policy["comparison"]["uncertainty"]["confidence"],
            "resamples": policy["comparison"]["uncertainty"]["resamples"],
            "standardized_max_critical_value": critical_value,
            "inconclusive_pairs": inconclusive_pairs,
            "unstable_cells": stability_issues,
            "metric_intervals": metric_intervals,
        },
        "pairwise_comparisons": observed_pair_rows,
        "representation_metrics": observed_metrics,
        "criteria": criteria_rows,
        "product_interpretation": {
            **policy["claim_boundary"],
            "positioning": positioning,
        },
    }
    if analyzer_record is not None:
        result["analyzer"] = analyzer_record
    result["verdict_id"] = canonical_sha256(result)
    return result


__all__ = ["evaluate_decision_fidelity_v2", "validate_decision_fidelity_policy_v2"]
