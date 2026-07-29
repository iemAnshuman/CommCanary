"""Deterministic exact-work qualification materialization contract."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from ..errors import SchemaError
from ..formats import QUALIFICATION_MATERIALIZATION_FORMAT, QUALIFICATION_REQUEST_FORMAT
from ..resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits, validate_json_mapping
from .json_codec import canonical_json_bytes
from .qualification import (
    QUALIFICATION_COMPUTE_RECIPE_METHOD,
    QUALIFICATION_EXECUTION_ADAPTER,
    QUALIFICATION_EXECUTOR_CONTRACT,
    QUALIFICATION_PROGRAM_ENCODING,
    QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY,
)
from .wire import as_float, as_int, require_format, validate_sha256

QUALIFICATION_MATERIALIZATION_FILENAME = "materialization.json"
QUALIFICATION_REPLAY_PROGRAM_FILENAME = "replay-program.json"

_TOP_LEVEL_FIELDS = {
    "format",
    "materialization_id",
    "request",
    "compute_work",
    "program",
    "executor",
    "claims",
}


def qualification_materialization_sha256(materialization: Mapping[str, Any]) -> str:
    """Identify a materialization by canonical content excluding its identifier."""

    stable = {key: value for key, value in materialization.items() if key != "materialization_id"}
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def validate_qualification_materialization(
    materialization: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    """Validate one request-bound exact-work materialization without reading files."""

    try:
        validate_json_mapping(materialization, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"qualification materialization violates JSON resource constraints: {exc}") from exc
    require_format(
        materialization,
        QUALIFICATION_MATERIALIZATION_FORMAT,
        "qualification materialization",
    )
    _require_exact_fields(materialization, _TOP_LEVEL_FIELDS, "qualification materialization")
    validate_sha256(
        materialization.get("materialization_id"),
        "qualification materialization materialization_id",
    )
    if qualification_materialization_sha256(materialization) != materialization.get("materialization_id"):
        raise SchemaError(
            "qualification materialization materialization_id does not match canonical materialization content"
        )

    request = _mapping(materialization.get("request"), "qualification materialization request")
    _require_exact_fields(
        request,
        {"format", "request_id", "manifest_sha256", "manifest_size_bytes"},
        "qualification materialization request",
    )
    if request.get("format") != QUALIFICATION_REQUEST_FORMAT:
        raise SchemaError("qualification materialization request.format is unsupported")
    validate_sha256(request.get("request_id"), "qualification materialization request.request_id")
    validate_sha256(
        request.get("manifest_sha256"),
        "qualification materialization request.manifest_sha256",
    )
    if as_int(request.get("manifest_size_bytes")) <= 0:
        raise SchemaError("qualification materialization request.manifest_size_bytes must be positive")

    compute_work = _mapping(
        materialization.get("compute_work"),
        "qualification materialization compute_work",
    )
    _require_exact_fields(
        compute_work,
        {
            "provenance",
            "method",
            "projection_sha256",
            "event_count",
            "rank_recipe_count",
            "operation_count",
            "rank_operation_counts",
            "source_kernel_count",
            "source_kernel_duration_us",
            "matmul_flop_count",
        },
        "qualification materialization compute_work",
    )
    if compute_work.get("provenance") != "source_trace_exact_rank_local_work":
        raise SchemaError("qualification materialization compute_work.provenance is unsupported")
    if compute_work.get("method") != QUALIFICATION_COMPUTE_RECIPE_METHOD:
        raise SchemaError("qualification materialization compute_work.method is unsupported")
    validate_sha256(
        compute_work.get("projection_sha256"),
        "qualification materialization compute_work.projection_sha256",
    )
    event_count = _positive_int(
        compute_work.get("event_count"),
        "qualification materialization compute_work.event_count",
    )
    if event_count > limits.max_stored_events:
        raise SchemaError(
            "qualification materialization compute_work.event_count exceeds "
            f"max_stored_events={limits.max_stored_events}"
        )
    rank_recipe_count = _positive_int(
        compute_work.get("rank_recipe_count"),
        "qualification materialization compute_work.rank_recipe_count",
    )
    if rank_recipe_count > limits.max_param_entries:
        raise SchemaError(
            "qualification materialization compute_work.rank_recipe_count exceeds "
            f"max_param_entries={limits.max_param_entries}"
        )
    operation_count = _nonnegative_int(
        compute_work.get("operation_count"),
        "qualification materialization compute_work.operation_count",
    )
    if operation_count > limits.max_param_compute_operations:
        raise SchemaError(
            "qualification materialization compute_work.operation_count exceeds "
            f"max_param_compute_operations={limits.max_param_compute_operations}"
        )
    rank_counts = _rank_operation_counts(
        compute_work.get("rank_operation_counts"),
        "qualification materialization compute_work.rank_operation_counts",
    )
    if sum(rank_counts.values()) != operation_count:
        raise SchemaError("qualification materialization rank operation counts do not sum to operation_count")
    _nonnegative_int(
        compute_work.get("source_kernel_count"),
        "qualification materialization compute_work.source_kernel_count",
    )
    _nonnegative_float(
        compute_work.get("source_kernel_duration_us"),
        "qualification materialization compute_work.source_kernel_duration_us",
    )
    _nonnegative_int(
        compute_work.get("matmul_flop_count"),
        "qualification materialization compute_work.matmul_flop_count",
    )

    program = _mapping(materialization.get("program"), "qualification materialization program")
    _require_exact_fields(
        program,
        {
            "path",
            "encoding",
            "sha256",
            "size_bytes",
            "entry_count",
            "compute_operation_count",
        },
        "qualification materialization program",
    )
    if program.get("path") != QUALIFICATION_REPLAY_PROGRAM_FILENAME:
        raise SchemaError("qualification materialization program.path is not canonical")
    if program.get("encoding") != QUALIFICATION_PROGRAM_ENCODING:
        raise SchemaError("qualification materialization program.encoding is unsupported")
    validate_sha256(program.get("sha256"), "qualification materialization program.sha256")
    if as_int(program.get("size_bytes")) <= 0:
        raise SchemaError("qualification materialization program.size_bytes must be positive")
    entry_count = _positive_int(
        program.get("entry_count"),
        "qualification materialization program.entry_count",
    )
    if entry_count > limits.max_param_entries:
        raise SchemaError(
            f"qualification materialization program.entry_count exceeds max_param_entries={limits.max_param_entries}"
        )
    if as_int(program.get("compute_operation_count")) != operation_count:
        raise SchemaError(
            "qualification materialization program.compute_operation_count does not match its compute-work audit"
        )

    executor = _mapping(materialization.get("executor"), "qualification materialization executor")
    expected_executor = {
        "contract": QUALIFICATION_EXECUTOR_CONTRACT,
        "adapter": QUALIFICATION_EXECUTION_ADAPTER,
        "upstream_param_compatibility": QUALIFICATION_UPSTREAM_PARAM_COMPATIBILITY,
        "timestamp_pacing": "disabled",
    }
    if dict(executor) != expected_executor:
        raise SchemaError("qualification materialization executor contract is unsupported")

    expected_claims = {
        "materialization": "request_bound",
        "compute_work_provenance": "source_trace_verified",
        "physical_execution": "not_included",
        "physical_measurement": "not_included",
        "qualification_verdict": "not_issued",
    }
    claims = _mapping(materialization.get("claims"), "qualification materialization claims")
    if dict(claims) != expected_claims:
        raise SchemaError("qualification materialization claims exceed the materialization-only assurance boundary")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _rank_operation_counts(value: Any, label: str) -> dict[int, int]:
    mapping = _mapping(value, label)
    if not mapping:
        raise SchemaError(f"{label} must contain at least one rank")
    result: dict[int, int] = {}
    for raw_rank, raw_count in mapping.items():
        if not isinstance(raw_rank, str):
            raise SchemaError(f"{label} keys must be canonical decimal rank strings")
        rank = as_int(raw_rank)
        if rank < 0 or str(rank) != raw_rank:
            raise SchemaError(f"{label} key {raw_rank!r} is not a canonical non-negative rank")
        result[rank] = _nonnegative_int(raw_count, f"{label}.{raw_rank}")
    if sorted(result) != list(range(max(result) + 1)):
        raise SchemaError(f"{label} must cover a dense rank domain")
    return result


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise SchemaError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise SchemaError(f"{label} contains unsupported fields: {', '.join(unknown)}")


def _nonnegative_float(value: Any, label: str) -> float:
    parsed = as_float(value)
    if parsed < 0.0:
        raise SchemaError(f"{label} must be non-negative")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    parsed = as_int(value)
    if parsed < 0:
        raise SchemaError(f"{label} must be non-negative")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    parsed = _nonnegative_int(value, label)
    if parsed == 0:
        raise SchemaError(f"{label} must be positive")
    return parsed


__all__ = [
    "QUALIFICATION_MATERIALIZATION_FILENAME",
    "QUALIFICATION_REPLAY_PROGRAM_FILENAME",
    "qualification_materialization_sha256",
    "validate_qualification_materialization",
]
