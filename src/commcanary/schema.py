"""Compatibility facade for CommCanary artifact contracts.

The implementation lives in :mod:`commcanary.artifacts`.  This module retains
the complete historical import surface so existing callers do not need to
migrate while artifact responsibilities evolve independently from engines.
"""

from __future__ import annotations

from .artifacts.canary import (
    ASSURANCE_STATES,
    CANARY_HASH_FIELD_NAMES,
    FIDELITY_ERROR_FIELDS,
    CanaryExpansionCounts,
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
    expand_sequence_motif,
    iter_canary_logical_events,
    iter_canary_stored_leaf_events,
    preflight_canary_expansion,
    validate_canary,
)
from .artifacts.comparison import (
    comparison_policy_evaluations,
    derive_comparison_verdict,
    validate_comparison,
)
from .artifacts.json_codec import canonical_json_bytes
from .artifacts.report import validate_report
from .artifacts.trace import validate_trace
from .artifacts.wire import (
    MAX_ABS_INTEGER,
    MAX_RANK_COUNT,
    MAX_TIME_US,
    PROTOCOL_FINGERPRINT_EXCLUDE,
    SUPPORTED_OPS,
    JsonDict,
    arrival_skew_us,
    as_float,
    as_int,
    average_wait_us,
    clean_private_keys,
    load_json,
    load_json_document,
    merge_metadata,
    normalize_arrival_offsets,
    normalize_ranks,
    replay_protocol_sha256,
    require_format,
    write_json,
)
from .errors import CommCanaryError, SchemaError
from .formats import (
    ARTIFACT_PROVENANCE_ALGORITHM,
    CANARY_FORMAT,
    CANARY_INTEGRITY_PROFILE,
    COMPARE_FORMAT,
    REPORT_FORMAT,
    TRACE_FORMAT,
)
from .resources import DEFAULT_RESOURCE_LIMITS, JsonResourceError, ResourceLimits
from .statistics import median, percentile, percentile_from_sorted, summarize_latencies

# Historical private monkeypatch target retained for repository compatibility.
_expand_sequence_motif = expand_sequence_motif

__all__ = [
    "ARTIFACT_PROVENANCE_ALGORITHM",
    "ASSURANCE_STATES",
    "CANARY_FORMAT",
    "CANARY_HASH_FIELD_NAMES",
    "CANARY_INTEGRITY_PROFILE",
    "COMPARE_FORMAT",
    "CanaryExpansionCounts",
    "CommCanaryError",
    "DEFAULT_RESOURCE_LIMITS",
    "FIDELITY_ERROR_FIELDS",
    "JsonDict",
    "JsonResourceError",
    "MAX_ABS_INTEGER",
    "MAX_RANK_COUNT",
    "MAX_TIME_US",
    "PROTOCOL_FINGERPRINT_EXCLUDE",
    "REPORT_FORMAT",
    "ResourceLimits",
    "SUPPORTED_OPS",
    "SchemaError",
    "TRACE_FORMAT",
    "arrival_skew_us",
    "as_float",
    "as_int",
    "average_wait_us",
    "canary_artifact_provenance_sha256",
    "canary_calibration_sha256",
    "canary_execution_sha256",
    "canary_scheduler_execution_sha256",
    "canonical_json_bytes",
    "clean_private_keys",
    "comparison_policy_evaluations",
    "derive_comparison_verdict",
    "iter_canary_logical_events",
    "iter_canary_stored_leaf_events",
    "load_json",
    "load_json_document",
    "median",
    "merge_metadata",
    "normalize_arrival_offsets",
    "normalize_ranks",
    "percentile",
    "percentile_from_sorted",
    "preflight_canary_expansion",
    "replay_protocol_sha256",
    "require_format",
    "summarize_latencies",
    "validate_canary",
    "validate_comparison",
    "validate_report",
    "validate_trace",
    "write_json",
]
