"""Public artifact-contract surface with no engine dependencies.

Application engines should import artifact types and contracts here (or from
the cohesive public submodules) rather than through the legacy
``commcanary.schema`` compatibility facade.
"""

from .canary import (
    ASSURANCE_STATES,
    CANARY_HASH_FIELD_NAMES,
    FIDELITY_ERROR_FIELDS,
    CanaryExpansionCounts,
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
    iter_canary_logical_events,
    iter_canary_stored_leaf_events,
    preflight_canary_expansion,
    validate_canary,
)
from .comparison import comparison_policy_evaluations, derive_comparison_verdict, validate_comparison
from .io import (
    PARAM_TRACE_POLICY,
    SENSITIVE_JSON_POLICY,
    SHAREABLE_HTML_POLICY,
    AtomicWritePolicy,
    SymlinkPolicy,
    TempPlacement,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)
from .json_codec import canonical_json_bytes, formatted_json_bytes
from .report import validate_report
from .schemas import load_schema_bytes
from .trace import validate_trace
from .wire import (
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

__all__ = [
    "ASSURANCE_STATES",
    "AtomicWritePolicy",
    "CANARY_HASH_FIELD_NAMES",
    "CanaryExpansionCounts",
    "FIDELITY_ERROR_FIELDS",
    "JsonDict",
    "MAX_ABS_INTEGER",
    "MAX_RANK_COUNT",
    "MAX_TIME_US",
    "PARAM_TRACE_POLICY",
    "PROTOCOL_FINGERPRINT_EXCLUDE",
    "SENSITIVE_JSON_POLICY",
    "SHAREABLE_HTML_POLICY",
    "SUPPORTED_OPS",
    "SymlinkPolicy",
    "TempPlacement",
    "arrival_skew_us",
    "as_float",
    "as_int",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "average_wait_us",
    "canary_artifact_provenance_sha256",
    "canary_calibration_sha256",
    "canary_execution_sha256",
    "canary_scheduler_execution_sha256",
    "canonical_json_bytes",
    "clean_private_keys",
    "comparison_policy_evaluations",
    "derive_comparison_verdict",
    "formatted_json_bytes",
    "iter_canary_logical_events",
    "iter_canary_stored_leaf_events",
    "load_json",
    "load_json_document",
    "load_schema_bytes",
    "merge_metadata",
    "normalize_arrival_offsets",
    "normalize_ranks",
    "preflight_canary_expansion",
    "replay_protocol_sha256",
    "require_format",
    "validate_canary",
    "validate_comparison",
    "validate_report",
    "validate_trace",
    "write_json",
]
