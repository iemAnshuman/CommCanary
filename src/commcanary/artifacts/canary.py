"""Stable facade for canary artifact contracts.

Canary expansion/accounting, hashing/projections, and validation are kept in
separate downward-only modules because they have distinct reasons to change.
"""

from .canary_expansion import (
    CanaryExpansionCounts,
    expand_sequence_motif,
    iter_canary_logical_events,
    iter_canary_stored_leaf_events,
    preflight_canary_expansion,
)
from .canary_hashes import (
    CANARY_HASH_FIELD_NAMES,
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
)
from .canary_validation import ASSURANCE_STATES, FIDELITY_ERROR_FIELDS, validate_canary

__all__ = [
    "ASSURANCE_STATES",
    "CANARY_HASH_FIELD_NAMES",
    "CanaryExpansionCounts",
    "FIDELITY_ERROR_FIELDS",
    "canary_artifact_provenance_sha256",
    "canary_calibration_sha256",
    "canary_execution_sha256",
    "canary_scheduler_execution_sha256",
    "expand_sequence_motif",
    "iter_canary_logical_events",
    "iter_canary_stored_leaf_events",
    "preflight_canary_expansion",
    "validate_canary",
]
