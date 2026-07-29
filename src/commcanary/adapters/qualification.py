"""Compatibility facade for exact-work qualification program materialization.

The independently verified projection and program contract live with artifact
contracts. This adapter path remains available for callers that discovered the
experimental API before the 0.3.0 release boundary.
"""

from ..artifacts.qualification_program import (
    qualification_compute_recipe_audit,
    qualification_compute_tensor_bytes,
    qualification_compute_tensor_elements,
    trace_to_qualification_program,
)

__all__ = [
    "qualification_compute_recipe_audit",
    "qualification_compute_tensor_bytes",
    "qualification_compute_tensor_elements",
    "trace_to_qualification_program",
]
