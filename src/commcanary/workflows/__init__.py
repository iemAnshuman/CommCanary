"""High-level workflows that compose services with external adapters."""

from .physical_canary import build_physical_canary_bundle, verify_physical_canary_bundle
from .qualification import materialize_qualification, verify_qualification_materialization

__all__ = [
    "build_physical_canary_bundle",
    "materialize_qualification",
    "verify_physical_canary_bundle",
    "verify_qualification_materialization",
]
