"""Compatibility facade for canary verification implementations."""

from .behavior import verify_canary_behavior
from .fidelity import verify_canary_fidelity

__all__ = ["verify_canary_behavior", "verify_canary_fidelity"]
