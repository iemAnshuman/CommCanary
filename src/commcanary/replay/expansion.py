"""Compatibility alias for bounded timing-record expansion."""

from __future__ import annotations

from ..artifacts.canary_expansion import (
    iter_canary_timing_samples,
    timing_sample_offsets,
)

_iter_timing_samples = iter_canary_timing_samples
_sample_offsets = timing_sample_offsets

__all__ = ["_iter_timing_samples", "_sample_offsets"]
