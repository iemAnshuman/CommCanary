"""Trace normalization and compression primitives."""

from ._constants import DEFAULT_TIMING_SAMPLE_LIMIT
from .compression import TimingPriorityTier, important_timing_indices
from .core import compile_trace_core
from .metrics import refresh_canary_hashes_and_size, update_size_metrics
from .normalization import grouped_event_summary

__all__ = [
    "DEFAULT_TIMING_SAMPLE_LIMIT",
    "TimingPriorityTier",
    "compile_trace_core",
    "grouped_event_summary",
    "important_timing_indices",
    "refresh_canary_hashes_and_size",
    "update_size_metrics",
]
