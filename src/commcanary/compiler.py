"""Compatibility facade for the pre-0.4 compiler import path.

Implementation now lives behind compilation, verification, and service
boundaries. Supported callers may continue importing the established names
from :mod:`commcanary.compiler`.
"""

from .behavior_config import ranking_relation as _ranking_relation  # noqa: F401 - legacy private seam
from .compilation import DEFAULT_TIMING_SAMPLE_LIMIT, TimingPriorityTier
from .compilation import grouped_event_summary as _grouped_event_summary  # noqa: F401 - legacy private seam
from .compilation import important_timing_indices as _important_timing_indices  # noqa: F401 - legacy private seam
from .compilation.metrics import json_size as _json_size
from .compilation.metrics import source_segment_sha256 as _source_segment_sha256  # noqa: F401 - golden seam
from .compilation.metrics import update_size_metrics
from .services.behavior_search import BehaviorSearchSizeKey, synthesize_behavioral_canary
from .services.behavior_search import behavior_search_size_key as _behavior_search_size_key  # noqa: F401
from .services.compile import compile_trace
from .verification.canary import verify_canary_behavior, verify_canary_fidelity


def _update_size_metrics(canary: dict[str, object]) -> None:
    """Preserve the characterized monkeypatch seam of the legacy module."""

    update_size_metrics(canary, size_calculator=_json_size)


__all__ = [
    "BehaviorSearchSizeKey",
    "DEFAULT_TIMING_SAMPLE_LIMIT",
    "TimingPriorityTier",
    "compile_trace",
    "synthesize_behavioral_canary",
    "verify_canary_behavior",
    "verify_canary_fidelity",
]
