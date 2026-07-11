"""Compatibility boundary for the historical combined ecosystem adapter.

Kineto import and PARAM export live in independent modules. This boundary keeps
the old function signatures and the logical-event iterator patch point used to
prove PARAM resource preflight happens before expansion.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from ..artifacts.canary import iter_canary_logical_events
from ..artifacts.wire import JsonDict
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from .kineto import kineto_trace_to_commcanary_trace, load_kineto_trace
from .param import export_param_comms_trace, write_param_comms_trace


def canary_to_param_comms_trace(
    canary: Mapping[str, Any],
    *,
    dtype: str = "float32",
    skip_unsupported: bool = False,
    compute_fill_us_per_gemm: Optional[float] = None,
    compute_fill_gemm_dim: int = 1024,
    overlap_structure: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> List[JsonDict]:
    """Export through PARAM while preserving the historical iterator patch seam."""

    return export_param_comms_trace(
        canary,
        dtype=dtype,
        skip_unsupported=skip_unsupported,
        compute_fill_us_per_gemm=compute_fill_us_per_gemm,
        compute_fill_gemm_dim=compute_fill_gemm_dim,
        overlap_structure=overlap_structure,
        limits=limits,
        logical_event_iterator=iter_canary_logical_events,
    )


__all__ = [
    "canary_to_param_comms_trace",
    "kineto_trace_to_commcanary_trace",
    "load_kineto_trace",
    "write_param_comms_trace",
]
