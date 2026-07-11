"""Compatibility facade and public API for deterministic replay."""

from __future__ import annotations

from typing import Any, Mapping

from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from .accumulator import ReplayAccumulator, _ReplaySampleValues
from .core import (
    DEFAULT_MAX_REPLAY_EVENTS,
    QUANTILE_METHOD,
    SIMULATION_MODEL_VERSION,
    SUPPORTED_ABLATIONS,
    replay_canary,
)


def verify_report_against_canary(
    report: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> dict[str, Any]:
    """Compatibility entry point for independent report recomputation."""

    from ..verification.report import verify_report_against_canary as verify

    return verify(report, canary, limits=limits)


__all__ = [
    "DEFAULT_MAX_REPLAY_EVENTS",
    "QUANTILE_METHOD",
    "ReplayAccumulator",
    "SIMULATION_MODEL_VERSION",
    "SUPPORTED_ABLATIONS",
    "_ReplaySampleValues",
    "replay_canary",
    "verify_report_against_canary",
]
