"""Shared artifact mutation helpers used by characterization tests."""

from __future__ import annotations

from commcanary.schema import (
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
)


def refresh_canary_hashes(canary):
    canary["compiler"]["execution_semantic_sha256"] = canary_execution_sha256(canary)
    canary["compiler"]["scheduler_execution_sha256"] = canary_scheduler_execution_sha256(canary)
    canary["compiler"]["calibration_evaluation_sha256"] = canary_calibration_sha256(canary)
    canary["compiler"]["artifact_provenance_sha256"] = canary_artifact_provenance_sha256(canary)
