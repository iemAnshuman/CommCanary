"""Versioned wire-format identifiers and package capability declarations.

This module is intentionally dependency-light so validators, producers, the
public API, and the CLI can share one definition without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

TRACE_FORMAT = "commcanary.trace.v1"
CANARY_FORMAT = "commcanary.canary.v2"
REPORT_FORMAT = "commcanary.report.v2"
COMPARE_FORMAT = "commcanary.compare.v2"
FIDELITY_VERIFICATION_FORMAT = "commcanary.fidelity_verification.v1"
BEHAVIOR_VERIFICATION_FORMAT = "commcanary.behavior_verification.v1"
REPORT_VERIFICATION_FORMAT = "commcanary.report_verification.v1"

CANONICAL_JSON_VERSION = "commcanary.canonical-json.v1"
CANARY_INTEGRITY_PROFILE = "commcanary.canary-integrity.v1"
ARTIFACT_PROVENANCE_ALGORITHM = "commcanary.artifact-provenance.v2"


@dataclass(frozen=True)
class FormatCapability:
    """One exact artifact version supported by this package."""

    artifact: str
    format_id: str
    schema: str
    read: bool
    write: bool
    migrate: bool
    semantic_validator: bool


FORMAT_CAPABILITIES: Tuple[FormatCapability, ...] = (
    FormatCapability(
        artifact="trace",
        format_id=TRACE_FORMAT,
        schema="schemas/commcanary.trace.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="canary",
        format_id=CANARY_FORMAT,
        schema="schemas/commcanary.canary.v2.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="report",
        format_id=REPORT_FORMAT,
        schema="schemas/commcanary.report.v2.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="comparison",
        format_id=COMPARE_FORMAT,
        schema="schemas/commcanary.compare.v2.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="fidelity_verification",
        format_id=FIDELITY_VERIFICATION_FORMAT,
        schema="schemas/commcanary.fidelity_verification.v1.schema.json",
        read=False,
        write=True,
        migrate=False,
        semantic_validator=False,
    ),
    FormatCapability(
        artifact="behavior_verification",
        format_id=BEHAVIOR_VERIFICATION_FORMAT,
        schema="schemas/commcanary.behavior_verification.v1.schema.json",
        read=False,
        write=True,
        migrate=False,
        semantic_validator=False,
    ),
    FormatCapability(
        artifact="report_verification",
        format_id=REPORT_VERIFICATION_FORMAT,
        schema="schemas/commcanary.report_verification.v1.schema.json",
        read=False,
        write=True,
        migrate=False,
        semantic_validator=False,
    ),
)


def format_capabilities() -> Tuple[FormatCapability, ...]:
    """Return the immutable exact-version support matrix."""

    return FORMAT_CAPABILITIES


__all__ = [
    "ARTIFACT_PROVENANCE_ALGORITHM",
    "BEHAVIOR_VERIFICATION_FORMAT",
    "CANARY_FORMAT",
    "CANARY_INTEGRITY_PROFILE",
    "CANONICAL_JSON_VERSION",
    "COMPARE_FORMAT",
    "FIDELITY_VERIFICATION_FORMAT",
    "FORMAT_CAPABILITIES",
    "FormatCapability",
    "REPORT_FORMAT",
    "REPORT_VERIFICATION_FORMAT",
    "TRACE_FORMAT",
    "format_capabilities",
]
