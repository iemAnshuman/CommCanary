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
BEHAVIOR_SEARCH_EVIDENCE_FORMAT = "commcanary.behavior_search_evidence.experimental.v1"
DOCTOR_REPORT_FORMAT = "commcanary.doctor_report.experimental.v1"
REPORT_VERIFICATION_FORMAT = "commcanary.report_verification.v1"
QUALIFICATION_REQUEST_V1_FORMAT = "commcanary.qualification_request.v1"
QUALIFICATION_REQUEST_FORMAT = "commcanary.qualification_request.v2"
QUALIFICATION_MATERIALIZATION_FORMAT = "commcanary.qualification_materialization.v1"
QUALIFICATION_POLICY_FORMAT = "commcanary.qualification_policy.v1"
QUALIFICATION_OBSERVATION_FORMAT = "commcanary.qualification_observation.v1"
QUALIFICATION_VERDICT_FORMAT = "commcanary.qualification_verdict.v1"
CHAKRA_PROJECTION_FORMAT = "commcanary.chakra_projection.v1"
PHYSICAL_CANARY_POLICY_FORMAT = "commcanary.physical_canary_policy.v1"
PHYSICAL_ORACLE_CORPUS_FORMAT = "commcanary.physical_oracle_corpus.v1"
PHYSICAL_DECISION_CANARY_FORMAT = "physical_decision_canary.v1"
PHYSICAL_GATE_OBSERVATION_FORMAT = "commcanary.physical_gate_observation.v1"
PHYSICAL_GATE_RESULT_FORMAT = "commcanary.physical_gate_result.v1"
PHYSICAL_SYNTHESIS_LEDGER_FORMAT = "commcanary.physical_synthesis_ledger.v1"
PHYSICAL_FIDELITY_CERTIFICATE_FORMAT = "commcanary.physical_fidelity_certificate.v1"
PHYSICAL_LEAKAGE_ASSESSMENT_FORMAT = "commcanary.physical_leakage_assessment.v1"
PHYSICAL_EXECUTION_MEASUREMENT_FORMAT = "commcanary.physical_execution_measurement.v1"
PHYSICAL_EXECUTION_EVIDENCE_SET_FORMAT = "commcanary.physical_execution_evidence_set.v1"

CANONICAL_JSON_VERSION = "commcanary.canonical-json.v1"
CANARY_INTEGRITY_PROFILE = "commcanary.canary-integrity.v1"
ARTIFACT_PROVENANCE_ALGORITHM = "commcanary.artifact-provenance.v2"
APPLICATION_MEASUREMENT_FORMAT = "commcanary.application_measurement.v1"
APPLICATION_ORACLE_FORMAT = "commcanary.application_oracle.v1"
APPLICATION_EVIDENCE_SET_FORMAT = "commcanary.application_evidence_set.v1"
ACTIVE_PHYSICAL_STUDY_LEDGER_FORMAT = "commcanary.active_physical_study_ledger.v1"
MEASUREMENT_SET_FORMAT = "commcanary.measurement_set.v1"
TRAFFIC_TRACE_FORMAT = "commcanary.traffic_trace.v1"
SERVING_MEASUREMENT_FORMAT = "commcanary.serving_measurement.v1"


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
        artifact="measurement_set",
        format_id=MEASUREMENT_SET_FORMAT,
        schema="schemas/commcanary.measurement_set.v1.schema.json",
        read=True,
        write=False,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="traffic_trace",
        format_id=TRAFFIC_TRACE_FORMAT,
        schema="schemas/commcanary.traffic_trace.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="serving_measurement",
        format_id=SERVING_MEASUREMENT_FORMAT,
        schema="schemas/commcanary.serving_measurement.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
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
    FormatCapability(
        artifact="qualification_request_legacy",
        format_id=QUALIFICATION_REQUEST_V1_FORMAT,
        schema="schemas/commcanary.qualification_request.v1.schema.json",
        read=True,
        write=False,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="qualification_request",
        format_id=QUALIFICATION_REQUEST_FORMAT,
        schema="schemas/commcanary.qualification_request.v2.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="qualification_materialization",
        format_id=QUALIFICATION_MATERIALIZATION_FORMAT,
        schema="schemas/commcanary.qualification_materialization.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="qualification_policy",
        format_id=QUALIFICATION_POLICY_FORMAT,
        schema="schemas/commcanary.qualification_policy.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="qualification_observation",
        format_id=QUALIFICATION_OBSERVATION_FORMAT,
        schema="schemas/commcanary.qualification_observation.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="qualification_verdict",
        format_id=QUALIFICATION_VERDICT_FORMAT,
        schema="schemas/commcanary.qualification_verdict.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="chakra_projection",
        format_id=CHAKRA_PROJECTION_FORMAT,
        schema="schemas/commcanary.chakra_projection.v1.schema.json",
        read=True,
        write=False,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="physical_canary_policy",
        format_id=PHYSICAL_CANARY_POLICY_FORMAT,
        schema="schemas/commcanary.physical_canary_policy.v1.schema.json",
        read=True,
        write=False,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="physical_oracle_corpus",
        format_id=PHYSICAL_ORACLE_CORPUS_FORMAT,
        schema="schemas/commcanary.physical_oracle_corpus.v1.schema.json",
        read=True,
        write=False,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="physical_decision_canary",
        format_id=PHYSICAL_DECISION_CANARY_FORMAT,
        schema="schemas/physical_decision_canary.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="physical_gate_observation",
        format_id=PHYSICAL_GATE_OBSERVATION_FORMAT,
        schema="schemas/commcanary.physical_gate_observation.v1.schema.json",
        read=True,
        write=False,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="physical_gate_result",
        format_id=PHYSICAL_GATE_RESULT_FORMAT,
        schema="schemas/commcanary.physical_gate_result.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="physical_execution_measurement",
        format_id=PHYSICAL_EXECUTION_MEASUREMENT_FORMAT,
        schema="schemas/commcanary.physical_execution_measurement.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="physical_execution_evidence_set",
        format_id=PHYSICAL_EXECUTION_EVIDENCE_SET_FORMAT,
        schema="schemas/commcanary.physical_execution_evidence_set.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="application_measurement",
        format_id=APPLICATION_MEASUREMENT_FORMAT,
        schema="schemas/commcanary.application_measurement.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="application_oracle",
        format_id=APPLICATION_ORACLE_FORMAT,
        schema="schemas/commcanary.application_oracle.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="application_evidence_set",
        format_id=APPLICATION_EVIDENCE_SET_FORMAT,
        schema="schemas/commcanary.application_evidence_set.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
    FormatCapability(
        artifact="active_physical_study_ledger",
        format_id=ACTIVE_PHYSICAL_STUDY_LEDGER_FORMAT,
        schema="schemas/commcanary.active_physical_study_ledger.v1.schema.json",
        read=True,
        write=True,
        migrate=False,
        semantic_validator=True,
    ),
)


def format_capabilities() -> Tuple[FormatCapability, ...]:
    """Return the immutable exact-version support matrix."""

    return FORMAT_CAPABILITIES


__all__ = [
    "ARTIFACT_PROVENANCE_ALGORITHM",
    "APPLICATION_MEASUREMENT_FORMAT",
    "APPLICATION_ORACLE_FORMAT",
    "APPLICATION_EVIDENCE_SET_FORMAT",
    "ACTIVE_PHYSICAL_STUDY_LEDGER_FORMAT",
    "BEHAVIOR_VERIFICATION_FORMAT",
    "BEHAVIOR_SEARCH_EVIDENCE_FORMAT",
    "CANARY_FORMAT",
    "CANARY_INTEGRITY_PROFILE",
    "CANONICAL_JSON_VERSION",
    "CHAKRA_PROJECTION_FORMAT",
    "COMPARE_FORMAT",
    "DOCTOR_REPORT_FORMAT",
    "FIDELITY_VERIFICATION_FORMAT",
    "FORMAT_CAPABILITIES",
    "FormatCapability",
    "MEASUREMENT_SET_FORMAT",
    "PHYSICAL_CANARY_POLICY_FORMAT",
    "PHYSICAL_DECISION_CANARY_FORMAT",
    "PHYSICAL_EXECUTION_MEASUREMENT_FORMAT",
    "PHYSICAL_EXECUTION_EVIDENCE_SET_FORMAT",
    "PHYSICAL_FIDELITY_CERTIFICATE_FORMAT",
    "PHYSICAL_GATE_OBSERVATION_FORMAT",
    "PHYSICAL_GATE_RESULT_FORMAT",
    "PHYSICAL_LEAKAGE_ASSESSMENT_FORMAT",
    "PHYSICAL_ORACLE_CORPUS_FORMAT",
    "PHYSICAL_SYNTHESIS_LEDGER_FORMAT",
    "QUALIFICATION_MATERIALIZATION_FORMAT",
    "QUALIFICATION_OBSERVATION_FORMAT",
    "QUALIFICATION_POLICY_FORMAT",
    "QUALIFICATION_REQUEST_FORMAT",
    "QUALIFICATION_REQUEST_V1_FORMAT",
    "QUALIFICATION_VERDICT_FORMAT",
    "REPORT_FORMAT",
    "SERVING_MEASUREMENT_FORMAT",
    "REPORT_VERIFICATION_FORMAT",
    "TRACE_FORMAT",
    "TRAFFIC_TRACE_FORMAT",
    "format_capabilities",
]
