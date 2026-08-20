"""Immutable construction and independent verification of physical canary bundles."""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..artifacts.application_oracle import (
    application_evidence_id,
    application_evidence_regression_decision,
    validate_application_evidence_set,
)
from ..artifacts.chakra import decode_chakra_execution_trace, load_chakra_execution_trace
from ..artifacts.io import SENSITIVE_JSON_POLICY, atomic_write_bytes
from ..artifacts.json_codec import canonical_json_bytes, formatted_json_bytes
from ..artifacts.physical_canary import (
    PHYSICAL_CANARY_STATUSES,
    SUPPORTED_PHYSICAL_DOMAIN,
    validate_chakra_projection,
    validate_physical_canary_policy,
    validate_physical_oracle_corpus,
    with_content_identity,
)
from ..artifacts.physical_execution import validate_physical_execution_evidence_set
from ..errors import CommCanaryIOError, SchemaError
from ..formats import (
    PHYSICAL_DECISION_CANARY_FORMAT,
    PHYSICAL_FIDELITY_CERTIFICATE_FORMAT,
    PHYSICAL_LEAKAGE_ASSESSMENT_FORMAT,
)
from ..qualification_io import VerifiedDirectory
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits, decode_bounded_json_bytes
from ..services.active_physical_synthesis import validate_active_physical_study_ledger
from ..services.physical_synthesis import PhysicalSynthesisResult, synthesize_physical_decision_canary
from .signing import public_key_sha256, sign_ed25519_payload, verify_ed25519_payload

PHYSICAL_CANARY_MANIFEST = "manifest.json"
PHYSICAL_CANARY_ET = "canary.et"
PHYSICAL_CANARY_POLICY = "policy.json"
PHYSICAL_CANARY_PROJECTION = "projection.json"
PHYSICAL_CANARY_EXECUTION_PROJECTION = "canary-projection.json"
PHYSICAL_CANARY_CORPUS = "oracle-corpus.json"
PHYSICAL_CANARY_LEDGER = "selection-ledger.json"
PHYSICAL_CANARY_ACTIVE_LEDGER = "active-study-ledger.json"
PHYSICAL_CANARY_APPLICATION_EVIDENCE = "application-evidence.json"
PHYSICAL_CANARY_PHYSICAL_EVIDENCE = "physical-evidence.json"
PHYSICAL_CANARY_CERTIFICATE = "fidelity-certificate.json"
PHYSICAL_CANARY_LEAKAGE = "leakage-assessment.json"
PHYSICAL_CANARY_SOURCE = "source.et"
PHYSICAL_CANARY_SIGNATURE = "owner-signature.json"
PHYSICAL_CANARY_MODES = frozenset({"internal", "private_exchange", "full_audit"})

_IMMUTABLE_BUNDLE_POLICY = replace(
    SENSITIVE_JSON_POLICY,
    artifact_label="physical decision canary artifact",
    create_parents=False,
    overwrite=False,
)


def build_physical_canary_bundle(
    source_et_path: str,
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    output_directory: str,
    *,
    corpus: Optional[Mapping[str, Any]] = None,
    active_ledger: Optional[Mapping[str, Any]] = None,
    application_evidence: Optional[Mapping[str, Any]] = None,
    physical_evidence: Optional[Mapping[str, Any]] = None,
    mode: str = "internal",
    owner_private_key: Optional[str] = None,
    owner_public_key: Optional[str] = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Dict[str, Any]:
    """Build a new immutable Chakra-backed physical decision canary bundle."""

    if mode not in PHYSICAL_CANARY_MODES:
        raise SchemaError(f"physical canary mode must be one of {sorted(PHYSICAL_CANARY_MODES)}")
    if mode == "private_exchange" and (owner_private_key is None or owner_public_key is None):
        raise SchemaError("private_exchange requires --owner-private-key and --owner-public-key")
    if mode != "private_exchange" and (owner_private_key is not None or owner_public_key is not None):
        raise SchemaError("owner signing keys are accepted only for private_exchange bundles")
    trace = load_chakra_execution_trace(source_et_path, limits=limits)
    validate_physical_canary_policy(policy, limits=limits)
    validate_chakra_projection(projection, trace, limits=limits)
    if corpus is not None:
        validate_physical_oracle_corpus(corpus, trace, projection, policy, limits=limits)
    if active_ledger is not None:
        if corpus is None:
            raise SchemaError("active physical study ledger requires its measured oracle corpus")
        if application_evidence is None:
            raise SchemaError("active physical study ledger requires its complete application evidence")
        if physical_evidence is None:
            raise SchemaError("active physical study ledger requires its complete physical evidence")
        validate_active_physical_study_ledger(active_ledger)
        _bind_active_ledger(active_ledger, trace.source_sha256, projection, policy, corpus)
    elif application_evidence is not None or physical_evidence is not None:
        raise SchemaError("application and physical evidence require an active physical study ledger")
    if application_evidence is not None:
        if corpus is None or active_ledger is None:
            raise SchemaError("application evidence requires its corpus and active study ledger")
        _bind_application_evidence(application_evidence, corpus, active_ledger, projection, policy)
    if physical_evidence is not None:
        if corpus is None or active_ledger is None:
            raise SchemaError("physical evidence requires its corpus and active study ledger")
        _bind_physical_evidence(physical_evidence, corpus, active_ledger, policy)
    synthesis = synthesize_physical_decision_canary(
        trace,
        projection,
        policy,
        corpus,
        limits=limits,
    )
    if active_ledger is not None:
        _bind_active_selection(active_ledger, synthesis)
    if mode == "private_exchange":
        if synthesis.status != "qualified_physical_decision_canary":
            raise SchemaError("private_exchange requires a qualified physical decision canary")
        if projection["privacy_review"]["opaque_chakra_attributes_reviewed"] is not True:
            raise SchemaError("private_exchange requires reviewed opaque Chakra attributes")
    executable_projection = _executable_projection(projection, synthesis)
    artifact_bytes = _bundle_artifact_bytes(
        trace_raw=trace.source_raw,
        projection=projection,
        policy=policy,
        corpus=corpus,
        active_ledger=active_ledger,
        application_evidence=application_evidence,
        physical_evidence=physical_evidence,
        synthesis=synthesis,
        executable_projection=executable_projection,
        mode=mode,
    )
    owner_key_sha256 = None if owner_public_key is None else public_key_sha256(owner_public_key)
    manifest = _bundle_manifest(
        mode=mode,
        status=synthesis.status,
        trace_sha256=trace.source_sha256,
        trace_size=trace.source_bytes,
        metadata_version=trace.metadata_version,
        projection=projection,
        executable_projection=executable_projection,
        policy=policy,
        corpus=corpus,
        active_ledger=active_ledger,
        application_evidence=application_evidence,
        physical_evidence=physical_evidence,
        synthesis=synthesis,
        artifact_bytes=artifact_bytes,
        owner_public_key_sha256=owner_key_sha256,
    )
    manifest_bytes = formatted_json_bytes(manifest, indent=2)
    if mode == "private_exchange":
        if owner_private_key is None or owner_public_key is None:
            raise SchemaError("private_exchange signing keys disappeared before publication")
        signature = sign_ed25519_payload(
            manifest_bytes,
            private_key=owner_private_key,
            expected_public_key=owner_public_key,
            signed_artifact=PHYSICAL_CANARY_MANIFEST,
        )
        artifact_bytes[PHYSICAL_CANARY_SIGNATURE] = formatted_json_bytes(signature, indent=2)
    artifact_bytes[PHYSICAL_CANARY_MANIFEST] = manifest_bytes
    _publish_new_directory(Path(output_directory), artifact_bytes)
    verified = verify_physical_canary_bundle(
        output_directory,
        trusted_owner_public_key=owner_public_key,
        limits=limits,
    )
    if verified != manifest:
        raise SchemaError("physical decision canary did not verify after publication")
    return copy.deepcopy(manifest)


def verify_physical_canary_bundle(
    bundle_directory: str,
    *,
    trusted_owner_public_key: Optional[str] = None,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Dict[str, Any]:
    """Recompute a bundle from exact descriptor-bound source artifacts."""

    directory = Path(bundle_directory)
    with VerifiedDirectory(directory, label="physical decision canary") as opened:
        names = opened.names()
        if PHYSICAL_CANARY_MANIFEST not in names:
            raise SchemaError("physical decision canary lacks manifest.json")
        manifest_snapshot = opened.read_json(
            PHYSICAL_CANARY_MANIFEST,
            limits=limits,
            require_object=True,
        )
        manifest = manifest_snapshot.value
        _validate_manifest_shape(manifest, limits=limits)
        signature_name = manifest["authenticity"]["signature_artifact"]
        expected_names = {PHYSICAL_CANARY_MANIFEST, *manifest["artifacts"]}
        if signature_name is not None:
            expected_names.add(signature_name)
        if names != expected_names:
            raise SchemaError(
                "physical decision canary inventory mismatch: "
                f"missing={sorted(expected_names - names)}, unexpected={sorted(names - expected_names)}"
            )
        snapshots = {name: opened.read_bytes(name, limits=limits) for name in sorted(names)}
        if opened.names() != names:
            raise SchemaError("physical decision canary inventory changed while it was verified")

    manifest_raw = snapshots[PHYSICAL_CANARY_MANIFEST].raw
    if signature_name is not None:
        if trusted_owner_public_key is None:
            raise SchemaError("private_exchange verification requires a trusted owner public key")
        signature_document = _decode_json_object(snapshots[signature_name].raw, "owner signature", limits)
        key_sha256 = verify_ed25519_payload(
            manifest_raw,
            signature_document,
            expected_signed_artifact=PHYSICAL_CANARY_MANIFEST,
            trusted_public_key=trusted_owner_public_key,
        )
        if key_sha256 != manifest["authenticity"]["owner_public_key_sha256"]:
            raise SchemaError("private exchange owner signature key does not match the manifest")

    for name, identity in manifest["artifacts"].items():
        snapshot = snapshots[name]
        if snapshot.sha256 != identity["sha256"] or snapshot.size_bytes != identity["size_bytes"]:
            raise SchemaError(f"physical decision canary artifact identity mismatch: {name}")

    if manifest["publication_mode"] == "private_exchange":
        _verify_private_exchange_artifacts(manifest, snapshots, limits=limits)
        return copy.deepcopy(manifest)

    source = snapshots[PHYSICAL_CANARY_SOURCE]
    trace = decode_chakra_execution_trace(source.raw, limits=limits)
    projection = _decode_json_object(snapshots[PHYSICAL_CANARY_PROJECTION].raw, "projection", limits)
    policy = _decode_json_object(snapshots[PHYSICAL_CANARY_POLICY].raw, "policy", limits)
    corpus = None
    if PHYSICAL_CANARY_CORPUS in snapshots:
        corpus = _decode_json_object(snapshots[PHYSICAL_CANARY_CORPUS].raw, "oracle corpus", limits)
    active_ledger = None
    if PHYSICAL_CANARY_ACTIVE_LEDGER in snapshots:
        active_ledger = _decode_json_object(
            snapshots[PHYSICAL_CANARY_ACTIVE_LEDGER].raw,
            "active study ledger",
            limits,
        )
        validate_active_physical_study_ledger(active_ledger)
        if corpus is None:
            raise SchemaError("physical decision canary active ledger lacks its oracle corpus")
        _bind_active_ledger(active_ledger, trace.source_sha256, projection, policy, corpus)
    application_evidence = None
    if PHYSICAL_CANARY_APPLICATION_EVIDENCE in snapshots:
        application_evidence = _decode_json_object(
            snapshots[PHYSICAL_CANARY_APPLICATION_EVIDENCE].raw,
            "application evidence",
            limits,
        )
    if active_ledger is not None:
        if application_evidence is None:
            raise SchemaError("active physical study bundle lacks its complete application evidence")
        if corpus is None:
            raise SchemaError("application evidence lacks its oracle corpus")
        _bind_application_evidence(application_evidence, corpus, active_ledger, projection, policy)
    elif application_evidence is not None:
        raise SchemaError("application evidence lacks its active study ledger")
    physical_evidence = None
    if PHYSICAL_CANARY_PHYSICAL_EVIDENCE in snapshots:
        physical_evidence = _decode_json_object(
            snapshots[PHYSICAL_CANARY_PHYSICAL_EVIDENCE].raw,
            "physical evidence",
            limits,
        )
    if active_ledger is not None:
        if physical_evidence is None:
            raise SchemaError("active physical study bundle lacks its complete physical evidence")
        if corpus is None:
            raise SchemaError("physical evidence lacks its oracle corpus")
        _bind_physical_evidence(physical_evidence, corpus, active_ledger, policy)
    elif physical_evidence is not None:
        raise SchemaError("physical evidence lacks its active study ledger")
    synthesis = synthesize_physical_decision_canary(
        trace,
        projection,
        policy,
        corpus,
        limits=limits,
    )
    if active_ledger is not None:
        _bind_active_selection(active_ledger, synthesis)
    executable_projection = _executable_projection(projection, synthesis)
    expected_bytes = _bundle_artifact_bytes(
        trace_raw=trace.source_raw,
        projection=projection,
        policy=policy,
        corpus=corpus,
        active_ledger=active_ledger,
        application_evidence=application_evidence,
        physical_evidence=physical_evidence,
        synthesis=synthesis,
        executable_projection=executable_projection,
        mode=str(manifest["publication_mode"]),
    )
    for name, raw in expected_bytes.items():
        if snapshots[name].raw != raw:
            raise SchemaError(f"physical decision canary artifact does not recompute: {name}")
    expected_manifest = _bundle_manifest(
        mode=str(manifest["publication_mode"]),
        status=synthesis.status,
        trace_sha256=trace.source_sha256,
        trace_size=trace.source_bytes,
        metadata_version=trace.metadata_version,
        projection=projection,
        executable_projection=executable_projection,
        policy=policy,
        corpus=corpus,
        active_ledger=active_ledger,
        application_evidence=application_evidence,
        physical_evidence=physical_evidence,
        synthesis=synthesis,
        artifact_bytes=expected_bytes,
        owner_public_key_sha256=None,
    )
    if expected_manifest != manifest:
        raise SchemaError("physical decision canary manifest does not recompute")
    return copy.deepcopy(manifest)


def _executable_projection(
    projection: Mapping[str, Any],
    synthesis: PhysicalSynthesisResult,
) -> Dict[str, Any]:
    """Project only the selected executable semantics onto ``canary.et``."""

    selected_nodes = set(synthesis.selected_node_ids)
    selected_regions = set(synthesis.selected_region_ids)
    nodes = [copy.deepcopy(row) for row in projection["nodes"] if int(row["node_id"]) in selected_nodes]
    regions = [copy.deepcopy(row) for row in projection["regions"] if str(row["region_id"]) in selected_regions]
    if {int(node_id) for region in regions for node_id in region["node_ids"]} != selected_nodes:
        raise SchemaError("selected physical regions do not exactly cover the reduced executable")
    reviewed_disclosures = sorted({str(value) for node in nodes for value in node.get("disclosures", [])})
    canary = decode_chakra_execution_trace(synthesis.canary_et)
    raw = {
        "format": projection["format"],
        "source_et": {
            "sha256": canary.source_sha256,
            "bytes": canary.source_bytes,
            "metadata_version": canary.metadata_version,
        },
        "capture_provenance": copy.deepcopy(projection["capture_provenance"]),
        "supported_domain": projection["supported_domain"],
        "nodes": nodes,
        "regions": regions,
        "privacy_review": {
            "opaque_chakra_attributes_reviewed": projection["privacy_review"]["opaque_chakra_attributes_reviewed"],
            "reviewed_disclosures": reviewed_disclosures,
        },
    }
    result = with_content_identity(raw, "projection_id")
    validate_chakra_projection(result, canary)
    return result


def _bind_active_ledger(
    active_ledger: Mapping[str, Any],
    source_et_sha256: str,
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> None:
    expected = {
        "source_et_sha256": source_et_sha256,
        "projection_id": projection["projection_id"],
        "policy_id": policy["policy_id"],
        "corpus_id": corpus["corpus_id"],
    }
    for field, value in expected.items():
        if active_ledger.get(field) != value:
            raise SchemaError(f"active physical study ledger does not match {field}")


def _bind_active_selection(
    active_ledger: Mapping[str, Any],
    synthesis: PhysicalSynthesisResult,
) -> None:
    selection = active_ledger.get("selection")
    if not isinstance(selection, Mapping):
        raise SchemaError("active physical study ledger selection is missing")
    if selection.get("selected_region_ids") != list(synthesis.selected_region_ids):
        raise SchemaError("active physical selection differs from the certified region selection")
    if selection.get("selected_node_ids") != list(synthesis.selected_node_ids):
        raise SchemaError("active physical selection differs from the certified node selection")
    if selection.get("canary_et_sha256") != hashlib.sha256(synthesis.canary_et).hexdigest():
        raise SchemaError("active physical selection differs from the certified canary bytes")


def _bind_application_evidence(
    application_evidence: Mapping[str, Any],
    corpus: Mapping[str, Any],
    active_ledger: Mapping[str, Any],
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    """Recompute corpus truth rows from the complete application evidence."""

    validate_application_evidence_set(application_evidence)
    evidence_set_id = application_evidence["evidence_set_id"]
    if active_ledger.get("application_evidence_set_id") != evidence_set_id:
        raise SchemaError("application evidence does not match the active study ledger")
    if application_evidence.get("runner_oci_digest") != policy["runner"]["oci_digest"]:
        raise SchemaError("application evidence runner does not match the physical policy")
    corpus_rows = {str(row["perturbation_id"]): row for row in corpus["perturbations"]}
    splits = application_evidence["splits"]
    expected_ids = {
        str(row["perturbation_id"]) for split in ("training", "holdout") for row in splits[split]["perturbations"]
    }
    if set(corpus_rows) != expected_ids:
        raise SchemaError("application evidence perturbation inventory does not match the oracle corpus")
    training_baseline = splits["training"]["baseline"]
    holdout_baseline = splits["holdout"]["baseline"]
    baseline_subject = corpus.get("baseline_subject_sha256")
    if (
        training_baseline.get("subject_sha256") != baseline_subject
        or holdout_baseline.get("subject_sha256") != baseline_subject
    ):
        raise SchemaError("application evidence baseline subject does not match the oracle corpus")
    holdout_access = active_ledger["holdout_access"]
    if application_evidence_id(holdout_baseline) != holdout_access["baseline_application_evidence_sha256"]:
        raise SchemaError("application evidence holdout baseline does not match the active study ledger")
    if set(active_ledger["predeclared_holdout_perturbation_ids"]) != {
        str(row["perturbation_id"]) for row in splits["holdout"]["perturbations"]
    }:
        raise SchemaError("application evidence holdout inventory does not match the active study ledger")

    application = corpus["application"]
    expected_application = {
        "ground_truth_kind": "application_ground_truth",
        "name": training_baseline["application"]["name"],
        "engine": training_baseline["application"]["engine"],
        "artifact_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "application": training_baseline["application"],
                    "workload": training_baseline["workload"],
                }
            )
        ).hexdigest(),
    }
    if application != expected_application:
        raise SchemaError("application evidence identity does not match the oracle corpus")

    threshold_pct = float(policy["severe_regression_threshold_pct"])
    world_size = int(policy["runner"]["world_size"])
    executed_collectives = sum(node["operation"] == "all_reduce" for node in projection["nodes"])
    executed_flops = sum(int(node["executed_flops"]) for node in projection["nodes"])
    for split_name, baseline in (("training", training_baseline), ("holdout", holdout_baseline)):
        for row in splits[split_name]["perturbations"]:
            perturbation_id = str(row["perturbation_id"])
            evidence = row["evidence"]
            corpus_row = corpus_rows[perturbation_id]
            decision = application_evidence_regression_decision(
                baseline,
                evidence,
                threshold_pct=threshold_pct,
            )
            runtime = float(evidence["summary"]["median_batch_latency_ms"]) / 1000.0
            expected_metrics = {
                "physical_runtime_seconds": runtime,
                "gpu_seconds": runtime * world_size,
                "gpu_count": world_size,
                "executed_collectives": executed_collectives,
                "executed_flops": executed_flops,
                "peak_memory_bytes": int(evidence["summary"]["max_observed_gpu_memory_used_bytes"]),
            }
            expected_row = {
                "perturbation_id": perturbation_id,
                "split": split_name,
                "application_decision": decision["decision"],
                "regression_magnitude_pct": decision["regression_magnitude_pct"],
                "severe": (decision["decision"] == "fail" and decision["regression_magnitude_pct"] >= threshold_pct),
                "application_metrics": expected_metrics,
                "application_evidence_sha256": application_evidence_id(evidence),
                "environment_sha256": evidence["environment_sha256"],
                "candidate_subject_sha256": evidence["subject_sha256"],
            }
            if corpus_row != expected_row:
                raise SchemaError(f"application evidence does not recompute oracle corpus row {perturbation_id!r}")


def _bind_physical_evidence(
    physical_evidence: Mapping[str, Any],
    corpus: Mapping[str, Any],
    active_ledger: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    """Recompute every corpus observation from its raw physical measurement."""

    validate_physical_execution_evidence_set(physical_evidence)
    if active_ledger.get("physical_evidence_set_id") != physical_evidence["evidence_set_id"]:
        raise SchemaError("physical evidence does not match the active study ledger")
    if physical_evidence.get("runner_oci_digest") != policy["runner"]["oci_digest"]:
        raise SchemaError("physical evidence runner does not match the physical policy")
    evidence_rows = physical_evidence["measurements"]
    by_measurement_id = {str(row["measurement"]["measurement_id"]): row for row in evidence_rows}
    requests = active_ledger["measurement_requests"]
    if set(by_measurement_id) != {str(request["measurement_id"]) for request in requests}:
        raise SchemaError("physical evidence inventory does not match the active study ledger")
    for request in requests:
        row = by_measurement_id[str(request["measurement_id"])]
        measurement = row["measurement"]
        expected_request = {
            "candidate_id": row["candidate_id"],
            "method": row["method"],
            "selected_region_ids": measurement["selected_region_ids"],
            "executable_sha256": measurement["executable_sha256"],
            "split": request["split"],
            "perturbation_id": measurement["perturbation_id"],
            "subject_sha256": measurement["subject_sha256"],
            "measurement_id": measurement["measurement_id"],
        }
        if request != expected_request:
            raise SchemaError("physical evidence does not recompute an active study request")

    for evaluation in corpus["evaluations"]:
        candidate_id = str(evaluation["candidate_id"])
        method = str(evaluation["method"])
        for observation in evaluation["observations"]:
            measurement_id = str(observation["evidence_sha256"])
            row = by_measurement_id.get(measurement_id)
            if row is None:
                raise SchemaError("physical corpus observation lacks its raw measurement")
            measurement = row["measurement"]
            if (
                row["candidate_id"] != candidate_id
                or row["method"] != method
                or measurement["perturbation_id"] != observation["perturbation_id"]
                or measurement["executable_sha256"] != evaluation["executable_sha256"]
                or measurement["selected_region_ids"] != evaluation["selected_region_ids"]
                or measurement["physical_metrics"] != observation["physical_metrics"]
            ):
                raise SchemaError("physical evidence does not recompute a corpus observation")


def _bundle_artifact_bytes(
    *,
    trace_raw: bytes,
    projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    corpus: Optional[Mapping[str, Any]],
    active_ledger: Optional[Mapping[str, Any]],
    application_evidence: Optional[Mapping[str, Any]],
    physical_evidence: Optional[Mapping[str, Any]],
    synthesis: PhysicalSynthesisResult,
    executable_projection: Mapping[str, Any],
    mode: str,
) -> Dict[str, bytes]:
    artifacts = {
        PHYSICAL_CANARY_ET: synthesis.canary_et,
        PHYSICAL_CANARY_POLICY: formatted_json_bytes(policy, indent=2),
        PHYSICAL_CANARY_PROJECTION: formatted_json_bytes(projection, indent=2),
        PHYSICAL_CANARY_EXECUTION_PROJECTION: formatted_json_bytes(executable_projection, indent=2),
        PHYSICAL_CANARY_LEDGER: formatted_json_bytes(synthesis.ledger, indent=2),
        PHYSICAL_CANARY_CERTIFICATE: formatted_json_bytes(synthesis.certificate, indent=2),
        PHYSICAL_CANARY_LEAKAGE: formatted_json_bytes(synthesis.leakage, indent=2),
        PHYSICAL_CANARY_SOURCE: trace_raw,
    }
    if corpus is not None:
        artifacts[PHYSICAL_CANARY_CORPUS] = formatted_json_bytes(corpus, indent=2)
    if active_ledger is not None:
        artifacts[PHYSICAL_CANARY_ACTIVE_LEDGER] = formatted_json_bytes(active_ledger, indent=2)
    if application_evidence is not None:
        artifacts[PHYSICAL_CANARY_APPLICATION_EVIDENCE] = formatted_json_bytes(
            application_evidence,
            indent=2,
        )
    if physical_evidence is not None:
        artifacts[PHYSICAL_CANARY_PHYSICAL_EVIDENCE] = formatted_json_bytes(
            physical_evidence,
            indent=2,
        )
    if mode == "private_exchange":
        return {
            name: artifacts[name]
            for name in (
                PHYSICAL_CANARY_ET,
                PHYSICAL_CANARY_POLICY,
                PHYSICAL_CANARY_EXECUTION_PROJECTION,
                PHYSICAL_CANARY_CERTIFICATE,
                PHYSICAL_CANARY_LEAKAGE,
                *(tuple() if active_ledger is None else (PHYSICAL_CANARY_ACTIVE_LEDGER,)),
            )
        }
    return artifacts


def _bundle_manifest(
    *,
    mode: str,
    status: str,
    trace_sha256: str,
    trace_size: int,
    metadata_version: str,
    projection: Mapping[str, Any],
    executable_projection: Mapping[str, Any],
    policy: Mapping[str, Any],
    corpus: Optional[Mapping[str, Any]],
    active_ledger: Optional[Mapping[str, Any]],
    application_evidence: Optional[Mapping[str, Any]],
    physical_evidence: Optional[Mapping[str, Any]],
    synthesis: PhysicalSynthesisResult,
    artifact_bytes: Mapping[str, bytes],
    owner_public_key_sha256: Optional[str],
) -> Dict[str, Any]:
    artifacts = {
        name: {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        for name, raw in sorted(artifact_bytes.items())
    }
    raw_manifest: Dict[str, Any] = {
        "format": PHYSICAL_DECISION_CANARY_FORMAT,
        "status": status,
        "publication_mode": mode,
        "supported_domain": SUPPORTED_PHYSICAL_DOMAIN,
        "source_et": {
            "sha256": trace_sha256,
            "size_bytes": trace_size,
            "metadata_version": metadata_version,
        },
        "projection_id": projection["projection_id"],
        "executable_projection_id": executable_projection["projection_id"],
        "policy_id": policy["policy_id"],
        "gate_policy": copy.deepcopy(dict(policy)),
        "runner": copy.deepcopy(policy["runner"]),
        "corpus_id": None if corpus is None else corpus["corpus_id"],
        "active_ledger_id": None if active_ledger is None else active_ledger["ledger_id"],
        "application_evidence_set_id": (
            None if application_evidence is None else application_evidence["evidence_set_id"]
        ),
        "physical_evidence_set_id": (None if physical_evidence is None else physical_evidence["evidence_set_id"]),
        "baseline_subject_sha256": (None if corpus is None else corpus["baseline_subject_sha256"]),
        "selection": copy.deepcopy(synthesis.ledger["selection"]),
        "certificate_id": synthesis.certificate["certificate_id"],
        "ledger_id": synthesis.ledger["ledger_id"],
        "leakage_assessment_id": synthesis.leakage["assessment_id"],
        "artifacts": artifacts,
        "scientific_roles": {
            "application_ground_truth": synthesis.certificate["claims"]["application_ground_truth"],
            "trace_derived_reference": "not_application_ground_truth",
            "exact_materialization_control": "conformance_control_not_included",
            "reduced_decision_canary": (
                "held_out_measured" if status == "qualified_physical_decision_canary" else "unproven"
            ),
        },
        "authenticity": (
            {
                "producer_signature": "detached_ed25519",
                "private_exchange": "owner_signed_source_withheld",
                "owner_public_key_sha256": owner_public_key_sha256,
                "signature_artifact": PHYSICAL_CANARY_SIGNATURE,
            }
            if mode == "private_exchange"
            else {
                "producer_signature": "absent",
                "private_exchange": "not_applicable",
                "owner_public_key_sha256": None,
                "signature_artifact": None,
            }
        ),
    }
    return with_content_identity(raw_manifest, "bundle_id")


def _validate_manifest_shape(
    manifest: Mapping[str, Any],
    *,
    limits: ResourceLimits,
) -> None:
    expected_fields = {
        "format",
        "bundle_id",
        "status",
        "publication_mode",
        "supported_domain",
        "source_et",
        "projection_id",
        "executable_projection_id",
        "policy_id",
        "gate_policy",
        "runner",
        "corpus_id",
        "active_ledger_id",
        "application_evidence_set_id",
        "physical_evidence_set_id",
        "baseline_subject_sha256",
        "selection",
        "certificate_id",
        "ledger_id",
        "leakage_assessment_id",
        "artifacts",
        "scientific_roles",
        "authenticity",
    }
    if set(manifest) != expected_fields:
        raise SchemaError("physical decision canary manifest fields are not closed")
    if manifest.get("format") != PHYSICAL_DECISION_CANARY_FORMAT:
        raise SchemaError(f"physical decision canary format must be {PHYSICAL_DECISION_CANARY_FORMAT!r}")
    expected_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in manifest.items() if key != "bundle_id"})
    ).hexdigest()
    if manifest.get("bundle_id") != expected_id:
        raise SchemaError("physical decision canary bundle_id does not match canonical content")
    if manifest.get("status") not in PHYSICAL_CANARY_STATUSES:
        raise SchemaError("physical decision canary status is unsupported")
    if manifest.get("publication_mode") not in PHYSICAL_CANARY_MODES:
        raise SchemaError("physical decision canary publication_mode is unsupported")
    if manifest.get("supported_domain") != SUPPORTED_PHYSICAL_DOMAIN:
        raise SchemaError("physical decision canary supported_domain is unsupported")

    source_et = manifest.get("source_et")
    if not isinstance(source_et, Mapping) or set(source_et) != {
        "sha256",
        "size_bytes",
        "metadata_version",
    }:
        raise SchemaError("physical decision canary source_et is malformed")
    _manifest_sha256(source_et.get("sha256"), "source_et.sha256")
    _manifest_positive_int(source_et.get("size_bytes"), "source_et.size_bytes")
    metadata_version = source_et.get("metadata_version")
    if not isinstance(metadata_version, str) or not metadata_version:
        raise SchemaError("physical decision canary source_et.metadata_version must be a non-empty string")
    for field in (
        "projection_id",
        "executable_projection_id",
        "policy_id",
        "certificate_id",
        "ledger_id",
        "leakage_assessment_id",
    ):
        _manifest_sha256(manifest.get(field), field)
    corpus_id = manifest.get("corpus_id")
    if corpus_id is not None:
        _manifest_sha256(corpus_id, "corpus_id")
    if manifest.get("status") == "blocked_missing_physical_oracle_corpus":
        if corpus_id is not None:
            raise SchemaError("missing-corpus physical canary unexpectedly binds a corpus_id")
        if manifest.get("baseline_subject_sha256") is not None:
            raise SchemaError("missing-corpus physical canary unexpectedly binds a baseline subject")
    elif corpus_id is None:
        raise SchemaError("physical decision canary status requires a corpus_id")
    else:
        _manifest_sha256(
            manifest.get("baseline_subject_sha256"),
            "baseline_subject_sha256",
        )
    active_ledger_id = manifest.get("active_ledger_id")
    application_evidence_set_id = manifest.get("application_evidence_set_id")
    physical_evidence_set_id = manifest.get("physical_evidence_set_id")
    if active_ledger_id is not None:
        _manifest_sha256(active_ledger_id, "active_ledger_id")
        if corpus_id is None:
            raise SchemaError("physical decision canary active ledger requires a corpus")
        _manifest_sha256(application_evidence_set_id, "application_evidence_set_id")
        _manifest_sha256(physical_evidence_set_id, "physical_evidence_set_id")
    elif application_evidence_set_id is not None or physical_evidence_set_id is not None:
        raise SchemaError("physical decision canary evidence requires an active ledger")

    gate_policy = manifest.get("gate_policy")
    if not isinstance(gate_policy, Mapping):
        raise SchemaError("physical decision canary gate_policy must be an object")
    validate_physical_canary_policy(gate_policy, limits=limits)
    if gate_policy.get("policy_id") != manifest.get("policy_id"):
        raise SchemaError("physical decision canary policy_id does not match gate_policy")
    if manifest.get("runner") != gate_policy.get("runner"):
        raise SchemaError("physical decision canary runner does not match gate_policy")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "source_et_sha256",
        "projection_id",
        "policy_id",
        "corpus_id",
        "candidate_id",
        "selected_region_ids",
        "selected_node_ids",
        "canary_et_sha256",
        "canary_et_bytes",
        "selection_sha256",
    }:
        raise SchemaError("physical decision canary selection is malformed")
    for selection_field in (
        "source_et_sha256",
        "projection_id",
        "policy_id",
        "canary_et_sha256",
        "selection_sha256",
    ):
        _manifest_sha256(selection.get(selection_field), f"selection.{selection_field}")
    if selection.get("corpus_id") != corpus_id:
        raise SchemaError("physical decision canary selection corpus_id mismatch")
    if selection.get("source_et_sha256") != source_et.get("sha256"):
        raise SchemaError("physical decision canary selection source identity mismatch")
    if selection.get("projection_id") != manifest.get("projection_id"):
        raise SchemaError("physical decision canary selection projection identity mismatch")
    if selection.get("policy_id") != manifest.get("policy_id"):
        raise SchemaError("physical decision canary selection policy identity mismatch")
    candidate_id = selection.get("candidate_id")
    if candidate_id is not None and (not isinstance(candidate_id, str) or not candidate_id):
        raise SchemaError("physical decision canary selection candidate_id is malformed")
    _manifest_sorted_unique_strings(selection.get("selected_region_ids"), "selection.selected_region_ids")
    selected_node_ids = selection.get("selected_node_ids")
    if (
        not isinstance(selected_node_ids, list)
        or not selected_node_ids
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in selected_node_ids)
        or len(selected_node_ids) != len(set(selected_node_ids))
    ):
        raise SchemaError("physical decision canary selection selected_node_ids is malformed")
    _manifest_positive_int(selection.get("canary_et_bytes"), "selection.canary_et_bytes")
    expected_selection_id = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in selection.items() if key != "selection_sha256"})
    ).hexdigest()
    if selection.get("selection_sha256") != expected_selection_id:
        raise SchemaError("physical decision canary selection_sha256 does not match canonical content")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise SchemaError("physical decision canary artifacts must be a non-empty object")
    if manifest.get("publication_mode") == "private_exchange":
        expected_artifacts = {
            PHYSICAL_CANARY_ET,
            PHYSICAL_CANARY_POLICY,
            PHYSICAL_CANARY_EXECUTION_PROJECTION,
            PHYSICAL_CANARY_CERTIFICATE,
            PHYSICAL_CANARY_LEAKAGE,
        }
        if active_ledger_id is not None:
            expected_artifacts.add(PHYSICAL_CANARY_ACTIVE_LEDGER)
    else:
        expected_artifacts = {
            PHYSICAL_CANARY_ET,
            PHYSICAL_CANARY_POLICY,
            PHYSICAL_CANARY_PROJECTION,
            PHYSICAL_CANARY_EXECUTION_PROJECTION,
            PHYSICAL_CANARY_LEDGER,
            PHYSICAL_CANARY_CERTIFICATE,
            PHYSICAL_CANARY_LEAKAGE,
            PHYSICAL_CANARY_SOURCE,
        }
        if corpus_id is not None:
            expected_artifacts.add(PHYSICAL_CANARY_CORPUS)
        if active_ledger_id is not None:
            expected_artifacts.add(PHYSICAL_CANARY_ACTIVE_LEDGER)
            expected_artifacts.add(PHYSICAL_CANARY_APPLICATION_EVIDENCE)
            expected_artifacts.add(PHYSICAL_CANARY_PHYSICAL_EVIDENCE)
    if set(artifacts) != expected_artifacts:
        raise SchemaError("physical decision canary artifact inventory is not exact")
    for name, raw_identity in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise SchemaError("physical decision canary artifact names must be canonical basenames")
        if not isinstance(raw_identity, Mapping) or set(raw_identity) != {"sha256", "size_bytes"}:
            raise SchemaError(f"physical decision canary artifact identity is malformed: {name}")
        _manifest_sha256(raw_identity.get("sha256"), f"artifacts.{name}.sha256")
        _manifest_nonnegative_int(raw_identity.get("size_bytes"), f"artifacts.{name}.size_bytes")
    if manifest.get("publication_mode") != "private_exchange":
        if artifacts[PHYSICAL_CANARY_SOURCE] != {
            "sha256": source_et["sha256"],
            "size_bytes": source_et["size_bytes"],
        }:
            raise SchemaError("physical decision canary source_et does not match source.et identity")
    if artifacts[PHYSICAL_CANARY_ET] != {
        "sha256": selection["canary_et_sha256"],
        "size_bytes": selection["canary_et_bytes"],
    }:
        raise SchemaError("physical decision canary selection does not match canary.et identity")

    scientific_roles = manifest.get("scientific_roles")
    if not isinstance(scientific_roles, Mapping) or set(scientific_roles) != {
        "application_ground_truth",
        "trace_derived_reference",
        "exact_materialization_control",
        "reduced_decision_canary",
    }:
        raise SchemaError("physical decision canary scientific_roles is malformed")
    if scientific_roles.get("application_ground_truth") not in {"measured", "unproven"}:
        raise SchemaError("physical decision canary application_ground_truth role is unsupported")
    if scientific_roles.get("trace_derived_reference") != "not_application_ground_truth":
        raise SchemaError("physical decision canary trace-derived role is malformed")
    if scientific_roles.get("exact_materialization_control") != "conformance_control_not_included":
        raise SchemaError("physical decision canary exact-materialization role is malformed")
    expected_reduced_role = (
        "held_out_measured" if manifest.get("status") == "qualified_physical_decision_canary" else "unproven"
    )
    if scientific_roles.get("reduced_decision_canary") != expected_reduced_role:
        raise SchemaError("physical decision canary reduced-canary role is malformed")

    authenticity = manifest.get("authenticity")
    if not isinstance(authenticity, Mapping) or set(authenticity) != {
        "producer_signature",
        "private_exchange",
        "owner_public_key_sha256",
        "signature_artifact",
    }:
        raise SchemaError("physical decision canary authenticity boundary is malformed")
    if manifest.get("publication_mode") == "private_exchange":
        if (
            authenticity.get("producer_signature") != "detached_ed25519"
            or authenticity.get("private_exchange") != "owner_signed_source_withheld"
            or authenticity.get("signature_artifact") != PHYSICAL_CANARY_SIGNATURE
        ):
            raise SchemaError("private exchange signature boundary is malformed")
        _manifest_sha256(authenticity.get("owner_public_key_sha256"), "authenticity.owner_public_key_sha256")
        if manifest.get("status") != "qualified_physical_decision_canary":
            raise SchemaError("private exchange manifest must be qualified")
    elif dict(authenticity) != {
        "producer_signature": "absent",
        "private_exchange": "not_applicable",
        "owner_public_key_sha256": None,
        "signature_artifact": None,
    }:
        raise SchemaError("unsigned physical canary authenticity boundary is malformed")


def _verify_private_exchange_artifacts(
    manifest: Mapping[str, Any],
    snapshots: Mapping[str, Any],
    *,
    limits: ResourceLimits,
) -> None:
    """Validate every source-withheld artifact the receiver can recompute."""

    policy = _decode_json_object(snapshots[PHYSICAL_CANARY_POLICY].raw, "policy", limits)
    validate_physical_canary_policy(policy, limits=limits)
    if policy != manifest.get("gate_policy"):
        raise SchemaError("private exchange policy bytes do not match the manifest gate policy")

    canary = decode_chakra_execution_trace(snapshots[PHYSICAL_CANARY_ET].raw, limits=limits)
    selection = manifest["selection"]
    if canary.node_ids != tuple(selection["selected_node_ids"]):
        raise SchemaError("private exchange canary node inventory does not match the signed selection")
    if canary.metadata_version != manifest["source_et"]["metadata_version"]:
        raise SchemaError("private exchange canary metadata version does not match the source commitment")
    executable_projection = _decode_json_object(
        snapshots[PHYSICAL_CANARY_EXECUTION_PROJECTION].raw,
        "canary projection",
        limits,
    )
    validate_chakra_projection(executable_projection, canary, limits=limits)
    if executable_projection.get("projection_id") != manifest.get("executable_projection_id"):
        raise SchemaError("private exchange canary projection does not match the manifest")

    leakage = _decode_json_object(snapshots[PHYSICAL_CANARY_LEAKAGE].raw, "leakage assessment", limits)
    if leakage.get("format") != PHYSICAL_LEAKAGE_ASSESSMENT_FORMAT:
        raise SchemaError("private exchange leakage assessment format is unsupported")
    if with_content_identity(leakage, "assessment_id") != leakage:
        raise SchemaError("private exchange leakage assessment identity does not recompute")
    if leakage.get("assessment_id") != manifest.get("leakage_assessment_id"):
        raise SchemaError("private exchange leakage assessment does not match the manifest")
    leakage_score = leakage.get("leakage_score")
    if not isinstance(leakage_score, (int, float)) or isinstance(leakage_score, bool):
        raise SchemaError("private exchange leakage score must be numeric")
    if float(leakage_score) > float(policy["privacy"]["maximum_leakage_score"]):
        raise SchemaError("private exchange leakage exceeds the signed policy")
    if leakage.get("opaque_chakra_attributes_reviewed") is not True:
        raise SchemaError("private exchange lacks the required opaque-attribute review")

    active_ledger_id = manifest.get("active_ledger_id")
    if active_ledger_id is not None:
        active = _decode_json_object(
            snapshots[PHYSICAL_CANARY_ACTIVE_LEDGER].raw,
            "active study ledger",
            limits,
        )
        validate_active_physical_study_ledger(active)
        if active.get("ledger_id") != active_ledger_id:
            raise SchemaError("private exchange active ledger identity does not match the manifest")
        if active.get("source_et_sha256") != manifest["source_et"]["sha256"]:
            raise SchemaError("private exchange active ledger source does not match the manifest")
        if active.get("projection_id") != manifest.get("projection_id"):
            raise SchemaError("private exchange active ledger projection does not match the manifest")
        if active.get("policy_id") != manifest.get("policy_id"):
            raise SchemaError("private exchange active ledger policy does not match the manifest")
        if active.get("corpus_id") != manifest.get("corpus_id"):
            raise SchemaError("private exchange active ledger corpus does not match the manifest")
        if active.get("application_evidence_set_id") != manifest.get("application_evidence_set_id"):
            raise SchemaError("private exchange application evidence commitment does not match the active ledger")
        if active.get("physical_evidence_set_id") != manifest.get("physical_evidence_set_id"):
            raise SchemaError("private exchange physical evidence commitment does not match the active ledger")
        active_selection = active.get("selection")
        if not isinstance(active_selection, Mapping):
            raise SchemaError("private exchange active ledger selection is missing")
        for field in ("selected_region_ids", "selected_node_ids", "canary_et_sha256"):
            if active_selection.get(field) != selection.get(field):
                raise SchemaError(f"private exchange active selection does not match {field}")

    certificate = _decode_json_object(
        snapshots[PHYSICAL_CANARY_CERTIFICATE].raw,
        "fidelity certificate",
        limits,
    )
    if certificate.get("format") != PHYSICAL_FIDELITY_CERTIFICATE_FORMAT:
        raise SchemaError("private exchange fidelity certificate format is unsupported")
    if with_content_identity(certificate, "certificate_id") != certificate:
        raise SchemaError("private exchange fidelity certificate identity does not recompute")
    if certificate.get("certificate_id") != manifest.get("certificate_id"):
        raise SchemaError("private exchange fidelity certificate does not match the manifest")
    if certificate.get("status") != manifest.get("status"):
        raise SchemaError("private exchange fidelity certificate status does not match the manifest")
    if certificate.get("selection_sha256") != selection.get("selection_sha256"):
        raise SchemaError("private exchange fidelity certificate selection does not match the manifest")
    if certificate.get("leakage_assessment_id") != leakage.get("assessment_id"):
        raise SchemaError("private exchange fidelity certificate names the wrong leakage assessment")
    claims = certificate.get("claims")
    if not isinstance(claims, Mapping) or claims.get("application_ground_truth") != "measured":
        raise SchemaError("private exchange fidelity certificate lacks measured application ground truth")
    if (
        claims.get("decision_preservation") != "held_out_measured"
        or claims.get("physical_runtime_reduction") != "held_out_measured"
    ):
        raise SchemaError("private exchange fidelity certificate lacks held-out physical claims")


def _manifest_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError(f"physical decision canary {label} must be a lowercase SHA-256")


def _manifest_nonnegative_int(value: Any, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError(f"physical decision canary {label} must be a non-negative integer")


def _manifest_positive_int(value: Any, label: str) -> None:
    _manifest_nonnegative_int(value, label)
    if value == 0:
        raise SchemaError(f"physical decision canary {label} must be positive")


def _manifest_sorted_unique_strings(value: Any, label: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise SchemaError(f"physical decision canary {label} must be sorted and unique")


def _decode_json_object(raw: bytes, label: str, limits: ResourceLimits) -> Dict[str, Any]:
    try:
        value = decode_bounded_json_bytes(raw, limits=limits)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SchemaError(f"physical decision canary {label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"physical decision canary {label} must be a JSON object")
    return value


def _publish_new_directory(target: Path, artifacts: Mapping[str, bytes]) -> None:
    if PHYSICAL_CANARY_MANIFEST not in artifacts:
        raise SchemaError("physical decision canary publication requires manifest.json")
    for name in artifacts:
        if Path(name).name != name:
            raise SchemaError("physical decision canary artifact names must be canonical basenames")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(mode=0o700)
    except OSError as exc:
        raise CommCanaryIOError(
            f"cannot prepare physical decision canary directory {target}: {exc}",
            path=str(target),
            operation="prepare physical decision canary directory",
        ) from exc
    created_names: list[str] = []
    try:
        # Reserve the final path with an exclusive mkdir.  Publishing into a
        # sibling staging directory and renaming it can replace an empty
        # directory that appears after a preflight existence check on POSIX.
        # The manifest is installed last, so an interrupted or concurrent read
        # of the reserved directory cannot be mistaken for a complete bundle.
        for name in sorted(set(artifacts).difference({PHYSICAL_CANARY_MANIFEST})):
            atomic_write_bytes(target / name, artifacts[name], policy=_IMMUTABLE_BUNDLE_POLICY)
            created_names.append(name)
        _fsync_directory(target)
        atomic_write_bytes(
            target / PHYSICAL_CANARY_MANIFEST,
            artifacts[PHYSICAL_CANARY_MANIFEST],
            policy=_IMMUTABLE_BUNDLE_POLICY,
        )
        created_names.append(PHYSICAL_CANARY_MANIFEST)
        _fsync_directory(target)
        _fsync_directory(target.parent)
    except BaseException as exc:
        _remove_incomplete_publication(target, created_names)
        if isinstance(exc, (SchemaError, CommCanaryIOError)):
            raise
        if isinstance(exc, OSError):
            raise CommCanaryIOError(
                f"cannot publish physical decision canary directory {target}: {exc}",
                path=str(target),
                operation="publish physical decision canary directory",
            ) from exc
        raise


def _remove_incomplete_publication(target: Path, created_names: list[str]) -> None:
    """Remove only entries this publication was authorized to create."""

    for name in reversed(created_names):
        try:
            (target / name).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # An unexpected replacement or extra entry leaves a visibly
            # incomplete fail-closed directory instead of widening deletion.
            return
    try:
        target.rmdir()
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PHYSICAL_CANARY_MODES",
    "build_physical_canary_bundle",
    "verify_physical_canary_bundle",
]
