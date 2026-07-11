from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Dict

import pytest  # type: ignore[import-not-found]

from experiments.rostam.harness import (
    CAMPAIGN_SCHEMA,
    MAX_CAMPAIGN_MATRIX_CELLS,
    MAX_CAMPAIGN_REPETITIONS,
    CampaignSpec,
    CanonicalJSONError,
    JSONResourceLimits,
    ManifestFreezeError,
    ManifestValidationError,
    PathContainmentError,
    RunManifest,
    UnsafeSlugError,
    build_run_manifest,
    canonical_json_bytes,
    canonical_sha256,
    contained_path,
    freeze_run_manifest,
    load_frozen_run,
    read_bounded_bytes,
    safe_slug,
    strict_json_loads,
)
from experiments.rostam.harness import manifest as manifest_module


def _digest(character: str) -> str:
    return character * 64


def _campaign_dict(*, shuffled: bool = False) -> Dict[str, Any]:
    configurations = [
        {
            "id": "nccl-2.19.3-default",
            "environment": {},
            "parameters": {"nccl": "2.19.3"},
            "expected_runtime": {"nccl_version_code": 21903},
        },
        {
            "id": "nccl-2.20.5-ring-ll",
            "environment": {"NCCL_PROTO": "LL", "NCCL_ALGO": "Ring"},
            "parameters": {"nccl": "2.20.5"},
            "expected_runtime": {"nccl_version_code": 22005},
        },
    ]
    workloads = [
        {
            "id": "capture",
            "producer_schema": "commcanary.experiment.capture-output.v1",
            "measurement_schema": "commcanary.experiment.artifact-set.v1",
            "parameters": {"tokens": 2},
            "depends_on": [],
        },
        {
            "id": "replay",
            "producer_schema": "commcanary.experiment.replay-output.v1",
            "measurement_schema": "commcanary.experiment.latency-series.v1",
            "parameters": {"warmup": 1},
            "depends_on": ["capture"],
        },
    ]
    inputs = [
        {"id": "configs", "sha256": _digest("b"), "size_bytes": 22},
        {"id": "source-wheel", "sha256": _digest("a"), "size_bytes": 11},
    ]
    if shuffled:
        configurations.reverse()
        workloads.reverse()
        inputs.reverse()
    return {
        "schema": CAMPAIGN_SCHEMA,
        "run_id": "rostam-golden-run",
        "campaign_id": "rostam-ranking",
        "repository": {
            "commit": "1" * 40,
            "dirty": False,
            "patch_sha256": None,
            "source_archive_sha256": _digest("c"),
        },
        "inputs": inputs,
        "axes": {
            "configurations": configurations,
            "workloads": workloads,
            "repetitions": 2,
        },
        "policy": {
            "aggregation": "median",
            "tie": {"kind": "max-iqr"},
        },
        "expected_site": {
            "site_id": "rostam",
            "scheduler": "slurm",
            "partition": "cuda-A100",
            "nodes": 1,
            "exclusive": True,
            "node_constraints": ["toranj0"],
            "account": None,
            "resources": {"gpus": 4},
        },
    }


def _manifest() -> RunManifest:
    return build_run_manifest(CampaignSpec.from_dict(_campaign_dict()))


def test_canonical_json_and_hash_are_order_independent() -> None:
    left = {"z": [3, 2, 1], "a": {"unicode": "λ", "flag": True}}
    right = {"a": {"flag": True, "unicode": "λ"}, "z": [3, 2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_bytes(left) == b'{"a":{"flag":true,"unicode":"\xce\xbb"},"z":[3,2,1]}'
    assert canonical_sha256(left) == canonical_sha256(right)


def test_strict_json_rejects_duplicates_and_non_finite_values() -> None:
    with pytest.raises(CanonicalJSONError, match="duplicate"):
        strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(CanonicalJSONError, match="non-standard"):
        strict_json_loads('{"a":NaN}')
    with pytest.raises(CanonicalJSONError, match="non-finite"):
        canonical_json_bytes({"a": float("inf")})


@pytest.mark.parametrize(  # type: ignore[misc]
    ("payload", "limits", "message"),
    [
        (b'{"value":1}', JSONResourceLimits(max_document_bytes=8), "max_document_bytes=8"),
        (b"[[[]]]", JSONResourceLimits(max_depth=2), "max_depth=2"),
        (b"[0,1,2]", JSONResourceLimits(max_items=2), "max_items=2"),
        ('{"value":"λλ"}', JSONResourceLimits(max_string_bytes=3), "max_string_bytes=3"),
        (b"12345", JSONResourceLimits(max_numeric_characters=4), "max_numeric_characters=4"),
    ],
)
def test_strict_json_resource_limits_fail_with_stable_errors(
    payload: str | bytes,
    limits: JSONResourceLimits,
    message: str,
) -> None:
    with pytest.raises(CanonicalJSONError, match=message):
        strict_json_loads(payload, limits=limits)


def test_canonical_encoder_uses_the_same_resource_limits() -> None:
    with pytest.raises(CanonicalJSONError, match="max_items=2"):
        canonical_json_bytes([0, 1, 2], limits=JSONResourceLimits(max_items=2))
    with pytest.raises(CanonicalJSONError, match="max_numeric_characters=4"):
        canonical_json_bytes(12345, limits=JSONResourceLimits(max_numeric_characters=4))


def test_depth_preflight_ignores_brackets_inside_json_strings() -> None:
    assert strict_json_loads(
        '{"literal":"[[{{\\"", "value":[1]}',
        limits=JSONResourceLimits(max_depth=2),
    ) == {"literal": '[[{{"', "value": [1]}


def test_bounded_file_reader_uses_a_max_plus_one_read(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b"123456789")
    with pytest.raises(CanonicalJSONError, match="max_bytes=8"):
        read_bounded_bytes(path, max_bytes=8, field="adversarial fixture")


def test_frozen_manifest_loader_applies_the_shared_byte_cap(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    frozen = freeze_run_manifest(_manifest(), tmp_path / "results")
    monkeypatch.setattr(
        manifest_module,
        "DEFAULT_JSON_LIMITS",
        JSONResourceLimits(max_document_bytes=8),
    )
    with pytest.raises(ManifestValidationError, match="max_bytes=8"):
        load_frozen_run(frozen.directory)


@pytest.mark.parametrize(  # type: ignore[misc]
    "value",
    ["../escape", "..", ".hidden", "Uppercase", "slash/name", "back\\slash", "x" * 65],
)
def test_safe_slug_rejects_traversal_and_unbounded_labels(value: str) -> None:
    with pytest.raises(UnsafeSlugError):
        safe_slug(value)


def test_contained_path_rejects_escape_and_resolved_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PathContainmentError):
        contained_path(root, "../outside")
    with pytest.raises(PathContainmentError):
        contained_path(root, "back\\slash")
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathContainmentError):
        contained_path(root, "link")


def test_shuffled_axes_and_objects_freeze_to_same_golden_manifest() -> None:
    ordered = build_run_manifest(CampaignSpec.from_dict(_campaign_dict()))
    shuffled = build_run_manifest(CampaignSpec.from_dict(_campaign_dict(shuffled=True)))
    assert ordered.to_json_bytes() == shuffled.to_json_bytes()
    assert ordered.sha256 == shuffled.sha256
    assert [cell.id for cell in ordered.cells] == sorted(cell.id for cell in ordered.cells)
    assert len(ordered.cells) == 8
    # These constants make accidental changes to canonicalization or identity
    # coverage an explicit contract revision rather than a quiet drift.
    assert ordered.campaign_sha256 == "a6524a9b9188c4c683fce0256c1cc498bf1ef732f0486903f2ae5926a9787e17"
    assert ordered.sha256 == "472e46b6a61527fa02ae72a891e24649b3778453321a66213a83b913ffb4eb78"
    assert [cell.id for cell in ordered.cells] == [
        "c-capture-nccl-2.19.3-defaul-r000000-d330cc32ab076e8d",
        "c-capture-nccl-2.19.3-defaul-r000001-f719943cf10dc74e",
        "c-capture-nccl-2.20.5-ring-l-r000000-3ac5a2e0738fe045",
        "c-capture-nccl-2.20.5-ring-l-r000001-efbd8a0c28024244",
        "c-replay-nccl-2.19.3-defaul-r000000-0c3f1595444ec96a",
        "c-replay-nccl-2.19.3-defaul-r000001-8825597606d7d748",
        "c-replay-nccl-2.20.5-ring-l-r000000-281bdc368371058d",
        "c-replay-nccl-2.20.5-ring-l-r000001-a762520bba525fc7",
    ]


def test_dependencies_resolve_to_same_config_and_repetition() -> None:
    manifest = _manifest()
    lookup = {(cell.configuration_id, cell.workload_id, cell.repetition): cell for cell in manifest.cells}
    for config in ("nccl-2.19.3-default", "nccl-2.20.5-ring-ll"):
        for repetition in (0, 1):
            capture = lookup[(config, "capture", repetition)]
            replay = lookup[(config, "replay", repetition)]
            assert replay.dependencies == (capture.id,)
            assert capture.dependencies == ()


@pytest.mark.parametrize("axis", ["configurations", "workloads"])  # type: ignore[misc]
def test_duplicate_axis_ids_are_rejected(axis: str) -> None:
    raw = _campaign_dict()
    raw["axes"][axis].append(copy.deepcopy(raw["axes"][axis][0]))
    with pytest.raises(ManifestValidationError, match="duplicate"):
        CampaignSpec.from_dict(raw)


def test_unknown_dependency_and_dependency_cycle_are_rejected() -> None:
    unknown = _campaign_dict()
    unknown["axes"]["workloads"][1]["depends_on"] = ["missing"]
    with pytest.raises(ManifestValidationError, match="unknown workloads"):
        CampaignSpec.from_dict(unknown)

    cyclic = _campaign_dict()
    cyclic["axes"]["workloads"][0]["depends_on"] = ["replay"]
    with pytest.raises(ManifestValidationError, match="cycle"):
        CampaignSpec.from_dict(cyclic)


def test_duplicate_missing_unknown_and_forged_cells_are_rejected() -> None:
    raw = _manifest().to_dict()

    duplicate = copy.deepcopy(raw)
    duplicate["cells"].append(copy.deepcopy(duplicate["cells"][0]))
    with pytest.raises(ManifestValidationError, match="duplicate cell ids"):
        RunManifest.from_dict(duplicate)

    missing = copy.deepcopy(raw)
    missing["cells"].pop()
    with pytest.raises(ManifestValidationError, match="missing cells"):
        RunManifest.from_dict(missing)

    unknown = copy.deepcopy(raw)
    unknown["cells"][0]["repetition"] = 999
    unknown["cells"] = sorted(unknown["cells"], key=lambda item: item["id"])
    with pytest.raises(ManifestValidationError, match="matrix mismatch"):
        RunManifest.from_dict(unknown)

    forged = copy.deepcopy(raw)
    forged["cells"][0]["identity_sha256"] = _digest("f")
    with pytest.raises(ManifestValidationError, match="forged identity"):
        RunManifest.from_dict(forged)


def test_campaign_drift_is_detected_even_if_manifest_json_is_well_formed() -> None:
    raw = _manifest().to_dict()
    raw["campaign"]["policy"]["aggregation"] = "mean"
    with pytest.raises(ManifestValidationError, match="campaign hash"):
        RunManifest.from_dict(raw)


def test_expected_site_rejects_observed_scheduler_metadata() -> None:
    raw = _campaign_dict()
    raw["expected_site"]["slurm_job_id"] = "12345"
    with pytest.raises(ManifestValidationError, match="unknown fields"):
        CampaignSpec.from_dict(raw)


def test_manifest_round_trip_is_exact_and_immutable() -> None:
    manifest = _manifest()
    round_tripped = RunManifest.from_json_bytes(manifest.to_json_bytes())
    assert round_tripped == manifest
    assert round_tripped.to_json_bytes() == manifest.to_json_bytes()
    with pytest.raises(Exception):
        manifest.cells[0].dependencies += ("something",)  # type: ignore[misc]


def test_freeze_is_deterministic_contained_and_refuses_reuse(tmp_path: Path) -> None:
    manifest = _manifest()
    root = tmp_path / "results"
    frozen = freeze_run_manifest(manifest, root)
    assert frozen.directory == root.resolve() / manifest.run_id
    assert frozen.manifest_path.read_bytes() == manifest.to_json_bytes()
    assert frozen.manifest_sha256 == hashlib.sha256(frozen.manifest_path.read_bytes()).hexdigest()
    assert frozen.checksum_path.read_text(encoding="ascii") == (f"{manifest.sha256}  run_manifest.json\n")
    assert os.stat(frozen.manifest_path).st_mode & 0o222 == 0
    loaded, loaded_frozen = load_frozen_run(frozen.directory)
    assert loaded == manifest
    assert loaded_frozen == frozen
    with pytest.raises(ManifestFreezeError, match="already exists"):
        freeze_run_manifest(manifest, root)


def test_frozen_manifest_tamper_and_directory_drift_are_detected(tmp_path: Path) -> None:
    manifest = _manifest()
    frozen = freeze_run_manifest(manifest, tmp_path / "results")
    os.chmod(frozen.manifest_path, 0o644)
    frozen.manifest_path.write_bytes(frozen.manifest_path.read_bytes() + b" ")
    with pytest.raises(ManifestValidationError, match="SHA-256 mismatch"):
        load_frozen_run(frozen.directory)

    second_root = tmp_path / "second-results"
    second = freeze_run_manifest(manifest, second_root)
    moved = second_root / "wrong-run-name"
    second.directory.rename(moved)
    with pytest.raises(ManifestValidationError, match="does not match"):
        load_frozen_run(moved)


def test_manifest_loader_rejects_unknown_or_missing_root_fields() -> None:
    raw = _manifest().to_dict()
    extra = copy.deepcopy(raw)
    extra["observed_scheduler"] = {"job_id": "1"}
    with pytest.raises(ManifestValidationError, match="unknown fields"):
        RunManifest.from_dict(extra)
    missing = copy.deepcopy(raw)
    del missing["campaign_sha256"]
    with pytest.raises(ManifestValidationError, match="missing required"):
        RunManifest.from_dict(missing)


def test_campaign_repetition_and_matrix_caps_precede_expansion() -> None:
    excessive_repetitions = _campaign_dict()
    excessive_repetitions["axes"]["repetitions"] = MAX_CAMPAIGN_REPETITIONS + 1
    with pytest.raises(ManifestValidationError, match="repetitions"):
        CampaignSpec.from_dict(excessive_repetitions)

    excessive_matrix = _campaign_dict()
    configuration = excessive_matrix["axes"]["configurations"][0]
    workload = excessive_matrix["axes"]["workloads"][0]
    excessive_matrix["axes"] = {
        "configurations": [configuration] * 101,
        "workloads": [workload] * 100,
        "repetitions": 10,
    }
    assert 101 * 100 * 10 > MAX_CAMPAIGN_MATRIX_CELLS
    with pytest.raises(ManifestValidationError, match="MAX_CAMPAIGN_MATRIX_CELLS"):
        CampaignSpec.from_dict(excessive_matrix)


def test_dirty_repository_requires_exact_patch_hash() -> None:
    raw = _campaign_dict()
    raw["repository"]["dirty"] = True
    with pytest.raises(ManifestValidationError, match="requires"):
        CampaignSpec.from_dict(raw)
    raw["repository"]["patch_sha256"] = _digest("d")
    campaign = CampaignSpec.from_dict(raw)
    assert campaign.repository.patch_sha256 == _digest("d")


def test_raw_json_duplicate_manifest_field_is_rejected() -> None:
    payload = _manifest().to_json_bytes().decode("utf-8")
    malicious = payload.replace('{"campaign":', '{"schema":"duplicate","campaign":', 1)
    with pytest.raises(CanonicalJSONError, match="duplicate"):
        RunManifest.from_json_bytes(malicious.encode("utf-8"))
