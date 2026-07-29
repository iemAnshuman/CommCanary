"""Reviewed, byte-proven bridge for joining evidence across repository states."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, cast

from ..harness import (
    ContractError,
    JSONResourceLimits,
    RepositoryState,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    read_bounded_bytes,
    strict_json_loads,
)

PathLike = Union[str, "Path"]

CROSS_COMMIT_COMPATIBILITY_SCHEMA = "commcanary.rostam.cross-commit-compatibility.v1"
_CONTRACT_LIMITS = JSONResourceLimits(
    max_document_bytes=4 * 1024 * 1024,
    max_depth=12,
    max_items=100_000,
    max_string_bytes=16_384,
    max_numeric_characters=128,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_ALLOWED_POLICY_DIFFERENCES = frozenset({"script_hashes"})
_PUBLICATION_FILENAMES = frozenset({"aggregate.json", "aggregate.csv", "paper-fragment.md"})
_ROSTAM_ROOT = Path(__file__).resolve().parent.parent


class CrossCommitCompatibilityError(ContractError):
    """Raised when an explicit cross-commit evidence bridge is not trustworthy."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossCommitCompatibilityError(f"{field} must be an object")
    return value


def _strict(value: Mapping[str, Any], field: str, expected: Sequence[str]) -> None:
    expected_fields = set(expected)
    missing = sorted(expected_fields - set(value))
    unknown = sorted(set(value) - expected_fields)
    if missing or unknown:
        raise CrossCommitCompatibilityError(f"{field} fields mismatch: missing={missing!r}, unknown={unknown!r}")


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CrossCommitCompatibilityError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str, *, maximum: int = 8192) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise CrossCommitCompatibilityError(f"{field} must be a bounded non-empty string")
    return value


def _optional_nonempty(value: Any, field: str) -> Optional[str]:
    return None if value is None else _nonempty(value, field)


def _finite_non_negative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CrossCommitCompatibilityError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CrossCommitCompatibilityError(f"{field} must be a finite non-negative number")
    return result


def _string_array(value: Any, field: str, *, safe_ids: bool = False) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise CrossCommitCompatibilityError(f"{field} must be an array")
    result = []
    for index, item in enumerate(value):
        text = _nonempty(item, f"{field}[{index}]")
        if safe_ids and _SAFE_ID_RE.fullmatch(text) is None:
            raise CrossCommitCompatibilityError(f"{field}[{index}] must be a safe identifier")
        result.append(text)
    if result != sorted(result) or len(set(result)) != len(result):
        raise CrossCommitCompatibilityError(f"{field} must be sorted and unique")
    return tuple(result)


def analysis_implementation_record() -> Dict[str, Any]:
    """Fingerprint every local analyzer, harness, CLI, and schema source byte."""

    candidates = [_ROSTAM_ROOT / "analyze.py"]
    candidates.extend(sorted((_ROSTAM_ROOT / "analysis").glob("*.py")))
    candidates.extend(sorted((_ROSTAM_ROOT / "harness").glob("*.py")))
    candidates.extend(sorted((_ROSTAM_ROOT / "schemas").glob("*.json")))
    files = []
    seen_paths = set()
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise CrossCommitCompatibilityError(f"analysis implementation file is missing or unsafe: {path}")
        relative = path.relative_to(_ROSTAM_ROOT).as_posix()
        if relative in seen_paths:
            continue
        seen_paths.add(relative)
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    files.sort(key=lambda item: cast(str, item["path"]))
    if not files:
        raise CrossCommitCompatibilityError("analysis implementation inventory is empty")
    return {
        "fingerprint": canonical_sha256({"files": files}),
        "files": files,
    }


@dataclass(frozen=True)
class CrossCommitCompatibility:
    """Canonical reviewed contract whose ground-truth bytes must be rechecked."""

    raw: Mapping[str, Any]
    sha256: str
    campaign_bindings: Tuple[Mapping[str, Any], ...]
    ground_truth_manifest_sha256s: Tuple[str, ...]
    regeneration_command: str
    baseline_config: Optional[str]
    candidate_config: Optional[str]
    relative_threshold_pct: float
    absolute_threshold_us: float
    publication_sha256: Mapping[str, str]
    raw_archive_verified: bool
    allowed_policy_fields: Tuple[str, ...]
    allowed_input_ids: Tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw_value: Any) -> "CrossCommitCompatibility":
        raw = _object(raw_value, "cross-commit compatibility contract")
        _strict(
            raw,
            "cross-commit compatibility contract",
            (
                "schema",
                "status",
                "analysis_implementation",
                "campaigns",
                "ground_truth",
                "allowed_differences",
                "contract_sha256",
            ),
        )
        if raw["schema"] != CROSS_COMMIT_COMPATIBILITY_SCHEMA:
            raise CrossCommitCompatibilityError(f"unsupported cross-commit compatibility schema {raw['schema']!r}")
        if raw["status"] != "reviewed":
            raise CrossCommitCompatibilityError("cross-commit compatibility contract must have status='reviewed'")

        implementation = _object(raw["analysis_implementation"], "analysis_implementation")
        _strict(implementation, "analysis_implementation", ("fingerprint", "files"))
        _sha256(implementation["fingerprint"], "analysis_implementation.fingerprint")
        files_raw = implementation["files"]
        if not isinstance(files_raw, list) or not files_raw:
            raise CrossCommitCompatibilityError("analysis_implementation.files must be a non-empty array")
        files = []
        for index, item_raw in enumerate(files_raw):
            item = _object(item_raw, f"analysis_implementation.files[{index}]")
            _strict(item, f"analysis_implementation.files[{index}]", ("path", "sha256", "size_bytes"))
            path = _nonempty(item["path"], f"analysis_implementation.files[{index}].path")
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise CrossCommitCompatibilityError("analysis implementation paths must be safe relative paths")
            size = item["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= 2**63 - 1:
                raise CrossCommitCompatibilityError("analysis implementation size must be a non-negative integer")
            files.append({"path": path, "sha256": _sha256(item["sha256"], "file sha256"), "size_bytes": size})
        if files != sorted(files, key=lambda item: cast(str, item["path"])) or len(
            {item["path"] for item in files}
        ) != len(files):
            raise CrossCommitCompatibilityError("analysis implementation files must be sorted and unique")
        if canonical_sha256({"files": files}) != implementation["fingerprint"]:
            raise CrossCommitCompatibilityError("analysis implementation fingerprint does not match its file inventory")

        campaigns_raw = raw["campaigns"]
        if not isinstance(campaigns_raw, list) or len(campaigns_raw) < 2:
            raise CrossCommitCompatibilityError("cross-commit compatibility requires at least two campaigns")
        campaigns = []
        repositories = set()
        for index, item_raw in enumerate(campaigns_raw):
            item = _object(item_raw, f"campaigns[{index}]")
            _strict(
                item,
                f"campaigns[{index}]",
                (
                    "manifest_sha256",
                    "run_id",
                    "campaign_id",
                    "selection_sha256",
                    "verdict_sha256",
                    "repository",
                ),
            )
            repository = RepositoryState.from_dict(item["repository"]).to_dict()
            repositories.add(canonical_sha256(repository))
            campaigns.append(
                {
                    "manifest_sha256": _sha256(item["manifest_sha256"], "campaign manifest_sha256"),
                    "run_id": _nonempty(item["run_id"], "campaign run_id"),
                    "campaign_id": _nonempty(item["campaign_id"], "campaign campaign_id"),
                    "selection_sha256": _sha256(item["selection_sha256"], "campaign selection_sha256"),
                    "verdict_sha256": _sha256(item["verdict_sha256"], "campaign verdict_sha256"),
                    "repository": repository,
                }
            )
        campaigns.sort(key=lambda item: cast(str, item["manifest_sha256"]))
        if list(campaigns_raw) != campaigns or len({item["manifest_sha256"] for item in campaigns}) != len(campaigns):
            raise CrossCommitCompatibilityError("campaign bindings must be sorted by unique manifest SHA-256")
        if len(repositories) != 2:
            raise CrossCommitCompatibilityError("cross-commit v1 contract must bind exactly two repository identities")

        ground = _object(raw["ground_truth"], "ground_truth")
        _strict(
            ground,
            "ground_truth",
            (
                "manifest_sha256s",
                "regeneration_command",
                "baseline_config",
                "candidate_config",
                "relative_threshold_pct",
                "absolute_threshold_us",
                "publication_sha256",
                "raw_archive_verified",
            ),
        )
        ground_manifests = _string_array(ground["manifest_sha256s"], "ground_truth.manifest_sha256s")
        for digest in ground_manifests:
            _sha256(digest, "ground_truth.manifest_sha256s[]")
        campaign_manifests = {item["manifest_sha256"] for item in campaigns}
        if not ground_manifests or not set(ground_manifests) < campaign_manifests:
            raise CrossCommitCompatibilityError(
                "ground-truth manifests must be a non-empty proper subset of contract campaigns"
            )
        repository_by_manifest = {
            str(item["manifest_sha256"]): canonical_sha256(item["repository"]) for item in campaigns
        }
        ground_repositories = {repository_by_manifest[digest] for digest in ground_manifests}
        if len(ground_repositories) != 1:
            raise CrossCommitCompatibilityError("ground-truth campaigns must come from exactly one repository identity")
        ground_repository = next(iter(ground_repositories))
        all_source_manifests = {
            manifest_sha256
            for manifest_sha256, repository_sha256 in repository_by_manifest.items()
            if repository_sha256 == ground_repository
        }
        if set(ground_manifests) != all_source_manifests:
            raise CrossCommitCompatibilityError(
                "ground truth must include every contract campaign from the source repository"
            )
        publication = _object(ground["publication_sha256"], "ground_truth.publication_sha256")
        if set(publication) != _PUBLICATION_FILENAMES:
            raise CrossCommitCompatibilityError("ground-truth publication must bind the exact publication file set")
        publication_sha256 = {
            filename: _sha256(publication[filename], f"ground_truth.publication_sha256.{filename}")
            for filename in sorted(publication)
        }
        raw_archive_verified = ground["raw_archive_verified"]
        if not isinstance(raw_archive_verified, bool):
            raise CrossCommitCompatibilityError("ground_truth.raw_archive_verified must be boolean")

        differences = _object(raw["allowed_differences"], "allowed_differences")
        _strict(differences, "allowed_differences", ("analysis_policy_fields", "input_ids"))
        policy_fields = _string_array(
            differences["analysis_policy_fields"],
            "allowed_differences.analysis_policy_fields",
            safe_ids=True,
        )
        unsupported_policy = sorted(set(policy_fields) - _ALLOWED_POLICY_DIFFERENCES)
        if unsupported_policy:
            raise CrossCommitCompatibilityError(f"unsupported cross-commit policy differences: {unsupported_policy!r}")
        input_ids = _string_array(differences["input_ids"], "allowed_differences.input_ids", safe_ids=True)

        contract_sha256 = _sha256(raw["contract_sha256"], "contract_sha256")
        stable = {key: value for key, value in raw.items() if key != "contract_sha256"}
        if canonical_sha256(stable) != contract_sha256:
            raise CrossCommitCompatibilityError("cross-commit contract SHA-256 does not recompute")

        result = cls(
            raw=dict(raw),
            sha256=contract_sha256,
            campaign_bindings=tuple(campaigns),
            ground_truth_manifest_sha256s=ground_manifests,
            regeneration_command=_nonempty(ground["regeneration_command"], "ground_truth.regeneration_command"),
            baseline_config=_optional_nonempty(ground["baseline_config"], "ground_truth.baseline_config"),
            candidate_config=_optional_nonempty(ground["candidate_config"], "ground_truth.candidate_config"),
            relative_threshold_pct=_finite_non_negative(
                ground["relative_threshold_pct"], "ground_truth.relative_threshold_pct"
            ),
            absolute_threshold_us=_finite_non_negative(
                ground["absolute_threshold_us"], "ground_truth.absolute_threshold_us"
            ),
            publication_sha256=publication_sha256,
            raw_archive_verified=raw_archive_verified,
            allowed_policy_fields=policy_fields,
            allowed_input_ids=input_ids,
        )
        result.verify_current_implementation()
        return result

    def verify_current_implementation(self) -> None:
        declared = _object(self.raw["analysis_implementation"], "analysis_implementation")
        if analysis_implementation_record() != declared:
            raise CrossCommitCompatibilityError(
                "current analysis implementation does not match the reviewed compatibility contract"
            )

    def provenance_summary(self) -> Dict[str, Any]:
        return {
            "schema": CROSS_COMMIT_COMPATIBILITY_SCHEMA,
            "status": "reviewed-byte-identical-ground-truth",
            "contract_sha256": self.sha256,
            "ground_truth_manifest_sha256s": list(self.ground_truth_manifest_sha256s),
            "ground_truth_publication_sha256": dict(self.publication_sha256),
            "allowed_policy_fields": list(self.allowed_policy_fields),
            "allowed_input_ids": list(self.allowed_input_ids),
        }


def load_cross_commit_compatibility(path_value: PathLike) -> CrossCommitCompatibility:
    path = Path(path_value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise CrossCommitCompatibilityError("cross-commit compatibility contract must be a real regular file")
    try:
        raw_bytes = read_bounded_bytes(
            path,
            max_bytes=_CONTRACT_LIMITS.max_document_bytes,
            field="cross-commit compatibility contract",
        )
        raw = strict_json_loads(raw_bytes, limits=_CONTRACT_LIMITS)
    except (OSError, UnicodeError, ContractError) as exc:
        raise CrossCommitCompatibilityError(f"cannot decode cross-commit compatibility contract: {exc}") from exc
    contract = CrossCommitCompatibility.from_mapping(raw)
    if raw_bytes != canonical_json_bytes(raw):
        raise CrossCommitCompatibilityError("cross-commit compatibility file must use canonical JSON bytes")
    return contract


__all__ = [
    "CROSS_COMMIT_COMPATIBILITY_SCHEMA",
    "CrossCommitCompatibility",
    "CrossCommitCompatibilityError",
    "analysis_implementation_record",
    "load_cross_commit_compatibility",
]
