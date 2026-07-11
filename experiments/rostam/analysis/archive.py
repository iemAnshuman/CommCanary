"""Post-run raw archive descriptors bound to exact selected evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

from ..harness import ContractError, JSONResourceLimits, file_sha256, read_bounded_bytes, sha256_hex, strict_json_loads
from .schemas import RAW_ARCHIVE_DESCRIPTOR_SCHEMA

PathLike = Union[str, "Path"]

ARCHIVE_DESCRIPTOR_SCHEMA = RAW_ARCHIVE_DESCRIPTOR_SCHEMA
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_URI_RE = re.compile(r"^(?:https|s3|gs|doi|urn|ipfs):[^\s\x00]+$")
_DESCRIPTOR_LIMITS = JSONResourceLimits(
    max_document_bytes=1024 * 1024,
    max_depth=8,
    max_items=10_000,
    max_string_bytes=4096,
    max_numeric_characters=128,
)


class ArchiveVerificationError(ContractError):
    """Raised when raw bytes are not bound to exact post-run evidence."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArchiveVerificationError(f"{field} must be an object")
    return value


def _strict(value: Mapping[str, Any], field: str, expected: Sequence[str]) -> None:
    expected_fields = set(expected)
    missing = sorted(expected_fields - set(value))
    unknown = sorted(set(value) - expected_fields)
    if missing or unknown:
        raise ArchiveVerificationError(f"{field} fields mismatch: missing={missing!r}, unknown={unknown!r}")


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArchiveVerificationError(f"{field} must be a lowercase SHA-256")
    return value


def _size(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise ArchiveVerificationError(f"{field} must be a non-negative integer")
    return cast(int, value)


def verify_archive_descriptor(
    expected_campaigns: Sequence[Mapping[str, str]],
    descriptor_path: Optional[PathLike],
    archive_path: Optional[PathLike],
) -> Dict[str, Any]:
    """Verify post-run evidence identities and archive bytes without publishing local paths."""

    if (descriptor_path is None) != (archive_path is None):
        raise ArchiveVerificationError("archive_descriptor and raw_archive must be supplied together")
    if descriptor_path is None or archive_path is None:
        return {"verified": False, "descriptor": None}
    if not expected_campaigns:
        raise ArchiveVerificationError("archive verification requires at least one trusted evidence binding")
    descriptor = Path(descriptor_path).expanduser()
    if descriptor.is_symlink() or not descriptor.is_file():
        raise ArchiveVerificationError("archive descriptor must be a real regular file")
    try:
        descriptor_bytes = read_bounded_bytes(
            descriptor,
            max_bytes=_DESCRIPTOR_LIMITS.max_document_bytes,
            field="archive descriptor",
        )
        raw = _object(strict_json_loads(descriptor_bytes, limits=_DESCRIPTOR_LIMITS), "archive descriptor")
    except (OSError, UnicodeError, ContractError) as exc:
        raise ArchiveVerificationError(f"cannot decode archive descriptor: {exc}") from exc
    descriptor_sha256 = sha256_hex(descriptor_bytes)
    descriptor_size = len(descriptor_bytes)
    _strict(raw, "archive descriptor", ("schema", "uri", "sha256", "size_bytes", "campaigns"))
    if raw["schema"] != ARCHIVE_DESCRIPTOR_SCHEMA:
        raise ArchiveVerificationError(f"unsupported archive descriptor schema {raw['schema']!r}")
    uri = raw["uri"]
    if not isinstance(uri, str) or len(uri) > 4096 or _URI_RE.fullmatch(uri) is None:
        raise ArchiveVerificationError("archive descriptor uri must be an immutable non-local URI label")
    archive_sha256 = _sha256(raw["sha256"], "archive descriptor.sha256")
    archive_size = _size(raw["size_bytes"], "archive descriptor.size_bytes")
    campaigns_raw = raw["campaigns"]
    if not isinstance(campaigns_raw, list) or not campaigns_raw:
        raise ArchiveVerificationError("archive descriptor.campaigns must be a non-empty array")
    declared_campaigns = []
    for index, entry_raw in enumerate(campaigns_raw):
        entry = _object(entry_raw, f"archive descriptor.campaigns[{index}]")
        _strict(
            entry,
            f"archive descriptor.campaigns[{index}]",
            (
                "run_id",
                "campaign_id",
                "repository_commit",
                "manifest_sha256",
                "selection_id",
                "selection_sha256",
                "verdict_sha256",
            ),
        )
        if not all(isinstance(entry[field], str) and entry[field] for field in entry):
            raise ArchiveVerificationError(f"archive descriptor.campaigns[{index}] contains an invalid identity")
        declared_campaigns.append(dict(entry))
    expected = sorted(
        (dict(item) for item in expected_campaigns), key=lambda item: (item["run_id"], item["campaign_id"])
    )
    if declared_campaigns != expected:
        raise ArchiveVerificationError("archive descriptor campaign identities do not match the trusted join")
    archive = Path(archive_path).expanduser()
    if archive.is_symlink() or not archive.is_file():
        raise ArchiveVerificationError("raw archive must be a real regular file")
    if archive.stat().st_size != archive_size or file_sha256(archive) != archive_sha256:
        raise ArchiveVerificationError("raw archive bytes mismatch the post-run evidence-bound descriptor")
    return {
        "verified": True,
        "descriptor": {
            "schema": ARCHIVE_DESCRIPTOR_SCHEMA,
            "uri": uri,
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            "campaigns": expected,
            "descriptor_sha256": descriptor_sha256,
            "descriptor_size_bytes": descriptor_size,
        },
    }
