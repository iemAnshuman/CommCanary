"""Deterministic JSON, hashing, identifiers, and path containment.

The experiment manifest is a commitment.  These helpers deliberately accept a
smaller value domain than Python's permissive :mod:`json` module so that two
callers cannot silently disagree about what was hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

PathLike = Union[str, "Path"]

_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
DEFAULT_SLUG_MAX_LENGTH = 64


@dataclass(frozen=True)
class JSONResourceLimits:
    """Resource envelope shared by experiment JSON readers and writers.

    ``max_items`` counts object members and array elements. ``max_depth``
    counts nested containers below the root. Numeric tokens are capped before
    Python converts them, avoiding arbitrarily large integer allocations.
    """

    max_document_bytes: int = 16 * 1024 * 1024
    max_depth: int = 64
    max_items: int = 1_000_000
    max_string_bytes: int = 1024 * 1024
    max_numeric_characters: int = 128

    def validate(self) -> None:
        for field, value in (
            ("max_document_bytes", self.max_document_bytes),
            ("max_depth", self.max_depth),
            ("max_items", self.max_items),
            ("max_string_bytes", self.max_string_bytes),
            ("max_numeric_characters", self.max_numeric_characters),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"JSON resource limit {field} must be a positive integer")


DEFAULT_JSON_LIMITS = JSONResourceLimits()
CHECKSUM_MAX_BYTES = 256


class ContractError(ValueError):
    """Base error for an invalid experiment contract."""


class CanonicalJSONError(ContractError):
    """Raised when a value cannot be represented by the canonical codec."""


class UnsafeSlugError(ContractError):
    """Raised when an identifier is not a safe, bounded path segment."""


class PathContainmentError(ContractError):
    """Raised when a requested path can escape its declared root."""


def _string(value: str, path: str, limits: JSONResourceLimits) -> str:
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise CanonicalJSONError(f"{path}: string is not valid UTF-8") from exc
    if size > limits.max_string_bytes:
        raise CanonicalJSONError(f"{path}: string exceeds max_string_bytes={limits.max_string_bytes}")
    return value


def _number(value: Union[int, float], path: str, limits: JSONResourceLimits) -> Union[int, float]:
    if isinstance(value, float) and not math.isfinite(value):
        raise CanonicalJSONError(f"{path}: non-finite floats are not JSON commitments")
    try:
        token = str(value)
    except ValueError as exc:
        raise CanonicalJSONError(
            f"{path}: number exceeds max_numeric_characters={limits.max_numeric_characters}"
        ) from exc
    if len(token) > limits.max_numeric_characters:
        raise CanonicalJSONError(f"{path}: number exceeds max_numeric_characters={limits.max_numeric_characters}")
    return value


def _normalize_json(
    value: Any,
    path: str = "$",
    *,
    limits: JSONResourceLimits = DEFAULT_JSON_LIMITS,
    depth: int = 0,
    item_count: list[int] | None = None,
) -> Any:
    """Return a JSON-native tree while enforcing the canonical value domain."""

    if item_count is None:
        limits.validate()
        item_count = [0]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _string(value, path, limits)
    if isinstance(value, int) and not isinstance(value, bool):
        return _number(value, path, limits)
    if isinstance(value, float):
        return _number(value, path, limits)
    if isinstance(value, Mapping):
        if depth >= limits.max_depth:
            raise CanonicalJSONError(f"{path}: JSON exceeds max_depth={limits.max_depth}")
        item_count[0] += len(value)
        if item_count[0] > limits.max_items:
            raise CanonicalJSONError(f"{path}: JSON exceeds max_items={limits.max_items}")
        normalized: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                raise CanonicalJSONError(f"{path}: object keys must be strings")
            normalized[_string(key, f"{path}.<key:{index}>", limits)] = _normalize_json(
                item,
                f"{path}.<value:{index}>",
                limits=limits,
                depth=depth + 1,
                item_count=item_count,
            )
        return normalized
    if isinstance(value, (list, tuple)):
        if depth >= limits.max_depth:
            raise CanonicalJSONError(f"{path}: JSON exceeds max_depth={limits.max_depth}")
        item_count[0] += len(value)
        if item_count[0] > limits.max_items:
            raise CanonicalJSONError(f"{path}: JSON exceeds max_items={limits.max_items}")
        return [
            _normalize_json(
                item,
                f"{path}[{index}]",
                limits=limits,
                depth=depth + 1,
                item_count=item_count,
            )
            for index, item in enumerate(value)
        ]
    raise CanonicalJSONError(f"{path}: unsupported JSON value type {type(value).__name__}")


def canonical_json_bytes(
    value: Any,
    *,
    limits: JSONResourceLimits = DEFAULT_JSON_LIMITS,
) -> bytes:
    """Encode *value* as the unique UTF-8 JSON byte representation we hash.

    The representation has sorted object keys, no insignificant whitespace,
    literal Unicode, and no trailing newline. Arrays retain their order.
    """

    normalized = _normalize_json(value, limits=limits)
    try:
        text = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        encoded = text.encode("utf-8")
        if len(encoded) > limits.max_document_bytes:
            raise CanonicalJSONError(f"canonical JSON exceeds max_document_bytes={limits.max_document_bytes}")
        return encoded
    except CanonicalJSONError:
        raise
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as exc:  # defensive
        raise CanonicalJSONError(str(exc)) from exc


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise CanonicalJSONError(f"non-standard JSON constant {value!r}")


def _preflight_structure(data: bytes, limits: JSONResourceLimits) -> None:
    """Reject excessive nesting/items before the decoder allocates them."""

    containers: list[bool] = []
    items = 0
    in_string = False
    escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            if containers:
                containers[-1] = True
            in_string = True
        elif byte in {0x5B, 0x7B}:  # [ or {
            if containers:
                containers[-1] = True
            containers.append(False)
            if len(containers) > limits.max_depth:
                raise CanonicalJSONError(f"JSON exceeds max_depth={limits.max_depth}")
        elif byte in {0x5D, 0x7D} and containers:  # ] or }
            if containers.pop():
                items += 1
        elif byte == 0x2C and containers:  # comma
            items += 1
        elif byte not in {0x09, 0x0A, 0x0D, 0x20} and containers:
            containers[-1] = True
        if items > limits.max_items:
            raise CanonicalJSONError(f"JSON exceeds max_items={limits.max_items}")


def strict_json_loads(
    data: Union[str, bytes, bytearray],
    *,
    limits: JSONResourceLimits = DEFAULT_JSON_LIMITS,
) -> Any:
    """Parse strict, resource-bounded JSON with deterministic failures."""

    limits.validate()
    try:
        encoded = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    except UnicodeEncodeError as exc:
        raise CanonicalJSONError("invalid JSON: input is not valid UTF-8") from exc
    if len(encoded) > limits.max_document_bytes:
        raise CanonicalJSONError(f"JSON exceeds max_document_bytes={limits.max_document_bytes}")
    _preflight_structure(encoded, limits)

    def bounded_integer(token: str) -> int:
        if len(token) > limits.max_numeric_characters:
            raise CanonicalJSONError(f"number exceeds max_numeric_characters={limits.max_numeric_characters}")
        return int(token)

    def bounded_float(token: str) -> float:
        if len(token) > limits.max_numeric_characters:
            raise CanonicalJSONError(f"number exceeds max_numeric_characters={limits.max_numeric_characters}")
        return float(token)

    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonstandard_constant,
            parse_float=bounded_float,
            parse_int=bounded_integer,
        )
    except CanonicalJSONError:
        raise
    except (OverflowError, RecursionError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CanonicalJSONError(f"invalid JSON: {exc}") from exc
    return _normalize_json(value, limits=limits)


def read_bounded_bytes(
    path: PathLike,
    *,
    max_bytes: int = DEFAULT_JSON_LIMITS.max_document_bytes,
    field: str = "file",
) -> bytes:
    """Read at most ``max_bytes + 1`` and reject oversized or unsafe files."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise CanonicalJSONError(f"{field} must be a real regular file")
    try:
        with source.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise CanonicalJSONError(f"cannot read {field}: {exc}") from exc
    if len(data) > max_bytes:
        raise CanonicalJSONError(f"{field} exceeds max_bytes={max_bytes}")
    return data


def read_bounded_text(
    path: PathLike,
    *,
    max_bytes: int,
    field: str,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> str:
    """Decode a bounded regular file without a later wholesale read."""

    data = read_bounded_bytes(path, max_bytes=max_bytes, field=field)
    try:
        return data.decode(encoding, errors=errors)
    except UnicodeError as exc:
        raise CanonicalJSONError(f"{field} is not valid {encoding}") from exc


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON representation of *value*."""

    return sha256_hex(canonical_json_bytes(value))


def file_sha256(path: PathLike) -> str:
    """Stream a file into SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_slug(
    value: str,
    *,
    field: str = "identifier",
    max_length: int = DEFAULT_SLUG_MAX_LENGTH,
) -> str:
    """Validate and return a lowercase ASCII identifier safe as one path part."""

    if not isinstance(value, str):
        raise UnsafeSlugError(f"{field} must be a string")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not value or len(value) > max_length:
        raise UnsafeSlugError(f"{field} must contain 1..{max_length} characters")
    if value in {".", ".."} or ".." in value or not _SLUG_RE.fullmatch(value):
        raise UnsafeSlugError(
            f"{field} must be lowercase ASCII letters, digits, dots, or hyphens "
            "and may not contain '..' or punctuation at either end"
        )
    return value


def contained_path(root: PathLike, *parts: str) -> Path:
    """Resolve a descendant path and prove that it remains below *root*.

    Every supplied part is a single relative path segment. Existing symlinks
    are resolved, so a symlink below the root cannot be used as an escape.
    """

    root_path = Path(root).expanduser().resolve()
    candidate = root_path
    for part in parts:
        if not isinstance(part, str) or not part:
            raise PathContainmentError("path parts must be non-empty strings")
        parsed = Path(part)
        if parsed.is_absolute() or len(parsed.parts) != 1 or part in {".", ".."} or "/" in part or "\\" in part:
            raise PathContainmentError(f"unsafe path segment {part!r}")
        candidate = candidate / part
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise PathContainmentError(f"path {resolved} escapes root {root_path}") from exc
    return resolved
