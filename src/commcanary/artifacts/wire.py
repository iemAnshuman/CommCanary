"""Shared artifact wire types, bounded parsing, and validation primitives.

This module is the dependency root for artifact contracts.  It deliberately has
no dependency on compiler, replay, comparison, capture, or CLI services.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from ..errors import SchemaError
from ..resources import (
    DEFAULT_RESOURCE_LIMITS,
    JsonResourceError,
    ResourceLimits,
    load_bounded_json,
    validate_json_mapping,
)
from .io import SENSITIVE_JSON_POLICY, atomic_write_json
from .json_codec import canonical_json_bytes

MAX_RANK_COUNT = 65536
MAX_ABS_INTEGER = 2**63 - 1
MAX_TIME_US = 1_000_000_000_000.0
SUPPORTED_OPS = {
    "all_reduce",
    "reduce_scatter",
    "all_gather",
    "all_to_all",
    "broadcast",
    "send",
    "recv",
    "point_to_point",
}
PROTOCOL_FINGERPRINT_EXCLUDE = {"sha256", "max_replay_events"}

JsonDict = Dict[str, Any]


def load_json_document(
    path: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Any:
    """Load one bounded JSON document without imposing a root shape."""

    try:
        return load_bounded_json(path, limits=limits)
    except FileNotFoundError as exc:
        raise SchemaError(f"{path} does not exist") from exc
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{path} is not UTF-8 JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc.msg}") from exc
    except JsonResourceError as exc:
        raise SchemaError(f"{path} violates JSON resource constraints: {exc}") from exc
    except RecursionError as exc:
        raise SchemaError(f"{path} exceeds the JSON parser nesting capacity") from exc
    except OSError as exc:
        raise SchemaError(f"cannot read {path}: {exc}") from exc
    except OverflowError as exc:
        raise SchemaError(f"{path} contains a number that is too large") from exc
    except ValueError as exc:
        raise SchemaError(f"{path} contains non-standard JSON: {exc}") from exc


def load_json(
    path: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    data = load_json_document(path, limits=limits)
    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a JSON object")
    return data


def write_json(path: str, data: Mapping[str, Any]) -> None:
    """Write deterministic JSON using the legacy private-artifact policy."""

    atomic_write_json(
        path,
        data,
        indent=2,
        policy=SENSITIVE_JSON_POLICY,
    )


def replay_protocol_sha256(
    protocol: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> str:
    try:
        validate_json_mapping(protocol, limits=limits)
    except JsonResourceError as exc:
        raise SchemaError(f"replay protocol violates JSON resource constraints: {exc}") from exc
    stable = {key: value for key, value in protocol.items() if key not in PROTOCOL_FINGERPRINT_EXCLUDE}
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def require_format(data: Mapping[str, Any], expected: str, label: str) -> None:
    if not isinstance(data, Mapping):
        raise SchemaError(f"{label} must be a JSON object")
    actual = data.get("format")
    if actual != expected:
        raise SchemaError(f"{label} format must be {expected!r}, got {actual!r}")


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise SchemaError(f"expected finite numeric value, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"expected numeric value, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise SchemaError(f"expected finite numeric value, got {value!r}")
    return parsed


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise SchemaError(f"expected integer value, got {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise SchemaError(f"expected integer value, got {value!r}")
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("-"):
            digits = stripped[1:]
        else:
            digits = stripped
        if digits and digits.isascii() and digits.isdigit():
            if len(digits) > 19:
                raise SchemaError(f"integer value is too large: {value!r}")
            try:
                parsed = int(stripped)
            except ValueError as exc:
                raise SchemaError(f"expected integer value, got {value!r}") from exc
        else:
            raise SchemaError(f"expected integer value, got {value!r}")
    else:
        raise SchemaError(f"expected integer value, got {value!r}")
    if abs(parsed) > MAX_ABS_INTEGER:
        raise SchemaError(f"integer value is too large: {value!r}")
    return parsed


def normalize_ranks(value: Any) -> List[int]:
    if value is None:
        raise SchemaError("event is missing required 'ranks'")
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise SchemaError("rank count must be positive")
        if value > MAX_RANK_COUNT:
            raise SchemaError(f"rank count must not exceed {MAX_RANK_COUNT}")
        return list(range(value))
    if not isinstance(value, list):
        raise SchemaError("'ranks' must be a rank list or rank count")
    ranks = [as_int(rank) for rank in value]
    if not ranks:
        raise SchemaError("'ranks' must not be empty")
    if any(rank < 0 for rank in ranks):
        raise SchemaError("'ranks' must contain only non-negative integers")
    if len(set(ranks)) != len(ranks):
        raise SchemaError("'ranks' must not contain duplicates")
    if len(ranks) > MAX_RANK_COUNT:
        raise SchemaError(f"'ranks' must not contain more than {MAX_RANK_COUNT} entries")
    return ranks


def normalize_arrival_offsets(event: Mapping[str, Any], ranks: List[int]) -> List[float]:
    raw = event.get("rank_arrival_us")
    if raw is None:
        skew = as_float(event.get("arrival_skew_us"), 0.0)
        if skew < 0.0:
            raise SchemaError("arrival_skew_us must be non-negative")
        if len(ranks) == 1:
            if skew > 0.001:
                raise SchemaError("a one-rank collective cannot have positive arrival skew")
            return [0.0]
        return [0.0 for _ in ranks[:-1]] + [max(0.0, skew)]

    if isinstance(raw, Mapping):
        validate_arrival_keys(raw, ranks, "rank_arrival_us", allow_subset=False)
        values = []
        for rank in ranks:
            if str(rank) in raw:
                value = as_float(raw[str(rank)])
            elif rank in raw:
                value = as_float(raw[rank])
            else:
                raise SchemaError(f"rank_arrival_us is missing rank {rank}")
            if value < 0.0:
                raise SchemaError("rank_arrival_us values must be non-negative")
            values.append(value)
    elif isinstance(raw, list):
        values = [as_float(item) for item in raw]
        if len(values) != len(ranks):
            raise SchemaError("rank_arrival_us list length must match ranks")
        if any(value < 0.0 for value in values):
            raise SchemaError("rank_arrival_us values must be non-negative")
    else:
        raise SchemaError("rank_arrival_us must be an object or list")

    minimum = min(values) if values else 0.0
    return [max(0.0, value - minimum) for value in values]


def arrival_skew_us(offsets: Iterable[float]) -> float:
    values = list(offsets)
    if not values:
        return 0.0
    return max(values) - min(values)


def average_wait_us(offsets: Iterable[float]) -> float:
    values = list(offsets)
    if not values:
        return 0.0
    latest = max(values)
    return sum(latest - value for value in values) / len(values)


def merge_metadata(base: Optional[Mapping[str, Any]], override: Optional[Mapping[str, Any]]) -> JsonDict:
    merged: JsonDict = dict(base or {})
    merged.update(dict(override or {}))
    return merged


def clean_private_keys(data: MutableMapping[str, Any]) -> JsonDict:
    return {key: value for key, value in data.items() if not key.startswith("_")}


def require_optional_mapping(data: Mapping[str, Any], key: str, label: str) -> None:
    if key in data and not isinstance(data.get(key), Mapping):
        raise SchemaError(f"{label} {key} must be an object")


def validate_point_to_point_metadata(event: Mapping[str, Any], ranks: List[int], label: str) -> None:
    op = str(event.get("op", ""))
    sender_present = "sender_rank" in event
    receiver_present = "receiver_rank" in event
    if op == "point_to_point":
        if not sender_present or not receiver_present:
            raise SchemaError(f"{label} point_to_point requires sender_rank and receiver_rank")
    if sender_present:
        sender = as_int(event.get("sender_rank"))
        if sender not in ranks:
            raise SchemaError(f"{label} sender_rank must be one of ranks")
    if receiver_present:
        receiver = as_int(event.get("receiver_rank"))
        if receiver not in ranks:
            raise SchemaError(f"{label} receiver_rank must be one of ranks")
    if sender_present and receiver_present and as_int(event.get("sender_rank")) == as_int(event.get("receiver_rank")):
        raise SchemaError(f"{label} sender_rank and receiver_rank must differ")
    if "message_sequence" in event and as_int(event.get("message_sequence")) < 0:
        raise SchemaError(f"{label} message_sequence must be non-negative")
    for key in ("tag", "channel"):
        if key in event and (not isinstance(event.get(key), str) or not event.get(key)):
            raise SchemaError(f"{label} {key} must be a non-empty string")


def validate_op(value: Any, label: str, *, custom: bool) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} op must be a non-empty string")
    if value not in SUPPORTED_OPS and not custom:
        raise SchemaError(f"{label} op {value!r} is unsupported; set custom_op=true for custom operations")


def validate_nonempty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} must be a non-empty string")


def validate_arrival_keys(raw: Any, ranks: List[int], label: str, *, allow_subset: bool) -> None:
    if not isinstance(raw, Mapping):
        raise SchemaError(f"{label} must be an object")
    expected = {str(rank) for rank in ranks}
    actual = {str(key) for key in raw.keys()}
    if allow_subset:
        if not actual:
            raise SchemaError(f"{label} must include at least one rank")
        unexpected = actual - expected
        if unexpected:
            raise SchemaError(f"{label} contains ranks outside ranks")
        return
    if actual != expected:
        raise SchemaError(f"{label} keys must exactly match ranks")


def validate_skew_matches_offsets(skew_us: float, offsets: Iterable[float], label: str) -> None:
    computed = arrival_skew_us(offsets)
    if abs(skew_us - computed) > 0.001:
        raise SchemaError(f"{label} arrival_skew_us must match arrival_offsets_us")


def validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SchemaError(f"{label} must be a 64-character lowercase SHA-256 hex digest")


__all__ = [
    "JsonDict",
    "MAX_ABS_INTEGER",
    "MAX_RANK_COUNT",
    "MAX_TIME_US",
    "PROTOCOL_FINGERPRINT_EXCLUDE",
    "SUPPORTED_OPS",
    "arrival_skew_us",
    "as_float",
    "as_int",
    "average_wait_us",
    "clean_private_keys",
    "load_json",
    "load_json_document",
    "merge_metadata",
    "normalize_arrival_offsets",
    "normalize_ranks",
    "replay_protocol_sha256",
    "require_format",
    "require_optional_mapping",
    "validate_arrival_keys",
    "validate_nonempty_string",
    "validate_op",
    "validate_point_to_point_metadata",
    "validate_sha256",
    "validate_skew_matches_offsets",
    "write_json",
]
