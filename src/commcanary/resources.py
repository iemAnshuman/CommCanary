"""Resource policies for reading untrusted JSON documents.

The standard-library JSON decoder has no input-size or nesting controls and
silently accepts duplicate object keys.  This module puts a small, reusable
boundary in front of it.  It deliberately does not import :mod:`schema`, so
schema-facing adapters can translate these low-level failures into the public
``SchemaError`` type without creating an import cycle.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Dict, List, Set, Tuple

MAX_CHECKED_COUNT = (1 << 63) - 1


@dataclass(frozen=True)
class ResourceLimits:
    """Hard limits applied while loading and expanding one artifact.

    ``max_json_items`` counts object members and array elements.  String limits
    apply independently to every object key and string value after UTF-8
    encoding.  Numeric-token length is bounded before converting an integer or
    float.  JSON depth counts nested objects and arrays, with the root container
    at depth one.  Stored limits cover the compact wire representation; expanded
    limits cover logical work before any motif, timing weight, replay, or PARAM
    output is materialized.
    """

    max_input_bytes: int = 64 * 1024 * 1024
    max_json_depth: int = 64
    max_json_items: int = 2_000_000
    max_json_string_bytes: int = 1024 * 1024
    max_json_number_chars: int = 1024
    max_stored_events: int = 1_000_000
    max_stored_timing_records: int = 2_000_000
    max_ranks: int = 65_536
    max_expanded_events: int = 1_000_000
    max_expanded_timing_records: int = 2_000_000
    max_replay_events: int = 1_000_000
    max_param_entries: int = 2_000_000
    max_capture_shards: int = 65_536
    max_capture_events: int = 1_000_000
    max_behavior_configurations: int = 32
    max_behavior_candidates: int = 4096
    max_behavior_ranking_comparisons: int = 10_000_000
    max_retained_ledger_rows: int = 10_000
    max_reduction_oracle_calls: int = 10_000

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field.name} must be an integer")
        if self.max_input_bytes < 1:
            raise ValueError("max_input_bytes must be positive")
        if self.max_json_depth < 1:
            raise ValueError("max_json_depth must be positive")
        if self.max_json_items < 0:
            raise ValueError("max_json_items must be non-negative")
        if self.max_json_string_bytes < 0:
            raise ValueError("max_json_string_bytes must be non-negative")
        if self.max_json_number_chars < 1:
            raise ValueError("max_json_number_chars must be positive")
        for name in (
            "max_stored_events",
            "max_stored_timing_records",
            "max_ranks",
            "max_expanded_events",
            "max_expanded_timing_records",
            "max_replay_events",
            "max_param_entries",
            "max_capture_shards",
            "max_capture_events",
            "max_behavior_configurations",
            "max_behavior_candidates",
            "max_behavior_ranking_comparisons",
            "max_retained_ledger_rows",
            "max_reduction_oracle_calls",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_behavior_configurations < 2:
            raise ValueError("max_behavior_configurations must be at least 2")


DEFAULT_RESOURCE_LIMITS = ResourceLimits()


class JsonResourceError(ValueError):
    """Raised when JSON violates a resource or representation boundary."""


def checked_add(left: int, right: int, *, label: str) -> int:
    """Add two non-negative work counts without allowing unbounded integers."""

    _require_count(left, label)
    _require_count(right, label)
    if left > MAX_CHECKED_COUNT - right:
        raise JsonResourceError(f"{label} exceeds the supported count range")
    return left + right


def checked_multiply(left: int, right: int, *, label: str) -> int:
    """Multiply two non-negative work counts with a fixed overflow boundary."""

    _require_count(left, label)
    _require_count(right, label)
    if left and right > MAX_CHECKED_COUNT // left:
        raise JsonResourceError(f"{label} exceeds the supported count range")
    return left * right


def require_within(value: int, limit: int, *, label: str) -> int:
    """Return *value* when it fits *limit*, otherwise fail before expansion."""

    _require_count(value, label)
    _require_count(limit, f"{label} limit")
    if value > limit:
        raise JsonResourceError(f"{label}={value} exceeds limit={limit}")
    return value


def _require_count(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise JsonResourceError(f"{label} must be a non-negative integer count")


def load_bounded_json(
    path: str,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> Any:
    """Read and decode one bounded UTF-8 JSON document.

    The byte cap and structural nesting scan happen before ``json.loads``.  The
    iterative post-walk then enforces limits that require decoded values and
    verifies that the resulting tree contains only JSON-native values.
    """

    with open(path, "rb") as handle:
        raw = handle.read(limits.max_input_bytes + 1)
    if len(raw) > limits.max_input_bytes:
        raise JsonResourceError(f"input exceeds max_input_bytes={limits.max_input_bytes}")

    text = raw.decode("utf-8")
    preflight_json_depth(text, max_depth=limits.max_json_depth)
    data = json.loads(
        text,
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_nonstandard_constant,
        parse_float=partial(
            _finite_json_float,
            max_chars=limits.max_json_number_chars,
        ),
        parse_int=partial(
            _bounded_json_int,
            max_chars=limits.max_json_number_chars,
        ),
    )
    validate_json_value(data, limits=limits)
    return data


def preflight_json_depth(text: str, *, max_depth: int) -> None:
    """Reject excessive object/array nesting without invoking the JSON parser."""

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > max_depth:
                raise JsonResourceError(f"JSON nesting exceeds max_json_depth={max_depth}")
        elif character in "]}" and depth:
            depth -= 1


def validate_json_value(
    data: Any,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    """Iteratively validate a decoded JSON tree against ``limits``."""

    _validate_json_tree(data, limits=limits, allow_mappings=False)


def validate_json_mapping(
    data: MappingABC[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    """Validate an in-memory JSON object without first copying it.

    Decoded files use concrete ``dict`` objects and should continue to call
    :func:`validate_json_value`. Public Python APIs historically accept the
    broader ``Mapping`` protocol, so this variant accepts mappings throughout
    the tree while retaining the same depth, item, string, numeric, and cycle
    checks. The preflight happens before callers normalize or deep-copy input.
    """

    if not isinstance(data, MappingABC):
        raise JsonResourceError("JSON document root must be an object")
    _validate_json_tree(data, limits=limits, allow_mappings=True)


def _validate_json_tree(
    data: Any,
    *,
    limits: ResourceLimits,
    allow_mappings: bool,
) -> None:
    item_count = 0
    active_containers: Set[int] = set()
    stack: List[Tuple[Any, int, bool]] = [(data, 1, False)]
    while stack:
        value, container_depth, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(value))
            continue

        is_mapping = isinstance(value, dict) or (allow_mappings and isinstance(value, MappingABC))
        if is_mapping:
            container_id = id(value)
            if container_id in active_containers:
                raise JsonResourceError("JSON values must not contain circular references")
            if container_depth > limits.max_json_depth:
                raise JsonResourceError(f"JSON nesting exceeds max_json_depth={limits.max_json_depth}")
            item_count = checked_add(item_count, len(value), label="JSON item count")
            _check_item_count(item_count, limits)
            active_containers.add(container_id)
            stack.append((value, container_depth, True))
            child_depth = container_depth + 1
            for key, child in value.items():
                if not isinstance(key, str):
                    raise JsonResourceError("JSON object keys must be strings")
                _check_string(key, limits)
                stack.append((child, child_depth, False))
        elif isinstance(value, list):
            container_id = id(value)
            if container_id in active_containers:
                raise JsonResourceError("JSON values must not contain circular references")
            if container_depth > limits.max_json_depth:
                raise JsonResourceError(f"JSON nesting exceeds max_json_depth={limits.max_json_depth}")
            item_count = checked_add(item_count, len(value), label="JSON item count")
            _check_item_count(item_count, limits)
            active_containers.add(container_id)
            stack.append((value, container_depth, True))
            child_depth = container_depth + 1
            stack.extend((child, child_depth, False) for child in value)
        elif isinstance(value, str):
            _check_string(value, limits)
        elif value is None or isinstance(value, (bool, int)):
            continue
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise JsonResourceError("JSON numbers must be finite")
        else:
            raise JsonResourceError(
                f"JSON values must use native object, array, string, number, boolean, or null types; "
                f"got {type(value).__name__}"
            )


def _object_without_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JsonResourceError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise JsonResourceError(f"non-standard JSON constant {value!r} is not allowed")


def _finite_json_float(value: str, *, max_chars: int) -> float:
    _check_number_chars(value, max_chars=max_chars)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise JsonResourceError(f"JSON number {value!r} is outside the finite float range")
    return parsed


def _bounded_json_int(value: str, *, max_chars: int) -> int:
    _check_number_chars(value, max_chars=max_chars)
    return int(value)


def _check_number_chars(value: str, *, max_chars: int) -> None:
    if len(value) > max_chars:
        raise JsonResourceError(f"JSON numeric token exceeds max_json_number_chars={max_chars}")


def _check_item_count(item_count: int, limits: ResourceLimits) -> None:
    if item_count > limits.max_json_items:
        raise JsonResourceError(f"JSON item count exceeds max_json_items={limits.max_json_items}")


def _check_string(value: str, limits: ResourceLimits) -> None:
    try:
        byte_count = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise JsonResourceError("JSON strings must contain valid Unicode scalar values") from exc
    if byte_count > limits.max_json_string_bytes:
        raise JsonResourceError(f"JSON string exceeds max_json_string_bytes={limits.max_json_string_bytes}")
