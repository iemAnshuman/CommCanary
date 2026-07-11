"""Single JSON encoding and error-mapping implementation for artifacts."""

from __future__ import annotations

import json
from typing import Any, Optional

from ..errors import SchemaError


def _encode_json(
    data: Any,
    *,
    context: str,
    indent: Optional[int],
    canonical: bool,
    trailing_newline: bool,
) -> bytes:
    try:
        if canonical:
            rendered = json.dumps(
                data,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        else:
            rendered = json.dumps(
                data,
                allow_nan=False,
                ensure_ascii=True,
                indent=indent,
                sort_keys=True,
            )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"{context}: {exc}") from exc
    if trailing_newline:
        rendered += "\n"
    return rendered.encode("utf-8")


def canonical_json_bytes(data: Any) -> bytes:
    """Encode the canonical JSON v1 byte representation."""

    return _encode_json(
        data,
        context="cannot canonicalize JSON",
        indent=None,
        canonical=True,
        trailing_newline=False,
    )


def formatted_json_bytes(data: Any, *, indent: int, trailing_newline: bool = True) -> bytes:
    """Encode deterministic, human-readable JSON for an artifact writer."""

    if not isinstance(indent, int) or isinstance(indent, bool) or indent < 0:
        raise ValueError("indent must be a non-negative integer")
    return _encode_json(
        data,
        context="cannot encode JSON",
        indent=indent,
        canonical=False,
        trailing_newline=trailing_newline,
    )


__all__ = ["canonical_json_bytes", "formatted_json_bytes"]
