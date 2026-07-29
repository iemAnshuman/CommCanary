"""Canonical collective data-type identities shared by artifact adapters."""

from __future__ import annotations

from typing import Any

from ..errors import SchemaError

_CANONICAL_DTYPE_BYTES = {
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
    "complex32": 4,
    "complex64": 8,
    "complex128": 16,
}

_DTYPE_ALIASES = {
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e4m3fnuz": "float8_e4m3fnuz",
    "float8_e5m2": "float8_e5m2",
    "float8_e5m2fnuz": "float8_e5m2fnuz",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float32": "float32",
    "double": "float64",
    "float64": "float64",
    "char": "int8",
    "int8": "int8",
    "byte": "uint8",
    "uint8": "uint8",
    "short": "int16",
    "int16": "int16",
    "int": "int32",
    "int32": "int32",
    "long": "int64",
    "int64": "int64",
    "bool": "bool",
    "complexhalf": "complex32",
    "complex32": "complex32",
    "complexfloat": "complex64",
    "complex64": "complex64",
    "complexdouble": "complex128",
    "complex128": "complex128",
}

CANONICAL_DTYPES = tuple(_CANONICAL_DTYPE_BYTES)
PARAM_DTYPES = frozenset(
    {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
    }
)
PARAM_COMPUTE_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


def normalize_dtype(value: Any, *, label: str = "dtype") -> str:
    """Return one canonical dtype spelling or fail closed."""

    key = str(value).strip().lower()
    canonical = _DTYPE_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(CANONICAL_DTYPES)
        raise SchemaError(f"unsupported {label} {value!r}; supported canonical dtypes: {supported}")
    return canonical


def require_canonical_dtype(value: Any, *, label: str = "dtype") -> str:
    """Validate that an artifact already uses the canonical spelling."""

    canonical = normalize_dtype(value, label=label)
    if value != canonical:
        raise SchemaError(f"{label} must use canonical spelling {canonical!r}")
    return canonical


def dtype_size_bytes(value: Any, *, label: str = "dtype") -> int:
    """Return element width for a supported canonical dtype or alias."""

    return _CANONICAL_DTYPE_BYTES[normalize_dtype(value, label=label)]


def require_param_dtype(value: Any, *, label: str = "PARAM dtype") -> str:
    """Return a canonical dtype supported by CommCanary's PARAM adapter."""

    canonical = normalize_dtype(value, label=label)
    if canonical not in PARAM_DTYPES:
        supported = ", ".join(dtype for dtype in CANONICAL_DTYPES if dtype in PARAM_DTYPES)
        raise SchemaError(f"unsupported {label} {value!r}; supported PARAM dtypes: {supported}")
    return canonical


def require_param_compute_dtype(value: Any, *, label: str = "PARAM compute dtype") -> str:
    """Return a canonical floating dtype suitable for matrix multiplication."""

    canonical = require_param_dtype(value, label=label)
    if canonical not in PARAM_COMPUTE_DTYPES:
        supported = ", ".join(dtype for dtype in CANONICAL_DTYPES if dtype in PARAM_COMPUTE_DTYPES)
        raise SchemaError(f"unsupported {label} {value!r}; supported compute dtypes: {supported}")
    return canonical


__all__ = [
    "CANONICAL_DTYPES",
    "PARAM_COMPUTE_DTYPES",
    "PARAM_DTYPES",
    "dtype_size_bytes",
    "normalize_dtype",
    "require_canonical_dtype",
    "require_param_compute_dtype",
    "require_param_dtype",
]
