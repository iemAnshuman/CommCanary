"""Offline access to the exact JSON Schemas shipped with CommCanary."""

from __future__ import annotations

from importlib import resources
from pathlib import Path, PurePosixPath

from ..errors import SchemaError
from ..formats import FormatCapability


def load_schema_bytes(capability: FormatCapability) -> bytes:
    """Return the immutable schema bytes declared by ``capability``.

    Wheels read package data through ``importlib.resources``. Editable source
    trees fall back to the repository ``schemas/`` path referenced by the same
    capability; the schema is never copied, modified, or fetched from a network.
    """

    relative = PurePosixPath(capability.schema)
    if len(relative.parts) != 2 or relative.parts[0] != "schemas" or relative.name != relative.parts[1]:
        raise SchemaError(f"invalid schema resource path {capability.schema!r}")
    try:
        return resources.files("commcanary.schemas").joinpath(relative.name).read_bytes()
    except FileNotFoundError:
        source_path = Path(__file__).resolve().parents[3] / relative
        try:
            return source_path.read_bytes()
        except OSError as source_error:
            raise SchemaError(f"schema resource is unavailable for {capability.format_id!r}") from source_error


__all__ = ["load_schema_bytes"]
