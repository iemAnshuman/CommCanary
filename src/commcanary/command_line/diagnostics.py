"""CLI version rendering, structured diagnostics, and elapsed-time helpers."""

from __future__ import annotations

import sys
import time
from typing import Any

from ..artifacts import canonical_json_bytes
from ..formats import CANONICAL_JSON_VERSION, format_capabilities
from ..replay import SIMULATION_MODEL_VERSION
from ..version import package_version


def version_text() -> str:
    lines = [
        f"commcanary {package_version()}",
        f"canonicalization: {CANONICAL_JSON_VERSION}",
        f"replay-model: {SIMULATION_MODEL_VERSION}",
        "formats:",
    ]
    lines.extend(f"  {capability.artifact}: {capability.format_id}" for capability in format_capabilities())
    return "\n".join(lines)


def emit_diagnostic(args: Any, *, event: str, exit_code: int, **fields: Any) -> None:
    payload = {
        "format": "commcanary.diagnostic.v1",
        "event": event,
        "command": getattr(args, "command", None),
        "exit_code": exit_code,
        **fields,
    }
    print(canonical_json_bytes(payload).decode("utf-8"), file=sys.stderr)


def elapsed_seconds(started: float) -> float:
    return round(max(0.0, time.monotonic() - started), 6)


__all__ = ["elapsed_seconds", "emit_diagnostic", "version_text"]
