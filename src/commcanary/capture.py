"""Compatibility facade for the historical :mod:`commcanary.capture` path.

The implementation lives in :mod:`commcanary.adapters.capture`. At runtime the
legacy module name resolves to that same module object so supported patching of
recorder state and loader preflight probes retains its historical behavior.
"""

from __future__ import annotations

import sys as _sys

from .adapters import capture as _implementation
from .adapters.capture import (
    NullRecorder,
    TraceRecorder,
    get_recorder,
    merge_trace_shards,
    record_collective,
)

__all__ = [
    "NullRecorder",
    "TraceRecorder",
    "get_recorder",
    "merge_trace_shards",
    "record_collective",
]

_sys.modules[__name__] = _implementation
