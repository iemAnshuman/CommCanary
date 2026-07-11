"""Compatibility facade for the historical :mod:`commcanary.interop` path.

The implementation lives in :mod:`commcanary.adapters.interop`. At runtime the
legacy module name resolves to that implementation module so existing preflight
probes that patch expansion helpers keep their historical behavior.
"""

from __future__ import annotations

import sys as _sys

from .adapters import interop as _implementation
from .adapters.interop import (
    canary_to_param_comms_trace,
    kineto_trace_to_commcanary_trace,
    load_kineto_trace,
    write_param_comms_trace,
)

__all__ = [
    "canary_to_param_comms_trace",
    "kineto_trace_to_commcanary_trace",
    "load_kineto_trace",
    "write_param_comms_trace",
]

_sys.modules[__name__] = _implementation
