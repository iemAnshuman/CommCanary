"""Stable, intentionally small CommCanary public package surface.

Adapters such as capture, Kineto, and PARAM remain in their documented
submodules. Research baselines and reduction are not part of this stable tier.
Public JSON inputs are treated as read-only and returned artifacts are detached.
"""

from .artifacts import JsonDict, load_json, validate_canary, validate_comparison, validate_report, validate_trace
from .compare import compare_reports
from .compiler import compile_trace, verify_canary_behavior, verify_canary_fidelity
from .errors import CommCanaryError, SchemaError
from .formats import (
    CANARY_FORMAT,
    CANONICAL_JSON_VERSION,
    COMPARE_FORMAT,
    REPORT_FORMAT,
    TRACE_FORMAT,
    FormatCapability,
    format_capabilities,
)
from .replay import replay_canary, verify_report_against_canary
from .resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from .version import __version__, package_version

__all__ = [
    "CANARY_FORMAT",
    "CANONICAL_JSON_VERSION",
    "COMPARE_FORMAT",
    "DEFAULT_RESOURCE_LIMITS",
    "REPORT_FORMAT",
    "TRACE_FORMAT",
    "CommCanaryError",
    "FormatCapability",
    "JsonDict",
    "ResourceLimits",
    "SchemaError",
    "__version__",
    "compare_reports",
    "compile_trace",
    "format_capabilities",
    "load_json",
    "package_version",
    "replay_canary",
    "validate_canary",
    "validate_comparison",
    "validate_report",
    "validate_trace",
    "verify_canary_behavior",
    "verify_canary_fidelity",
    "verify_report_against_canary",
]
