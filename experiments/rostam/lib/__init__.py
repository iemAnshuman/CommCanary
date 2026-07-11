"""Manifest-bound Rostam planning and physical-result adapters.

Nothing in this package submits work at import time.  Scheduler submission is
an explicit second step over an immutable, content-addressed plan.
"""

from .catalog import CATALOG_SCHEMA, Catalog, CatalogValidationError, load_catalog
from .physical_results import (
    DEFAULT_PARAM_TRACE_LIMITS,
    ParamTraceLimits,
    PhysicalResultError,
    adapt_physical_measurement,
    load_validated_param_trace,
    validate_overlap_trace,
    validate_param_trace,
)
from .submission import SubmissionPlanError, build_submission_plan

__all__ = [
    "CATALOG_SCHEMA",
    "Catalog",
    "CatalogValidationError",
    "DEFAULT_PARAM_TRACE_LIMITS",
    "ParamTraceLimits",
    "PhysicalResultError",
    "SubmissionPlanError",
    "adapt_physical_measurement",
    "build_submission_plan",
    "load_catalog",
    "load_validated_param_trace",
    "validate_overlap_trace",
    "validate_param_trace",
]
