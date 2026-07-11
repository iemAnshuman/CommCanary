"""Compatibility facade for the historical :mod:`commcanary.compare` path.

New code may import the same API from :mod:`commcanary.comparison`.
"""

from .comparison import (
    ComparisonReasonCode,
    ComparisonThresholdPolicy,
    compare_reports,
    comparison_reason_codes,
)

__all__ = [
    "ComparisonReasonCode",
    "ComparisonThresholdPolicy",
    "compare_reports",
    "comparison_reason_codes",
]
