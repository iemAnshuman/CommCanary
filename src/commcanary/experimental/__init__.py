"""Explicitly unstable research baselines and reduction services.

These names are importable for experiment code but are not covered by the
top-level API deprecation promise. Existing module paths remain compatible.
"""

from ..baselines import (
    clustering_representative_baseline_trace,
    frequency_representative_baseline_trace,
    isolated_collective_baseline_trace,
    random_sampling_baseline_trace,
    stratified_sampling_baseline_trace,
)
from ..services.reduction import ddmin_ranking_reduction

__all__ = [
    "clustering_representative_baseline_trace",
    "ddmin_ranking_reduction",
    "frequency_representative_baseline_trace",
    "isolated_collective_baseline_trace",
    "random_sampling_baseline_trace",
    "stratified_sampling_baseline_trace",
]
