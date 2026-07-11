"""Stable compiler constants shared inside the compilation boundary."""

DEFAULT_TIMING_SAMPLE_LIMIT = 128
US_TOLERANCE = 1e-6
FIDELITY_FIELDS = (
    "max_gap_error_us",
    "max_skew_error_us",
    "max_arrival_offset_error_us",
    "max_compute_before_error_us",
    "max_overlap_error_us",
    "max_pressure_error",
    "max_observed_exposed_error_us",
    "max_prefix_gap_error_us",
)

__all__ = ["DEFAULT_TIMING_SAMPLE_LIMIT"]
