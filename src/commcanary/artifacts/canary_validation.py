"""Canary artifact structural, integrity, and fidelity validation."""

from __future__ import annotations

from typing import Any, List, Mapping, MutableMapping, Sequence, Tuple

from ..errors import SchemaError
from ..formats import (
    ARTIFACT_PROVENANCE_ALGORITHM,
    CANARY_FORMAT,
    CANARY_INTEGRITY_PROFILE,
)
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits
from .canary_expansion import (
    iter_canary_logical_events,
    iter_canary_stored_leaf_events,
    preflight_canary_expansion,
    stored_event_timing_record_count,
)
from .canary_hashes import (
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
)
from .wire import (
    MAX_TIME_US,
    as_float,
    as_int,
    normalize_ranks,
    require_format,
    require_optional_mapping,
    validate_arrival_keys,
    validate_nonempty_string,
    validate_op,
    validate_point_to_point_metadata,
    validate_sha256,
    validate_skew_matches_offsets,
)

ASSURANCE_STATES = (
    "structurally_valid",
    "internally_consistent",
    "source_corresponding",
    "model_recomputed",
    "behaviorally_verified",
)
FIDELITY_ERROR_FIELDS = (
    "max_gap_error_us",
    "max_skew_error_us",
    "max_arrival_offset_error_us",
    "max_compute_before_error_us",
    "max_overlap_error_us",
    "max_pressure_error",
    "max_observed_exposed_error_us",
    "max_prefix_gap_error_us",
)


def _validate_stored_event_source_blocks(events: Sequence[Any], *, profiled_integrity: bool) -> None:
    """Validate source metadata on every stored event, including motif wrappers."""

    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        label = f"canary event {event_index}"
        _validate_stored_event_source(event, label, profiled_integrity=profiled_integrity)
        if event.get("program") != "sequence_motif":
            continue
        children = event.get("events", [])
        if not isinstance(children, list):
            continue
        for child_index, child in enumerate(children):
            if isinstance(child, Mapping):
                _validate_stored_event_source(
                    child,
                    f"{label} child {child_index}",
                    profiled_integrity=profiled_integrity,
                )


def _validate_stored_event_source(
    event: Mapping[str, Any],
    label: str,
    *,
    profiled_integrity: bool,
) -> None:
    source = event.get("source")
    if not isinstance(source, Mapping):
        raise SchemaError(f"{label} must contain a source object")
    if "count" not in source:
        raise SchemaError(f"{label} source.count is required")
    if profiled_integrity:
        for field in ("first_id", "last_id", "digest"):
            if field not in source:
                raise SchemaError(f"{label} source.{field} is required")
    if "digest" in source:
        validate_sha256(source.get("digest"), f"{label} source.digest")


def validate_canary(
    canary: Mapping[str, Any],
    *,
    allow_legacy_unverified: bool = False,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> None:
    require_format(canary, CANARY_FORMAT, "canary")
    require_optional_mapping(canary, "workload", "canary")
    require_optional_mapping(canary, "system", "canary")
    require_optional_mapping(canary, "compiler", "canary")
    compiler = canary.get("compiler")
    if not isinstance(compiler, Mapping):
        raise SchemaError("canary must contain a compiler object")
    integrity_profile = compiler.get("integrity_profile")
    profiled_integrity = integrity_profile == CANARY_INTEGRITY_PROFILE
    if not profiled_integrity:
        if integrity_profile is not None:
            raise SchemaError(f"unsupported canary integrity profile {integrity_profile!r}")
        if not allow_legacy_unverified:
            raise SchemaError("canary is missing the required integrity profile")
    elif compiler.get("artifact_provenance_algorithm") != ARTIFACT_PROVENANCE_ALGORITHM:
        raise SchemaError("canary compiler.artifact_provenance_algorithm is unsupported")
    source_format = canary.get("source_format")
    if profiled_integrity and source_format is None:
        raise SchemaError("canary source_format is required by the integrity profile")
    if source_format is not None:
        validate_nonempty_string(source_format, "canary source_format")
    events = canary.get("events")
    if not isinstance(events, list):
        raise SchemaError("canary must contain an 'events' list")

    expansion = preflight_canary_expansion(events, limits=limits)
    _validate_stored_event_source_blocks(events, profiled_integrity=profiled_integrity)
    actual_stored_recursive_records = sum(
        stored_event_timing_record_count(event) for event in iter_canary_stored_leaf_events(events, limits=limits)
    )
    actual_stored_approximate_records = sum(
        _event_approximate_record_count(event) for event in iter_canary_stored_leaf_events(events, limits=limits)
    )

    total_repeat = 0
    all_leaf_observed_flags: List[bool] = []
    actual_recursive_records = 0
    actual_approximate_records = 0
    actual_encoded_gap_total = 0.0
    actual_compute_uncertain_events = 0
    actual_fidelity_maxima = {field: 0.0 for field in FIDELITY_ERROR_FIELDS}
    for index, event in enumerate(iter_canary_logical_events(events, limits=limits)):
        if not isinstance(event, Mapping):
            raise SchemaError(f"canary event {index} must be an object")
        for key in ("op", "bytes", "ranks", "repeat"):
            if key not in event:
                raise SchemaError(f"canary event {index} is missing {key!r}")
        validate_op(event.get("op"), f"canary event {index}", custom=event.get("custom_op") is True)
        for text_key in ("phase", "group"):
            if text_key not in event:
                raise SchemaError(f"canary event {index} is missing {text_key!r}")
            validate_nonempty_string(event.get(text_key), f"canary event {index} {text_key}")
        if as_int(event.get("bytes")) <= 0:
            raise SchemaError(f"canary event {index} bytes must be positive")
        repeat = as_int(event.get("repeat"))
        if repeat <= 0:
            raise SchemaError(f"canary event {index} repeat must be positive")
        total_repeat += repeat

        source = event.get("source")
        if not isinstance(source, Mapping):
            raise SchemaError(f"canary event {index} must contain a source object")
        if "count" not in source:
            raise SchemaError(f"canary event {index} source.count is required")
        if as_int(source.get("count")) != repeat:
            raise SchemaError(f"canary event {index} source.count must match repeat")
        if profiled_integrity and "digest" not in source:
            raise SchemaError(f"canary event {index} source.digest is required")
        if "digest" in source:
            validate_sha256(source.get("digest"), f"canary event {index} source.digest")
        if "sampled_timing_records" in source and as_int(source.get("sampled_timing_records")) <= 0:
            raise SchemaError(f"canary event {index} source.sampled_timing_records must be positive")
        if "execution_occurrence_base" in event and as_int(event.get("execution_occurrence_base")) < 0:
            raise SchemaError(f"canary event {index} execution_occurrence_base must be non-negative")
        if "concurrent_groups" in event and as_int(event.get("concurrent_groups")) <= 0:
            raise SchemaError(f"canary event {index} concurrent_groups must be positive")

        ranks = normalize_ranks(event.get("ranks"))
        if len(ranks) > limits.max_ranks:
            raise SchemaError(f"canary event {index} rank count exceeds resource policy limit={limits.max_ranks}")
        if "rank_count" in event and as_int(event.get("rank_count")) != len(ranks):
            raise SchemaError(f"canary event {index} rank_count must match ranks")
        if "rank_arrival_us" in event:
            validate_arrival_keys(
                event.get("rank_arrival_us", {}),
                ranks,
                f"canary event {index} rank_arrival_us",
                allow_subset=event.get("partial_rank_arrival") is True,
            )
        if "arrival_offsets_us" in event:
            offsets = event.get("arrival_offsets_us")
            if not isinstance(offsets, list) or len(offsets) != len(ranks):
                raise SchemaError(f"canary event {index} arrival_offsets_us must match ranks")
            parsed_offsets = [as_float(value) for value in offsets]
            if any(value < 0.0 for value in parsed_offsets):
                raise SchemaError(f"canary event {index} arrival offsets must be non-negative")
            if "arrival_skew_us" in event and as_float(event.get("arrival_skew_us")) < 0.0:
                raise SchemaError(f"canary event {index} arrival_skew_us must be non-negative")
            if len(ranks) == 1 and as_float(event.get("arrival_skew_us"), 0.0) > 0.001:
                raise SchemaError(f"canary event {index} one-rank skew must be zero")
        validate_point_to_point_metadata(event, ranks, f"canary event {index}")

        samples = event.get("timing_samples")
        if not isinstance(samples, list) or not samples:
            raise SchemaError(f"canary event {index} must contain non-empty timing_samples")
        weight_total = 0
        source_indices: List[int] = []
        intervals: List[Tuple[int, int]] = []
        for sample_index, sample in enumerate(samples):
            label = f"canary event {index} timing sample {sample_index}"
            if not isinstance(sample, Mapping):
                raise SchemaError(f"{label} must be an object")
            _validate_timing_record(sample, ranks, label, repeat=repeat)
            actual_recursive_records += 1
            actual_encoded_gap_total += _timing_record_gap_sum(sample, label)
            actual_compute_uncertain_events += _timing_record_logical_uncertain_weight(sample)
            _accumulate_fidelity_maxima(actual_fidelity_maxima, sample)
            if sample.get("approximation") == "bounded_interval":
                actual_approximate_records += 1

            sample_weight = as_int(sample.get("weight", 1))
            if sample_weight <= 0:
                raise SchemaError(f"{label} weight must be positive")
            pattern = sample.get("timing_pattern")
            if pattern is not None:
                if not isinstance(pattern, list) or not pattern:
                    raise SchemaError(f"{label} timing_pattern must be non-empty")
                pattern_repeats = as_int(sample.get("pattern_repeats", 1))
                if pattern_repeats <= 0:
                    raise SchemaError(f"{label} pattern_repeats must be positive")
                pattern_weight = 0
                pattern_source_indices: List[int] = []
                pattern_gap_sum = 0.0
                for pattern_index, pattern_sample in enumerate(pattern):
                    pattern_label = f"{label} pattern record {pattern_index}"
                    if not isinstance(pattern_sample, Mapping):
                        raise SchemaError(f"{pattern_label} must be an object")
                    if "timing_pattern" in pattern_sample:
                        raise SchemaError(f"{pattern_label} must not contain a nested timing_pattern")
                    _validate_timing_record(pattern_sample, ranks, pattern_label, repeat=repeat)
                    actual_recursive_records += 1
                    _accumulate_fidelity_maxima(actual_fidelity_maxima, pattern_sample)
                    if pattern_sample.get("approximation") == "bounded_interval":
                        actual_approximate_records += 1
                    pattern_entry_weight = as_int(pattern_sample.get("weight", 1))
                    if pattern_entry_weight <= 0:
                        raise SchemaError(f"{pattern_label} weight must be positive")
                    pattern_weight += pattern_entry_weight
                    pattern_gap_sum += _timing_record_gap_sum(pattern_sample, pattern_label)
                    if "source_index" in pattern_sample:
                        pattern_source_indices.append(as_int(pattern_sample.get("source_index")))
                    all_leaf_observed_flags.append("observed_exposed_us" in pattern_sample)
                if sample_weight != pattern_weight * pattern_repeats:
                    raise SchemaError(f"{label} pattern weight must match sample weight")
                _validate_source_indices(pattern_source_indices, f"{label} pattern")
                parent_gap_sum = _timing_record_gap_sum(sample, label)
                expected_gap_sum = pattern_gap_sum * pattern_repeats
                if abs(parent_gap_sum - expected_gap_sum) > 1e-6:
                    raise SchemaError(f"{label} gap_sum_us must match its repeated timing_pattern")
            else:
                _validate_non_pattern_gap_sum(sample, label)
                all_leaf_observed_flags.append("observed_exposed_us" in sample)

            weight_total += sample_weight
            if "source_index" in sample:
                source_indices.append(as_int(sample.get("source_index")))
            intervals.append(_sample_interval(sample, label))

        _validate_source_indices(source_indices, f"canary event {index} timing samples")
        _validate_intervals(intervals, f"canary event {index} timing samples")
        if intervals and (intervals[0][0] != 0 or intervals[-1][1] != repeat - 1):
            raise SchemaError(f"canary event {index} timing samples must cover the full repeat interval")
        if weight_total != repeat:
            raise SchemaError(f"canary event {index} timing sample weights must sum to repeat")
        if repeat == 1 and len(samples) == 1 and isinstance(samples[0], Mapping) and "timing_pattern" not in samples[0]:
            _validate_event_summary_matches_single_sample(event, samples[0], ranks, f"canary event {index}")

    if all_leaf_observed_flags and any(all_leaf_observed_flags) and not all(all_leaf_observed_flags):
        raise SchemaError("observed_exposed_us must be present on all timing records or none")

    if "source_events" not in compiler:
        raise SchemaError("canary compiler.source_events is required")
    source_events = as_int(compiler.get("source_events"))
    if source_events < 0:
        raise SchemaError("canary compiler.source_events must be non-negative")
    if source_events != total_repeat:
        raise SchemaError("canary compiler.source_events must match event repeats")
    if "canary_events" in compiler and as_int(compiler.get("canary_events")) != len(events):
        raise SchemaError("canary compiler.canary_events must match stored events")
    if (
        "expanded_canary_events" in compiler
        and as_int(compiler.get("expanded_canary_events")) != expansion.logical_events
    ):
        raise SchemaError("canary compiler.expanded_canary_events must match expanded events")
    if (
        "recursive_timing_records" in compiler
        and as_int(compiler.get("recursive_timing_records")) != actual_recursive_records
    ):
        raise SchemaError("canary compiler.recursive_timing_records must match logical timing records")
    if (
        "approximate_timing_records" in compiler
        and as_int(compiler.get("approximate_timing_records")) != actual_approximate_records
    ):
        raise SchemaError("canary compiler.approximate_timing_records must match logical timing records")
    if (
        "stored_recursive_timing_records" in compiler
        and as_int(compiler.get("stored_recursive_timing_records")) != actual_stored_recursive_records
    ):
        raise SchemaError("canary compiler.stored_recursive_timing_records must match stored timing records")
    if (
        "stored_approximate_timing_records" in compiler
        and as_int(compiler.get("stored_approximate_timing_records")) != actual_stored_approximate_records
    ):
        raise SchemaError("canary compiler.stored_approximate_timing_records must match stored timing records")
    capture_uncertainty = compiler.get("capture_uncertainty")
    if actual_compute_uncertain_events:
        if not isinstance(capture_uncertainty, Mapping):
            raise SchemaError("canary compiler.capture_uncertainty is required for uncertain timing records")
        if as_int(capture_uncertainty.get("compute_fields_uncertain_events")) != actual_compute_uncertain_events:
            raise SchemaError("canary compiler.capture_uncertainty compute count must match timing records")
        status = capture_uncertainty.get("status")
        if not isinstance(status, str) or not status:
            raise SchemaError("canary compiler.capture_uncertainty.status must be a non-empty string")
    elif capture_uncertainty is not None:
        if not isinstance(capture_uncertainty, Mapping):
            raise SchemaError("canary compiler.capture_uncertainty must be an object")
        if as_int(capture_uncertainty.get("compute_fields_uncertain_events"), 0) != 0:
            raise SchemaError("canary compiler.capture_uncertainty contradicts timing records")
    required_hashes = (
        "source_trace_sha256",
        "source_normalized_sha256",
        "execution_semantic_sha256",
        "scheduler_execution_sha256",
        "calibration_evaluation_sha256",
        "artifact_provenance_sha256",
    )
    for hash_key in required_hashes:
        if profiled_integrity and hash_key not in compiler:
            raise SchemaError(f"canary compiler.{hash_key} is required")
        if hash_key in compiler:
            validate_sha256(compiler.get(hash_key), f"canary compiler.{hash_key}")
    if (
        "source_normalized_sha256" in compiler
        and "source_trace_sha256" in compiler
        and compiler.get("source_normalized_sha256") != compiler.get("source_trace_sha256")
    ):
        raise SchemaError("canary compiler.source_normalized_sha256 must match source_trace_sha256")
    if "execution_semantic_sha256" in compiler and compiler.get("execution_semantic_sha256") != canary_execution_sha256(
        canary, limits=limits
    ):
        raise SchemaError("canary compiler.execution_semantic_sha256 does not match executable events")
    if "scheduler_execution_sha256" in compiler and compiler.get(
        "scheduler_execution_sha256"
    ) != canary_scheduler_execution_sha256(canary, limits=limits):
        raise SchemaError("canary compiler.scheduler_execution_sha256 does not match scheduler events")
    if "calibration_evaluation_sha256" in compiler and compiler.get(
        "calibration_evaluation_sha256"
    ) != canary_calibration_sha256(canary, limits=limits):
        raise SchemaError("canary compiler.calibration_evaluation_sha256 does not match calibration fields")
    if "artifact_provenance_sha256" in compiler and compiler.get(
        "artifact_provenance_sha256"
    ) != canary_artifact_provenance_sha256(canary):
        raise SchemaError("canary compiler.artifact_provenance_sha256 does not match artifact fields")
    for integer_key in (
        "source_bytes",
        "canary_bytes",
        "timing_sample_limit",
        "timing_group_count",
        "expanded_canary_events",
        "sequence_motif_count",
        "stored_recursive_timing_records",
        "stored_approximate_timing_records",
    ):
        if integer_key in compiler and as_int(compiler.get(integer_key)) < 0:
            raise SchemaError(f"canary compiler.{integer_key} must be non-negative")
    timing_limit_mode = compiler.get("timing_sample_limit_mode")
    if timing_limit_mode is not None and timing_limit_mode not in {"uniform", "per_group"}:
        raise SchemaError("canary compiler.timing_sample_limit_mode is invalid")
    raw_group_limits = compiler.get("timing_sample_limits_by_group")
    if raw_group_limits is not None:
        if not isinstance(raw_group_limits, Mapping):
            raise SchemaError("canary compiler.timing_sample_limits_by_group must be an object")
        default_limit = as_int(compiler.get("timing_sample_limit"), 0)
        group_count = as_int(compiler.get("timing_group_count"), 0)
        for raw_group, raw_limit in raw_group_limits.items():
            group_id = as_int(raw_group)
            limit = as_int(raw_limit)
            if group_id < 0:
                raise SchemaError("canary compiler.timing_sample_limits_by_group keys must be non-negative")
            if group_count and group_id >= group_count:
                raise SchemaError("canary compiler.timing_sample_limits_by_group references an unknown group")
            if limit < 2:
                raise SchemaError("canary compiler.timing_sample_limits_by_group values must be at least 2")
            if default_limit and limit > default_limit:
                raise SchemaError(
                    "canary compiler.timing_sample_limits_by_group values must not exceed timing_sample_limit"
                )

    fidelity = compiler.get("fidelity")
    if actual_approximate_records and fidelity is None:
        raise SchemaError("canary compiler.fidelity is required for approximate timing records")
    if fidelity is not None:
        if not isinstance(fidelity, Mapping):
            raise SchemaError("canary compiler.fidelity must be an object")
        mode = fidelity.get("mode")
        if mode not in {"lossless_timing", "bounded_approximate"}:
            raise SchemaError("canary compiler.fidelity.mode is invalid")
        approximate = as_int(fidelity.get("approximate_timing_records"), 0)
        if approximate != actual_approximate_records:
            raise SchemaError("canary compiler.fidelity approximate count must match timing records")
        if (mode == "lossless_timing") != (actual_approximate_records == 0):
            raise SchemaError("canary compiler.fidelity.mode contradicts approximation records")
        for key in (*FIDELITY_ERROR_FIELDS, "source_gap_total_us", "encoded_gap_total_us", "total_gap_error_us"):
            if key in fidelity and as_float(fidelity.get(key)) < 0.0:
                raise SchemaError(f"canary compiler.fidelity.{key} must be non-negative")
        for key, expected in actual_fidelity_maxima.items():
            if key not in fidelity:
                raise SchemaError(f"canary compiler.fidelity.{key} is required")
            if abs(as_float(fidelity.get(key)) - expected) > 1e-6:
                raise SchemaError(f"canary compiler.fidelity.{key} does not match timing records")
        if "encoded_gap_total_us" in fidelity:
            if abs(as_float(fidelity.get("encoded_gap_total_us")) - actual_encoded_gap_total) > 1e-6:
                raise SchemaError("canary compiler.fidelity.encoded_gap_total_us does not match timing records")
        if all(key in fidelity for key in ("source_gap_total_us", "encoded_gap_total_us", "total_gap_error_us")):
            expected_error = abs(
                as_float(fidelity.get("source_gap_total_us")) - as_float(fidelity.get("encoded_gap_total_us"))
            )
            if abs(expected_error - as_float(fidelity.get("total_gap_error_us"))) > 1e-6:
                raise SchemaError("canary compiler.fidelity.total_gap_error_us is inconsistent")

    fidelity_budget = compiler.get("fidelity_budget")
    if fidelity_budget is not None:
        if not isinstance(fidelity_budget, Mapping):
            raise SchemaError("canary compiler.fidelity_budget must be an object")
        if fidelity is None:
            raise SchemaError("canary compiler.fidelity_budget requires compiler.fidelity")
        for key, budget in fidelity_budget.items():
            if key not in FIDELITY_ERROR_FIELDS:
                raise SchemaError(f"canary compiler.fidelity_budget contains unknown field {key!r}")
            budget_value = as_float(budget)
            if budget_value < 0.0:
                raise SchemaError(f"canary compiler.fidelity_budget.{key} must be non-negative")
            if as_float(fidelity.get(key), 0.0) > budget_value + 1e-6:
                raise SchemaError(f"canary compiler.fidelity.{key} exceeds recorded fidelity budget")

    tail_signal = compiler.get("tail_signal")
    has_observed = bool(all_leaf_observed_flags and all(all_leaf_observed_flags))
    if tail_signal == "observed_exposed_us" and not has_observed:
        raise SchemaError("canary compiler.tail_signal requires observed timing records")
    if tail_signal == "structural-proxy" and has_observed:
        raise SchemaError("canary compiler.tail_signal contradicts observed timing records")


def _event_approximate_record_count(event: Mapping[str, Any]) -> int:
    samples = event.get("timing_samples")
    if not isinstance(samples, list):
        return 0
    total = 0
    for sample in samples:
        if isinstance(sample, Mapping):
            total += int(sample.get("approximation") == "bounded_interval")
            pattern = sample.get("timing_pattern")
            if isinstance(pattern, list):
                total += sum(
                    int(isinstance(child, Mapping) and child.get("approximation") == "bounded_interval")
                    for child in pattern
                )
    return total


def _validate_timing_record(sample: Mapping[str, Any], ranks: List[int], label: str, *, repeat: int) -> None:
    offsets = sample.get("arrival_offsets_us")
    if not isinstance(offsets, list) or len(offsets) != len(ranks):
        raise SchemaError(f"{label} arrival_offsets_us must match ranks")
    numeric_offsets = []
    for offset in offsets:
        parsed_offset = as_float(offset)
        if parsed_offset < 0.0:
            raise SchemaError("arrival offsets must be non-negative")
        numeric_offsets.append(parsed_offset)
    if "arrival_skew_us" in sample:
        validate_skew_matches_offsets(as_float(sample.get("arrival_skew_us")), numeric_offsets, label)
    for numeric_key in (
        "gap_us",
        "gap_sum_us",
        "compute_before_us",
        "compute_overlap_us",
        "compute_pressure",
        "observed_exposed_us",
        "max_gap_error_us",
        "max_skew_error_us",
        "max_arrival_offset_error_us",
        "max_compute_before_error_us",
        "max_overlap_error_us",
        "max_pressure_error",
        "max_observed_exposed_error_us",
        "representative_gap_error_us",
        "max_prefix_gap_error_us",
    ):
        if numeric_key in sample and as_float(sample.get(numeric_key), 0.0) < 0.0:
            raise SchemaError(f"{label} {numeric_key} must be non-negative")
        if (
            numeric_key in sample
            and numeric_key.endswith("_us")
            and as_float(sample.get(numeric_key), 0.0) > MAX_TIME_US
        ):
            raise SchemaError(f"{label} {numeric_key} exceeds maximum supported duration")
    approximation = sample.get("approximation")
    if approximation is not None and approximation != "bounded_interval":
        raise SchemaError(f"{label} approximation is unsupported")
    if approximation == "bounded_interval":
        _validate_bounded_interval_evidence(sample, label)
    if "source_start" in sample and "source_end" in sample:
        expected_weight = as_int(sample.get("source_end")) - as_int(sample.get("source_start")) + 1
        if as_int(sample.get("weight", 1)) != expected_weight:
            raise SchemaError(f"{label} weight must match source interval length")
    elif as_int(sample.get("weight", 1)) != 1:
        raise SchemaError(f"{label} weight above one requires source_start and source_end")
    if "compute_fields_uncertain" in sample and not isinstance(sample.get("compute_fields_uncertain"), bool):
        raise SchemaError(f"{label} compute_fields_uncertain must be a boolean")
    if "uncertain_weight" in sample:
        uncertain_weight = as_int(sample.get("uncertain_weight"))
        weight = as_int(sample.get("weight"), 1)
        if uncertain_weight < 0 or uncertain_weight > weight:
            raise SchemaError(f"{label} uncertain_weight must be between zero and weight")
        if uncertain_weight and sample.get("compute_fields_uncertain") is not True:
            raise SchemaError(f"{label} uncertain_weight requires compute_fields_uncertain")
    if "source_index" in sample:
        source_index = as_int(sample.get("source_index"))
        if source_index < 0 or source_index >= repeat:
            raise SchemaError(f"{label} source_index must be within repeat")
    if "source_start" in sample or "source_end" in sample:
        if "source_start" not in sample or "source_end" not in sample:
            raise SchemaError(f"{label} source interval requires both source_start and source_end")
        source_start = as_int(sample.get("source_start"))
        source_end = as_int(sample.get("source_end"))
        if source_start < 0 or source_end < source_start or source_end >= repeat:
            raise SchemaError(f"{label} source interval must be ordered and within repeat")


def _accumulate_fidelity_maxima(maxima: MutableMapping[str, float], sample: Mapping[str, Any]) -> None:
    for key in FIDELITY_ERROR_FIELDS:
        if key in sample:
            maxima[key] = max(maxima[key], as_float(sample.get(key)))


def _validate_bounded_interval_evidence(sample: Mapping[str, Any], label: str) -> None:
    required = (
        "source_index",
        "source_start",
        "source_end",
        "weight",
        "source_count",
        "representative_source_index",
        "source_segment_sha256",
        "source_gap_sum_us",
        "gap_sum_us",
        "representative_selection_method",
        "error_vector",
        "max_gap_error_us",
        "max_skew_error_us",
        "max_arrival_offset_error_us",
        "max_compute_before_error_us",
        "max_overlap_error_us",
        "max_pressure_error",
        "representative_gap_error_us",
        "max_prefix_gap_error_us",
    )
    for key in required:
        if key not in sample:
            raise SchemaError(f"{label} bounded interval missing {key!r}")
    if "observed_exposed_us" in sample and "max_observed_exposed_error_us" not in sample:
        raise SchemaError(f"{label} bounded interval missing 'max_observed_exposed_error_us'")
    if as_int(sample.get("source_count")) != as_int(sample.get("weight")):
        raise SchemaError(f"{label} source_count must match weight")
    representative_source_index = as_int(sample.get("representative_source_index"))
    if not as_int(sample.get("source_start")) <= representative_source_index <= as_int(sample.get("source_end")):
        raise SchemaError(f"{label} representative_source_index must be within source interval")
    if abs(as_float(sample.get("source_gap_sum_us")) - as_float(sample.get("gap_sum_us"))) > 1e-6:
        raise SchemaError(f"{label} source_gap_sum_us must match gap_sum_us")
    validate_sha256(sample.get("source_segment_sha256"), f"{label} source_segment_sha256")
    method = sample.get("representative_selection_method")
    if not isinstance(method, str) or not method:
        raise SchemaError(f"{label} representative_selection_method must be a non-empty string")
    error_vector = sample.get("error_vector")
    if not isinstance(error_vector, Mapping):
        raise SchemaError(f"{label} error_vector must be an object")
    expected_error_fields = (
        "max_gap_error_us",
        "max_skew_error_us",
        "max_arrival_offset_error_us",
        "max_compute_before_error_us",
        "max_overlap_error_us",
        "max_pressure_error",
        "max_observed_exposed_error_us",
        "representative_gap_error_us",
        "max_prefix_gap_error_us",
    )
    for field in expected_error_fields:
        if field in sample:
            if field not in error_vector:
                raise SchemaError(f"{label} error_vector missing {field!r}")
            if abs(as_float(error_vector.get(field)) - as_float(sample.get(field))) > 1e-6:
                raise SchemaError(f"{label} error_vector.{field} must match {field}")
        elif field in error_vector:
            raise SchemaError(f"{label} error_vector contains absent field {field!r}")


def _timing_record_logical_uncertain_weight(sample: Mapping[str, Any]) -> int:
    if "uncertain_weight" in sample:
        return as_int(sample.get("uncertain_weight"))
    pattern = sample.get("timing_pattern")
    if isinstance(pattern, list) and pattern:
        repeats = as_int(sample.get("pattern_repeats"), 1)
        return (
            sum(_timing_record_logical_uncertain_weight(child) for child in pattern if isinstance(child, Mapping))
            * repeats
        )
    if sample.get("compute_fields_uncertain") is True:
        return as_int(sample.get("weight"), 1)
    return 0


def _validate_event_summary_matches_single_sample(
    event: Mapping[str, Any],
    sample: Mapping[str, Any],
    ranks: List[int],
    label: str,
) -> None:
    if "arrival_offsets_us" in event:
        event_offsets = event.get("arrival_offsets_us")
        sample_offsets = sample.get("arrival_offsets_us")
        if not isinstance(event_offsets, list) or len(event_offsets) != len(ranks):
            raise SchemaError(f"{label} arrival_offsets_us must match ranks")
        if not isinstance(sample_offsets, list) or len(sample_offsets) != len(ranks):
            raise SchemaError(f"{label} timing sample arrival_offsets_us must match ranks")
        for left, right in zip(event_offsets, sample_offsets):
            if abs(as_float(left) - as_float(right)) > 0.001:
                raise SchemaError(f"{label} arrival_offsets_us must match its timing sample")
        if "arrival_skew_us" in event:
            validate_skew_matches_offsets(as_float(event.get("arrival_skew_us")), event_offsets, label)
    for key in ("gap_us", "compute_before_us", "compute_overlap_us", "observed_exposed_us"):
        if key in event and key in sample and abs(as_float(event.get(key)) - as_float(sample.get(key))) > 0.001:
            raise SchemaError(f"{label} {key} must match its timing sample")
    if "compute_pressure" in event and "compute_pressure" in sample:
        if abs(as_float(event.get("compute_pressure")) - as_float(sample.get("compute_pressure"))) > 1e-6:
            raise SchemaError(f"{label} compute_pressure must match its timing sample")


def _timing_record_gap_sum(sample: Mapping[str, Any], label: str) -> float:
    weight = as_int(sample.get("weight"), 1)
    if weight <= 0:
        raise SchemaError(f"{label} weight must be positive")
    if "gap_sum_us" in sample:
        return as_float(sample.get("gap_sum_us"))
    return as_float(sample.get("gap_us"), 0.0) * weight


def _validate_non_pattern_gap_sum(sample: Mapping[str, Any], label: str) -> None:
    if "gap_sum_us" not in sample:
        return
    weight = as_int(sample.get("weight"), 1)
    expected = as_float(sample.get("gap_us"), 0.0) * weight
    actual = as_float(sample.get("gap_sum_us"))
    # gap_us is stored to nine decimal places; permit only the accumulated
    # representational error implied by that rounding.
    tolerance = max(1e-6, weight * 1e-9)
    if abs(actual - expected) > tolerance:
        raise SchemaError(f"{label} gap_sum_us must equal gap_us times weight")


def _validate_source_indices(source_indices: List[int], label: str) -> None:
    if source_indices != sorted(source_indices):
        raise SchemaError(f"{label} source_index values must be ordered")
    if len(source_indices) != len(set(source_indices)):
        raise SchemaError(f"{label} source_index values must be unique")


def _sample_interval(sample: Mapping[str, Any], label: str) -> Tuple[int, int]:
    if "source_start" in sample or "source_end" in sample:
        start = as_int(sample.get("source_start"))
        end = as_int(sample.get("source_end"))
    elif "source_index" in sample:
        start = end = as_int(sample.get("source_index"))
    else:
        raise SchemaError(f"{label} source interval is required")
    return start, end


def _validate_intervals(intervals: List[Tuple[int, int]], label: str) -> None:
    previous_end = -1
    for start, end in intervals:
        if start <= previous_end:
            raise SchemaError(f"{label} source intervals must be ordered and non-overlapping")
        if start != previous_end + 1:
            raise SchemaError(f"{label} source intervals must be contiguous")
        previous_end = end


__all__ = ["ASSURANCE_STATES", "FIDELITY_ERROR_FIELDS", "validate_canary"]
