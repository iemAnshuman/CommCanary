from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

TRACE_FORMAT = "commcanary.trace.v1"
CANARY_FORMAT = "commcanary.canary.v2"
REPORT_FORMAT = "commcanary.report.v2"
COMPARE_FORMAT = "commcanary.compare.v2"
MAX_RANK_COUNT = 65536
MAX_ABS_INTEGER = 2**63 - 1
SUPPORTED_OPS = {"all_reduce", "reduce_scatter", "all_gather", "all_to_all", "broadcast", "send", "recv"}
PROTOCOL_FINGERPRINT_EXCLUDE = {"sha256", "max_replay_events"}
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


class CommCanaryError(Exception):
    """Base exception for expected CommCanary failures."""


class SchemaError(CommCanaryError):
    """Raised when a trace, canary, or report does not match the MVP schema."""


JsonDict = Dict[str, Any]


def load_json(path: str) -> JsonDict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=_reject_json_constant)
    except FileNotFoundError as exc:
        raise SchemaError(f"{path} does not exist") from exc
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{path} is not UTF-8 JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc.msg}") from exc
    except OSError as exc:
        raise SchemaError(f"cannot read {path}: {exc}") from exc
    except OverflowError as exc:
        raise SchemaError(f"{path} contains a number that is too large") from exc
    except ValueError as exc:
        raise SchemaError(f"{path} contains non-standard JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a JSON object")
    return data


def write_json(path: str, data: Mapping[str, Any]) -> None:
    target = Path(path)
    fd = None
    temp_name = ""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp_name, target)
    except (ValueError, OverflowError) as exc:
        raise SchemaError(f"cannot write {path}: JSON contains a non-finite number") from exc
    except TypeError as exc:
        raise SchemaError(f"cannot write {path}: data is not JSON serializable: {exc}") from exc
    except OSError as exc:
        raise SchemaError(f"cannot write {path}: {exc}") from exc
    finally:
        if fd is not None:
            os.close(fd)
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"cannot canonicalize JSON: {exc}") from exc


def replay_protocol_sha256(protocol: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in protocol.items() if key not in PROTOCOL_FINGERPRINT_EXCLUDE}
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def canary_execution_sha256(canary: Mapping[str, Any]) -> str:
    stable = {
        "format": canary.get("format"),
        "events": [
            _execution_event_projection(event)
            for event in canary.get("events", [])
            if isinstance(event, Mapping)
        ],
    }
    return hashlib.sha256(canonical_json_bytes(stable)).hexdigest()


def _execution_event_projection(event: Mapping[str, Any]) -> JsonDict:
    projected: JsonDict = {}
    for key in ("phase", "op", "bytes", "ranks", "group", "concurrent_groups"):
        if key in event:
            projected[key] = event.get(key)
    ranks = event.get("ranks")
    if isinstance(ranks, list):
        projected["rank_count"] = len(ranks)
    elif "rank_count" in event:
        projected["rank_count"] = event.get("rank_count")
    samples = event.get("timing_samples")
    if isinstance(samples, list):
        projected["timing_samples"] = [
            _execution_timing_projection(sample)
            for sample in samples
            if isinstance(sample, Mapping)
        ]
    return projected


def _execution_timing_projection(sample: Mapping[str, Any]) -> JsonDict:
    projected: JsonDict = {}
    weight = as_int(sample.get("weight"), 1)
    if "gap_sum_us" in sample:
        projected["gap_sum_us"] = round(as_float(sample.get("gap_sum_us")), 9)
    elif "gap_us" in sample:
        projected["gap_sum_us"] = round(as_float(sample.get("gap_us"), 0.0) * weight, 9)
    for key in (
        "arrival_offsets_us",
        "compute_before_us",
        "compute_overlap_us",
        "compute_pressure",
        "observed_exposed_us",
        "weight",
        "pattern_repeats",
    ):
        if key in sample:
            projected[key] = sample.get(key)
    pattern = sample.get("timing_pattern")
    if isinstance(pattern, list):
        projected["timing_pattern"] = [
            _execution_timing_projection(child)
            for child in pattern
            if isinstance(child, Mapping)
        ]
    return projected


def require_format(data: Mapping[str, Any], expected: str, label: str) -> None:
    if not isinstance(data, Mapping):
        raise SchemaError(f"{label} must be a JSON object")
    actual = data.get("format")
    if actual != expected:
        raise SchemaError(f"{label} format must be {expected!r}, got {actual!r}")


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise SchemaError(f"expected finite numeric value, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"expected numeric value, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise SchemaError(f"expected finite numeric value, got {value!r}")
    return parsed


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise SchemaError(f"expected integer value, got {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise SchemaError(f"expected integer value, got {value!r}")
        parsed = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("-"):
            digits = stripped[1:]
        else:
            digits = stripped
        if digits and digits.isdigit():
            if len(digits) > 19:
                raise SchemaError(f"integer value is too large: {value!r}")
            try:
                parsed = int(stripped)
            except ValueError as exc:
                raise SchemaError(f"expected integer value, got {value!r}") from exc
        else:
            raise SchemaError(f"expected integer value, got {value!r}")
    else:
        raise SchemaError(f"expected integer value, got {value!r}")
    if abs(parsed) > MAX_ABS_INTEGER:
        raise SchemaError(f"integer value is too large: {value!r}")
    return parsed


def normalize_ranks(value: Any) -> List[int]:
    if value is None:
        raise SchemaError("event is missing required 'ranks'")
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise SchemaError("rank count must be positive")
        if value > MAX_RANK_COUNT:
            raise SchemaError(f"rank count must not exceed {MAX_RANK_COUNT}")
        return list(range(value))
    if not isinstance(value, list):
        raise SchemaError("'ranks' must be a rank list or rank count")
    ranks = [as_int(rank) for rank in value]
    if not ranks:
        raise SchemaError("'ranks' must not be empty")
    if any(rank < 0 for rank in ranks):
        raise SchemaError("'ranks' must contain only non-negative integers")
    if len(set(ranks)) != len(ranks):
        raise SchemaError("'ranks' must not contain duplicates")
    if len(ranks) > MAX_RANK_COUNT:
        raise SchemaError(f"'ranks' must not contain more than {MAX_RANK_COUNT} entries")
    return ranks


def normalize_arrival_offsets(event: Mapping[str, Any], ranks: List[int]) -> List[float]:
    raw = event.get("rank_arrival_us")
    if raw is None:
        skew = as_float(event.get("arrival_skew_us"), 0.0)
        if skew < 0.0:
            raise SchemaError("arrival_skew_us must be non-negative")
        if len(ranks) == 1:
            if skew > 0.001:
                raise SchemaError("a one-rank collective cannot have positive arrival skew")
            return [0.0]
        return [0.0 for _ in ranks[:-1]] + [max(0.0, skew)]

    if isinstance(raw, Mapping):
        _validate_arrival_keys(raw, ranks, "rank_arrival_us", allow_subset=False)
        values = []
        for rank in ranks:
            if str(rank) in raw:
                value = as_float(raw[str(rank)])
            elif rank in raw:
                value = as_float(raw[rank])
            else:
                raise SchemaError(f"rank_arrival_us is missing rank {rank}")
            if value < 0.0:
                raise SchemaError("rank_arrival_us values must be non-negative")
            values.append(value)
    elif isinstance(raw, list):
        values = [as_float(item) for item in raw]
        if len(values) != len(ranks):
            raise SchemaError("rank_arrival_us list length must match ranks")
        if any(value < 0.0 for value in values):
            raise SchemaError("rank_arrival_us values must be non-negative")
    else:
        raise SchemaError("rank_arrival_us must be an object or list")

    minimum = min(values) if values else 0.0
    return [max(0.0, value - minimum) for value in values]


def arrival_skew_us(offsets: Iterable[float]) -> float:
    values = list(offsets)
    if not values:
        return 0.0
    return max(values) - min(values)


def average_wait_us(offsets: Iterable[float]) -> float:
    values = list(offsets)
    if not values:
        return 0.0
    latest = max(values)
    return sum(latest - value for value in values) / len(values)


def median(values: Iterable[float]) -> float:
    return percentile(values, 50.0)


def percentile(values: Iterable[float], q: float) -> float:
    return percentile_from_sorted(sorted(as_float(value) for value in values), q)


def percentile_from_sorted(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latencies(values: Iterable[float]) -> JsonDict:
    data = sorted(as_float(value) for value in values)
    if not data:
        return {
            "count": 0,
            "median_us": 0.0,
            "p95_us": 0.0,
            "p99_us": 0.0,
            "max_us": 0.0,
            "mean_us": 0.0,
        }
    return {
        "count": len(data),
        "median_us": round(percentile_from_sorted(data, 50.0), 3),
        "p95_us": round(percentile_from_sorted(data, 95.0), 3),
        "p99_us": round(percentile_from_sorted(data, 99.0), 3),
        "max_us": round(data[-1], 3),
        "mean_us": round(sum(data) / len(data), 3),
    }


def validate_trace(trace: Mapping[str, Any], *, allow_partial_arrivals: bool = False) -> None:
    require_format(trace, TRACE_FORMAT, "trace")
    _require_optional_mapping(trace, "workload", "trace")
    _require_optional_mapping(trace, "system", "trace")
    events = trace.get("events")
    if not isinstance(events, list):
        raise SchemaError("trace must contain an 'events' list")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise SchemaError(f"trace event {index} must be an object")
        if "op" not in event:
            raise SchemaError(f"trace event {index} is missing 'op'")
        _validate_op(event.get("op"), f"trace event {index}", custom=event.get("custom_op") is True)
        for text_key in ("phase", "group"):
            if text_key in event:
                _validate_nonempty_string(event.get(text_key), f"trace event {index} {text_key}")
        if "bytes" not in event:
            raise SchemaError(f"trace event {index} is missing 'bytes'")
        if as_int(event.get("bytes")) <= 0:
            raise SchemaError(f"trace event {index} bytes must be positive")
        ranks = normalize_ranks(event.get("ranks"))
        if "rank_count" in event and as_int(event.get("rank_count")) != len(ranks):
            raise SchemaError(f"trace event {index} rank_count must match ranks")
        if allow_partial_arrivals and event.get("partial_rank_arrival") and isinstance(event.get("rank_arrival_us"), Mapping):
            _validate_arrival_keys(
                event.get("rank_arrival_us", {}),
                ranks,
                f"trace event {index} rank_arrival_us",
                allow_subset=True,
            )
            for value in event.get("rank_arrival_us", {}).values():
                if as_float(value) < 0.0:
                    raise SchemaError(f"trace event {index} rank_arrival_us values must be non-negative")
        else:
            offsets = normalize_arrival_offsets(event, ranks)
            if "arrival_skew_us" in event and event.get("rank_arrival_us") is not None:
                _validate_skew_matches_offsets(
                    as_float(event.get("arrival_skew_us")),
                    offsets,
                    f"trace event {index}",
                )
        for numeric_key in (
            "start_us",
            "gap_us",
            "compute_before_us",
            "compute_overlap_us",
            "compute_pressure",
            "observed_exposed_us",
        ):
            if numeric_key in event:
                if as_float(event.get(numeric_key)) < 0.0:
                    raise SchemaError(f"trace event {index} {numeric_key} must be non-negative")
        if "concurrent_groups" in event and as_int(event.get("concurrent_groups")) <= 0:
            raise SchemaError(f"trace event {index} concurrent_groups must be positive")


def validate_canary(canary: Mapping[str, Any]) -> None:
    require_format(canary, CANARY_FORMAT, "canary")
    _require_optional_mapping(canary, "workload", "canary")
    _require_optional_mapping(canary, "system", "canary")
    _require_optional_mapping(canary, "compiler", "canary")
    compiler = canary.get("compiler")
    if not isinstance(compiler, Mapping):
        raise SchemaError("canary must contain a compiler object")
    events = canary.get("events")
    if not isinstance(events, list):
        raise SchemaError("canary must contain an 'events' list")

    total_repeat = 0
    all_leaf_observed_flags: List[bool] = []
    actual_recursive_records = 0
    actual_approximate_records = 0
    actual_encoded_gap_total = 0.0
    actual_fidelity_maxima = {field: 0.0 for field in FIDELITY_ERROR_FIELDS}
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise SchemaError(f"canary event {index} must be an object")
        for key in ("op", "bytes", "ranks", "repeat"):
            if key not in event:
                raise SchemaError(f"canary event {index} is missing {key!r}")
        _validate_op(event.get("op"), f"canary event {index}", custom=event.get("custom_op") is True)
        for text_key in ("phase", "group"):
            if text_key not in event:
                raise SchemaError(f"canary event {index} is missing {text_key!r}")
            _validate_nonempty_string(event.get(text_key), f"canary event {index} {text_key}")
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
        if "digest" in source:
            _validate_sha256(source.get("digest"), f"canary event {index} source.digest")
        if "sampled_timing_records" in source and as_int(source.get("sampled_timing_records")) <= 0:
            raise SchemaError(f"canary event {index} source.sampled_timing_records must be positive")
        if "concurrent_groups" in event and as_int(event.get("concurrent_groups")) <= 0:
            raise SchemaError(f"canary event {index} concurrent_groups must be positive")

        ranks = normalize_ranks(event.get("ranks"))
        if "rank_count" in event and as_int(event.get("rank_count")) != len(ranks):
            raise SchemaError(f"canary event {index} rank_count must match ranks")
        if "rank_arrival_us" in event:
            _validate_arrival_keys(
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

        samples = event.get("timing_samples")
        if not isinstance(samples, list) or not samples:
            raise SchemaError(f"canary event {index} must contain non-empty timing_samples")
        weight_total = 0
        source_indices: List[int] = []
        intervals: List[tuple] = []
        for sample_index, sample in enumerate(samples):
            label = f"canary event {index} timing sample {sample_index}"
            if not isinstance(sample, Mapping):
                raise SchemaError(f"{label} must be an object")
            _validate_timing_record(sample, ranks, label, repeat=repeat)
            actual_recursive_records += 1
            actual_encoded_gap_total += _timing_record_gap_sum(sample, label)
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
        raise SchemaError("canary compiler.canary_events must match events")
    if "recursive_timing_records" in compiler and as_int(compiler.get("recursive_timing_records")) != actual_recursive_records:
        raise SchemaError("canary compiler.recursive_timing_records must match timing records")
    if "approximate_timing_records" in compiler and as_int(compiler.get("approximate_timing_records")) != actual_approximate_records:
        raise SchemaError("canary compiler.approximate_timing_records must match timing records")
    if "source_trace_sha256" in compiler:
        _validate_sha256(compiler.get("source_trace_sha256"), "canary compiler.source_trace_sha256")
    if "execution_semantic_sha256" in compiler:
        _validate_sha256(compiler.get("execution_semantic_sha256"), "canary compiler.execution_semantic_sha256")
        if compiler.get("execution_semantic_sha256") != canary_execution_sha256(canary):
            raise SchemaError("canary compiler.execution_semantic_sha256 does not match executable events")
    for integer_key in ("source_bytes", "canary_bytes", "timing_sample_limit"):
        if integer_key in compiler and as_int(compiler.get(integer_key)) < 0:
            raise SchemaError(f"canary compiler.{integer_key} must be non-negative")

    fidelity = compiler.get("fidelity")
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
            tolerance = 1e-6 if key == "max_pressure_error" else 1e-6
            if abs(as_float(fidelity.get(key)) - expected) > tolerance:
                raise SchemaError(f"canary compiler.fidelity.{key} does not match timing records")
        if "encoded_gap_total_us" in fidelity:
            if abs(as_float(fidelity.get("encoded_gap_total_us")) - actual_encoded_gap_total) > 1e-6:
                raise SchemaError("canary compiler.fidelity.encoded_gap_total_us does not match timing records")
        if all(key in fidelity for key in ("source_gap_total_us", "encoded_gap_total_us", "total_gap_error_us")):
            expected_error = abs(
                as_float(fidelity.get("source_gap_total_us"))
                - as_float(fidelity.get("encoded_gap_total_us"))
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

def validate_report(report: Mapping[str, Any]) -> None:
    require_format(report, REPORT_FORMAT, "report")
    _require_optional_mapping(report, "workload", "report")
    _require_optional_mapping(report, "system", "report")
    _require_optional_mapping(report, "canary_summary", "report")
    _require_optional_mapping(report, "backend", "report")

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SchemaError("report must contain a metrics object")
    required_metrics = {
        "count": "count",
        "median_us": "latency",
        "p95_us": "latency",
        "p99_us": "latency",
        "max_us": "latency",
        "mean_us": "latency",
        "arrival_skew_median_us": "latency",
        "arrival_skew_p95_us": "latency",
        "arrival_skew_max_us": "latency",
        "avg_rank_wait_median_us": "latency",
        "communication_hidden_pct": "percent",
    }
    for key, kind in required_metrics.items():
        if key not in metrics:
            raise SchemaError(f"report metrics missing {key!r}")
        if kind == "count":
            if as_int(metrics.get(key)) < 0:
                raise SchemaError("report metric count must be non-negative")
        elif kind == "percent":
            value = as_float(metrics.get(key))
            if not 0.0 <= value <= 100.0:
                raise SchemaError(f"report metric {key} must be between 0 and 100")
        elif as_float(metrics.get(key)) < 0.0:
            raise SchemaError(f"report metric {key} must be non-negative")
    _validate_latency_quantiles(metrics, "report metrics")

    canary = report.get("canary")
    if not isinstance(canary, Mapping) or not canary.get("sha256"):
        raise SchemaError("report must contain canary.sha256")
    _validate_sha256(canary.get("sha256"), "canary.sha256")
    if "execution_semantic_sha256" in canary:
        _validate_sha256(canary.get("execution_semantic_sha256"), "canary.execution_semantic_sha256")
    if "source_events" not in canary:
        raise SchemaError("report canary.source_events is required")
    source_events = as_int(canary.get("source_events"))
    if source_events < 0:
        raise SchemaError("report canary.source_events must be non-negative")

    model = report.get("simulation_model")
    if not isinstance(model, Mapping) or not isinstance(model.get("version"), str) or not model.get("version"):
        raise SchemaError("report must contain simulation_model.version")
    protocol = report.get("replay_protocol")
    if not isinstance(protocol, Mapping) or not protocol.get("sha256"):
        raise SchemaError("report must contain replay_protocol.sha256")
    required_protocol = ("model_name", "model_version", "seed", "iterations", "quantile_method", "bandwidth_unit")
    for key in required_protocol:
        if key not in protocol:
            raise SchemaError(f"report replay_protocol missing {key!r}")
    for key in ("model_name", "model_version", "quantile_method", "bandwidth_unit"):
        if not isinstance(protocol.get(key), str) or not protocol.get(key):
            raise SchemaError(f"report replay_protocol.{key} must be a non-empty string")
    if not isinstance(protocol.get("seed"), int) or isinstance(protocol.get("seed"), bool):
        raise SchemaError("report replay_protocol.seed must be an integer")
    if not isinstance(protocol.get("iterations"), int) or isinstance(protocol.get("iterations"), bool):
        raise SchemaError("report replay_protocol.iterations must be an integer")
    if as_int(protocol.get("iterations")) <= 0:
        raise SchemaError("report replay_protocol.iterations must be positive")
    _validate_sha256(protocol.get("sha256"), "replay_protocol.sha256")
    if replay_protocol_sha256(protocol) != protocol.get("sha256"):
        raise SchemaError("replay_protocol.sha256 does not match replay protocol fields")

    backend = report.get("backend", {})
    if not isinstance(backend, Mapping):
        raise SchemaError("report backend must be an object")
    _validate_report_backend(backend)
    for key in ("seed", "iterations", "bandwidth_unit"):
        if key in backend and backend.get(key) != protocol.get(key):
            raise SchemaError(f"report backend.{key} must match replay_protocol.{key}")
    if model.get("version") != protocol.get("model_version"):
        raise SchemaError("simulation_model.version must match replay_protocol.model_version")
    if model.get("name") and model.get("name") != protocol.get("model_name"):
        raise SchemaError("simulation_model.name must match replay_protocol.model_name")

    expected_count = source_events * as_int(protocol.get("iterations"))
    if as_int(metrics.get("count")) != expected_count:
        raise SchemaError("report metrics.count must match source events times iterations")

    breakdowns: Dict[str, List[Mapping[str, Any]]] = {}
    for breakdown_key in ("by_phase", "by_op"):
        rows = report.get(breakdown_key, [])
        if not isinstance(rows, list):
            raise SchemaError(f"report {breakdown_key} must be a list")
        if as_int(metrics.get("count")) > 0 and not rows:
            raise SchemaError(f"report {breakdown_key} must not be empty when metrics.count is positive")
        names = set()
        validated_rows: List[Mapping[str, Any]] = []
        for row_index, row in enumerate(rows):
            label = f"report {breakdown_key} row {row_index}"
            if not isinstance(row, Mapping):
                raise SchemaError(f"{label} must be an object")
            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise SchemaError(f"{label} name must be a non-empty string")
            if name in names:
                raise SchemaError(f"report {breakdown_key} names must be unique")
            names.add(name)
            _validate_breakdown_row(row, label)
            validated_rows.append(row)
        row_total = sum(as_int(row.get("count")) for row in validated_rows)
        if rows and row_total != as_int(metrics.get("count")):
            raise SchemaError(f"report {breakdown_key} counts must sum to metrics.count")
        _reconcile_breakdown_summary(metrics, validated_rows, breakdown_key)
        breakdowns[breakdown_key] = validated_rows

    calibration = report.get("calibration")
    if calibration is not None:
        _validate_calibration(calibration)
        if as_int(calibration.get("count")) != as_int(metrics.get("count")):
            raise SchemaError("report calibration.count must match metrics.count")

    samples = report.get("samples")
    if samples is not None:
        if not isinstance(samples, list):
            raise SchemaError("report samples must be a list")
        if len(samples) != as_int(metrics.get("count")):
            raise SchemaError("report sample count must match metrics.count")
        _reconcile_report_samples(report, samples, breakdowns)


def _validate_breakdown_row(row: Mapping[str, Any], label: str) -> None:
    required = ("count", "median_us", "p95_us", "p99_us", "max_us", "mean_us")
    for key in required:
        if key not in row:
            raise SchemaError(f"{label} missing {key!r}")
    if as_int(row.get("count")) <= 0:
        raise SchemaError(f"{label} count must be positive")
    for key in required[1:]:
        if as_float(row.get(key)) < 0.0:
            raise SchemaError(f"{label} {key} must be non-negative")
    _validate_latency_quantiles(row, label)


def _reconcile_breakdown_summary(
    overall: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], label: str
) -> None:
    if not rows:
        return
    total_count = sum(as_int(row.get("count")) for row in rows)
    weighted_mean = sum(
        as_float(row.get("mean_us")) * as_int(row.get("count")) for row in rows
    ) / total_count
    if abs(weighted_mean - as_float(overall.get("mean_us"))) > 0.0021:
        raise SchemaError(f"report {label} weighted mean must match metrics.mean_us")
    row_max = max(as_float(row.get("max_us")) for row in rows)
    if abs(row_max - as_float(overall.get("max_us"))) > 0.0011:
        raise SchemaError(f"report {label} maximum must match metrics.max_us")


def _validate_calibration(calibration: Any) -> None:
    if not isinstance(calibration, Mapping):
        raise SchemaError("report calibration must be an object")
    required = (
        "signal",
        "count",
        "mean_absolute_error_us",
        "median_absolute_error_us",
        "p95_absolute_error_us",
        "max_absolute_error_us",
        "mean_bias_us",
        "mean_absolute_percentage_error_pct",
        "percentage_count",
    )
    for key in required:
        if key not in calibration:
            raise SchemaError(f"report calibration missing {key!r}")
    if calibration.get("signal") != "observed_exposed_us":
        raise SchemaError("report calibration.signal is unsupported")
    count = as_int(calibration.get("count"))
    percentage_count = as_int(calibration.get("percentage_count"))
    if count <= 0 or percentage_count < 0 or percentage_count > count:
        raise SchemaError("report calibration counts are invalid")
    for key in (
        "mean_absolute_error_us",
        "median_absolute_error_us",
        "p95_absolute_error_us",
        "max_absolute_error_us",
        "mean_absolute_percentage_error_pct",
    ):
        if as_float(calibration.get(key)) < 0.0:
            raise SchemaError(f"report calibration {key} must be non-negative")
    median_error = as_float(calibration.get("median_absolute_error_us"))
    p95_error = as_float(calibration.get("p95_absolute_error_us"))
    max_error = as_float(calibration.get("max_absolute_error_us"))
    if not median_error <= p95_error <= max_error:
        raise SchemaError("report calibration absolute-error quantiles are unordered")
    as_float(calibration.get("mean_bias_us"))


def _validate_report_backend(backend: Mapping[str, Any]) -> None:
    if "bandwidth_gbps" in backend and as_float(backend.get("bandwidth_gbps")) <= 0.0:
        raise SchemaError("report backend.bandwidth_gbps must be positive")
    if "latency_floor_us" in backend and as_float(backend.get("latency_floor_us")) < 0.0:
        raise SchemaError("report backend.latency_floor_us must be non-negative")
    if "compute_pressure" in backend and as_float(backend.get("compute_pressure")) < 0.0:
        raise SchemaError("report backend.compute_pressure must be non-negative")
    if "overlap_efficiency" in backend:
        overlap_efficiency = as_float(backend.get("overlap_efficiency"))
        if not 0.0 <= overlap_efficiency <= 1.0:
            raise SchemaError("report backend.overlap_efficiency must be between 0 and 1")


def _reconcile_report_samples(
    report: Mapping[str, Any],
    samples: Sequence[Any],
    breakdowns: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    required_sample_keys = {
        "index",
        "iteration",
        "phase",
        "op",
        "group",
        "first_arrival_us",
        "last_arrival_us",
        "collective_start_us",
        "completion_us",
        "total_us",
        "hidden_us",
        "exposed_us",
        "arrival_skew_us",
        "avg_rank_wait_us",
        "compute_overlap_us",
        "collective_us",
        "queue_wait_us",
    }
    exposed: List[float] = []
    skew: List[float] = []
    wait: List[float] = []
    hidden_total = 0.0
    total_sum = 0.0
    phase_values: Dict[str, List[float]] = {}
    op_values: Dict[str, List[float]] = {}
    observed_flags: List[bool] = []
    observed_values: List[float] = []
    seen_indices: List[int] = []
    group_available: Dict[tuple, float] = {}
    iterations = as_int(report.get("replay_protocol", {}).get("iterations"), 1)
    source_events = as_int(report.get("canary", {}).get("source_events"))
    overlap_efficiency = as_float(report.get("backend", {}).get("overlap_efficiency"), 0.0)
    if not 0.0 <= overlap_efficiency <= 1.0:
        raise SchemaError("report backend.overlap_efficiency must be between 0 and 1")

    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise SchemaError(f"report sample {index} must be an object")
        missing = sorted(required_sample_keys - set(sample.keys()))
        if missing:
            raise SchemaError(f"report sample {index} missing {missing[0]!r}")
        if not isinstance(sample.get("phase"), str) or not sample.get("phase"):
            raise SchemaError(f"report sample {index} phase must be a non-empty string")
        if not isinstance(sample.get("op"), str) or not sample.get("op"):
            raise SchemaError(f"report sample {index} op must be a non-empty string")
        if not isinstance(sample.get("group"), str) or not sample.get("group"):
            raise SchemaError(f"report sample {index} group must be a non-empty string")
        scheduler_resource = sample.get("scheduler_resource", sample.get("group"))
        if not isinstance(scheduler_resource, str) or not scheduler_resource:
            raise SchemaError(f"report sample {index} scheduler_resource must be a non-empty string")
        sequence_index = as_int(sample.get("index"))
        iteration = as_int(sample.get("iteration"))
        if sequence_index != index:
            raise SchemaError("report sample indices must be contiguous and ordered")
        if iteration < 0 or iteration >= iterations:
            raise SchemaError(f"report sample {index} iteration is outside replay_protocol.iterations")
        if source_events > 0 and iteration != index // source_events:
            raise SchemaError(f"report sample {index} iteration does not match replay partitioning")
        seen_indices.append(sequence_index)
        for key in required_sample_keys - {"index", "iteration", "phase", "op", "group"}:
            if as_float(sample.get(key)) < 0.0:
                raise SchemaError(f"report sample {index} {key} must be non-negative")
        first_arrival_us = as_float(sample.get("first_arrival_us"))
        last_arrival_us = as_float(sample.get("last_arrival_us"))
        arrival_skew_us = as_float(sample.get("arrival_skew_us"))
        avg_rank_wait_us = as_float(sample.get("avg_rank_wait_us"))
        collective_start_us = as_float(sample.get("collective_start_us"))
        queue_wait_us = as_float(sample.get("queue_wait_us"))
        collective_us = as_float(sample.get("collective_us"))
        completion_us = as_float(sample.get("completion_us"))
        total_us = as_float(sample.get("total_us"))
        hidden_us = as_float(sample.get("hidden_us"))
        exposed_us = as_float(sample.get("exposed_us"))
        compute_overlap_us = as_float(sample.get("compute_overlap_us"))
        tolerance = 0.0051
        if abs((last_arrival_us - first_arrival_us) - arrival_skew_us) > tolerance:
            raise SchemaError(f"report sample {index} last_arrival_us must match arrival_skew_us")
        if avg_rank_wait_us > arrival_skew_us + tolerance:
            raise SchemaError(f"report sample {index} avg_rank_wait_us must not exceed arrival_skew_us")
        group_key = (iteration, scheduler_resource)
        expected_start = max(last_arrival_us, group_available.get(group_key, 0.0))
        if abs(collective_start_us - expected_start) > tolerance:
            raise SchemaError(f"report sample {index} collective_start_us is inconsistent")
        if abs(queue_wait_us - (collective_start_us - last_arrival_us)) > tolerance:
            raise SchemaError(f"report sample {index} queue_wait_us is inconsistent")
        if abs(completion_us - (collective_start_us + collective_us)) > tolerance:
            raise SchemaError(f"report sample {index} completion_us is inconsistent")
        if abs(total_us - (arrival_skew_us + queue_wait_us + collective_us)) > tolerance:
            raise SchemaError(f"report sample {index} total_us decomposition is inconsistent")
        if hidden_us > total_us or abs((hidden_us + exposed_us) - total_us) > 0.0021:
            raise SchemaError(f"report sample {index} hidden_us + exposed_us must equal total_us")
        maximum_hideable_us = min(total_us, max(0.0, compute_overlap_us) * overlap_efficiency)
        if abs(hidden_us - maximum_hideable_us) > tolerance:
            raise SchemaError(f"report sample {index} hidden_us does not match deterministic overlap model")
        exposed.append(exposed_us)
        skew.append(arrival_skew_us)
        wait.append(avg_rank_wait_us)
        hidden_total += hidden_us
        total_sum += total_us
        group_available[group_key] = completion_us
        phase_values.setdefault(str(sample.get("phase")), []).append(exposed_us)
        op_values.setdefault(str(sample.get("op")), []).append(exposed_us)
        has_observed = "observed_exposed_us" in sample
        observed_flags.append(has_observed)
        if has_observed:
            observed = as_float(sample.get("observed_exposed_us"))
            if observed < 0.0:
                raise SchemaError(f"report sample {index} observed_exposed_us must be non-negative")
            observed_values.append(observed)

    if seen_indices != list(range(len(samples))):
        raise SchemaError("report sample indices must be unique and contiguous")

    metrics = report.get("metrics", {})
    expected_metrics = summarize_latencies(exposed)
    ordered_skew = sorted(skew)
    ordered_wait = sorted(wait)
    hidden_pct = (hidden_total / total_sum * 100.0) if total_sum else 0.0
    expected_metrics.update(
        {
            "arrival_skew_median_us": round(percentile_from_sorted(ordered_skew, 50.0), 3),
            "arrival_skew_p95_us": round(percentile_from_sorted(ordered_skew, 95.0), 3),
            "arrival_skew_max_us": round(ordered_skew[-1], 3) if ordered_skew else 0.0,
            "avg_rank_wait_median_us": round(percentile_from_sorted(ordered_wait, 50.0), 3),
            "communication_hidden_pct": round(hidden_pct, 2),
        }
    )
    for key, expected in expected_metrics.items():
        tolerance = 0.02 if key == "communication_hidden_pct" else 0.0021
        if abs(as_float(metrics.get(key)) - as_float(expected)) > tolerance:
            raise SchemaError(f"report metrics.{key} does not match included samples")

    for breakdown_key, values_by_name in (("by_phase", phase_values), ("by_op", op_values)):
        expected_rows: Dict[str, JsonDict] = {}
        for name, values in values_by_name.items():
            row = {"name": name}
            row.update(summarize_latencies(values))
            expected_rows[name] = row
        actual_rows = {str(row.get("name")): row for row in breakdowns.get(breakdown_key, [])}
        if set(actual_rows) != set(expected_rows):
            raise SchemaError(f"report {breakdown_key} names do not match included samples")
        for name, expected_row in expected_rows.items():
            actual_row = actual_rows[name]
            for key in ("count", "median_us", "p95_us", "p99_us", "max_us", "mean_us"):
                if key == "count":
                    if as_int(actual_row.get(key)) != as_int(expected_row.get(key)):
                        raise SchemaError(f"report {breakdown_key} row {name!r} count does not match samples")
                elif abs(as_float(actual_row.get(key)) - as_float(expected_row.get(key))) > 0.0021:
                    raise SchemaError(f"report {breakdown_key} row {name!r} {key} does not match samples")

    calibration = report.get("calibration")
    if any(observed_flags) and not all(observed_flags):
        raise SchemaError("report samples must either all include observed_exposed_us or none")
    if all(observed_flags) and observed_flags:
        if calibration is None:
            raise SchemaError("report with observed samples must contain calibration")
        errors = [modeled - observed for modeled, observed in zip(exposed, observed_values)]
        absolute = sorted(abs(value) for value in errors)
        percentage = [
            abs(modeled - observed) / observed * 100.0
            for modeled, observed in zip(exposed, observed_values)
            if observed > 0.0
        ]
        expected_calibration = {
            "count": len(errors),
            "mean_absolute_error_us": round(sum(absolute) / len(absolute), 3),
            "median_absolute_error_us": round(percentile_from_sorted(absolute, 50.0), 3),
            "p95_absolute_error_us": round(percentile_from_sorted(absolute, 95.0), 3),
            "max_absolute_error_us": round(absolute[-1], 3),
            "mean_bias_us": round(sum(errors) / len(errors), 3),
            "mean_absolute_percentage_error_pct": round(sum(percentage) / len(percentage), 3)
            if percentage
            else 0.0,
            "percentage_count": len(percentage),
        }
        for key, expected in expected_calibration.items():
            if key in {"count", "percentage_count"}:
                if as_int(calibration.get(key)) != expected:
                    raise SchemaError(f"report calibration.{key} does not match samples")
            elif abs(as_float(calibration.get(key)) - as_float(expected)) > 0.01:
                raise SchemaError(f"report calibration.{key} does not match samples")
    elif calibration is not None:
        raise SchemaError("report calibration requires observed_exposed_us samples")

def merge_metadata(base: Optional[Mapping[str, Any]], override: Optional[Mapping[str, Any]]) -> JsonDict:
    merged: JsonDict = dict(base or {})
    merged.update(dict(override or {}))
    return merged


def clean_private_keys(data: MutableMapping[str, Any]) -> JsonDict:
    return {key: value for key, value in data.items() if not key.startswith("_")}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"constant {value!r} is not allowed")


def _require_optional_mapping(data: Mapping[str, Any], key: str, label: str) -> None:
    if key in data and not isinstance(data.get(key), Mapping):
        raise SchemaError(f"{label} {key} must be an object")


def _validate_op(value: Any, label: str, *, custom: bool) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} op must be a non-empty string")
    if value not in SUPPORTED_OPS and not custom:
        raise SchemaError(f"{label} op {value!r} is unsupported; set custom_op=true for custom operations")


def _validate_nonempty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} must be a non-empty string")


def _validate_arrival_keys(raw: Any, ranks: List[int], label: str, *, allow_subset: bool) -> None:
    if not isinstance(raw, Mapping):
        raise SchemaError(f"{label} must be an object")
    expected = {str(rank) for rank in ranks}
    actual = {str(key) for key in raw.keys()}
    if allow_subset:
        if not actual:
            raise SchemaError(f"{label} must include at least one rank")
        unexpected = actual - expected
        if unexpected:
            raise SchemaError(f"{label} contains ranks outside ranks")
        return
    if actual != expected:
        raise SchemaError(f"{label} keys must exactly match ranks")


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
        _validate_skew_matches_offsets(as_float(sample.get("arrival_skew_us")), numeric_offsets, label)
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
    approximation = sample.get("approximation")
    if approximation is not None and approximation != "bounded_interval":
        raise SchemaError(f"{label} approximation is unsupported")
    if approximation == "bounded_interval":
        _validate_bounded_interval_evidence(sample, label)
    if "weight" in sample and "source_start" in sample and "source_end" in sample:
        expected_weight = as_int(sample.get("source_end")) - as_int(sample.get("source_start")) + 1
        if as_int(sample.get("weight")) != expected_weight:
            raise SchemaError(f"{label} weight must match source interval length")
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
        "gap_sum_us",
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
            _validate_skew_matches_offsets(as_float(event.get("arrival_skew_us")), event_offsets, label)
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


def _sample_interval(sample: Mapping[str, Any], label: str) -> tuple:
    if "source_start" in sample or "source_end" in sample:
        start = as_int(sample.get("source_start"))
        end = as_int(sample.get("source_end"))
    elif "source_index" in sample:
        start = end = as_int(sample.get("source_index"))
    else:
        raise SchemaError(f"{label} source interval is required")
    return start, end


def _validate_intervals(intervals: List[tuple], label: str) -> None:
    previous_end = -1
    for start, end in intervals:
        if start <= previous_end:
            raise SchemaError(f"{label} source intervals must be ordered and non-overlapping")
        previous_end = end


def _validate_skew_matches_offsets(skew_us: float, offsets: Iterable[float], label: str) -> None:
    computed = arrival_skew_us(offsets)
    if abs(skew_us - computed) > 0.001:
        raise SchemaError(f"{label} arrival_skew_us must match arrival_offsets_us")


def _validate_latency_quantiles(metrics: Mapping[str, Any], label: str) -> None:
    keys = ("median_us", "p95_us", "p99_us", "max_us")
    if not all(key in metrics for key in keys):
        return
    median_us, p95_us, p99_us, max_us = (as_float(metrics.get(key)) for key in keys)
    if not median_us <= p95_us <= p99_us <= max_us:
        raise SchemaError(f"{label} must satisfy median_us <= p95_us <= p99_us <= max_us")
    if "mean_us" in metrics and as_float(metrics.get("mean_us")) > max_us:
        raise SchemaError(f"{label} mean_us must be no greater than max_us")
    skew_keys = ("arrival_skew_median_us", "arrival_skew_p95_us", "arrival_skew_max_us")
    if all(key in metrics for key in skew_keys):
        skew_median, skew_p95, skew_max = (as_float(metrics.get(key)) for key in skew_keys)
        if not skew_median <= skew_p95 <= skew_max:
            raise SchemaError(f"{label} must satisfy arrival_skew_median_us <= arrival_skew_p95_us <= arrival_skew_max_us")


def _validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SchemaError(f"{label} must be a 64-character lowercase SHA-256 hex digest")
