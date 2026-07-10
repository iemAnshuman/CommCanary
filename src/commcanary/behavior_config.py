"""Private normalization for behavioral replay configuration mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterator, List, Mapping, Optional, Sequence, Tuple, TypedDict

from .schema import JsonDict, SchemaError, as_float, as_int

_MAX_BEHAVIOR_CONFIGURATIONS = 32
_BEHAVIORAL_RANKING_METRICS = ("median_us", "p95_us", "p99_us", "mean_us")


@dataclass(frozen=True)
class _BehaviorConfiguration(Mapping[str, Any]):
    """One immutable, normalized behavioral replay configuration."""

    name: str
    bandwidth_gbps: float
    latency_floor_us: float
    compute_pressure: float
    overlap_efficiency: float
    iterations: int
    seed: int
    max_replay_events: int

    _FIELDS: ClassVar[Tuple[str, ...]] = (
        "name",
        "bandwidth_gbps",
        "latency_floor_us",
        "compute_pressure",
        "overlap_efficiency",
        "iterations",
        "seed",
        "max_replay_events",
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self._FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._FIELDS)

    def __len__(self) -> int:
        return len(self._FIELDS)


class _BehaviorReplayArguments(TypedDict):
    bandwidth_gbps: float
    latency_floor_us: float
    compute_pressure: float
    overlap_efficiency: float
    iterations: int
    seed: int
    max_replay_events: int


_REPLAY_DEFAULTS: _BehaviorReplayArguments = {
    "bandwidth_gbps": 55.0,
    "latency_floor_us": 7.5,
    "compute_pressure": 0.55,
    "overlap_efficiency": 0.72,
    "iterations": 1,
    "seed": 7,
    "max_replay_events": 1_000_000,
}

_DEFAULT_BEHAVIOR_CONFIGURATIONS = (
    {"name": "baseline"},
    {"name": "low_latency", "latency_floor_us": 3.5},
    {"name": "high_bandwidth", "bandwidth_gbps": 110.0, "latency_floor_us": 10.0},
    {"name": "overlap_friendly", "latency_floor_us": 9.0, "overlap_efficiency": 0.95},
    {"name": "congested", "bandwidth_gbps": 28.0, "compute_pressure": 0.95},
)

_REPLAY_ARGUMENT_KEYS = tuple(_REPLAY_DEFAULTS)
_ALLOWED_CONFIGURATION_KEYS = frozenset(("name", *_REPLAY_ARGUMENT_KEYS))


def _normalize_behavior_configurations(
    configurations: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[_BehaviorConfiguration, ...]:
    """Validate and detach public mapping inputs into canonical replay configs."""

    raw_configurations: Any = (
        _DEFAULT_BEHAVIOR_CONFIGURATIONS if configurations is None else configurations
    )
    if isinstance(raw_configurations, (str, bytes, bytearray)) or not isinstance(
        raw_configurations, Sequence
    ):
        raise SchemaError("behavior configurations must be a sequence of mappings")
    count = len(raw_configurations)
    if count < 2:
        raise SchemaError("behavior configurations must contain at least two configurations")
    if count > _MAX_BEHAVIOR_CONFIGURATIONS:
        raise SchemaError(
            "behavior configurations must contain at most "
            f"{_MAX_BEHAVIOR_CONFIGURATIONS} configurations"
        )

    normalized: List[_BehaviorConfiguration] = []
    names = set()
    for index, raw_config in enumerate(raw_configurations):
        if not isinstance(raw_config, Mapping):
            raise SchemaError(f"behavior configuration {index} must be a mapping")
        unknown_keys = [key for key in raw_config if key not in _ALLOWED_CONFIGURATION_KEYS]
        if unknown_keys:
            rendered = ", ".join(sorted(repr(key) for key in unknown_keys))
            raise SchemaError(
                f"behavior configuration {index} contains unknown keys: {rendered}"
            )

        raw_name = raw_config.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise SchemaError(
                f"behavior configuration {index} name must be a non-empty string"
            )
        name = raw_name.strip()
        if name in names:
            raise SchemaError(
                f"behavior configurations must use unique configuration names; duplicate {name!r}"
            )
        names.add(name)

        merged: JsonDict = {**_REPLAY_DEFAULTS, **dict(raw_config), "name": name}
        config = _BehaviorConfiguration(
            name=name,
            bandwidth_gbps=_positive_float(
                merged["bandwidth_gbps"], f"behavior configuration {name!r} bandwidth_gbps"
            ),
            latency_floor_us=_non_negative_float(
                merged["latency_floor_us"],
                f"behavior configuration {name!r} latency_floor_us",
            ),
            compute_pressure=_non_negative_float(
                merged["compute_pressure"],
                f"behavior configuration {name!r} compute_pressure",
            ),
            overlap_efficiency=_overlap_efficiency(
                merged["overlap_efficiency"],
                f"behavior configuration {name!r} overlap_efficiency",
            ),
            iterations=_positive_int(
                merged["iterations"], f"behavior configuration {name!r} iterations"
            ),
            seed=_integer(merged["seed"], f"behavior configuration {name!r} seed"),
            max_replay_events=_positive_int(
                merged["max_replay_events"],
                f"behavior configuration {name!r} max_replay_events",
            ),
        )
        normalized.append(config)
    return tuple(normalized)


def _behavioral_replay_args(
    config: Mapping[str, Any],
) -> _BehaviorReplayArguments:
    """Detach replay keyword arguments from a normalized configuration."""

    return {
        "bandwidth_gbps": as_float(config["bandwidth_gbps"]),
        "latency_floor_us": as_float(config["latency_floor_us"]),
        "compute_pressure": as_float(config["compute_pressure"]),
        "overlap_efficiency": as_float(config["overlap_efficiency"]),
        "iterations": as_int(config["iterations"]),
        "seed": as_int(config["seed"]),
        "max_replay_events": as_int(config["max_replay_events"]),
    }


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed: float = as_float(value)
    except SchemaError as exc:
        raise SchemaError(f"{label} is invalid: {exc}") from exc
    if parsed <= 0.0:
        raise SchemaError(f"{label} must be positive")
    return parsed


def _non_negative_float(value: Any, label: str) -> float:
    try:
        parsed: float = as_float(value)
    except SchemaError as exc:
        raise SchemaError(f"{label} is invalid: {exc}") from exc
    if parsed < 0.0:
        raise SchemaError(f"{label} must be non-negative")
    return parsed


def _overlap_efficiency(value: Any, label: str) -> float:
    try:
        parsed: float = as_float(value)
    except SchemaError as exc:
        raise SchemaError(f"{label} is invalid: {exc}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise SchemaError(f"{label} must be between 0 and 1")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    parsed = _integer(value, label)
    if parsed < 1:
        raise SchemaError(f"{label} must be at least 1")
    return parsed


def _integer(value: Any, label: str) -> int:
    try:
        parsed: int = as_int(value)
    except SchemaError as exc:
        raise SchemaError(f"{label} is invalid: {exc}") from exc
    return parsed
