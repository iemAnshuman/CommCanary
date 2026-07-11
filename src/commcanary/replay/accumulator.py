"""Replay sample accumulation and the single wire-rounding boundary."""

from __future__ import annotations

import copy
from array import array
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ..artifacts.wire import JsonDict, as_float
from ..statistics import percentile_from_sorted, summarize_latencies


@dataclass(frozen=True)
class _ReplaySampleValues:
    """One detached full-precision sample shared by every report consumer."""

    payload: JsonDict
    exposed_us: float
    arrival_skew_us: float
    avg_rank_wait_us: float
    hidden_us: float
    total_us: float
    phase: str
    op: str
    observed_exposed_us: Optional[float]

    @classmethod
    def from_mapping(cls, sample: Mapping[str, Any]) -> "_ReplaySampleValues":
        payload = copy.deepcopy(dict(sample))
        return cls(
            payload=payload,
            exposed_us=as_float(payload.get("exposed_us")),
            arrival_skew_us=as_float(payload.get("arrival_skew_us")),
            avg_rank_wait_us=as_float(payload.get("avg_rank_wait_us")),
            hidden_us=as_float(payload.get("hidden_us")),
            total_us=as_float(payload.get("total_us")),
            phase=str(payload.get("phase", "unknown")),
            op=str(payload.get("op", "unknown")),
            observed_exposed_us=(
                as_float(payload.get("observed_exposed_us")) if "observed_exposed_us" in payload else None
            ),
        )

    def to_wire(self) -> JsonDict:
        """Apply the sole presentation rounding boundary."""

        return _round_sample(self.payload)


class ReplayAccumulator:
    def __init__(self, *, include_samples: bool) -> None:
        self.include_samples = include_samples
        self.samples: List[JsonDict] = []
        self.exposed = array("d")
        self.skew = array("d")
        self.wait = array("d")
        self.hidden_total = 0.0
        self.total = 0.0
        self.phase_values: Dict[str, array[float]] = {}
        self.op_values: Dict[str, array[float]] = {}
        self.observed = array("d")
        self.modeled_for_observed = array("d")

    def add(self, sample: Mapping[str, Any]) -> None:
        values = _ReplaySampleValues.from_mapping(sample)
        self.exposed.append(values.exposed_us)
        self.skew.append(values.arrival_skew_us)
        self.wait.append(values.avg_rank_wait_us)
        self.hidden_total += values.hidden_us
        self.total += values.total_us
        self.phase_values.setdefault(values.phase, array("d")).append(values.exposed_us)
        self.op_values.setdefault(values.op, array("d")).append(values.exposed_us)
        if values.observed_exposed_us is not None:
            self.observed.append(values.observed_exposed_us)
            self.modeled_for_observed.append(values.exposed_us)
        if self.include_samples:
            self.samples.append(values.to_wire())

    def metrics(self) -> JsonDict:
        skew = sorted(self.skew)
        wait = sorted(self.wait)
        hidden_pct = (self.hidden_total / self.total * 100.0) if self.total else 0.0
        result = summarize_latencies(self.exposed)
        result.update(
            {
                "arrival_skew_median_us": round(percentile_from_sorted(skew, 50.0), 3),
                "arrival_skew_p95_us": round(percentile_from_sorted(skew, 95.0), 3),
                "arrival_skew_max_us": round(skew[-1], 3) if skew else 0.0,
                "avg_rank_wait_median_us": round(percentile_from_sorted(wait, 50.0), 3),
                "communication_hidden_pct": round(hidden_pct, 2),
            }
        )
        return result

    def breakdown(self, key: str) -> List[JsonDict]:
        buckets = self.phase_values if key == "phase" else self.op_values
        rows: List[JsonDict] = []
        for label, values in sorted(buckets.items()):
            row: JsonDict = {"name": label}
            row.update(summarize_latencies(values))
            rows.append(row)
        return rows

    def calibration(self) -> Optional[JsonDict]:
        if not self.observed:
            return None
        errors = [model - observed for model, observed in zip(self.modeled_for_observed, self.observed)]
        absolute = sorted(abs(value) for value in errors)
        percentage = [
            abs(model - observed) / observed * 100.0
            for model, observed in zip(self.modeled_for_observed, self.observed)
            if observed > 0.0
        ]
        return {
            "signal": "observed_exposed_us",
            "count": len(errors),
            "mean_absolute_error_us": round(sum(absolute) / len(absolute), 3),
            "median_absolute_error_us": round(percentile_from_sorted(absolute, 50.0), 3),
            "p95_absolute_error_us": round(percentile_from_sorted(absolute, 95.0), 3),
            "max_absolute_error_us": round(absolute[-1], 3),
            "mean_bias_us": round(sum(errors) / len(errors), 3),
            "mean_absolute_percentage_error_pct": round(sum(percentage) / len(percentage), 3) if percentage else 0.0,
            "percentage_count": len(percentage),
        }


def _round_sample(sample: Mapping[str, Any]) -> JsonDict:
    total_us = round(as_float(sample.get("total_us")), 3)
    hidden_us = min(total_us, round(as_float(sample.get("hidden_us")), 3))
    rounded: JsonDict = {}
    for key, value in sample.items():
        if isinstance(value, float):
            rounded[key] = round(value, 9) if key == "gap_us" else round(value, 3)
        else:
            rounded[key] = value
    rounded["total_us"] = total_us
    rounded["hidden_us"] = hidden_us
    rounded["exposed_us"] = round(total_us - hidden_us, 3)
    return rounded
