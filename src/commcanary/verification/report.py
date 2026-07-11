"""Independent model-recomputation verification for replay reports."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from ..artifacts.canary import validate_canary
from ..artifacts.report import validate_report
from ..artifacts.wire import JsonDict, as_float, as_int
from ..formats import REPORT_VERIFICATION_FORMAT
from ..replay.core import DEFAULT_MAX_REPLAY_EVENTS, replay_canary
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits


def verify_report_against_canary(
    report: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> JsonDict:
    """Recompute a replay report from its declared backend settings."""

    validate_report(report, limits=limits)
    validate_canary(canary, limits=limits)
    backend = report.get("backend", {})
    protocol = report.get("replay_protocol", {})
    recomputed = replay_canary(
        canary,
        backend_label=str(backend.get("label", "simulated-nccl")),
        bandwidth_gbps=as_float(backend.get("bandwidth_gbps"), 55.0),
        latency_floor_us=as_float(backend.get("latency_floor_us"), 7.5),
        compute_pressure=as_float(backend.get("compute_pressure"), 0.55),
        overlap_efficiency=as_float(backend.get("overlap_efficiency"), 0.72),
        iterations=as_int(protocol.get("iterations")),
        seed=as_int(protocol.get("seed")),
        include_samples="samples" in report,
        max_replay_events=as_int(protocol.get("max_replay_events"), DEFAULT_MAX_REPLAY_EVENTS),
        ablations=protocol.get("ablations", []),
        limits=limits,
    )
    checks = [
        _report_verification_check("canary", recomputed.get("canary"), report.get("canary")),
        _report_verification_check(
            "simulation_model",
            recomputed.get("simulation_model"),
            report.get("simulation_model"),
        ),
        _report_verification_check(
            "replay_protocol",
            recomputed.get("replay_protocol"),
            report.get("replay_protocol"),
        ),
        _report_verification_check("backend", recomputed.get("backend"), report.get("backend")),
        _report_verification_check("workload", recomputed.get("workload"), report.get("workload")),
        _report_verification_check(
            "canary_summary",
            recomputed.get("canary_summary"),
            report.get("canary_summary"),
        ),
        _report_verification_check("metrics", recomputed.get("metrics"), report.get("metrics")),
        _report_verification_check("by_phase", recomputed.get("by_phase"), report.get("by_phase")),
        _report_verification_check("by_op", recomputed.get("by_op"), report.get("by_op")),
        _report_verification_check("calibration", recomputed.get("calibration"), report.get("calibration")),
    ]
    if "samples" in report:
        checks.append(_report_verification_check("samples", recomputed.get("samples"), report.get("samples")))
    passed = all(check["status"] == "pass" for check in checks)
    return {
        "format": REPORT_VERIFICATION_FORMAT,
        "status": "model_recomputed" if passed else "failed",
        "assurance_state": "model_recomputed" if passed else "structurally_valid",
        "checks": checks,
    }


def _report_verification_check(name: str, expected: Any, actual: Any) -> JsonDict:
    check: JsonDict = {"name": name, "status": "pass" if expected == actual else "fail"}
    if check["status"] == "fail":
        check["expected"] = copy.deepcopy(expected)
        check["actual"] = copy.deepcopy(actual)
    return check


__all__ = ["verify_report_against_canary"]
