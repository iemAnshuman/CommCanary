from __future__ import annotations

from commcanary import (
    CANARY_FORMAT,
    REPORT_FORMAT,
    TRACE_FORMAT,
    FormatCapability,
    JsonDict,
    compile_trace,
    format_capabilities,
    replay_canary,
    validate_report,
    validate_trace,
)


def compile_and_replay(trace: JsonDict) -> JsonDict:
    validate_trace(trace)
    canary = compile_trace(trace, allow_empty=True)
    assert canary["format"] == CANARY_FORMAT
    report = replay_canary(canary)
    validate_report(report)
    assert report["format"] == REPORT_FORMAT
    return report


trace: JsonDict = {"format": TRACE_FORMAT, "workload": {"name": "typed-example"}, "events": []}
capabilities: tuple[FormatCapability, ...] = format_capabilities()
report = compile_and_replay(trace)
assert len(capabilities) == 7
assert report["metrics"]["count"] == 0
