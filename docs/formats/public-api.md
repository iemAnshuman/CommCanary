# Characterized Python imports and types

The top-level package is the stable tier for format IDs, core errors and limits,
artifact validation/loading, compile/replay/compare, the three verification
operations, package version, and the exact-version capability query. Capture,
ecosystem adapters, and research operations remain in documented submodules.

The following Python 3.9+ example is executable and checked by the contract
suite and by strict mypy. `JsonDict` is the current wire-object alias; it is not
a promise that arbitrary unvalidated dictionaries are canonical artifacts.

<!-- golden-example:start -->
```python
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
```
<!-- golden-example:end -->

`format_capabilities()` returns an immutable tuple of frozen
`FormatCapability` values. Each value names one exact format ID, its repository
schema path, read/write support, migration support, and whether a runtime
semantic validator exists. The query reports capabilities; it does not load a
schema or imply that schema shape establishes semantic assurance.

The top-level names are the supported stable import tier. Callers should not
import underscore-prefixed implementation details; capture, Kineto/PARAM
interop, research baselines, and reduction keep their explicit submodule tiers.
