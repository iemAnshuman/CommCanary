# Python API stability

CommCanary has three API tiers. A name being importable is not by itself a
stability promise.

## Stable top-level API

The names in `commcanary.__all__` are the supported library surface for the 0.3
line:

```python
from commcanary import (
    ResourceLimits,
    compare_reports,
    compile_trace,
    format_capabilities,
    load_json,
    replay_canary,
    validate_canary,
    validate_comparison,
    validate_report,
    validate_trace,
    verify_canary_behavior,
    verify_canary_fidelity,
    verify_report_against_canary,
)
```

The top level also exports `__version__`, the exact trace/canary/report/compare
format constants, `CANONICAL_JSON_VERSION`, `FormatCapability`,
`DEFAULT_RESOURCE_LIMITS`, `CommCanaryError`, `SchemaError`, and the current
`JsonDict` wire alias.

Inputs are caller-owned and treated as read-only. Returned artifacts are
detached JSON snapshots: mutating an input after the call or an output after the
call does not mutate the other object or a sibling result. Runtime validation is
authoritative even when static wire aliases are provided.

Typed policy values are available from their documented domain modules without
expanding the top-level namespace:

```python
from commcanary.behavior_config import (
    BehaviorConfiguration,
    parse_behavior_configurations,
    preflight_behavior_ranking_work,
)
from commcanary.compare import (
    ComparisonReasonCode,
    ComparisonThresholdPolicy,
    comparison_reason_codes,
)
```

`parse_behavior_configurations(None)` selects the immutable defaults; an empty
sequence is invalid. Names are nonempty and unique, unknown keys and invalid
ranges fail, and ranking work is preflighted before candidate evaluation.
`ComparisonThresholdPolicy` validates one immutable set of thresholds while the
legacy keyword arguments remain compatible. Comparison v2 does not add a new
reason-code field: the stable structured codes are the existing
`evaluations[].metric` values returned by `comparison_reason_codes()`.

Within a released minor line, compatible changes may add optional keyword-only
parameters with defaults, add result fields inside documented extension/open
objects, or add a new exact format capability. Removing/renaming a top-level
name, changing a default that affects semantics, changing an exit/reason code,
or altering canonical bytes/hashes requires a documented deprecation or a
versioned format/API break.

## Adapter API

Capture and ecosystem conversions are supported through explicit submodules:

```python
from commcanary.capture import TraceRecorder, record_collective
from commcanary.interop import (
    canary_to_param_comms_trace,
    kineto_trace_to_commcanary_trace,
)
```

Their wire outputs and safety properties are supported, but third-party runtime
compatibility (PyTorch Kineto and PARAM) is constrained by the versions and
fixtures documented in the format and research material. They are intentionally
not imported at the package top level.

## Experimental API

Research baselines, decision-only reduction, benchmark tooling, and the Rostam
campaign harness can change between minor releases. Existing module paths remain
available for the current compatibility period, while research functions have
an explicit namespace:

```python
from commcanary.experimental import (
    ddmin_ranking_reduction,
    isolated_collective_baseline_trace,
)
```

They are not promoted to the stable top level. Persisted experimental records
carry their own schema IDs; consume those schemas rather than relying on
internal Python classes.

## Version and capabilities

`commcanary.__version__` comes from installed distribution metadata. In an
unbuilt source checkout it reads the same `[project].version` build metadata;
if neither is available it safely reports `0+unknown` rather than inventing a
release version.

`format_capabilities()` returns an immutable tuple with exact format IDs, schema
paths, read/write support, migration support, and whether an independent
runtime semantic validator exists. CommCanary never silently migrates an
artifact during load.

## Deprecation policy

For a documented stable Python name, normal removals receive a warning for at
least one released minor version and name the intended removal version. Security
or correctness defects may require an immediate fail-closed behavior change;
those changes are called out in the changelog with a migration or explicit
opt-in compatibility path where safe.

Wire compatibility is governed by the exact format ID and
[`docs/formats/compatibility.md`](formats/compatibility.md), not by this Python
deprecation window.
