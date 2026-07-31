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
    kineto_traces_to_commcanary_trace,
    load_kineto_trace_with_identity,
)
```

Their wire outputs and safety properties are supported, but third-party runtime
compatibility (PyTorch Kineto and PARAM) is constrained by the versions and
fixtures documented in the format and research material. They are intentionally
not imported at the package top level.

`kineto_traces_to_commcanary_trace` requires at least two distributed profiles
and a complete contribution from every rank participating in each imported
operation. Supply additive `clock_offsets_us={rank: offset}` for every imported
rank, or set `assume_shared_clock=True` as an explicit zero-offset assertion.
Omitting both preserves unknown arrival skew and prevents compilation.
`load_kineto_trace_with_identity(path)` returns the parsed profile plus a
path-free `{"sha256": ..., "size_bytes": ...}` identity computed from the same
bounded bytes. The CLI attaches these records automatically; in-memory
conversion functions cannot infer original file-byte identity.

Qualification request composition is available through its owning modules:

```python
from commcanary.artifacts import validate_qualification_request
from commcanary.services import (
    prepare_qualification_request,
    verify_qualification_request,
)
from commcanary.workflows import (
    materialize_qualification,
    verify_qualification_materialization,
)
from commcanary.execution import (
    execute_qualification_materialization,
    preflight_qualification_execution,
)
```

The preparation service accepts an already imported Kineto trace and compiled
canary. It requires a source-bound canonical communication dtype and a complete
per-rank contiguous-GEMM recipe derived between issue and an explicit wait for
every event. `all_reduce` and `reduce_scatter` additionally require a
source-bound `reduction_op`; broadcasts require a source-bound `root_rank`.
Kineto-backed traces require exact input/output element counts and normalized
split evidence. The service will not fit elapsed gaps to compute, assume SUM,
infer a root from group order, or reconstruct missing message shapes.

`materialize_qualification(...)` takes only the verified request directory and
a new output directory. It deterministically emits asynchronous collective
issue, each rank's exact `m×k @ k×n` recipe, and an immediate explicit wait.
There is no target timing calibration or duration quantization. The manifest
records the canonical recipe projection hash, per-rank operation counts,
source kernel observations, mathematical FLOPs, and exact program identity.
`verify_qualification_materialization(...)` revalidates the source request and
recomputes the audit and program bytes exactly. Both preserve an explicit
conforming-adapter requirement and make no current upstream PARAM execution
claim.

`preflight_qualification_execution(...)` revalidates the request and
materialization, then returns a frozen execution plan only after operation,
request/wait, rank-domain, repeated-work, retained-sample, GEMM, and tensor
allocation checks pass. It also validates and retains an explicit
`distributed_timeout_seconds` value under the shared resource ceiling.
`execute_qualification_materialization(...)` lazily
imports PyTorch after that preflight and can run the full materialized
collective operation set while applying each exact rectangular recipe only to
its owner. Reduction collectives dispatch the
exact bound operator and the untimed correctness pass checks results under
those semantics. It preallocates and budgets both GEMM inputs plus the reused
output before execution. Its return value is a bound reference diagnostic, not
a supported physical-observation format. The implementation remains outside
the stable top-level facade while physical conformance is unproven.

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

Behavior search is also an experimental evidence-producing API:

```python
from commcanary.compiler import (
    synthesize_behavioral_canary,
    validate_behavior_search_evidence,
)

evidence = {}
canary = synthesize_behavioral_canary(trace, evidence_output=evidence)
validate_behavior_search_evidence(evidence, canary)
```

The canary remains executable without the ledger, but it contains only a
compact search summary and the ledger's exact canonical-byte identity. Retain
the detached evidence when the synthesis history must be audited.

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
