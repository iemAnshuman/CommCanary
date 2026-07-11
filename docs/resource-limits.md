# Resource limits

CommCanary treats trace, canary, report, capture, and ecosystem-adapter input as
untrusted. `ResourceLimits` is the immutable policy shared by JSON loading,
validation, expansion, hashing, replay, verification, behavior search,
reduction, capture merging, and PARAM export. The default instance is
`DEFAULT_RESOURCE_LIMITS` in `commcanary.resources`.

The policy is a safety ceiling, not a performance target or a promise that an
artifact at every limit will fit on every machine. Callers should normally use
smaller limits for constrained services and should apply independent process
memory and time limits when handling hostile input.

## Default policy

| Field | Default | What it bounds |
|---|---:|---|
| `max_input_bytes` | 67,108,864 | Bytes read from one JSON file |
| `max_json_depth` | 64 | Nested object/array depth, with the root container at depth 1 |
| `max_json_items` | 2,000,000 | Total object members plus array elements |
| `max_json_string_bytes` | 1,048,576 | UTF-8 bytes in one key or string value |
| `max_json_number_chars` | 1,024 | Characters in one integer or floating-point token |
| `max_stored_events` | 1,000,000 | Event records in the compact artifact, including motif children |
| `max_stored_timing_records` | 2,000,000 | Timing records stored in the compact artifact |
| `max_ranks` | 65,536 | Distinct ranks accepted by a trace |
| `max_expanded_events` | 1,000,000 | Logical events after motif and repeat expansion |
| `max_expanded_timing_records` | 2,000,000 | Logical timing records after repeat/weight expansion |
| `max_replay_events` | 1,000,000 | Logical events across all replay iterations |
| `max_param_entries` | 2,000,000 | Entries produced by one PARAM export |
| `max_capture_shards` | 65,536 | Shards considered by one capture merge |
| `max_capture_events` | 1,000,000 | Aggregate events accepted by one capture merge |
| `max_behavior_configurations` | 32 | Named configurations in one behavior-verification matrix |
| `max_behavior_candidates` | 4,096 | Candidates evaluated by one behavior search |
| `max_behavior_ranking_comparisons` | 10,000,000 | Pairwise comparisons in one behavior-ranking evaluation |
| `max_retained_ledger_rows` | 10,000 | Candidate/refinement diagnostics retained by one search |
| `max_reduction_oracle_calls` | 10,000 | Oracle calls allowed by one reduction |

All calculated work counts use non-negative checked arithmetic with a fixed
maximum of `2**63 - 1`. Overflow is rejected even if a configured limit would
otherwise be larger.

## Enforcement order

The byte and nesting checks run before the standard JSON decoder. Loading then
rejects duplicate object keys, non-standard constants (`NaN` and infinities),
oversized numeric tokens, invalid Unicode scalar values, and trees above the
item or string limits.

Compact event, timing, and rank counts are checked during validation. Motif
repeats, timing weights, replay iterations, behavior-search ranges, reduction
calls, capture shard counts, and PARAM output sizes are preflighted with checked
arithmetic before the relevant generator, repeat loop, or output list begins.
Validation and the operation that follows it receive the same policy object so
an artifact cannot be accepted with one ceiling and expanded with another.

Already-decoded report and comparison objects receive the same JSON depth,
item, and string preflight as file-loaded artifacts. Report replay counts are
checked before breakdown/sample reconciliation, and replay-protocol hashing
preflights its input before canonical encoding. `compare_reports`, report
recomputation verification, and replay's final report validation forward the
caller's exact `ResourceLimits` instance.

`TraceRecorder` and `TraceRecorder.from_env` accept `limits=`. Workload, system,
and per-event metadata are checked before normalization or deep copy. A recorder
rejects the next event before append when it would exceed either
`max_capture_events` or `max_stored_events`; capture merge also reapplies the
same in-memory JSON preflight when a custom shard loader is supplied.

Errors at the public schema boundary are reported as `SchemaError`; the
lower-level JSON policy uses `JsonResourceError`. A limit failure is deterministic
for the same input and policy, but its message is diagnostic text rather than a
stable parsing interface.

## Supplying a stricter policy

Python entry points that may load or expand untrusted data accept a keyword-only
`limits=` argument. Construct a complete immutable policy, usually by replacing
selected values on the default:

```python
from dataclasses import replace

from commcanary.compiler import compile_trace
from commcanary.resources import DEFAULT_RESOURCE_LIMITS

service_limits = replace(
    DEFAULT_RESOURCE_LIMITS,
    max_input_bytes=8 * 1024 * 1024,
    max_stored_events=100_000,
    max_expanded_events=100_000,
    max_replay_events=200_000,
)

canary = compile_trace(trace, limits=service_limits)
```

Use the same object for a complete pipeline:

```python
from commcanary.replay import replay_canary
from commcanary.schema import validate_canary

validate_canary(canary, limits=service_limits)
report = replay_canary(canary, limits=service_limits)
```

Limits must be integers (booleans are rejected). Most ceilings must be positive;
JSON item/string ceilings may be zero, and the behavior-configuration ceiling
must be at least two. A per-operation convenience argument such as
`max_replay_events` or `max_oracle_calls` may make a call stricter but cannot
raise the shared policy ceiling.

The command-line interface currently uses the documented default policy. A
deployment that requires tenant-specific limits should call the Python API or
place the CLI inside an independently bounded worker process rather than editing
the global default at runtime.

## Changes and compatibility

Changing a default limit does not change artifact semantics, canonical JSON, or
semantic hashes. It can change whether a large artifact is accepted, so default
changes are reviewed as compatibility and security changes. Code must not raise
a limit implicitly to make a fixture pass; callers opt into any broader policy
explicitly and remain responsible for the execution environment.
