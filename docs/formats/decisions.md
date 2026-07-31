# Wire-contract decisions

These decisions characterize the 0.3 wire contracts before internal module
moves. They are normative for the published formats where they describe
observable behavior. Items explicitly marked as gaps require a later format or
API decision; tests must not silently reinterpret them.

## ADR-001: Loading, parsing, and validation are distinct

**Status:** Accepted for 0.3; canonical parsing remains a named gap.

### Context

An artifact can be syntactically valid JSON while having the wrong format,
noncanonical field types, or inconsistent derived values. Conflating these
steps makes it unclear whether validation may rewrite a caller-owned object.

### Decision

`load_json_document` and `load_json` perform bounded, strict JSON decoding. They
reject invalid UTF-8, duplicate object keys, nonstandard/nonfinite numbers,
excessive numeric tokens, and configured resource-limit violations. Loading
does not select an artifact version and does not migrate it.

`validate_trace`, `validate_canary`, `validate_report`, and
`validate_comparison` inspect an already decoded object and never mutate it.
They require one exact `format` value and enforce the cross-field semantics
documented in the compatibility matrix.

The portable JSON Schemas describe canonical wire types. Current Python runtime
validators still accept numeric strings and integral floating-point values at
some `as_int`/`as_float` call sites. That compatibility coercion is validation
behavior only: the original object remains unchanged, producers do not emit
those forms, and the schemas reject them.

### Consequences

- There is no silent load-time migration.
- Validation success does not imply that a noncanonical numeric representation
  was rewritten.
- Cross-language producers must emit schema-native integers and numbers.
- A future `parse_*` API should return a new canonical value and must make its
  ownership/migration policy explicit.
- **Gap:** numeric coercion is broader than the canonical schema contract and
  should be removed or isolated behind an explicit compatibility parser in a
  future breaking release.

## ADR-002: Unknown fields are opaque, not understood extensions

**Status:** Accepted characterization for existing v1/v2 formats.

### Context

The current runtime validators generally allow unknown fields. There is no
reserved extension namespace with negotiated semantics, yet workloads and
systems need room for producer metadata.

### Decision

Unknown fields remain accepted wherever the current validator accepts them.
Objects named `extensions` receive no special treatment; they are ordinary
unknown or metadata fields. A consumer must not infer support for an extension
merely because validation preserved or ignored it.

The selected full source trace, including unknown top-level and event fields,
is protected by `source_trace_sha256`/`source_normalized_sha256`. Compilation
may omit unknown root or event fields from the executable canary, so executable
semantic hashes may remain unchanged while the source commitment changes.
Nested workload/system metadata that is copied is detached from caller-owned
objects. Unknown fields stored in a profiled canary are covered by full artifact
provenance unless explicitly excluded by that projection.

Maps are closed only where the runtime is closed. For example,
`fidelity_budget` rejects unknown budget names; general artifact and metadata
objects do not.

### Consequences

- `additionalProperties` is intentionally enabled in the published schemas for
  open objects.
- Unknown-field acceptance is not a forward-compatibility guarantee for a new
  `format` value.
- Producers should use reverse-domain names inside an `extensions` object to
  reduce collisions, but 0.3 does not assign that convention semantics.
- **Gap:** rejecting unknown semantic fields while preserving a formally named
  extension namespace requires a new format decision and migration story.

## ADR-003: Canonical JSON is versioned and Python-specific

**Status:** Accepted as `commcanary.canonical-json.v1`.

### Context

Hash agreement requires exact bytes. “Sorted JSON” is insufficient unless
escaping, separators, number rendering, duplicate handling, and string encoding
are fixed.

### Decision

Canonical JSON v1 is the UTF-8 encoding produced by Python `json.dumps` with:

- `sort_keys=True`;
- `separators=(",", ":")`;
- `allow_nan=False`;
- the default `ensure_ascii=True` behavior.

Object keys are sorted using Python string ordering. Non-ASCII characters are
therefore emitted as lowercase `\u` escapes by the standard encoder. JSON
booleans and null use their JSON spellings. Integers use decimal notation.
Finite floats use the supported Python runtime's shortest round-trippable JSON
rendering; negative zero is preserved as `-0.0`. No Unicode normalization or
numeric normalization is applied.

Strict loaders reject duplicate keys before canonicalization. In-memory inputs
must already contain JSON-native string keys and finite values. Canonicalization
does not preserve raw input whitespace, key order, escape choice, or numeric
token spelling.

Literal UTF-8, hexadecimal, and SHA-256 vectors live in
`tests/fixtures/contracts/hash_vectors.v1.json`. They run on every supported
Python version and cover a trace-v1 → canary-v2 → report-v2 chain.

### Consequences

- This is not RFC 8785/JCS and should not be labeled as such.
- Other languages must reproduce the literal vectors, including Python float
  and escaping behavior, before claiming hash interoperability.
- Raw artifact bytes are not committed by current source hashes.
- A future canonicalization algorithm must use a new version ID; changing bytes
  in place would invalidate stored commitments.

## ADR-004: Semantic determinism is separate from byte identity

**Status:** Accepted.

### Context

Compilation and replay contain deterministic algorithms but artifacts also
contain timestamps and host metadata. Release-build reproducibility is a third,
separate concern.

### Decision

For the same canonical semantic input and options:

- compilation must reproduce source, execution/scheduler, calibration, and
  artifact-provenance hashes;
- replay with the same seed, model, protocol, and backend settings must
  reproduce metrics and samples;
- flat versus sequence-motif encodings and flat runs versus repeated timing
  patterns must reproduce scheduler semantics and replay results;
- different replay seeds may intentionally produce different deterministic
  jitter sequences.

Full artifact byte identity is not promised. `created_at` is volatile and is
excluded from profiled canary provenance and report canary identity. Reports
also record host platform/Python metadata. Tests compare deterministic replay
documents only after removing `created_at` and keep literal per-seed metrics.

Reproducible wheel/sdist bytes are a release-build property, not an artifact
semantic property.

### Consequences

- A changed timestamp alone is not a semantic or provenance failure.
- Seed and replay protocol are part of model reproducibility; the enforcement
  ceiling `max_replay_events` is deliberately excluded from the protocol hash.
- Python matrix failures against literal vectors are compatibility failures,
  not values to regenerate casually.

## ADR-005: Integrity profiles express consistency, not authenticity

**Status:** Accepted for profiled canary v2.

### Context

Canary v2 existed before the complete integrity commitment set. Hash fields can
be syntactically valid while not matching their protected projections.

### Decision

Current producers emit `commcanary.canary-integrity.v1` with
`commcanary.artifact-provenance.v2`. The profile requires:

- source trace and normalized-source aliases;
- execution and scheduler aliases;
- calibration evaluation;
- full artifact provenance;
- `first_id`, `last_id`, and `digest` on every stored source block.

Runtime canary validation recomputes internal semantic, calibration, and
artifact projections. Source-assisted fidelity verification separately
reconstructs source correspondence and catches an internally rehashed producer
mutation. The integrity/correspondence summary remains:

1. `structurally_valid`;
2. `internally_consistent`;
3. `source_corresponding`;
4. `model_recomputed`.

Deterministic-model behavior preservation and physical/authenticity properties
are independent claim dimensions. In particular,
`model_behavior_preserved` does not assert physical execution, conformance, or
decision fidelity.

Legacy unprofiled canary v2 is readable only via the explicit
`allow_legacy_unverified=True` opt-in and does not satisfy the published current
profile schema. There is no automatic migration.

### Consequences

- Hash syntax alone establishes no assurance level.
- A producer can recompute an unsigned artifact; hashes do not authenticate the
  producer or authorization.
- Verification-output schemas cover shape only. There are not yet standalone
  semantic validators for their aggregate status/check agreement.
- Signatures or attestations require a separate authenticity design.

## ADR-006: Metadata is potentially sensitive and redaction is explicit

**Status:** Accepted privacy boundary; automated redaction remains a gap.

### Context

Communication traces can contain workload, topology, host, process, rank,
cluster, and source identifiers. Hashing a value does not make the surrounding
artifact safe to publish, and copied metadata can propagate across artifacts.

### Decision

All caller/adapter metadata is treated as potentially sensitive. CommCanary 0.3
does not automatically redact it. Redaction must happen before capture output is
accepted or before compilation; modifying protected metadata afterward changes
commitments and requires recompilation.

| Data family | Typical source | Propagation | Integrity/privacy note |
|---|---|---|---|
| `workload` | capture/CLI/importer/user | trace → canary → report; fidelity checks copy expected/actual values | Full objects may contain model, tenant, path, dataset, or job names. |
| `system` | capture/importer/user | trace → canary; fidelity checks | May contain rank, topology, backend, host, process, runtime, or cluster identifiers. Default replay reports do not copy the canary's `system` object. |
| Event IDs and source bounds | recorder/importer/user | trace IDs → canary `source.first_id`/`last_id`; verification output on mismatch | Bounds remain plaintext; digests do not redact them. |
| Event metadata/unknown fields | recorder/importer/user | fully committed in selected source; only executable fields necessarily enter canary | An ignored executable extension can still leak through retained source artifacts or verification diagnostics. |
| Backend label/settings | replay caller | report → comparison | Labels can reveal hardware/site naming conventions. |
| Host platform and Python | replay host | report `host` | Automatically emitted environment fingerprint; review before publication. |
| Capture session/rank/process data | capture runtime | trace `system` and event metadata | Can correlate workers and attempts. |
| Capture failure bundle | failed capture | workload name, session ID, child code, shard filenames/sizes/digests | Command and environment are deliberately omitted, but filenames and workload/session values still require review. |
| Timestamps | producers | trace/canary/report/comparison | Operational metadata; volatile exclusions do not make it nonsensitive. |

Public release workflows must use an allowlist or a reviewed, redacted source
rather than assuming unknown fields are harmless. Removing a secret from a
published artifact does not remove it from prior copies, logs, hashes, or
external caches.

### Consequences

- No artifact is safe to publish merely because runtime validation passes.
- Verification failure objects may echo expected/actual metadata and should be
  handled with the same classification as their inputs.
- **Gap:** there is no first-class redaction policy, metadata classification,
  or privacy-safe export mode. Those require source changes and explicit tests.

## ADR-007: Missing compute overlap is unknown, never an implicit zero

**Status:** Accepted fail-closed timing boundary.

### Context

`compute_overlap_us: 0.0` is a physical claim that no measured compute was
concurrent with the communication event. Earlier readers also produced that
value when the field was absent. That made an unmeasured event executable as if
zero overlap had been observed, even though overlap materially changes proxy
fidelity.

### Decision

A trace event has known overlap only when it carries `compute_overlap_us`.
Producers that know the measurement is unavailable should emit
`compute_overlap_unknown: true`; absence of both fields has the same unknown
meaning for legacy traces. Carrying both fields is invalid, as is
`compute_overlap_unknown: false` without a value.

General trace validation may inspect an unknown-overlap trace. Workflows that
compile, behavior-search, reduce, or preserve source overlap in a baseline
require known overlap for every selected event and fail closed otherwise.
`compute_overlap_us: 0.0` remains valid when it was measured or deliberately
constructed. In particular, the isolated-collective negative control explicitly
sets zero because removing overlap is that baseline's declared transformation.

The Kineto importer derives overlap only when a selected collective's unique
external correlation id links to complete communication-kernel intervals and
the trace carries a complete, placeable compute-kernel inventory. The value is
the union of intersections between those communication intervals and
non-communication kernels on other streams of the same device. Same-stream
work and other communication kernels are excluded; overlapping compute
intervals are not double-counted. Missing, malformed, or ambiguous evidence
preserves the unknown marker. The algorithm is locally mutation-tested, but
its physical proxy fidelity remains an experiment rather than a supported
result.

Multiple Kineto profiles reuse the capture reconciliation contract rather than
defining a second distributed-event format. Each profile must declare a unique
distributed rank and consistent world size/backend. Rank-local events are
matched by invocation ordinal within an exact process-group participant domain;
one missing participant or any operation/size/group disagreement rejects the
merge. Point-to-point records additionally require explicit source and
destination ranks.

`record_param_comms` does not name `BroadcastOptions.rootRank`. The importer
may derive `root_rank` only from a containing `c10d::broadcast_` CPU event on
the same pid/tid whose concrete dispatcher inputs encode a non-negative root
inside the process group. The field is included in compilation, compression,
fidelity, semantic hashes, baselines, rank reconciliation, and replay
materialization. Missing root evidence leaves the observational trace
inspectable, but qualification and PARAM-derived export refuse it. A root on a
non-broadcast, a root outside the participant set, or disagreement between
rank-local profiles is invalid. Group order is not a root-selection policy.

`record_param_comms` also lacks a canonical reduction-operator field. For
`all_reduce` and `reduce_scatter`, import derives `reduction_op` only from
uniquely linked NCCL communication-kernel names when every linked kernel has
one recognized and consistent operator token. Recognized semantics are SUM,
PRODUCT, MIN, MAX, and AVG. Missing links, generic or unrecognized names,
incomplete kernel records, and conflicting names leave the field unknown
instead of guessing. Multi-rank reconciliation preserves the operator only
when every participant derives the same value; conflicting known operators
are invalid, while mixed known/unknown evidence is marked incomplete. General
trace and canary validation permit the field to be absent for observational
use, but qualification refuses any reduction collective without it.

Kineto also supplies exact input/output element counts and input/output split
lists. Import normalizes and retains that evidence instead of treating
`max(in, out)` as a sufficient execution contract. Rank-local contributions
must agree on the complete shape evidence before merge. Qualification
independently checks standard collective ratios, dtype-derived bytes, and the
absence of skipped zero/missing-size records. Explicit split vectors remain
inspectable source evidence but are not materializable: the reference executor
binds an equal-split-only `all_to_all` policy and refuses rather than silently
reconstructing a different shape.

Kineto timestamps are not assumed to share a clock. A complete additive
rank-offset map, or an explicit shared-clock assertion equivalent to all-zero
offsets, permits full `rank_arrival_us` derivation. Without either, merged
events carry `arrival_skew_unknown: true`, and compilation refuses them. The
merge retains `compute_by_rank`; the executable scalar overlap is the minimum
known rank-local value, and any unknown rank leaves the logical event unknown.

### Consequences

- Existing traces that omitted overlap remain structurally inspectable but
  cannot produce a canary or fidelity claim without an explicit measurement.
- Published source hashes change when a deliberate zero is added; golden
  vectors therefore include the field.
- Import records both derived/unknown totals and per-event derivation status,
  so one unresolved collective keeps the later compile boundary closed.

## ADR-008: Qualification requests are portable pre-execution evidence

**Status:** Accepted for `commcanary.qualification_request.v1` and
`commcanary.qualification_materialization.v1`.

### Context

A model owner and hardware lab need one exchange boundary before either party
can discuss an acceptance result. Shipping only a canary loses the source needed
to recompute fidelity. Shipping a PARAM trace as the portable request is also
wrong: it neither binds the trace-to-canary proof nor proves that executable
compute came from the source rather than a duration fit. An owner-generated
program is acceptable only when the exchange also binds exact source-derived
work and lets the recipient regenerate it independently.

### Decision

A qualification request is a closed four-file directory containing a manifest,
source trace, canary, and fidelity verification. The manifest binds the exact
bytes of the other files, the canary's six source/execution/calibration/provenance
commitments, and a canonical request ID. Directory verification rejects missing
or extra files, symlinks, byte mismatches, semantic invalidity, rehashed source
tampering, and manifest-to-canary disagreement. Preparation also requires a
source-bound canonical dtype on every canary event, a source-bound reduction
operator on every reduction collective, and proves that the compact program can
be lowered to the narrow exact-work collective vocabulary without expanding
the executable.

The format has a fixed request-only claim boundary:

- source correspondence is verified;
- physical measurement is not included;
- physical fidelity is unproven;
- no qualification verdict is issued; and
- deterministic executable materialization is required. The request binds the
  communication dtype set, source-derived
  reduction-operator set, source-validated per-event message shapes,
  equal-split-only `all_to_all` policy, exact per-rank compute-recipe projection
  and counts, single-inflight issue/work/wait structure, timestamp-pacing
  policy, and the disclosure that GEMM shapes and dtypes are shared.

The manifest carries content identity, not a signature or producer
authentication. Preparation never overwrites an existing directory and installs
the manifest last, so an interrupted directory cannot look complete.

The receiving side creates a closed two-file materialization without target
timing calibration. Its manifest binds the exact request manifest, canonical
source-work projection, per-rank operation counts, source kernel observations,
deterministic program bytes, entry count, and
async-issue/exact-rank-work/explicit-wait semantics. Independent verification
revalidates the request and regenerates the program byte-for-byte.

The program encoding is named
`commcanary.source-bound-compute-recipe.v2`; it is not called an upstream PARAM
executable. Each compute entry carries exact rank-local contiguous GEMM shapes
derived between issue and explicit wait. Start gaps, source durations, Python
overhead, and idle time do not control executable work. Schedules requiring
multiple in-flight collectives or unsupported compute need a real dependency
graph and refuse; this format intentionally does not duplicate one. Current upstream PARAM removed
basic/Kineto parsing in favor of Chakra host execution traces, while the pinned
historical basic replayer both hardwires blocking execution and ignores the
rank-aware extension. Both request and materialization therefore bind
`upstream_param_compatibility: not_claimed` and
`conforming-adapter-required`. A format resemblance cannot substitute for an
executor conformance proof.

An in-package torch.distributed reference executor may consume this verified
program, but its diagnostic deliberately is not added to the supported format
matrix. Local pure and injected-runtime tests establish fail-closed parsing,
allocation/work budgets, operation dispatch, deterministic communication-data
checks under exact source-bound reduction operators, rank aggregation, and the
separation of per-operation latency from max-rank whole-program makespan; they
do not establish CUDA/NCCL timing or semantic conformance. The correctness pass
is separately resource-counted and untimed so a wrong result cannot become a
timing success and validation work cannot contaminate the measured makespan.
Pure tests additionally prove that timing-only mutations cannot change the
program, recipe-shape mutations must change it, and per-rank work accounting
recomputes exactly. The materialization
continues to say `conforming-adapter-required` until the reference executor
passes the physical gate.

### Consequences

- The receiving party can verify what the owner supplied before execution.
- `source.trace.json` enables independent fidelity recomputation but reveals
  more workload structure than a canary alone and requires source-boundary
  privacy review.
- The reference target-execution implementation must be physically validated,
  after which a physical-observation contract can bind the conforming runtime
  and raw evidence before the final `qualify` workflow issues an acceptance
  verdict. Source-work provenance and materialized program bytes are already
  closed by the pre-execution materialization format.

## Characterized gaps requiring later source or format changes

The current suite records, but does not paper over, these remaining gaps:

- numeric-string/integral-float coercion in runtime validators;
- open unknown semantic fields and no reserved extension semantics;
- Python-specific canonical JSON rather than a language-neutral standard;
- generic input paths still lack raw-byte commitments; Kineto CLI imports bind
  exact source-profile hashes/sizes and qualification requests bind their
  emitted trace/canary/fidelity file bytes;
- no standalone semantic validators for verification result artifacts;
- comparison has structured `evaluations[].metric` identifiers but no dedicated
  stable `reason_codes` array;
- no automatic privacy redaction/classification;
- no authenticity/signature layer.
