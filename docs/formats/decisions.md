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
mutation. The assurance ladder remains:

1. `structurally_valid`;
2. `internally_consistent`;
3. `source_corresponding`;
4. `model_recomputed`;
5. `behaviorally_verified`.

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

## Characterized gaps requiring later source or format changes

The current suite records, but does not paper over, these remaining gaps:

- numeric-string/integral-float coercion in runtime validators;
- open unknown semantic fields and no reserved extension semantics;
- Python-specific canonical JSON rather than a language-neutral standard;
- no raw-input-byte commitment;
- no standalone semantic validators for verification result artifacts;
- comparison has structured `evaluations[].metric` identifiers but no dedicated
  stable `reason_codes` array;
- no automatic privacy redaction/classification;
- no authenticity/signature layer.
