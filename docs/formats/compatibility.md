# Wire-format support and compatibility

CommCanary publishes JSON Schema Draft 2020-12 documents in `schemas/` for
portable, canonical wire shape. The schemas are deliberately not a substitute
for the Python runtime validators: they do not recompute hashes, expand motifs,
reconcile counts, or derive verdicts.

The immutable `commcanary.format_capabilities()` query is the executable source
for these exact IDs and support flags. `commcanary --version` prints the same
matrix together with `commcanary.canonical-json.v1` and the replay model.

## Support matrix

| Artifact | Exact format | Schema | Produced by this version | Consumed by this version | Runtime semantic validator | Automatic migration |
|---|---|---|---|---|---|---|
| Source trace | `commcanary.trace.v1` | `commcanary.trace.v1.schema.json` | Yes | Yes | `validate_trace` | None |
| Canary | `commcanary.canary.v2` | `commcanary.canary.v2.schema.json` | Yes, with `commcanary.canary-integrity.v1` | Yes | `validate_canary` | None |
| Replay report | `commcanary.report.v2` | `commcanary.report.v2.schema.json` | Yes | Yes | `validate_report` | None |
| Report comparison | `commcanary.compare.v2` | `commcanary.compare.v2.schema.json` | Yes | Python validation only | `validate_comparison` | None |
| Fidelity verification | `commcanary.fidelity_verification.v1` | `commcanary.fidelity_verification.v1.schema.json` | Yes | No general artifact reader | None | None |
| Behavior verification | `commcanary.behavior_verification.v1` | `commcanary.behavior_verification.v1.schema.json` | Yes | No general artifact reader | None | None |
| Report verification | `commcanary.report_verification.v1` | `commcanary.report_verification.v1.schema.json` | Yes | No general artifact reader | None | None |

“Consumed” means that a supported CLI or Python workflow accepts the artifact
as input. A JSON file being loadable is not a format-support promise. No format
currently has an automatic migration path, and load/validation never mutates an
artifact into another version.

Canary v2 documents without an integrity profile are a narrow compatibility
case. `validate_canary(..., allow_legacy_unverified=True)` can inspect them only
when the caller explicitly opts in. The published v2 schema describes the
current profiled artifact and therefore requires the integrity profile and its
six commitments. An unprofiled legacy document must not be presented as meeting
the current schema or as internally consistent.

## Schema boundary

Schema validation proves only that a document has the portable shape of the
declared format. The runtime layer remains authoritative for semantic checks.

| Artifact | Important checks intentionally left to runtime |
|---|---|
| Trace | rank-count equality, exact arrival-map coverage, skew derivation, endpoint membership, point-to-point requirements, custom-op opt-in, resource budgets |
| Canary | all digest recomputation and alias equality, source-block commitments, motif expansion, repeat/count equality, timing interval coverage, fidelity maxima and budgets, resource budgets |
| Report | replay-protocol digest, model/protocol/backend agreement, count derivation, quantile ordering, breakdown and sample reconciliation, deterministic scheduling equations |
| Comparison | embedded metric deltas, compatibility consistency, policy evaluation derivation, uncertainty effects, final verdict derivation |
| Verification outputs | agreement between individual checks, aggregate status, and assurance state |

This boundary is executable in `tests/contracts/test_json_schemas.py`:

- **valid** fixtures pass JSON Schema and the runtime validator when one exists;
- **invalid** mutations break portable shape and fail both layers when a runtime
  validator exists;
- **tampered** mutations preserve portable shape, pass JSON Schema, and fail the
  runtime semantic validator when one exists.

Verification result formats currently have producers but no corresponding
runtime validators. Their tampered fixtures document that JSON Schema cannot
establish that a claimed status follows from the included checks.

## Type and extension policy

The published schemas describe canonical JSON values: counts are JSON integers,
measurements are JSON numbers, booleans are booleans, and SHA-256 values are
64-character lowercase hexadecimal strings. Some current Python validation
paths still coerce numeric strings or integral floating-point values. That is a
reader implementation detail, not a portable wire guarantee; other languages
should emit and require the schema types.

Unknown fields are allowed where the current formats and validators allow
metadata or forward additions. Their presence does not imply that an older
consumer understands their semantics. Closed maps, such as `fidelity_budget`,
are closed only where the runtime validator also rejects unknown keys. There is
no blanket forward-compatibility guarantee across a new `format` value.

## Using the schemas

Each artifact schema and `common.schema.json` has an absolute `$id`. Consumers
should register all schema documents locally and resolve references by `$id`;
validation must not depend on network retrieval. The contract test suite uses
exactly that offline registry model.

Schema validation alone corresponds to structural shape, not to any stronger
assurance state. In particular, a syntactically correct SHA-256 value is not
evidence that its protected projection was recomputed, and no digest establishes
producer authenticity.
