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
| Qualification request (legacy) | `commcanary.qualification_request.v1` | `commcanary.qualification_request.v1.schema.json` | No | Yes | `validate_qualification_request` plus directory-level `verify_qualification_request` | None |
| Qualification request (current) | `commcanary.qualification_request.v2` | `commcanary.qualification_request.v2.schema.json` | Yes | Yes | `validate_qualification_request` plus directory-level `verify_qualification_request` | None |
| Qualification materialization | `commcanary.qualification_materialization.v1` | `commcanary.qualification_materialization.v1.schema.json` | Yes | Yes | `validate_qualification_materialization` plus request-assisted `verify_qualification_materialization` | None |
| Qualification policy | `commcanary.qualification_policy.v1` | `commcanary.qualification_policy.v1.schema.json` | Yes | Yes | `validate_qualification_policy` | None |
| Qualification observation | `commcanary.qualification_observation.v1` | `commcanary.qualification_observation.v1.schema.json` | Yes | Yes | `validate_qualification_observation` | None |
| Qualification verdict | `commcanary.qualification_verdict.v1` | `commcanary.qualification_verdict.v1.schema.json` | Yes | Yes | `validate_qualification_verdict` plus `evaluate_qualification_observations` | None |

Behavior search additionally emits the explicitly experimental
`commcanary.behavior_search_evidence.experimental.v1` sidecar. It is omitted
from `format_capabilities()` because it is research evidence rather than a
stable product interchange format. Its portable shape is published as
`commcanary.behavior_search_evidence.experimental.v1.schema.json`, and
`validate_behavior_search_evidence` verifies its exact-byte digest and selected
executable identity against the canary.

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
| Trace | canonical dtype and reduction operator when present, reduction-field operation scope, rank-count equality, exact arrival-map coverage, skew derivation, known-overlap requirements for canary-producing workflows, endpoint membership, point-to-point requirements, custom-op opt-in, resource budgets |
| Canary | canonical dtype and reduction operator when present, reduction-field operation scope, all digest recomputation and alias equality, source-block commitments, motif expansion, repeat/count equality, timing interval coverage, fidelity maxima and budgets, resource budgets |
| Report | replay-protocol digest, model/protocol/backend agreement, count derivation, quantile ordering, breakdown and sample reconciliation, deterministic scheduling equations |
| Comparison | embedded metric deltas, compatibility consistency, policy evaluation derivation, uncertainty effects, final verdict derivation |
| Verification outputs | agreement between individual checks, aggregate status, and assurance state |
| Experimental behavior-search evidence | exact canonical-byte digest bound by the canary, selected executable identity, and separation from executable canary bytes |
| Qualification request | canonical request ID, version-specific closed inventory references, byte identities, canary commitment bindings, exact fidelity recomputation, v2 policy identity/application binding, request-only claim boundary, supported execution materializability, communication dtype/reduction-operator agreement, exact Kineto input/output message-shape evidence, equal-split-only `all_to_all`, canonical exact rank-local GEMM-recipe projection, disabled timestamp pacing, and explicit shape/dtype privacy disclosure |
| Qualification materialization | canonical materialization ID, exact request-manifest binding, exact source-work projection and per-rank operation counts, source kernel observations, mathematical FLOPs, closed two-file inventory, exact program bytes/count, deterministic request-assisted regeneration, source-bound rank-aware issue/work/wait semantics, and no-execution/no-verdict claim boundary |
| Qualification policy | canonical policy ID, mandatory sample and warmup counts, absolute/relative acceptance boundary, deterministic bootstrap method/seed/count/confidence, explicit noise and environment comparability limits, and fixed four-state handling for incomplete, unstable, incompatible, and incorrect observations |
| Qualification observation | canonical observation ID, request/materialization/policy bindings, role and metric semantics, environment identity, raw positive samples, warmup/discard/correctness counts, and an unsigned raw-observation claim boundary |
| Qualification verdict | canonical verdict ID, exact policy and observation bindings, closed reason-code vocabulary, recomputable medians/difference/threshold/noise/confidence interval, and exactly one of pass/fail/inconclusive/incomparable |

This boundary is executable in `tests/contracts/test_json_schemas.py`:

- **valid** fixtures pass JSON Schema and the runtime validator when one exists;
- **invalid** mutations break portable shape and fail both layers when a runtime
  validator exists;
- **tampered** mutations preserve portable shape, pass JSON Schema, and fail the
  runtime semantic validator when one exists.

Verification result formats currently have producers but no corresponding
runtime validators. Their tampered fixtures document that JSON Schema cannot
establish that a claimed status follows from the included checks.
Qualification requests and materializations are different: each manifest has a
semantic validator, while directory verification additionally reads the fixed
inventory. Materialization verification also revalidates its request and
regenerates the program bytes rather than trusting the nearby program digest.
Policies, observations, and verdicts are independent artifacts so a policy can
reinterpret retained measurements without pretending that execution happened
again. Their SHA-256 identities establish content integrity, not signer
identity.

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

For trace timing, absence of `compute_overlap_us` means unknown rather than
zero. `validate_trace` can inspect that legacy shape, but canary-producing
workflows require a known value on every selected event. Producers can make the
unknown state explicit with `compute_overlap_unknown: true`; the runtime rejects
that marker if a value is also present.
