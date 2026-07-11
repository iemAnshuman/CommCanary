# Integrity and assurance

CommCanary reports assurance as a ladder. The `assurance_state` field names the
highest level demonstrated by that operation; existing `status` values remain
unchanged for API and CLI compatibility.

| `assurance_state` | What was demonstrated |
| --- | --- |
| `structurally_valid` | Required shapes, types, ranges, and cross-field constraints were accepted. |
| `internally_consistent` | Stored commitments and derived canary fields recompute from the artifact. |
| `source_corresponding` | A supplied source trace independently reproduces the protected source-derived fields and commitments. |
| `model_recomputed` | A report matches a deterministic rerun of its declared model and protocol. |
| `behaviorally_verified` | The declared behavior, tail, and configuration-ranking checks pass against the full source. |

The ladder is cumulative only for the artifact being checked. For example, a
report that fails model recomputation may still be structurally valid, while
its referenced canary may separately be internally consistent.

The verification APIs expose the ladder as follows:

| API | Existing successful `status` | Successful `assurance_state` |
| --- | --- | --- |
| `verify_canary_fidelity` | `source_verified` | `source_corresponding` |
| `verify_report_against_canary` | `model_recomputed` | `model_recomputed` |
| `verify_canary_behavior` | `behaviorally_verified` | `behaviorally_verified` |

Profiled v2 canaries must pass internal hash recomputation during
`validate_canary`. Legacy artifacts are accepted only with
`allow_legacy_unverified=True`; that opt-in establishes structural readability,
not the missing integrity commitments.

## Digest coverage

All JSON projections below use UTF-8 canonical JSON with sorted object keys,
compact separators, and non-finite numbers rejected.

| Commitment | Covered projection | Deliberate exclusions and limits |
| --- | --- | --- |
| `compiler.source_trace_sha256` | The selected trace after event ordering and any `max_events` selection, including its remaining top-level fields, workload, system, and full selected event objects. | This is a normalized-JSON commitment, not a digest of the original file bytes. Whitespace, input key order, and unselected events are not covered. |
| `compiler.source_normalized_sha256` | Exactly the same projection as `source_trace_sha256`. | Compatibility alias; it is not an independent guarantee. |
| Leaf `event.source.digest` | The ordered source event IDs represented by that stored leaf. Each canonical ID is followed by a NUL separator before SHA-256 is finalized. | Event contents other than IDs are excluded and are protected by the other source and semantic checks. |
| Motif-wrapper `event.source.digest` | Canonical `{"sources": [...]}` containing the independently ordered leaf-source digests for every motif occurrence. | It summarizes the leaf ID commitments; it does not replace source event-content verification. |
| `compiler.execution_semantic_sha256` | Logical expanded event identity, operation, bytes, ranks, point-to-point identity, group/concurrency, execution occurrence, and executable timing runs (gap, arrivals, overlap, and pressure). | Workload/system metadata, source bookkeeping, fidelity/error metadata, observed calibration values, and timestamps are excluded. |
| `compiler.scheduler_execution_sha256` | Exactly the same projection as `execution_semantic_sha256`. | Compatibility alias; it is not an independent guarantee. |
| `compiler.calibration_evaluation_sha256` | Logical expanded event identity plus the ordered `observed_exposed_us` runs used for calibration. | Source bookkeeping, workload/system metadata, fidelity/error metadata, and scheduler-only timing fields are excluded. |
| `compiler.artifact_provenance_sha256` | The full profiled canary: format, source format, workload, system, events, compiler metadata, and the other stored commitments. | Excludes top-level `created_at` and, inside `compiler`, the self-referential `artifact_provenance_sha256` plus derived `canary_bytes` and `byte_compression_ratio`. |
| Report `canary.sha256` | The full supplied canary except top-level `created_at`. | It identifies canary content for report recomputation; it is not a source-trace commitment. |
| Report `replay_protocol.sha256` | The declared replay protocol fields. | Excludes its own `sha256` and the enforcement-only `max_replay_events` ceiling. |

CommCanary does not currently preserve or commit the raw input artifact bytes.
If exact byte provenance matters, retain the input separately with its own
digest or attestation.

## What validation catches

The integrity profile requires both source-digest aliases, both execution-digest
aliases, the calibration commitment, artifact provenance, and `first_id`,
`last_id`, and `digest` on every stored source block, including sequence-motif
wrappers and their stored children. Removing a commitment or editing a protected
field without updating hashes fails internal validation.

A producer can always edit an unsigned artifact and recompute its internal
hashes. Source-assisted verification therefore does not trust those hashes: it
reconstructs `source_format`, workload/system correspondence, event signatures,
bounded-interval commitments, and every stored source ID bound and digest from
the supplied trace. A rehashed producer mutation can remain
`internally_consistent`, but it cannot become `source_corresponding` to the
unchanged source.

`created_at` is intentionally volatile and excluded. Changing it alone does not
invalidate a canary.

## Tamper evidence is not authenticity

SHA-256 detects changed content relative to a trusted digest or a trusted source.
It does not identify the producer, prove authorization, or prevent an attacker
from replacing an artifact and all nearby hashes. CommCanary currently provides
no signature, certificate, transparency log, or external attestation. Use a
signature or supply-chain attestation when producer authenticity is required.
