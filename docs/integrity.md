# Integrity and claim dimensions

CommCanary uses `assurance_state` as a compact compatibility summary for
structure, integrity, source correspondence, and deterministic report
recomputation. Product claims are independent dimensions, not further rungs on
that sequence.

| `assurance_state` | What was demonstrated |
| --- | --- |
| `structurally_valid` | Required shapes, types, ranges, and cross-field constraints were accepted. |
| `internally_consistent` | Stored commitments and derived canary fields recompute from the artifact. |
| `source_corresponding` | A supplied source trace independently reproduces the protected source-derived fields and commitments. |
| `model_recomputed` | A report matches a deterministic rerun of its declared model and protocol. |

The summary is cumulative only for the artifact being checked. For example, a
report that fails model recomputation may still be structurally valid, while
its referenced canary may separately be internally consistent.

The verification APIs expose the ladder as follows:

| API | Successful `status` | Successful `assurance_state` |
| --- | --- | --- |
| `verify_canary_fidelity` | `source_verified` (full) or `partial_source_verified` (prefix) | `source_corresponding` |
| `verify_report_against_canary` | `model_recomputed` | `model_recomputed` |
| `verify_canary_behavior` | `model_behavior_preserved` | `source_corresponding` |

Fidelity output records `source_coverage: full | partial`. Qualification
accepts only `source_verified` with full event coverage.

Behavior verification carries a separate `claims` object. Its successful
simulator comparison sets `model_behavior_preservation: pass`, while
`physical_execution`, `physical_conformance`, `physical_decision_fidelity`,
and `producer_authenticity` remain respectively `not_observed`, `unproven`,
`not_measured`, and `unsigned`. Those values cannot be upgraded by simulator
replay.

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
| `compiler.execution_semantic_sha256` | Logical expanded event identity, operation, dtype and reduction operator when present, broadcast root, bytes, ranks, point-to-point identity, group/concurrency, execution occurrence, and executable timing runs (gap, arrivals, overlap, and pressure). | Workload/system metadata, source bookkeeping, fidelity/error metadata, observed calibration values, and timestamps are excluded. |
| `compiler.scheduler_execution_sha256` | Exactly the same projection as `execution_semantic_sha256`. | Compatibility alias; it is not an independent guarantee. |
| `compiler.calibration_evaluation_sha256` | Logical expanded event identity plus the ordered `observed_exposed_us` runs used for calibration. | Source bookkeeping, workload/system metadata, fidelity/error metadata, and scheduler-only timing fields are excluded. |
| `compiler.artifact_provenance_sha256` | The full profiled canary: format, source format, workload, system, events, compiler metadata, and the other stored commitments. | Excludes top-level `created_at` and, inside `compiler`, the self-referential `artifact_provenance_sha256` plus derived `canary_bytes` and `byte_compression_ratio`. |
| Report `canary.sha256` | The full supplied canary except top-level `created_at`. | It identifies canary content for report recomputation; it is not a source-trace commitment. |
| Report `replay_protocol.sha256` | The declared replay protocol fields. | Excludes its own `sha256` and the enforcement-only `max_replay_events` ceiling. |

CommCanary does not preserve raw input artifacts inside compiled outputs.
The `import-kineto` CLI is the one exact-byte source boundary: it hashes the
same bounded bytes it decodes and records each profile's SHA-256 and byte size
under `system.kineto_source_profiles` without recording a path or contents.
Those identities are included in the normalized source and full artifact
provenance commitments after compilation. Retain the profiles separately if
another party must rehash them. For other input paths, exact raw-byte
provenance still requires a separately retained digest or attestation.

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

Qualification-directory verification additionally reruns the source fidelity
check and the non-expanding PARAM materializability preflight. It requires
source-bound per-event dtype and recomputes the declared communication dtype
set. Every `all_reduce` and `reduce_scatter` must carry a supported source-bound
reduction operator, and verification recomputes the exact declared operator
set. Every broadcast must carry a source-bound root inside its process group.
For Kineto sources, exact input/output element counts and normalized split
lists must be present and agree across ranks; verification recomputes the
operation-specific shape and dtype-derived byte count and refuses skipped
zero/missing-size events. Explicit splits are outside the current executor
contract. Reduction operator and root rank are part of source correspondence
and the execution-semantic commitment; no verifier or executor assumes SUM,
reconstructs a convenient shape, or derives a root from group order. The
request ID also covers the equal-split-only `all_to_all` policy, exact
source-derived rank-local GEMM recipe projection, single-inflight
issue/work/wait structure, disabled timestamp pacing, and explicit
shape-and-dtype privacy disclosure. These commitments make materialization
inputs deterministic.

Request verification opens the real bundle directory once, opens each expected
artifact relative to that directory descriptor with no-follow semantics where
available, and checks the opened descriptor with `fstat`. Each bounded file is
read once; its hash and parsed JSON come from those same bytes. Materialization
uses the verified in-memory source and canary snapshots, so replacing a nearby
pathname after verification cannot change the generated program.

Materialization verification then binds and reuses the exact request manifest,
recomputes the source-work projection, exact per-rank operation counts, source
kernel observations, mathematical FLOPs, and rederives
`replay-program.json` byte-for-byte. Source timing is retained only as an
observation and cannot change the executable recipe. Target calibration,
elapsed-gap fill, synthetic rank-arrival fill, and duration quantization are
absent. The canonical ID and program digest detect nearby mutation, but an
executor is still required separately, upstream PARAM compatibility is not
claimed, and no physical run or verdict is attested.

Reference execution starts from the descriptor-bound directory verifier rather
than trusting or reopening a nearby replay file. Every rank independently
revalidates the request, materialization, program bytes, and exact-work audit
before PyTorch import or process-group initialization. Pure preflight then validates request/wait
lifetimes, message shapes, process groups, exact rectangular recipes, repeated
work, retained samples, and tensor allocations. Rank-local operation counts
must match their encoded recipes and are budgeted before allocation. The union
of encoded groups must equal the launched dense rank domain, and each encoded
group gets a distinct runtime identity; extra
idle ranks and process-group aliasing are rejected. Preflight receives the
same parsed in-memory program whose single-read bytes matched the manifest.
After
initialization, the executor checks the actual rank, world size, and backend
against the launch contract. Aggregation requires exactly one matching sample
from every expected participant for each measured operation; duplicates cannot
mask a missing rank. Before measurement, a separately resource-counted,
untimed pass uses deterministic rank-dependent patterns to validate every
collective output under its exact source-bound reduction operator and every
receive endpoint, exchanges exact check counts across the launched world, and
fails all ranks if any data result is wrong. Group-local route identifiers are
encoded across enough element lanes for a collision-checked capacity bound;
probe values also bind the request. Dtype-aware, bounded analytical probes
distinguish SUM, AVG, MIN, MAX, and PRODUCT and expose rank-block,
reduce-scatter-shard, or destination-chunk permutations. Measured iterations retain both
per-operation issue-to-wait samples and whole-program wall time per rank; the
decision-facing makespan is the maximum rank duration, not a median of
collective calls. The resulting physical envelope binds both artifact IDs and
the program digest and is revalidated by the trusted analysis pipeline. It is
still self-reported execution evidence: no signature or environment
attestation authenticates the lab.

## Rostam execution and evidence boundary

New Rostam campaigns do not import experiment code from the mutable checkout.
Campaign preparation deterministically packages every Python source under the
`experiments.rostam` package, including package initializers, harness code,
producers, adapters, and analysis modules, into one content-addressed zipapp.
The campaign manifest binds the zipapp bytes and its mechanically generated
source inventory. A small standard-library launch shim verifies and privately
stages the separately bound bootstrap; that bootstrap verifies and privately
stages the zipapp before starting isolated Python with `PYTHONPATH` and
`PYTHONHOME` removed. Mutation tests alter every inventoried Python file and
require rejection before executor startup.

The decision-gate bootstrap treats the bound CommCanary wheel as a capability,
not a verified pathname. It keeps the source descriptor open while copying the
exact hashed bytes into a private `0700` directory, rehashes the read-only
copy, imports only from that copy, and retains it through process termination.
Replacing the original wheel before top-level import or before a later
submodule import cannot change the loaded package.

Manifest inputs and dependency artifacts follow the same rule when a path must
cross a subprocess boundary: the executor copies bytes from the verified
descriptor into the attempt's private workspace and passes only that staged
path. Harness verification returns an immutable byte snapshot. Loaders hash
and parse those same bytes instead of reopening the original pathname.
Analysis therefore cannot observe a same-size replacement after verification.

These controls establish project-code and payload identity relative to the
frozen campaign. They do not bind the base interpreter, standard library,
dynamic loader, installed Torch/CUDA userland, or host driver into one process
image. The term `frozen executor` binds the executor zipapp and analyzer bytes;
it is not whole-environment attestation. The replicated v2 campaign remains
freeze-blocked until a content-addressed runtime image is manifest-bound.
These controls also do not authenticate the campaign owner or lab, or upgrade
an `inconclusive` physical verdict.

`created_at` is intentionally volatile and excluded. Changing it alone does not
invalidate a canary.

## Tamper evidence is not authenticity

SHA-256 detects changed content relative to a trusted digest or a trusted source.
It does not identify the producer, prove authorization, or prevent an attacker
from replacing an artifact and all nearby hashes. CommCanary currently provides
no signature, certificate, transparency log, or external attestation. Use a
signature or supply-chain attestation when producer authenticity is required.
