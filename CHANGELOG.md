# Changelog

## 0.3.0 - Unreleased

### Integrity and safety

- Made absent compute/communication overlap an explicit unknown instead of
  silently coercing it to `0.0`. Canary compilation, behavior search,
  reduction, and overlap-preserving baselines now require a measured or
  deliberately constructed value.
- Kineto import now derives overlap only from unique external-id linkage to
  complete NCCL kernel intervals and the union of concurrent compute kernels
  on other streams of the same device. Missing/malformed/ambiguous evidence
  remains explicitly unknown, per-event reasons and totals are recorded, and
  cross-configuration physical decision fidelity remains an uncompleted
  experiment. Both ranks of the public 2-GPU ResNet-50 profile from PyTorch
  issue #131462 import with 7/7 overlap-known events and compile losslessly as
  a real-format check.
- `import-kineto` now accepts multiple rank-local profiles and reconciles them
  through the existing fail-closed capture contract. Missing or conflicting
  rank contributions fail; rank-local compute values are retained; arrival
  skew is usable only after an explicit shared-clock assertion or a complete
  additive clock-offset map. The public two-rank profile merges 14 records into
  7 logical collectives and compiles losslessly under that explicit assertion.
- `import-kineto` now binds every successfully decoded source profile by its
  exact byte SHA-256, byte size, and distributed rank without recording local
  paths or filenames. The same bounded bytes are hashed and parsed in one pass;
  multi-rank identities are deterministic by rank and survive compilation
  inside the source and artifact-provenance commitments.
- Kineto clock origins are now transient import state rather than shareable
  metadata. Events remain rebased and explicitly calibrated cross-rank arrival
  offsets remain intact, while raw monotonic starts and wall-clock base times
  are omitted from imported traces.
- Kineto collective dtype is now normalized and preserved per event through
  capture merge, compilation identity, fidelity checks, semantic hashes, and
  PARAM element-count export. Qualification preparation refuses missing or
  unsupported dtype and other non-materializable PARAM semantics instead of
  silently applying a global float32 default or rounding byte counts up to a
  different communicated volume.
- Broadcast root rank is now recovered from a containing same-thread
  `c10d::broadcast_` event's concrete dispatcher inputs, then preserved through
  capture reconciliation, compilation, fidelity, semantic hashes, baselines,
  materialization, and reference execution. Missing, out-of-group, or
  conflicting roots fail closed at the relevant boundary; qualification and
  export never guess the first process-group rank.
- Reduction operators are now explicit execution semantics. Kineto import
  derives SUM, PRODUCT, MIN, MAX, or AVG only from consistent uniquely linked
  NCCL kernel names; compilation, fidelity, semantic hashes, operation
  identities, and materialization preserve the result. General observational
  traces may leave it unknown, but qualification refuses missing reduction
  semantics instead of inheriting PyTorch's SUM default. The reference executor
  dispatches the exact bound `ReduceOp` and checks its result before timing.
- Kineto input/output element counts and split vectors are now normalized,
  retained, and compared across rank-local profiles instead of being collapsed
  irreversibly to `max(in, out)`. Qualification independently checks standard
  operation ratios, dtype-derived bytes, and skipped-size inventory, binds an
  equal-split-only `all_to_all` policy, and refuses explicit or ambiguous split
  evidence rather than generating a different executable shape.
- Added explicit assurance states for structural validity, internal
  consistency, source correspondence, model recomputation, and behavioral
  verification; documented that embedded hashes are not authenticity.
- Recompute profiled canary provenance and source commitments recursively,
  including motif wrappers/children, and independently verify source IDs,
  bounds, and digests so a producer-side rehash cannot forge source
  correspondence.
- Added one immutable `ResourceLimits` policy across bounded JSON loading,
  validation, motif/timing preflight, replay, behavior search, reduction,
  capture merge, and PARAM export; duplicate keys, non-finite constants,
  excessive nesting, checked-count overflow, and over-budget expansion fail
  before iteration/materialization.
- Hardened capture path containment, direct-output ownership across processes,
  fork/global-recorder lifecycle, linear rank-domain comparison, and bounded
  checksum-preserving failure bundles.
- Public compile/replay/compare/baseline/reduction/verification/interop outputs
  are detached from caller-owned nested input.

### Contracts and API

- Added `commcanary.qualification_request.v1` and the
  `prepare-qualification`/`verify-qualification` owner-to-lab workflow. A new,
  fixed-inventory directory binds exact trace/canary/fidelity bytes and all
  canary commitments, recomputes source fidelity independently, rejects
  symlinks and rehashed semantic tampering, and explicitly states that physical
  measurement, physical fidelity, and a qualification verdict are absent.
  The request now binds source-derived communication dtypes, reduction
  operators, validated message shapes, and equal-split `all_to_all` policy plus
  a canonical projection of each rank's exact source-derived contiguous GEMM
  recipe. It discloses that shared GEMM shapes and dtypes may reveal model
  structure, disables timestamp pacing, and rejects missing or unsupported
  recipes instead of reconstructing elapsed time as compute.
- Added `commcanary.qualification_materialization.v1` and the
  `materialize-qualification`/`verify-materialization` receiving-lab workflow.
  It binds exact request-manifest bytes, the source-work projection, per-rank
  operation counts, source kernel observations, mathematical FLOPs, and
  deterministic replay-program bytes/count. Independent verification
  regenerates both the audit and the program byte-for-byte. The executable
  sequence is asynchronous collective issue, the exact rectangular GEMM
  recipe belonging to each rank, and an immediate explicit wait; target
  calibration, elapsed-gap fill, synthetic arrival fill, and duration
  quantization are not accepted. It explicitly requires a conforming adapter
  and withholds execution, measurement, and verdict claims.
- Added a request-bound `execute-materialization` torch.distributed reference
  runner. Every rank revalidates the request/materialization and preflights
  process groups, all supported collective and point-to-point operations,
  request/wait lifetimes, floating compute dtype, exact rectangular GEMM
  dimensions, repeated work, retained samples, and tensor allocation before
  importing PyTorch. Each rank executes only its declared recipe; different
  arrival behavior therefore emerges from source-bound work rather than
  synthetic delay. The runner aggregates rank-local issue-to-wait timings into
  a bound diagnostic while explicitly withholding physical-conformance,
  fidelity, observation-format, and qualification-verdict claims. All three
  GEMM matrices, including the reused output, are preallocated and included in
  the per-rank memory proof so the timed loop cannot hide output allocation.
  Default-group initialization and every encoded subgroup now use one explicit
  positive distributed timeout (300 seconds by default, capped at 3,600 by
  `ResourceLimits`) and the rank-0 diagnostic records it, instead of inheriting
  PyTorch's backend-dependent 10- or 30-minute defaults.
- Executed the exact-work reference path on four A100-PCIE-40GB GPUs. The
  request/materialization-bound source and replay medians were 1,434.112 us and
  1,541.0015 us (+7.4533578967%), with 32/32 deterministic data checks passing.
  The retained observation deliberately issues no qualification verdict: one
  configuration and a post-observation tolerance cannot establish decision
  fidelity.
- Corrected the interoperability boundary: current upstream PARAM removed
  `basic` and Kineto trace parsing in favor of Chakra host execution traces,
  while CommCanary's pinned historical basic replayer is blocking. Legacy
  `export-param` remains for reviewed integrations, but qualification artifacts
  now state `upstream_param_compatibility: not_claimed` instead of presenting
  that encoding as a current upstream executable.
- Relicensed from MIT to Apache License 2.0 for its express patent grant and
  mandatory attribution, added a `NOTICE` file that ships in the wheel and
  sdist, and reserved the CommCanary name under section 6. See ADR 0009.
- Published Draft 2020-12 schemas, literal canonical/hash vectors,
  compatibility/unknown-field/coercion decisions, equivalence/determinism
  characterization, and exact comparison boundary fixtures for every supported
  artifact family.
- Added an immutable format-capability query, metadata-derived package version,
  deliberate top-level stable API, explicit experimental namespace, and PEP 561
  typed-package marker.
- Stabilized CLI exits: 1 for a valid negative verdict, 2 for usage, 3 for
  CommCanary application errors, 4 for child/workload failure, and 130 for
  interruption. `--version` reports package/format/canonicalization/model
  versions and `--diagnostics-json` emits JSON Lines on stderr.
- Added lifecycle timing and bounded-work progress diagnostics for behavior
  search and reduction, rejects method-inapplicable baseline flags, and makes
  `render-html` the primary spelling while retaining `report` as a deprecated
  compatibility alias through 0.4.
- The module-level capture helper now has a typed signature and supports the
  clearer `byte_count=` spelling while retaining `bytes=` compatibility.
- HTML reports declare a self-contained content-security policy, structurally
  escape untrusted content, and explicitly show summary-only data when samples
  are unavailable instead of synthesizing a distribution.

### Engineering and reproducibility

- Split artifact contracts, compilation, replay, verification, services,
  comparison, adapters, and reporting by dependency boundary behind tested
  compatibility facades; an AST gate now rejects upward imports, cycles,
  unclassified modules, and cross-boundary private imports.

- Added one canonical fast/full/release verification command with Ruff, strict
  mypy, coverage policy, schema/shell/workflow/docs checks, reproducible build,
  exact-wheel installation tests, artifact inventory, SHA256SUMS, and SPDX 2.3
  SBOM generation.
- Release staging now includes the reviewed docs, schemas, examples, benchmark,
  experiment, test, and verification sources referenced by the README; archive
  inspection rejects missing members, the unreproducible historical paper, and
  private/generated paths, while release mode requires a clean HEAD and unique
  dated changelog identity.
- CI tests supported Python versions from built artifacts, pins every action to
  a reviewed full commit SHA, separates low-privilege release building from
  OIDC publishing, enables signed PyPI attestations, and reviews/updates
  dependencies automatically.
- Added deterministic local 1K/10K/100K benchmark fixtures and an isolated
  wall/RSS/allocation/semantic-hash runner.
- Added a pinned weekly scale-observation workflow that retains deterministic
  three-repeat results but does not fabricate a regression threshold before a
  stable runner history is reviewed.
- Added immutable experiment manifests, terminal attempts, explicit retry
  selection, fail-closed completeness, a bounded shell-free local cell runner,
  and a golden mini-campaign without SLURM.
- Bounded every experiment control/result JSON reader, campaign expansion, and
  physical stdout/stderr path; exit-time output bursts are truncated and
  recorded, while mocked bounded probes capture driver, GPU, topology,
  binding, clock, Python, Torch, CUDA, and NCCL observations in attempts.
- Added a completeness-gated multi-campaign analyzer that binds physical rows
  to manifest workloads, configurations, dependencies, selected attempts,
  runtime identities, inputs, and trace hashes before deriving ranking,
  Kendall, regression, cost, aggregate, and paper-fragment outputs.
- Added a post-run archive descriptor bound to exact manifests, selections,
  and completeness verdicts; legacy directory-glob analysis now requires an
  explicit unsafe flag and watermarks every JSON/Markdown output.
- Replaced mutable third-party patching with a reviewed PARAM commit/archive,
  contextual patch, and preimage/postimage hash contract; overlap/shared
  catalogs fail closed until their Rostam-only GEMM calibration is supplied
  and manifest-bound.
- Executed new manifest-bound core, shared-replay, and explicit-overlap
  campaigns on Rostam with 160/160, 40/40, and 80/80 selected successful cells
  respectively, persisted zero-issue completeness verdicts, verified normalized
  raw archives, and byte-identical regenerated JSON/CSV/Markdown publications.
  The trusted join over all 280 selected cells also regenerates byte-for-byte.
- Published the complete non-workspace Rostam control plane, normalized raw
  archives, target environment evidence, exact-work bundle, and generated
  publications. Large normalized archives use Git LFS; redundant cluster
  staging workspaces remain excluded.
- Fixed the trusted-join guard so campaigns of different catalog profiles can
  be joined: it compares the analysis-relevant policy subset instead of whole
  policy documents, whose `catalog_profile` and `input_paths` differ by
  construction. Input identity is still enforced by `(sha256, size_bytes)` per
  input id, and divergent analysis semantics still fail closed.
- Added an opt-in cross-commit evidence bridge. A candidate/reviewed contract
  binds exact manifests, selections, verdicts, two repository identities, and
  every analyzer/harness/schema byte. Mixed-repository analysis is accepted
  only after the current implementation regenerates the complete historical
  publication byte-for-byte; policy/input exemptions must exactly equal the
  observed manifest differences, and the ordinary strict join is unchanged.
- Gave the publication serializer an explicit 4,000,000-item JSON budget. A
  280-cell joined aggregate measures 1,013,696 items in 8,101,263 bytes and
  exceeded the shared 1,000,000-item default; readers of untrusted input keep
  their original budgets.

- Added `--overlap-structure` to `export-param`: collectives are emitted for
  asynchronous issue, but only the source `compute_overlap_us` slice of the
  following gap precedes the explicit wait. The remainder and all rank-arrival
  fill are serialized, zero source overlap cannot create concurrency, and
  overlap crossing the next communication start refuses as an unrepresented
  dependency graph. Issue entries carry an `issue` marker so parsers separate
  issue lines from completion-bearing wait lines.
- Added compute-fill mode to `export-param`
  (`--compute-fill-us-per-gemm`, `--compute-fill-gemm-dim`):
  inter-collective gaps export as `{"compute": "gemm"}` entries instead of
  idle timestamps so a conforming executor can apply interference. Timestamp
  pacing must stay disabled; deterministic export alone is not physical replay
  evidence.

### Research fidelity

- Strengthened `verify-behavior` so it separately reports representation fidelity, source verification, behavioral fidelity, and configuration-ranking status.
- Added queue-wait distribution checks, phase/op behavior checks, tail-event recall, and pairwise backend ranking agreement across latency metrics.
- Added source commitments to bounded timing intervals: source count, source segment SHA-256, source gap sum, representative-selection method, representative index, and a complete error vector.
- Added separate source-normalized, scheduler-execution, calibration-evaluation, and artifact/provenance fingerprints.
- Added an adversarial ranking-inversion scaffold and tests that label too-small behaviorally lossy canaries as unverified.
- Added replay-equivalent sequence motif compression for exact repeated multi-event programs, with scheduler-hash equivalence to flat encodings.
- Added fail-closed behavior-gated compilation via `--require-behavior-verification`.
- Added behavior-search compilation that exhaustively searches timing sample limits and selects the smallest source-, behavior-, and ranking-verified canary.
- Added greedy per-group behavior-search refinement so quiet signature groups can use lower timing budgets while ranking-sensitive groups retain detail.
- Strengthened `verify-report` so forged canary identity, replay protocol, backend, workload, or canary-summary metadata fails model recomputation.
- Added research baseline trace generators for isolated-collective, random-sampling, frequency-representative, and clustering-representative controls, plus a `commcanary baseline` CLI.
- Added a stratified sampling baseline generator (`baseline --method stratified`), the kill-condition control named by RESEARCH_SPEC.md.
- Added a ddmin-style decision-preserving reducer (`commcanary reduce`) that minimizes a trace under a pairwise configuration-ranking oracle, as a generic property-preserving reduction baseline for behavior-search comparisons.
- Closed a canary validator gap: every timing sample now needs a weight that matches its declared source interval (single-index records are weight one), and sample intervals must tile the repeat range contiguously, so occurrences can no longer be silently double-counted or dropped.
- Rejected non-ASCII digit strings in integer parsing.
- `examples/make_synthetic_trace.py` now writes `llama70b_tp8_trace_long.json` instead of silently overwriting the small checked-in fixture.

### Ecosystem interop

- Added `commcanary import-kineto`: single-rank observational import of `record_param_comms` collective metadata (op, dtype, element counts, process-group ranks, timestamps) from PyTorch profiler traces (torch >= 2.2); timestamps are rebased to the trace start, truncated non-uniform rank lists fail closed instead of fabricating membership, unmapped collectives become `custom_op` events, control ops are skipped and counted, and no cross-rank skew or overlap is invented.
- Added per-invocation `import-kineto --max-input-bytes` and
  `--max-json-items` overrides for trusted local profiler output while keeping
  both bounded defaults fail-closed for untrusted input.
- Added `commcanary export-param`: expands a canary's full event program
  (motifs, patterns, run-length weights) into the historical PARAM
  comms-replay “basic” JSON encoding with element counts, asymmetric size
  conventions for `all_gather`/`reduce_scatter`, process-group ids, matched
  send/recv entries, and cumulative timestamps. This remains a legacy/pinned
  integration encoding, not a current upstream PARAM execution claim.
- Tightened `verify-behavior` so it replays the full normalized source trace by default and marks prefix/subset canaries as partial-source rather than behaviorally verified.
- Added simulator ablation controls for skew, overlap, ordering, rare tails, queue-reset gaps, pressure, and observed exposed latency.
- Strengthened point-to-point semantics with sender/receiver, tag, channel, message sequence, and send/recv observations.

## 0.2.0

### Correctness

- Preserved exact sub-microsecond timing through periodic compression and replay.
- Rejected ambiguous mixed timestamp/gap traces and conflicting timing fields.
- Added queue-aware deterministic scheduler model v4 and counter-based randomness.
- Reconciled report metrics, breakdowns, samples, and calibration data.
- Hardened one-rank skew, pattern sums, interval coverage, integer, and finite-number validation.

### Research fidelity

- Added optional measured `observed_exposed_us` as a joint timing feature.
- Added model calibration error to reports.
- Added lossless compilation mode and explicit approximation budgets.
- Added prefix-cumulative-gap error and serialized-byte compression metrics.
- Added phase- and operation-level regression localisation.

### Capture

- Added UUID-qualified shard names and generation-ordered saves.
- Rejects mixed capture sessions, conflicting workload/session metadata, missing or duplicate ranks, and partially observed latency.
- Preserves clock uncertainty instead of inventing cross-rank skew.
- Validates recorder inputs before storage.

### Performance

- Uses incremental source hashing, direct timing comparisons, compact replay arrays, and event-local counter-based random generation.
