# Changelog

## 0.3.0 - Unreleased

### Integrity and safety

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
  catalogs remain fail-closed until their Rostam-only GEMM calibration is
  supplied.

- Added `--overlap-structure` to `export-param`: collectives are emitted for asynchronous issue with explicit `wait` entries placed after the next gap's gemm entries, reconstructing compute/communication concurrency; issue entries carry an `issue` marker so parsers separate issue lines from completion-bearing wait lines.
- Added compute-fill mode to `export-param` (`--compute-fill-us-per-gemm`, `--compute-fill-gemm-dim`): inter-collective gaps export as PARAM `{"compute": "gemm"}` entries instead of idle timestamps, so physical replay reproduces compute/communication interference. Replay compute-filled traces without `--use-timestamp`.

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
- Added `commcanary export-param`: expands a canary's full event program (motifs, patterns, run-length weights) into a PARAM comms-replay "basic" JSON trace with element counts, PARAM's asymmetric size conventions for `all_gather`/`reduce_scatter`, process-group ids, matched send/recv entry pairs (with `src_rank`/`dst_rank`) per point-to-point transfer, and cumulative `startTime_ns` timestamps for `--use-timestamp` replay — a physical NCCL execution path for minimized canaries.
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
