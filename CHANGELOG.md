# Changelog

## 0.3.0 - Unreleased

### Decision safety

- Hardened both application drivers after all four A100s on `toranj1` went
  into "GPU requires reset" on 2026-08-04 while job `180257` was running on
  them. The node was reset on 2026-08-20. Whether the job caused the state is
  not established: the same driver/GSP reset failure had hit `toranj0` and
  `toranj1` under other work in July. CUDA graphs and
  peer-to-peer custom all-reduce -- both outside the domain
  `docs/product-status.md` declares qualified -- now default **off** and must be
  opted into with `--allow-cuda-graphs` / `--allow-custom-all-reduce`, which
  warn when used. Both drivers refuse to start on GPUs already reporting a
  pending reset, both carry a `--max-runtime-seconds` watchdog that terminates
  the process so a hung engine cannot hold GPUs, and the vLLM driver now
  releases its engine, worker processes, and process group in a `finally`
  instead of leaving that to interpreter exit.
- Phase reduction, an experimental library service with no command yet
  (`commcanary.experimental.phase_representative_reduction`), now samples
  proportionally instead of by typicality.
  Percentiles are proportions, so an artifact only reproduces them if its phase
  mix matches the source. Medoid-only selection put p95/p99 58-80% under;
  reserving budget for the longest-iteration clusters recovered p99 but pushed
  the median 237% over. Retention is proportional with a floor for
  tail-carrying clusters, and timeline rebasing uses each cluster's own cadence.

- Fixed the GEMM correctness probe in the physical runner, which could not run
  at all. `a` and `b` are two-dimensional views, so `float(a[0].item())` raised
  `RuntimeError: a Tensor with N elements cannot be converted to Scalar` for
  every GEMM wider than one column. The only test covering it substitutes a
  fake torch, and the runner had never executed on a GPU, so this survived
  until the first real four-rank run. Both axes are now indexed.
- Recalibrated the injected-skew floor from measurement rather than assumption.
  The monotonic spin carries a fixed ~0.35 us overhead from its own clock reads,
  so `MINIMUM_RESOLVABLE_SKEW_US` moves from 0.5 to 4.0 -- the point where
  relative error first falls below 10 percent. Measured on A100: 4 us -> 4.35 us
  (+8.7%), 10 us -> 10.36 us (+3.6%), 20 us -> 20.36 us (+1.8%), 100 us ->
  100.36 us (+0.4%).
- Added `experiments/rostam/smoke_buffer_pool{_worker.py,.sbatch}`, a runner
  conformance check that issues more same-dtype collectives per region than
  there are buffer slots so the pool wraps and the settle-before-reuse path
  runs. It issues no qualification verdict and writes no campaign evidence.

- Fixed the four-process correctness conformance, which had failed in CI on
  `main` since at least 2026-08-03 and therefore never validated anything.
  `_torch_reduction_op` guarded on whether PyTorch exposed a `ReduceOp`
  attribute, but Gloo publishes `ReduceOp.AVG` and rejects it at call time with
  a bare `RuntimeError`, so a program the guard reported as executable failed
  mid-collective. The guard now consults backend capability and refuses with a
  named reason, and the conformance worker plans only the reductions the active
  backend can execute. Its plan constants are derived rather than written down;
  at five reductions they reproduce the previous 11 / (9, 9, 9, 10) / 38
  exactly. Verified on four ranks against torch 2.4.1+cu121 on Rostam.

- The physical gate can now return `inconclusive`. It previously offered only
  `pass`/`fail`/`incomparable`, compared baseline and candidate medians against
  a percentage threshold, and accepted a single sample per metric, so a
  difference the measurement could not resolve was reported as a pass. The
  qualification path in the same package already used a four-state verdict with
  a percentile bootstrap and an interquartile stability bound, and declared the
  2026-08-01 campaign `inconclusive` on exactly that basis; the product surface
  did not. Gate metrics are now screened for a predeclared `minimum_samples`
  replicate floor and a `noise.max_relative_iqr_pct` stability bound, then
  decided against a seeded percentile-bootstrap interval on the
  regression-oriented median difference. A mandatory metric whose interval
  straddles its acceptance boundary makes the gate `inconclusive` rather than
  passing it. The result validator independently rechecks the rule from the
  reported interval. `physical_canary_policy.v1` gains `minimum_samples`,
  `uncertainty`, and `noise`; `physical_gate_result.v1` gains per-metric
  interval, replicate, and dispersion fields.
- The physical runner no longer lets concurrent all-reduces alias one buffer.
  A single collective tensor per dtype was shared by every all-reduce in a
  region while `async_op=True` work stayed in flight until the region drained,
  so two same-dtype collectives reduced into overlapping memory and NCCL could
  serialise them — destroying the compute/communication overlap the canary
  exists to measure. Buffers are now a bounded pool and a slot's previous work
  is settled before reuse.
- Injected per-rank skew no longer uses `time.sleep`, whose granularity cannot
  resolve the single-digit microsecond skews these traces carry. A monotonic
  spin replaces it, skew below the resolvable floor is refused rather than
  silently inflated, and evidence records the mechanism as
  `host_issue_monotonic_spin` so it is not mistaken for a GPU-side arrival
  guarantee.
- A false-positive rate computed over zero passing perturbations is no longer
  reported as `0.0` satisfying the qualification gate. The denominator is
  published as `passing_perturbations` and the gate refuses a vacuous rate.

### Integrity and safety

- Signing and verifying private-exchange bundles now checks that `openssl` is
  OpenSSL 3.0 or newer before using it. The `openssl` that ships with macOS is
  LibreSSL, which has no Ed25519, and signing used to fail inside it with
  "unable to load key".

- Clustering features in `baselines` no longer fabricate concurrency. Absent
  `compute_pressure` defaulted to `0.5`, a midpoint asserting the event was half
  loaded, and absent overlap and preceding compute defaulted to `0.0`; each now
  carries a known-indicator so an undeclared field can only match another
  undeclared field.
- Chakra capture now distinguishes unknown overlap from measured zero overlap
  with an `overlap-unknown` tag, instead of tagging an undeclared trace as
  genuinely non-overlapping.
- Added phase-representative repetition reduction. Segmentation comes from a
  declared `iteration_index`, an explicit period, or an opt-in compute-gap
  heuristic; an undeclared repetition unit is refused rather than inferred from
  event structure, because every compression objective over the collective
  stream alone is maximised by discarding that structure. Retained iterations
  are rebased onto a contiguous timeline and carry weights.
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
- Added explicit integrity/correspondence summaries and independent claim
  dimensions. Simulator-relative success is now `model_behavior_preserved`;
  physical execution, conformance, decision fidelity, and authenticity remain
  separately unobserved or unproven.
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

- Added `commcanary.traffic_trace.v1`, the frozen request sequence of a
  sustained serving run, with `synthesize_traffic_trace` for synthetic traffic
  that is marked as such and refused where qualification is claimed; and
  `commcanary.serving_measurement.v1`, sustained output throughput at a fixed
  p99 time-to-first-token budget over a declared steady-state window. A
  sustained serving harness and a vLLM serving engine produce it.
- `commcanary.physical_execution_measurement.v1` now requires a `cost` block
  (setup phases, measured, instrumentation and total seconds, steady-state
  seconds per iteration) that must add up. This changes an unreleased format;
  no record without it exists.
- Workloads can declare step boundaries during capture with
  `begin_iteration`; traces carry an optional, all-or-nothing, non-decreasing
  `iteration_index` per event.
- Removed the unused `commcanary.replay.expansion` alias.

- Added the first `physical_decision_canary.v1` compiler contract above Chakra
  ET. A bounded, dependency-validating reader retains complete protobuf
  messages byte-for-byte; source-bound projections declare the narrow GEMM and
  all-reduce domain, candidate regions, work, feature coverage, and disclosure.
  An active counterexample-guided synthesizer measures training candidates,
  freezes selection before opening holdout evidence, and refuses qualification
  for absent, synthetic, incomplete, unsafe, or privacy-violating evidence. A
  reduced candidate must execute a strict subset of source nodes. Corpora bind
  the runner, application and stack subjects, environments, executable bytes,
  and GPU-seconds arithmetic. Audit bundles retain the complete application and
  physical measurement sets and recompute corpus rows; private bundles withhold
  those records behind an Ed25519-signed manifest. Instrumented capture and
  Kineto import can emit Chakra ET plus its projection directly. Immutable
  `build` bundles re-run synthesis during verification, while `gate` emits
  JSON, HTML, JUnit, and SARIF from bundle-, runner-, executable-, subject-,
  evidence-, and environment-bound observations. A real held-out campaign
  remains an open product requirement.

- Split behavior-search candidate/refinement ledgers from executable canaries.
  The compact canary summary binds an experimental evidence sidecar by exact
  canonical byte hash and selected executable identity, and the documented
  objective now says "smallest verified candidate found in the declared search
  space" rather than claiming to minimize the enlarged final artifact.

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

- Added `python -m experiments.rostam.restore_workspaces`, which restores the
  workspace files a campaign's selected attempts reference from its raw
  archive, after checking the archive against its descriptor and free disk
  space. Without it no publication could be regenerated from a clone; with it
  and the recorded analyzer, `shared-replay-20260720-r2-primary` regenerates
  byte for byte.
- Stopped ignoring `experiments/rostam/results/`, whose tracked evidence would
  otherwise never show new campaign output in `git status`; only campaign
  staging workspaces are ignored.
- CI and the declared support range now include Python 3.14.
- `docs/cli.md` documents `demo`, `fidelity`, `compile`, `replay` and
  `compare`, and a test keeps every public subcommand documented.
- Split `validate_canary` into named checks with the same order and messages;
  a differential run over 3,000 damaged canaries found no change.

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

- The trusted join now carries 95% repetition-bootstrap intervals and a
  sensitivity analysis that removes `nccl-2.20.5-tree-ll`
  (`paper/arxiv/scripts/trusted_join_uncertainty.py`, regenerated under
  test). Communication-only replay's failure to preserve the ranking survives
  both; its two-pair deficit against the microbenchmark does not, and the
  README, claims registry and design notes no longer claim it.
- Prepared an arXiv revision of the paper in `paper/arxiv/`, leaving the
  published v0.4 reconstruction untouched.

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
