# CommCanary research contract

> **Evidence status (2026-08-01):** the tracked Rostam decision gate contains a
> complete immutable manifest, attempt ledger, selection, completeness verdict,
> raw archive, publication, and bound evaluator source. Exact-work replay
> reached 26/28 pair agreement, but the predeclared outcome is `inconclusive`
> because pair uncertainty crossed decision boundaries and one configuration
> failed the stability limit. This evidence covers exact-work replay. Reduced
> canary size, execution cost, and decision fidelity remain unmeasured.

## Defensible paper claim

> Automatically synthesize the smallest model-free communication canary that
> preserves a real distributed-inference regression—including rank-arrival
> skew, compute overlap, burst order, and tail latency—better than isolated
> collective microbenchmarks.

A generic trace recorder or simulator is not the novelty. Existing trace and
benchmark systems already cover those functions. The research contribution
must be **tail-aware, workload-faithful minimisation** and must be evaluated by
its ability to reproduce regressions and preserve configuration rankings.

## Product and evidence boundary

The physical workflow has two distinct modes:

- The **exact qualification capsule** reconstructs the complete source-derived
  collective/GEMM program. It tests portable reconstruction, provenance, and
  independent replay. It makes no compression or cost-reduction claim.
- The **reduced decision canary** must physically execute a smaller
  representation. It must report decision fidelity, false-positive and
  false-negative regressions, replay-time reduction, and serialized-size
  reduction against exact replay, stratified sampling, random sampling, and
  ddmin.

The August 1 gate evaluated only the first mode. `exact_work` is now named as a
positive conformance control in new campaign contracts. A reduced physical
representation remains future work.

## Research questions

### RQ1 — When do isolated collective microbenchmarks mislead?

Find ranking inversions where a conventional microbenchmark prefers
configuration A, while the full inference workload prefers B because of skew,
overlap, queueing, or burst structure.

### RQ2 — Which trace properties are necessary?

Ablate arrival skew, compute overlap, operation order, rare tail windows,
message-size correlations, and queue-reset gaps. Measure the loss in latency
prediction, regression detection, and configuration ranking.

### RQ3 — How small can a faithful canary be?

Optimise a multi-objective target:

- serialized artifact size;
- physical replay duration;
- p50/p95/p99 error;
- exposed communication error;
- configuration-ranking disagreement;
- regression-detection precision and recall.

Event count alone is not a valid compression metric.

### RQ4 — Does it generalise?

Evaluate unseen combinations of serving engine, model family, GPU generation,
node topology, collective library/version, and workload intensity. Include both
injected and naturally occurring regressions.

## Implemented in this repository

- strict trace, canary, and report validation;
- deterministic queue-aware replay simulation;
- exact ordered periodic/run-length timing encoding;
- bounded interval encoding with explicit fidelity errors, budgets, source
  segment commitments, and representative-selection metadata;
- exact total-gap preservation and prefix-gap error reporting;
- joint preservation of skew, offsets, overlap, pressure, and observed latency;
- optional measured `observed_exposed_us` signal and replay calibration;
- fail-closed distributed shard merge and clock-uncertainty propagation;
- phase/operation regression localisation;
- behavior verification for p50/p95/p99/max/mean, queue waits, hidden
  communication, phase/op behavior, tail-event recall, and pairwise
  configuration rankings;
- separate source-normalized, scheduler-execution, calibration-evaluation,
  artifact/provenance, and replay-protocol fingerprints;
- replay-equivalent sequence motif compression for exact repeated multi-event
  programs, with flat/motif scheduler-hash equivalence;
- fail-closed behavior-gated compilation for canaries that must pass source,
  behavioral, and ranking verification;
- behavior-search compilation that exhaustively searches a declared global
  timing-sample budget range, then greedily lowers per-signature-group timing
  budgets when source, behavioral, and ranking verification still pass;
- model-recomputed report verification that rejects forged canary identity,
  replay protocol, backend, workload, or canary-summary metadata;
- research baseline generators for isolated collectives, random sampling,
  frequency representatives, clustering representatives, and stratified
  sampling (the declared kill-condition control);
- a ddmin-style decision-preserving reducer whose oracle preserves pairwise
  configuration rankings only, as the generic property-preserving reduction
  baseline (it demonstrably degenerates to single-event subsets on the
  synthetic scaffold, motivating the stricter behavioral gate);
- simulator ablation controls for skew, overlap, ordering, rare tails, queue
  reset gaps, pressure, and observed exposed latency;
- principled point-to-point identity fields for send/recv pairs;
- a synthetic ranking-inversion scaffold contrasting isolated collective
  results, full workload replay, and verified/unverified canaries;
- single- and multi-rank observational import for PyTorch Kineto profiler
  traces (`record_param_comms` collective metadata; no invented skew;
  multi-rank arrivals require an explicit shared-clock assertion or complete
  offset map; overlap is derived by unioning cross-stream compute intersections
  with linked NCCL kernel intervals only when kernel evidence is complete and
  unambiguous, otherwise it remains explicitly unknown and compilation
  refuses; broadcast root is recovered from the containing
  `c10d::broadcast_` dispatcher inputs, and missing/conflicting root evidence
  later refuses physical materialization rather than defaulting to rank 0;
  reduction operators are derived only from consistent uniquely linked NCCL
  kernel names, and incomplete evidence later refuses qualification rather
  than defaulting to SUM; exact input/output element counts and split vectors
  are retained and cross-rank checked, and qualification refuses unsupported or
  skipped shape evidence rather than rebuilding a convenient tensor shape);
- a legacy PARAM-basic-derived trace exporter that expands a canary's full
  event program while preserving source-bound collective dtype and reduction
  operator per event; the pinned research harness consumes it, but current
  upstream PARAM compatibility is explicitly not claimed;
- a portable, source-verified qualification-request bundle whose closed
  manifest binds exact source/canary/fidelity bytes while withholding any
  physical-fidelity or acceptance verdict, rejecting unsupported execution
  semantics before writing, binding communication dtype/reduction sets,
  source-validated message shapes and equal-split-only `all_to_all`,
  and a canonical projection of exact source-derived rank-local contiguous
  GEMM recipes; timestamp pacing and target compute calibration are disabled;
- a request-bound target materialization whose manifest binds the
  source-work projection, exact generated program, and required source-bound
  rank-aware async-issue/exact-work/explicit-wait executor semantics, with
  independent byte-for-byte regeneration, exact per-rank and bounded total
  rectangular GEMM work, and explicit no-execution/no-measurement/no-verdict
  claims.
- a content-addressed Rostam executor artifact that mechanically includes the
  experiment package, is verified and privately staged by a standard-library
  bootstrap, starts with `-I -S` before any project import, and dispatches all
  child Python processes without processing `.pth` or `sitecustomize` files;
- stdin-spooled, hash-bound SLURM wrappers, exact configuration-specific NCCL
  inputs, and a deterministic complete patched-PARAM artifact that is privately
  staged only for PARAM commands;
- descriptor-to-private-file staging for the manifest-bound CommCanary wheel,
  campaign inputs, dependency artifacts, and selected analysis bytes;
- operator-distinguishing and route-distinguishing collective correctness
  probes, with mutation tests and a mandatory four-process CPU/Gloo
  conformance test in a dedicated pinned-Torch CI job;
- a separately versioned, not-yet-executed exact-capsule follow-up design with
  eight complete allocation blocks, 24 balanced representation rotations,
  dynamic policy-margin bootstrap draws, and simultaneous uncertainty over all
  pair margins and reported criteria.

## Not implemented—and required before a strong systems-paper claim

- broader physical CUDA/NCCL evidence beyond the first Rostam decomposition:
  a 4x A100-PCIE single-node result now exists via `experiments/rostam/`,
  while multi-node, NVLink-class, and multi-hardware evaluations remain open;
- importers for Chakra ET, Nsight Systems, or serving-engine traces;
- an official current Chakra/PARAM replay path for
  `commcanary.source-bound-compute-recipe.v2`. The bounded
  in-package torch.distributed reference implementation covers the five
  exactly materialized collectives and has run on the narrow four-A100 gate,
  but it does not establish current upstream interoperability;
- execution of the frozen independent-allocation follow-up and broader
  hardware validation. The first exact-work decision gate is `inconclusive`;
  the new v2 campaign machinery has no physical observations, and neither
  campaign executes a reduced canary or establishes runtime or artifact-size
  reduction;
- synthetic compute kernels calibrated to preserve interference;
- simulator-side compute scheduling: after readiness is normalized,
  `compute_before_us` is descriptive and does not create a compute queue,
  shared-SM or memory-bandwidth contention, or compute/network dependencies;
- dependency-graph and communicator reconstruction;
- full per-window/per-motif optimisation that directly minimises canary size
  subject to ranking preservation across multiple target configurations;
  current behavior-search searches global timing budgets plus greedy per-group
  refinements, but it still does not search the true Pareto frontier over
  windows, motifs, and event-program structure;
- delta debugging or sequence minimisation against a real regression oracle;
- privacy leakage analysis;
- multi-engine, multi-model, multi-generation hardware evaluation.

## Fidelity contract

Every compiled artifact states whether timing is lossless or bounded. Bounded
records expose maximum gap, skew, arrival-offset, overlap, observed-latency, and
prefix-cumulative-gap errors. Users may specify budgets or require losslessness;
compilation fails instead of silently violating the contract.

When measured exposed latency is absent, “tail-aware” means structural tail
preservation, not demonstrated p99 preservation. The report labels this mode
`structural-proxy`. `verify-behavior` is a deterministic-model gate, not a
physical-fidelity gate: without passing source verification, full-source
coverage, model-metric comparison, and pairwise ranking checks, the artifact
cannot be described as `model_behavior_preserved`. `verify-behavior` compares
against the full normalized source trace by default; prefix-only or subset
canaries are labelled `partial_source_verified` and cannot receive that claim.
`compile --require-behavior-verification` applies that same gate at
artifact-generation time. `compile --behavior-search` goes further by searching
the declared timing sample limit range, then greedily lowering per-group timing
budgets when the verifier still passes. It is a verified minimization heuristic,
not a proof of global optimality.

## Required baselines

1. `nccl-tests` or the corresponding isolated collective microbenchmark;
2. random event/window sampling;
3. frequency- or clustering-based sampling;
4. a manually configured communication benchmark;
5. full trace replay as an accuracy upper bound;
6. CommCanary with each preservation mechanism ablated.

This repository now includes simulator-side baseline trace generators for items
1-3, including clustering-representative and stratified-sampling negative
controls (the latter is the declared kill-condition comparison), plus a
ddmin-style decision-preserving reducer as a generic minimisation baseline.
The physical `nccl-tests` baseline still needs real hardware execution and
comparable measurement methodology.

## Most decisive first experiment

Construct or find multiple configurations where isolated collective tests and a
real inference workload disagree on ranking. Then test whether a generated
canary retains the full workload’s ranking. Without ranking inversions, the
motivation is much weaker.

## Evaluation design

Use at least two serving engines, multiple model families and message regimes,
two GPU generations, and both single- and multi-node deployments. Split
workloads and configurations into generation and held-out evaluation sets so
the canary is not scored only on the trace from which it was derived.

Report confidence intervals and repeatability. Treat approximately 10–15%
latency error, 90%+ pairwise ranking agreement, and orders-of-magnitude size or
runtime reduction as evaluation targets—not predeclared results.

## Success and kill conditions

Proceed toward a full paper when the canary materially outperforms simple
sampling and isolated collectives on held-out ranking and regression detection,
while being substantially smaller or faster than full replay.

Reframe or stop if simple stratified sampling matches the method, if there are
no meaningful microbenchmark/workload ranking inversions, or if the physical
replay cannot preserve overlap and skew well enough to distinguish target
configurations.

## Terminology

Use **model-free** or **weight-free** for artifacts that omit model weights and
prompts. Do not claim “privacy-safe” without a formal threat model and leakage
evaluation.
