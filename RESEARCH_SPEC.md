# CommCanary research contract

> **Evidence status (2026-07-11):** the paper and legacy Rostam notes report an
> earlier narrow physical campaign, but its complete raw attempt archive is not
> tracked in this checkout. The reported values are historical evidence, not a
> campaign independently regenerable by the current local gate. A new strong
> claim must use the immutable manifest, attempt ledger, completeness, and
> hash-verification workflow in `experiments/rostam/`.

## Defensible paper claim

> Automatically synthesize the smallest model-free communication canary that
> preserves a real distributed-inference regression—including rank-arrival
> skew, compute overlap, burst order, and tail latency—better than isolated
> collective microbenchmarks.

A generic trace recorder or simulator is not the novelty. Existing trace and
benchmark systems already cover those functions. The research contribution
must be **tail-aware, workload-faithful minimisation** and must be evaluated by
its ability to reproduce regressions and preserve configuration rankings.

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
- a single-rank observational importer for PyTorch Kineto profiler traces
  (`record_param_comms` collective metadata; no invented skew or overlap);
- a PARAM comms-replay "basic" trace exporter that expands a canary's full
  event program for physical NCCL execution via facebookresearch/param.

## Not implemented—and required before a strong systems-paper claim

- broader physical CUDA/NCCL evidence beyond the first Rostam decomposition:
  a 4x A100-PCIE single-node result now exists via `experiments/rostam/`,
  while multi-node, NVLink-class, and multi-hardware evaluations remain open;
- importers for Chakra ET, Nsight Systems, or serving-engine traces, and
  multi-rank merged import for PyTorch profiler traces (the current Kineto
  importer is single-rank and observational);
- synthetic compute kernels calibrated to preserve interference;
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
`structural-proxy`. `verify-behavior` is the gate for stronger behavioral
claims: without a passing source verification, full-source coverage, behavioral
metric comparison, and pairwise ranking check, the artifact must be described as
behaviorally unverified. `verify-behavior` compares against the full normalized
source trace by default; prefix-only or subset canaries are labelled
`partial_source_verified` and cannot receive a strong behavioral claim.
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
