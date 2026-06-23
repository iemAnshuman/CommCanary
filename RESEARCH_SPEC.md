# CommCanary research contract

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
- bounded interval encoding with explicit fidelity errors and budgets;
- exact total-gap preservation and prefix-gap error reporting;
- joint preservation of skew, offsets, overlap, pressure, and observed latency;
- optional measured `observed_exposed_us` signal and replay calibration;
- fail-closed distributed shard merge and clock-uncertainty propagation;
- phase/operation regression localisation;
- canary and replay-protocol fingerprints.

## Not implemented—and required before a strong systems-paper claim

- physical CUDA/NCCL execution of the generated canary;
- importers for Chakra, PyTorch profiler, Nsight Systems, or serving-engine
  traces;
- synthetic compute kernels calibrated to preserve interference;
- dependency-graph and communicator reconstruction;
- optimisation that directly preserves rankings across multiple target
  configurations;
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
`structural-proxy`.

## Required baselines

1. `nccl-tests` or the corresponding isolated collective microbenchmark;
2. random event/window sampling;
3. frequency- or clustering-based sampling;
4. a manually configured communication benchmark;
5. full trace replay as an accuracy upper bound;
6. CommCanary with each preservation mechanism ablated.

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
