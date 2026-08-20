# CommCanary roadmap

CommCanary's target product is a workload-specific performance unit test:

> Given a private distributed-AI application and a regression policy, produce
> the shortest physical test that safely predicts whether a stack change
> should ship, then retain enough evidence to audit that claim.

Trace portability is an input to that product, not its novelty. Chakra is the
external execution-trace carrier. CommCanary owns policy-conditioned physical
minimization, asymmetric regression safety, and the resulting evidence bundle.

## Current boundary

The repository now contains two deliberately separate paths.

The older research path imports Kineto evidence, compiles compact exact-replay
artifacts, materializes the complete logical program, and retains fail-closed
Rostam evidence. Its August 1 matrix remains useful historical evidence, but
its `source` wire key refers to a trace-derived reference, not the original
application. The replicated v2 design used one compiled instruction path for
both that reference and the exact materialization control. It is therefore
retired as a product experiment. A future conformance check should use two
representative configurations and only enough repetitions to estimate the
measurement floor.

The new product path adds:

- bounded reading of official length-delimited Chakra ET protobuf streams,
  including gzip expansion limits, graph validation, and byte-preserving
  unknown fields;
- an exact source commitment and owner-supplied semantic projection for the
  narrow qualified domain;
- dependency-closed physical region selection;
- active counterexample-guided search whose site executor measures missing
  candidates, with holdout application evidence unavailable until the exact
  selection is frozen;
- first-class runtime, GPU-seconds, collective, FLOP, peak-memory,
  false-negative, false-positive, and pair-decision metrics;
- content-addressed vLLM and SGLang runner build contracts, exact application
  and stack subjects, environments, raw application and physical evidence,
  and each reduced Chakra executable;
- immutable `physical_decision_canary.v1` bundles; and
- Ed25519 owner-signed private exchange; and
- `gate` output as JSON, HTML, JUnit, and SARIF.

This is implementation, not completed scientific evidence. No real application
corpus or held-out physical qualification is checked in. Synthetic corpora are
accepted for tests but can only produce `blocked_synthetic_evidence`. Building
without a corpus produces `blocked_missing_physical_oracle_corpus`. A frozen
Rostam study must still execute successfully before its runner and evidence can
support a product claim.

## First supported wedge

The only physical-canary domain currently admitted is:

```text
single-node tensor-parallel dense-decoder inference
with GEMM and all-reduce dependency structure
```

The projection rejects unsupported operations with an explicit
`unsupported_for_physical_canary` reason. Expert-parallel all-to-all,
multi-node communication, CUDA graphs, multi-stream execution, KV-cache
pressure, and congestion are not qualified by this domain.

## Product workflow

The intended surface is `capture → build → gate`.

`capture`, `build`, and `gate` now exist. Instrumented capture can emit a source
trace, Chakra ET, and its semantic projection in one run. `import-kineto` has
the same projection path for supported complete profiles. `build` accepts the
ET, projection, predeclared policy, and an optional evidence group. It emits a
non-overwriting bundle and immediately re-verifies it. `gate` re-verifies that
bundle before comparing baseline and candidate observations.

Capture does not infer complete collective and GEMM semantics from an arbitrary
uninstrumented process. The child or its profiler must expose the all-rank
evidence required by the supported domain.

## Physical synthesis

The first synthesis implementation is an active, fail-closed
counterexample-guided loop:

1. Select the smallest recorded region set that covers the policy's required
   features.
2. Ask the site executor for missing training measurements.
3. Add the region that best removes the current severe false negatives.
4. Prune measured subsets while every training gate still passes.
5. Freeze the region and Chakra-node commitment.
6. Open the held-out application callback for the first time, then measure the
   frozen candidate on those perturbations.
7. Issue a qualified certificate only if every held-out policy gate passes.

The executor is injectable. The Rostam implementation freezes candidate bytes,
uses fresh append-only SLURM attempts, validates every returned measurement,
and records the complete request ledger. Bundle verification recomputes corpus
observations from the retained raw evidence.

Random sampling, stratified sampling, ddmin, independent exact replay, and a
communication-only microbenchmark are corpus methods, not privileged oracles.
A policy can require all of them before qualification.

## Evidence required before a product claim

Thresholds belong in the policy before measurements are collected. The first
real study should require at least:

1. actual application ground truth;
2. 10× minimum held-out physical runtime reduction;
3. zero false negatives for held-out regressions at or above the declared
   severity boundary;
4. no more than 10% false positives;
5. at least 90% pair-decision agreement;
6. stable decisions across independent allocations and days;
7. independent exact replay and the declared reduction baselines;
8. a second engine or materially different workload;
9. one operator other than the author completing the workflow from the docs;
10. a digest-pinned OCI or Apptainer runner.

Until those measurements exist, CommCanary remains a research alpha with a
physical-canary compiler contract. It is not a validated performance gate.

## Privacy and exchange modes

`internal` and `full_audit` bundles retain the source ET, projection, policy,
search ledger, and oracle corpus when present. The leakage assessment scores
five declared inference categories and separately records whether opaque
Chakra attributes were reviewed. This score is an audit aid, not an
information-theoretic privacy proof.

`private_exchange` requires an Ed25519 key pair and a qualified bundle. It omits
raw source, corpus, application, and physical evidence while signing their
manifest commitments. The receiver supplies the trusted public key and
revalidates the reduced ET, policy, active ledger, leakage assessment, and
certificate.

## Research and release boundary

Rostam campaign freezing, cross-commit joins, paper regeneration, and archived
exact-replay evidence remain under `experiments/rostam/`. They are research
operations and should not define the normal installed workflow.

The product research package now contains independent vLLM and SGLang plans,
plus a separate digest-locked study for the vLLM issue 2971 version pattern.
That historical study uses the issue-date model revision and real weights but
runs on four A100s with a fixed offline request set, so it cannot claim an exact
reproduction or causality.

Version 0.3.0 remains unreleased. A release still requires the repository's
explicit release gate and a separate tag and publication decision. If the
narrow physical study cannot meet its predeclared reduction and safety gates,
the product must be described as a verifiable exact workload capsule rather
than a fast regression canary.
