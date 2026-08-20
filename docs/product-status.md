# Product status

CommCanary 0.3.0 is an unreleased **research alpha**. The implementation is
ready for a predeclared physical study. It is not ready to be sold or described
as a validated performance gate.

The product hypothesis is:

> Given a private distributed-AI workload and a regression policy, produce the
> shortest physical test that safely predicts whether a stack change should
> ship, with enough evidence to audit the decision.

This file defines the reader-facing evidence boundary. The machine-readable
counterpart is [`claims/public-claims.yaml`](../claims/public-claims.yaml).
`HANDOFF.md` remains authoritative for the live repository and Rostam
checkpoint.

## Readiness summary

| Area | Status | Meaning |
|---|---|---|
| `capture → build → gate` | Implemented and tested | The commands and fail-closed artifact contracts exist. |
| Chakra graph ingestion | Implemented and tested | Length-delimited ET input, graph validation, and dependency-closed selection exist. |
| Active candidate search | Implemented and tested | Training/holdout separation and selection-before-holdout access are enforced locally. |
| vLLM and SGLang application drivers | Implemented and unit-tested with fakes | They have not produced a real perturbation corpus. |
| Four-rank physical runner | Implemented and tested locally | Its product OCI/SIF image has not been built on Rostam. |
| Exact-work decision fidelity | Promising but inconclusive | Point estimates passed; uncertainty and one unstable configuration prevented qualification. |
| Reduced physical runtime | Unmeasured | No reduced physical canary has run on Rostam. |
| Held-out regression safety | Unproven | No real held-out result exists. |
| Independent operation | Unproven | The quickstart exists; an independent operator has not completed it. |
| Production CI gate | Not validated | Do not use this label. |

## Supported domain

The first qualification study is deliberately narrower than the eventual
commercial scope:

```text
single node
exactly four NVIDIA GPUs for the first study
tensor-parallel dense-decoder inference
vLLM first; SGLang as the planned replication
GEMM and all-reduce dependency structure
```

The future-plan scope may expand this to four through eight NVIDIA GPUs only
after the four-GPU study produces stable evidence. The current study does not
qualify expert-parallel all-to-all, multi-node communication, CUDA graphs,
multi-stream execution, KV-cache pressure, or network congestion.

General Kineto import and PARAM export are compatibility surfaces. They do not
expand the qualified product domain.

## What is implemented

### Trace and graph boundary

- Chakra ET ingestion validates message framing, node identity, dependencies,
  acyclicity, and resource limits.
- Complete instrumented capture or supported Kineto input can produce a
  source-bound Chakra projection.
- Candidate selections must be dependency-closed and execute fewer Chakra
  nodes than the source.
- Unsupported operations fail with an explicit
  `unsupported_for_physical_canary` reason.

### Search and evidence boundary

- Counterexample-guided search uses training perturbations while the selection
  is mutable.
- The selected candidate is committed before the holdout callback can load
  held-out application evidence.
- Application, physical-execution, policy, ledger, and bundle identities are
  content-addressed and independently revalidated.
- Synthetic corpora can exercise the workflow but cannot qualify a canary.

### Execution and exchange boundary

- vLLM and SGLang offline tensor-parallel application drivers emit application
  measurements. Their local tests use fakes; this is implementation evidence,
  not physical qualification.
- The four-rank physical runner dispatches supported collectives and GEMMs with
  correctness and telemetry checks.
- The Rostam workflow can build a digest-pinned OCI archive and Apptainer SIF,
  freeze a study, submit append-only attempts, resume it, and bind the result.
- Full-audit bundles retain evidence. Private-exchange bundles may withhold raw
  evidence while signing content commitments with Ed25519.
- JSON, HTML, JUnit, and SARIF gate outputs exist.

`capture` runs the supplied workload command with the same authority as running
that command directly. The Python process is not a sandbox.

## Existing physical evidence

### Exact-work configuration study

The frozen `decision-gate-20260801-r3` campaign measured eight configurations
and 28 pairs on four A100 GPUs. Its point estimates were:

| Metric | Result |
|---|---:|
| Pair agreement | 26/28 (92.86%) |
| Kendall tau-b | 0.857 |
| Median error | 1.55% |
| 95th-percentile error | 4.05% |
| False negatives | 1 |
| False positives | 1 |

The frozen outcome is **inconclusive**. Bootstrap intervals crossed decision
boundaries and one configuration exceeded the predeclared stability limit.

This experiment replayed the complete source-derived program. It measured
neither reduction nor GPU-cost savings.

### Same-node exact-work diagnostic

Source job `177966` and replay job `178515` ran on the same node. The source
median was 1,434.112 microseconds; replay was 1,541.0015 microseconds, a
+7.45% signed error. All 32 deterministic data checks passed. The retained
claims remain `physical_fidelity: unproven`,
`multi_configuration_ranking: not_measured`, and
`qualification_verdict: not_issued`.

### Evidence that does not exist

- No reduced physical canary execution.
- No real application perturbation corpus.
- No held-out application or candidate result.
- No product runner built on Rostam.
- No independent-operator completion.
- No customer or production deployment.

## Claims boundary

The claims registry is the source for website, README, and narrator language.
Its initial categories are summarized here.

### Allowed

- The `capture → build → gate` surface exists.
- Chakra ingestion and dependency-closed candidate construction exist.
- The exact-work point estimates above are published, with an inconclusive
  verdict.
- Signed private exchange is implemented.

### Allowed only after evidence

| Claim | Required evidence |
|---|---|
| Predicts stack-change decisions | Zero severe held-out false negatives, no more than 10% false positives, and at least 90% pair-decision agreement across independent allocations and days. |
| Reduces physical runtime by at least 10× | A frozen reduced candidate measures at least 10× lower runtime while every safety gate passes. |
| Preserves held-out regressions | Selection is frozen before holdout access and the predeclared held-out gates pass. |
| Safe for CI gating | The physical gates pass, a second engine or materially different workload replicates the result, and an independent operator succeeds. |

### Forbidden now

- Production validated.
- Privacy safe or anonymous.
- Works across arbitrary models.
- Multi-node qualified.
- Production-ready GPU regression platform.

## Privacy boundary

CommCanary artifacts can omit prompts and model weights, but communication
sizes, operation order, topology, timing, and provenance can still disclose
deployment details. The leakage assessment is an audit aid, not a privacy
proof. A signature proves possession of a key; it identifies an owner only
when the receiver obtained that public key through a trusted channel.

## Release boundary

Publishing 0.3.0 requires all of the following:

1. A recorded outcome for the vLLM physical study, including an honest negative
   or inconclusive outcome.
2. An independent operator completing the documented workflow and reporting
   friction.
3. `python -m tools.verify --release` passing from a supported development
   environment.
4. README, website, and social claims agreeing with the claims registry.
5. A separate explicit decision to tag and publish the release.

A focused test suite, a successful synthetic bundle, or a promising point
estimate is not release evidence.

## Precedence

Read `HANDOFF.md` before any Rostam operation. Generated physical evidence and
its frozen verdict take precedence over prose. No documentation update may
turn an `inconclusive` result into a pass or imply held-out evidence that does
not exist.
