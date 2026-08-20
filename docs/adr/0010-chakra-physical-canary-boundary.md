# ADR 0010: Use Chakra as the physical-canary carrier

Status: Accepted (2026-08-05)

## Context

CommCanary's compact trace and motif formats reduce stored representation, but
their executable expansion can still contain the complete logical workload.
That is useful for exact replay and audit, but it does not establish the cost or
regression-safety claim expected of a performance canary.

Portable execution traces and owner-to-vendor workload exchange are also
existing ecosystem capabilities. Maintaining a second general workload graph
would spend engineering effort outside CommCanary's intended contribution and
make interoperability harder.

## Decision

Use MLCommons Chakra ET as the external carrier for physical workload graphs.
CommCanary reads the official length-delimited protobuf framing, validates the
dependency graph, and retains selected metadata and node messages byte-for-byte.
An exact, owner-supplied projection binds the workload semantics, executable
work, candidate regions, and disclosure categories that cannot be inferred
safely from opaque attributes.

Reserve `physical_decision_canary.v1` for a dependency-closed subgraph that
executes fewer operations than its source and is evaluated against measured
application decisions. Selection may use only training perturbations. A
held-out decision-preservation claim requires a frozen selection commitment,
actual application measurements, predeclared false-negative, false-positive,
agreement, cost, and privacy gates, and complete candidate observations.

Exact materialization remains a conformance and measurement-floor control. A
trace-derived executor is named `trace_derived_reference`; it is not
application ground truth. The installed product surface targets `capture`,
`build`, and `gate`, while campaign freezing and publication remain research
operations.

## Consequences

- Chakra carries the executable graph; CommCanary does not define a competing
  general graph format.
- Storage compression and physical execution reduction are separate claims and
  metrics.
- Missing and synthetic application evidence can produce only blocked bundles.
- The first supported domain is single-node tensor-parallel dense-decoder
  inference with GEMM and all-reduce dependency structure.
- Instrumented capture can emit Chakra ET and its projection directly. The
  product study has digest-pinned vLLM and SGLang runner builders, a measured
  active counterexample loop, append-only Rostam execution, raw evidence
  bundles, and owner-signed private exchange.
- A completed held-out physical campaign, its 10× cost and regression-safety
  result, and an independent operator trial remain release blockers for the
  product claim.
