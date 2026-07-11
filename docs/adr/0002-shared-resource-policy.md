# ADR 0002: Shared resource policy

- Status: Accepted
- Date: 2026-07-11

## Context

Compact communication artifacts can describe much more logical work than their
wire size suggests. JSON depth and item counts, motif repeats, timing weights,
replay iterations, behavior-search ranges, capture shards, and ecosystem export
entries can otherwise trigger excessive allocation or CPU work. Checking only
after expansion is too late, and validating under one ceiling before executing
under another creates a time-of-check/time-of-use gap.

Python integers do not overflow naturally, so arbitrary-precision arithmetic
can also spend work on attacker-controlled counts unless the supported count
range is explicit.

## Decision

All untrusted-input paths accept one frozen `ResourceLimits` value and pass that
same instance through loading, semantic validation, counting, hashing,
expansion, replay, verification, search, reduction, capture merge, and export.

Counts that control allocation or iteration are preflighted with checked
non-negative addition and multiplication, capped at `2**63 - 1`, and compared
with the applicable policy ceiling before expansion starts. Per-operation
arguments may lower a ceiling but may not exceed the shared policy. Defaults
are conservative compatibility limits, not an entitlement to machine resources.

The complete field table and override example live in
[`../resource-limits.md`](../resource-limits.md). Boundary and “generator must
not start” tests are the executable contract.

## Consequences

- A limit failure is deterministic for a given artifact and policy.
- Library deployments can impose tenant-specific ceilings without global state.
- Raising a default is reviewed as a security and compatibility change even
  though semantic hashes do not change.
- Host-level time and memory isolation is still required for hostile workloads;
  application limits are defense in depth, not a sandbox.
- Some operations perform an extra counting pass to reject work before
  materialization.

## Alternatives considered

- Independent per-function constants were rejected because they allow a value
  accepted by validation to expand under a different limit downstream.
- Catching `MemoryError` or timing out after expansion was rejected because it
  does not prevent resource exhaustion and is not deterministic.
- A mutable process-global policy was rejected because concurrent callers could
  affect each other and tests would depend on execution order.
- Removing all compressed encodings was rejected because compact, bounded
  logical representations are a core artifact feature.

