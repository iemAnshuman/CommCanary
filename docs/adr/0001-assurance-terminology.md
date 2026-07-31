# ADR 0001: Assurance terminology

- Status: Accepted
- Date: 2026-07-11

## Context

CommCanary validates artifact structure, recomputes internal commitments,
compares artifacts with a supplied source, reruns a deterministic model, and
checks behavior/ranking preservation. Those operations establish different
facts. A single “verified” or “valid” label obscures whether a source was
supplied and can accidentally imply producer authenticity.

Self-contained SHA-256 commitments are tamper evidence only relative to trusted
content or a trusted digest. An artifact producer can modify unsigned content
and recompute every embedded hash.

## Decision

CommCanary uses these ordered machine-readable integrity/correspondence states:

1. `structurally_valid`;
2. `internally_consistent`;
3. `source_corresponding`;
4. `model_recomputed`.

An operation reports only the highest state it actually demonstrated. It does
not infer a higher state from lower-level success, and it does not silently
upgrade a legacy artifact whose required commitments are absent.
The ordered summary stops there. Model-behavior preservation, physical
execution, physical conformance, physical decision fidelity, and producer
authenticity are independent claim dimensions. `verify_canary_behavior`
therefore returns `model_behavior_preserved`, `model_behavior_unproven`, or
`failed`, plus a `claims` object. A successful deterministic-simulator
comparison never implies that hardware was executed or that physical decisions
were preserved.

No state means authentic, authorized, signed, or attributable. Authenticity
would require a separately designed signature or attestation system with an
external trust root.

The exact field coverage and API mapping live in
[`../integrity.md`](../integrity.md). Validators and mutation tests are the
executable contract.

## Consequences

- Callers can distinguish local consistency, source correspondence,
  deterministic-model preservation, and unobserved physical claims.
- Negative results can retain evidence about a lower assurance level without
  being described as fully verified.
- User-facing documentation must name the demonstrated property instead of
  using “secure,” “authentic,” or an unqualified “verified.”
- Adding authenticity later is a new protocol decision, not a rename of an
  existing hash.

## Alternatives considered

- One boolean `verified` flag was rejected because it cannot express which
  evidence was evaluated.
- Treating a valid embedded SHA-256 digest as authenticity was rejected because
  the producer controls both content and digest.
- Extending one assurance ladder through physical product claims was rejected
  because those claims are neither cumulative nor established by the same
  evidence.
