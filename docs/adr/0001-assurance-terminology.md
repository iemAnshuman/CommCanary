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

CommCanary uses these ordered machine-readable assurance states:

1. `structurally_valid`;
2. `internally_consistent`;
3. `source_corresponding`;
4. `model_recomputed`;
5. `behaviorally_verified`.

An operation reports only the highest state it actually demonstrated. It does
not infer a higher state from lower-level success, and it does not silently
upgrade a legacy artifact whose required commitments are absent. Existing
public `status` values remain during the compatibility period; new code uses
`assurance_state` to communicate the precise claim.

No state means authentic, authorized, signed, or attributable. Authenticity
would require a separately designed signature or attestation system with an
external trust root.

The exact field coverage and API mapping live in
[`../integrity.md`](../integrity.md). Validators and mutation tests are the
executable contract.

## Consequences

- Callers can distinguish local consistency from correspondence to a supplied
  source.
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
- Removing legacy `status` fields immediately was rejected because it would
  create an avoidable v2 API break.

