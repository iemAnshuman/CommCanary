# ADR 0008: Keep the historical paper outside product distributions

Status: Accepted (2026-07-11)

## Context

`paper/draft.md` reports a historical Rostam campaign whose complete raw attempt
archive, immutable manifest, terminal attempts, selection, and completeness
verdict are not available in this repository. Its numeric tables therefore
cannot satisfy the current experiment subsystem's evidence contract. Shipping
that draft in a release sdist would blur a reviewed software artifact with
claims the release cannot independently regenerate.

## Decision

Retain the draft in the repository as clearly labelled historical source, but
exclude `paper/` from wheel and sdist staging. It is not a current CommCanary
release claim. The manifest-driven analyzer remains the only path for new
publication material: a future paper may enter the distribution only after a
complete selected campaign regenerates its aggregate and paper fragment from
validated evidence and CI byte-compares the committed outputs.

This decision preserves prior work without presenting it as reproducible
release evidence. The Git repository, unlike the product distribution, remains
the archive of that historical source.

## Consequences

- Release inspection fails if any `paper/` member appears in an sdist.
- Historical percentages and hardware observations must remain labelled as
  unverified by the current checkout.
- A future publication promotion is an explicit review: commit the complete
  evidence boundary (or immutable external archive hash), completeness verdict,
  generated fragment, and a passing regeneration comparison before changing
  this ADR.
