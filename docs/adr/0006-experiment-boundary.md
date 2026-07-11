# ADR 0006: Planned and observed experiment evidence

- Status: Accepted
- Date: 2026-07-11

## Context

A cluster campaign must be defined before submission, while job IDs, nodes,
scheduler state, failures, and retries exist only after execution. Mutating one
manifest with both kinds of information makes its identity unstable and can
hide changed inputs or discarded failures.

## Decision

Each campaign starts with one immutable, content-addressed run manifest. It
owns expected cells, exact commands, code/config/input/script hashes,
environment requirements, site constraints, and measurement policies. Its
`run_id` does not depend on observed scheduler values.

Submission observations go to an append-only ledger. Every attempt has an
immutable terminal record, including failures and exclusions; retries never
overwrite earlier attempts. Exactly one selected attempt per expected cell
feeds a fail-closed completeness verdict. Aggregates and paper fragments are
generated only from the validated selected set and identify the manifest and
raw archive hashes from which they derive.

Repository-local preparation may validate scripts, schemas, manifests,
selection, analysis, and a miniature non-SLURM campaign. It must not invent
module versions, partitions, accounts, node topology, tool builds, performance
measurements, or other values that require the target site. Those remain
explicit handoff inputs until collected on Rostam.

## Consequences

- A failed or retried cell remains part of the evidence trail.
- Resume and reuse are allowed only when immutable inputs match.
- A polished aggregate cannot be produced from a silently incomplete matrix.
- Pre-cluster verification can finish without claiming that physical execution
  or environment reproduction succeeded.

## Alternatives considered

- Globbing a shared result directory was rejected because stale and unexpected
  cells can enter an aggregate.
- Updating the frozen manifest with scheduler observations was rejected because
  it changes campaign identity after submission.
- Filling unknown site values from developer machines was rejected because it
  would fabricate evidence.
