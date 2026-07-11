# ADR 0005: Runtime, development, and CI dependency policy

- Status: Accepted
- Date: 2026-07-11

## Context

CommCanary processes integrity-sensitive JSON and is intended to run in
restricted workload environments. An unconstrained dependency graph expands
the release and supply-chain surface. Conversely, an unenforced collection of
local tools allows CI, packaging, and developer checks to drift.

## Decision

The installed runtime core remains Python-standard-library-only unless a later
ADR demonstrates a concrete correctness or maintenance benefit that outweighs
the added dependency. Test, schema, experiment, and development tools live in
named optional extras with bounded compatible version ranges.

`python -m tools.verify` is the canonical gate used locally and in CI. GitHub
Actions are pinned to full commit SHAs with a reviewed version comment.
Dependency review and automated update proposals report changes; they do not
silently relax a bound or replace a tested release artifact.

Release jobs publish the exact wheel and sdist that the lower-privilege build
job tested. Their member inventory, hashes, SBOM, and provenance metadata are
derived from those bytes. Experiment environments have a separate immutable
resolution/lock record because GPU and site packages are not package-runtime
dependencies.

## Consequences

- A normal `pip install commcanary` does not install optional schema or
  experiment tooling.
- Updating a development dependency is a reviewable gate change.
- Exact source-release and experiment reproducibility are measured separately.
- A security fix may deliberately narrow a range before an upstream release is
  adopted.

## Alternatives considered

- Floating CI action tags were rejected because the executed code could change
  without a repository diff.
- Installing all experiment packages as runtime dependencies was rejected
  because most users do not execute physical replay.
- Rebuilding in the publish job was rejected because the uploaded bytes would
  differ from the tested artifacts.
