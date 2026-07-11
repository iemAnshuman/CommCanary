# ADR 0004: Public API stability tiers

- Status: Accepted
- Date: 2026-07-11

## Context

The original package exposed implementation modules informally. Refactoring
large compiler and schema modules without an explicit surface would either
freeze every helper forever or break downstream users unpredictably.

## Decision

The names in `commcanary.__all__` form the stable top-level API for the 0.3
line. It contains version and error types, format capability discovery, shared
resource configuration, high-level load/validate/compile/replay/compare
operations, and their core verification operations. Tests assert the exact
export set and run a strict typed downstream example against an installed
wheel.

Capture, Kineto, PARAM, and presentation functionality is public only through
its documented adapter module. Research baselines and reduction are available
through `commcanary.experimental`; they may evolve faster and are not promoted
to the stable top level. A leading underscore is private regardless of whether
Python can import it.

Stable names receive at least one minor release of deprecation notice before
removal, unless retaining them creates a demonstrated security or correctness
hazard. Compatibility facades preserve documented legacy imports while code is
moved. Wire-format compatibility is governed separately by the format matrix;
an API deprecation cannot silently migrate an artifact.

All public mapping inputs are read-only and all returned documents are detached
snapshots. Runtime validation is authoritative even where `TypedDict` examples
improve static checking.

## Consequences

- Internal modules can be split behind tested compatibility facades.
- Merely importing an internal helper does not turn it into supported API.
- Adapter and experimental stability is explicit rather than inferred from the
  top-level package.
- Deprecation messages name the replacement and planned removal version.

## Alternatives considered

- Exporting every existing function was rejected because it would make module
  decomposition an API break.
- Exposing no stable Python API was rejected because the CLI and experiments
  need the same tested engineering operations.
- Treating research reduction as stable core was rejected until its policy and
  performance contract mature.
