# Architecture and dependency boundaries

CommCanary is organized as a dependency-directed functional core with explicit
imperative edges. File count is not a goal; each package owns one reason to
change, and the repository rejects imports that reverse the direction below.

```text
foundation
  errors · formats · resources · statistics · version
      ↓
artifacts
  wire/JSON · trace · canary expansion/hashes/validation
  report · comparison contracts · atomic I/O · packaged schemas
      ↓
sibling engines
  compilation       replay       comparison
       \              /             /
        verification/reference paths
                    ↓
application services
  compile orchestration · behavior search · reduction
                    ↓
imperative edges
  capture · Kineto/PARAM adapters · HTML reporting · CLI
```

The exact policy is executable in `tools/import_boundaries.py` and runs in the
canonical verification gate. It parses every package module with the standard
library AST, requires every module to have a boundary, rejects upward edges,
cross-boundary private imports, and strongly connected dependency cycles.
Compatibility facades have narrow, named exceptions; new implementation code
cannot depend on them.

## Component responsibilities

| Boundary | Package/modules | Owns | Does not own |
|---|---|---|---|
| Foundation | `errors`, `formats`, `resources`, `statistics`, `version` | dependency-free errors, format IDs, checked limits, numeric summaries, package identity | artifact shape or workflow policy |
| Artifact contracts | `artifacts/` | strict loading, canonical JSON, artifact validation, expansion preflight, semantic projections/hashes, atomic files, schema resources | compilation, replay, or verification decisions |
| Compilation | `compilation/` | trace normalization, grouping, timing compression, sequence motifs, fidelity metrics, compile core | calling a verifier or choosing a searched candidate |
| Replay | `replay/` | timing expansion under budget, deterministic scheduler/noise, accumulation and report production | report attestation or comparison policy |
| Comparison | `comparison/` | immutable thresholds, structured evaluation codes, report comparison and verdict production | replay or presentation |
| Verification | `verification/` | source/fidelity recomputation, behavior/ranking checks, model-recomputed report verification | producer shortcuts for the calculation being attested |
| Services | `services/` | verified compile orchestration, behavior search, decision-only reduction | files, environment, HTML, or CLI parsing |
| Adapters | `adapters/` | recorder lifecycle, shard reconciliation, Kineto conversion, PARAM export | domain policy not inherent to the external format |
| Reporting | `reporting/` | escaped, self-contained HTML presentation | synthetic samples or domain mutation |
| CLI | `cli`, `command_line/` | compatibility entry point, parser, stable exits, progress/diagnostics, subprocess and output orchestration | domain calculations |
| Experimental | `experimental/`, `baselines` | explicitly research-tier baselines and reduction access | stable top-level API promises |

`operation_identity` supplies named projections rather than a flag-driven tuple:
compression, scheduler ordering/resource, capture coalescing, deterministic
noise, and baseline grouping each state exactly which fields define identity.
Adding an operation field requires reviewing each projection independently.

## Data flow and assurance

```text
capture / Kineto
       │
       ▼
  trace.v1 ── compile service ──▶ canary.v2 ── replay core ──▶ report.v2
       │                              │                          │
       └──── source reference ────────┘                          ▼
                              fidelity/behavior checks      comparison.v2
                                                                 │
                                         model recomputation ────┘
```

Artifact validation establishes structure and internal consistency. Fidelity
verification additionally requires a supplied source; report verification
reruns the declared model; behavioral verification evaluates declared behavior
and ranking tolerances. These are cumulative assurance states, not producer
authenticity. Producer and verifier may share codecs and field definitions, but
bounded-interval source recomputation, report recomputation, and semantic-digest
checks retain independent calculations.

All compact representations are counted under one immutable `ResourceLimits`
instance before expansion. Public mapping inputs are treated as read-only and
outputs are detached snapshots. Semantic hashes exclude only documented
volatile/self-referential fields; clocks, files, environment, subprocesses, and
progress live outside the functional core.

## Compatibility facades

The historical modules `schema`, `compiler`, `compare`, `capture`, `interop`,
`html_report`, `reduce`, and `cli`, plus the historical `replay` package surface,
preserve documented imports during the 0.3 compatibility window. They contain
wiring, aliases, and explicitly characterized monkeypatch seams—not parallel
implementations. Tests assert facade object identity and dependency ownership. New
packages import `artifacts` or the owning engine directly and never route back
through a facade.

Normal stable-API removals receive at least one released minor version of
deprecation. Wire-format compatibility remains governed separately by exact
format IDs and the compatibility matrix.

## Extension points

- A new artifact version starts in `formats`, a Draft 2020-12 schema, literal
  valid/invalid/tampered fixtures, canonical/hash vectors, and a compatibility
  decision. It is not inferred from unknown fields.
- A new communication operation must define validation, identity projections,
  compilation/replay semantics, resource counting, independent contract
  fixtures, and adapter mappings before being described as supported.
- A new external ecosystem belongs in `adapters`; it translates to or from the
  canonical artifacts and fails closed when the external semantics cannot be
  represented honestly.
- A new comparison threshold is a typed policy/evaluation change with external
  boundary vectors. Human reason prose is not the machine code.
- Experimental producers use their own schema IDs and import only documented
  public or explicitly experimental engineering APIs.

## Experiment boundary

`experiments/rostam/` is not an upward dependency of the installed package. It
consumes public artifacts and maintains its own immutable campaign, attempts,
selection, completeness, physical-result, and aggregate contracts. Expected
site inputs are frozen before submission; observed scheduler values are
append-only evidence. See the [artifact-evaluation handoff](artifact-evaluation.md)
for the exact point where verified local preparation ends and authorized Rostam
execution begins.
