# CommCanary Engineering Plan

## Implementation checkpoint — 2026-07-11 (evening; supersedes the morning entry)

This live checkpoint is still before any Rostam login, setup, scheduler,
Torch/GPU, NCCL, PARAM execution, or cluster command. The implementation is
now committed on `codex/engineering-plan-implementation` as a reviewable
series ending at `521b8fb7909933c29d73fb820bbae4015eb30ff4`, the full
reproducible gate is green against that commit, and the exact artifacts are
built, hashed, and bound in the pending Rostam environment contract. 0.3.0
remains unreleased and untagged. The publication decision was made on
2026-07-11: `origin/main` had only GitHub's 2026-07-01 license-boilerplate
init commit (an Apache-2.0 LICENSE file, no code, preserved as
`archive/github-init`), the project license is and remains MIT across
LICENSE, packaging metadata, and README, and `main` was set to this verified
history rather than merging two unrelated lines.

| Phase | Checkpoint status | What is done | What remains |
|---|---|---|---|
| 0 — reconcile state | **Complete** | The assembled tree is committed as a reviewable dependency-ordered series (hygiene, core split, schemas, gate/CI, tests, benchmarks, experiments, docs, toolchain fix, binding). The worktree is clean; private context and generated output are ignored and untracked. Published 2026-07-11: `main` set to this history, the review branch pushed, and the unrelated GitHub init commit (Apache LICENSE boilerplate, no code) preserved as `archive/github-init`. | — |
| 1 — integrity and safety | **Complete** | Integrity ladder, recursive provenance recomputation, detached outputs, path containment, capture hardening, and bounded public resource policies are implemented with adversarial tests. | — |
| 2 — wire contracts | **Complete** | Published schemas, strict canonical JSON/hash vectors, compatibility rules, runtime schema mirror, and installed-wheel contract tests are implemented and re-verified from a clean archive. | — |
| 3 — guardrails | **Complete** | Canonical local/CI verification, Ruff, strict mypy (whole experiment tree included), statement and per-responsibility branch floors, import boundaries, shell/workflow/docs checks, and exact installed-wheel tests are green. The three floors left short by the interrupted edits (capture, compiler+integrity, replay+verification) were closed with behavior-asserting branch tests. | — |
| 4 — characterization | **Complete** | Semantic, hash, comparison-boundary, CLI-exit, public-import, and golden artifact characterization is in place and passing. | — |
| 5 — architecture | **Complete** | Artifact, compilation, replay, verification, comparison, service, adapter, reporting, and CLI packages sit behind compatibility facades; the monolithic test file is split by capability. | — |
| 6 — API and CLI | **Complete** | Stable API/version/format capability surface, typed package marker, CLI error taxonomy, diagnostics/progress, deprecation behavior, and honest HTML output are implemented; installed-wheel and archive checks passed. | — |
| 7 — experiment subsystem | **Locally complete; physical evidence pending** | Immutable manifests/cell IDs, append-only attempts, explicit selection, fail-closed completeness, local non-SLURM runner, thin scheduler wrappers, exact submission planning, distinct physical producer schemas, reviewed PARAM patch evidence, and structurally complete core/overlap/shared catalogs are implemented and gate-verified. The wheel commit and SHA-256 are now bound in the pending environment contract. | Environment resolution, GEMM calibration, shared trace capture, campaign execution, archive bytes, and generated physical claims are Rostam-only. |
| 8 — measured optimization | **Locally complete** | Deterministic 1K/10K/100K fixtures, isolated wall/RSS/allocation measurements, semantic hashes, smoke/scheduled workflows, and measured capture/PARAM optimizations are implemented. No threshold was fabricated from one runner. | Accumulate stable-runner history before choosing regression thresholds; this is not a Rostam prerequisite. |
| 9 — docs and release | **Complete up to the release boundary** | Docs, ADRs, contributor/security files, reproducible package tooling, SBOM/inventory/checksums, the paper-publication boundary, reviewable commits, a verified clean `git archive HEAD`, exact retained wheel/sdist with recorded hashes, the environment-contract binding, and the pre-Rostam handoff record in `docs/artifact-evaluation.md` are done. A fresh-venv toolchain gap (setuptools <70.1 cannot build wheels without the separate `wheel` package) was fixed by raising the dev-extra floor. | Tagging, publishing, and supported-platform CI remain deliberate later release actions; 0.3.0 stays `Unreleased` until then. |

### Last known verification state (commit `521b8fb`)

- `python -m tools.verify --reproducible --artifact-dir dist --metadata-dir
  release-metadata` passes end to end on macOS arm64 / CPython 3.10.14:
  **699 tests**, 90.47% statement coverage (86% floor), every
  per-responsibility branch floor (capture 89.7%/70, compiler+integrity
  87.6%/85, replay+verification 80.1%/75, resource limits 90.0%/90, others
  above floor), whole-tree strict mypy, Ruff and formatting, import
  boundaries, schema/shell/workflow/README checks, two byte-identical
  fixed-epoch builds, and installed-wheel tests outside the checkout.
- A `git archive HEAD` checkout passed all 699 tests in a fresh virtualenv
  against the exact tested wheel.
- Exact artifacts (hashes also in `release-metadata/SHA256SUMS`, inventory,
  and SPDX SBOM): `commcanary-0.3.0-py3-none-any.whl`
  `416dbea60943cf5ff93282547b1c350edb8880e0cc3e719bf6d1aef794a6738e`;
  `commcanary-0.3.0.tar.gz`
  `49443d276dea934e06494a30159ff247a53d461bf048febe095da667e2058014`.
  Both are bound to source commit `521b8fb7909933c29d73fb820bbae4015eb30ff4`
  in `experiments/rostam/constraints/environment-contract.json`.

### Remaining order before Rostam

Repository-side work is finished. What remains is Rostam-side collection per
`docs/artifact-evaluation.md`: resolve the two locked environments, record
platform/ABI/resolver evidence, rebuild and hash-verify the bound wheel,
calibrate GEMM, capture the shared trace, then plan and submit campaigns
through the fail-closed harness. Tagging and publishing 0.3.0 remain
deliberate later release actions.

_Original roadmap re-audited 2026-07-10. Its baseline observations below are
historical; the live implementation status is the checkpoint above._

This is an engineering plan, not a research agenda. The research material is
useful as a demanding workload and as an artifact-reproducibility test, but the
goal here is a small, trustworthy, maintainable software system: explicit
contracts, bounded work, independent verification, clean dependency direction,
excellent packaging, and one repeatable quality gate.

## Executive verdict

CommCanary has a stronger correctness instinct than most early-stage Python
projects: it rejects many malformed artifacts, uses deterministic replay,
escapes HTML, writes core JSON atomically, and has 126 fast tests. The live
working tree builds as a wheel, installs outside the checkout, and passes its
suite. Those are real strengths.

It is not yet release-ready or refactor-ready. Four issues precede architecture
work:

1. **Repository state is split across three releases.** `HEAD` is a coherent
   older 0.2.0 tree; the index and working tree contain a partially assembled
   0.3.0; important 0.3.0 modules and workflows are untracked. The old plan's
   claim that a fresh clone of `HEAD` is import-broken is false.
2. **The advertised tamper evidence is incomplete.** Compiler fingerprints are
   optional, artifact provenance is not recomputed, and metadata tampering can
   still produce a `source_verified` result.
3. **Untrusted inputs can consume unbounded resources before the advertised
   replay limit runs.** Motif, pattern, weight, and PARAM expansions happen
   before effective budgets; JSON loading is unbounded and accepts duplicate
   keys.
4. **Public results are mutable aliases of caller inputs.** Mutating nested
   input metadata after compile/replay/compare can silently mutate the returned
   artifact and invalidate its stored hashes.

The right sequence is therefore: reconcile the release state; seal integrity,
ownership, path, and resource contracts; install guardrails and characterization
tests; then simplify and split the architecture behind compatibility facades.

## Corrections to the previous plan

These are not wording nits; they change the implementation order.

- `HEAD` does **not** import untracked `interop.py` or `reduce.py`. A
  `git archive HEAD` checkout runs 104 tests and its CLI help successfully.
  The danger is committing tracked CLI changes without their untracked
  dependencies, not a currently broken commit.
- “No confirmed runtime-correctness bug of consequence” is not supportable.
  Integrity, aliasing, path traversal, and pre-limit expansion are correctness
  or security defects.
- Do **not** replace the independently implemented interval producer and
  verifier with one shared computation. That would create a common-mode failure
  in the component meant to attest the producer.
- Mixed trace/canary/report version numbers are not themselves a defect, and
  there is no evidence that automatic v1-to-v2 migration is currently needed.
  Define a compatibility policy and collect real legacy fixtures first.
- A histogram cannot be reconstructed honestly from median/p95/p99. When raw
  samples are absent, render a summary-only panel and say samples are absent.
- Binary search over behavior-search limits is invalid until monotonic
  acceptance is proved. Use bounded exhaustive candidates, caching, safe
  parallelism, or dependency-aware recomputation.
- File-length ceilings such as 800 source lines or 600 test lines are weak
  proxies. Enforce dependency direction, cohesion, complexity, stable public
  boundaries, and test behavior instead.
- `tests/__init__.py` is unnecessary. Do not add it without a concrete import
  reason.
- `paper/` is not equivalent to the private `context.md`. Either make the paper
  a reproducible part of this artifact or publish it separately by explicit
  decision.
- Remove claims about “90 agents,” “79 findings,” and a private local transcript.
  The durable evidence ledger is in this file.

## Audit baseline

The following results describe the live working tree unless stated otherwise.

| Area | Observed state | Engineering implication |
|---|---|---|
| Git | `master`; no tags; many `M`/`MM` files; untracked `.github/`, `LICENSE`, `docs/`, `experiments/`, `paper/`, `interop.py`, and `reduce.py` | Assemble a deliberate release; never use an indiscriminate `git add .` |
| Committed tree | `HEAD` is coherent 0.2.0; 104 tests pass; CLI help works | Correct the old plan's broken-HEAD claim |
| Working tree | 126 tests pass in about 5.2 s on macOS/Python 3.10 | Good refactor safety net, but only one locally exercised platform |
| Coverage | 4,453 statements, 638 missed: about 86%; `schema.py` and `capture.py` are 79%, CLI 83% | Preserve the baseline; target critical branches and invariants rather than chasing a vanity percentage |
| Wheel | Working-tree wheel builds, installs, CLI-smokes, and passes 126 tests outside `src/` | Packaging works locally, but CI never tests this path |
| Sdist | Omits tests, examples, docs, and changelog while its README references absent examples | Define an sdist-content policy and test it |
| Reproducible build | Two normal builds differ by timestamps; fixed `SOURCE_DATE_EPOCH` makes them identical | Define semantic determinism separately from byte-reproducible release artifacts |
| Ruff | 6 unused-import failures; default formatting would change 20 files | Land configuration and a formatter-only commit before semantic refactors |
| Mypy | 3 default errors; 15 strict errors in 8 files | Establish a measured ratchet; do not advertise `py.typed` prematurely |
| Module shape | `compiler.py` 2,430 lines; `schema.py` 1,942; `capture.py` 738; single 3,515-line test file | Responsibilities and contracts are tangled, though the import graph is mostly acyclic |
| Public API | `__init__.py` exports three format constants only; no package version; compare format omitted | Define a deliberately small stable surface and an experimental namespace |
| CI/release | All workflows untracked; floating action tags; `PYTHONPATH=src`; publish rebuilds an untested artifact | Make local and CI verification identical; publish the exact tested artifact |
| Experiments | Result directory ignored; no run manifest; analyzer globs a shared directory and silently drops failures | Results cannot yet support a reproducible engineering artifact |
| Shell | All current `.sh`/`.sbatch` files pass `bash -n`; a dry-run submits the expected eight micro configs | Syntax is sound; semantics, completeness, and environment provenance remain unverified locally |
| Scope not exercised | No GPU/NCCL/PARAM/SLURM execution, no remote/PyPI settings audit, no Python 3.9/3.11-3.13 runtime run in this audit | Keep these as explicit release risks, not inferred successes |

## Engineering principles

1. **Integrity claims are capabilities, not optional decorations.** An artifact
   is structurally valid, internally consistent, source-corresponding, or model-
   recomputed; those levels must not be conflated under “verified.”
2. **Verifier independence beats superficial deduplication.** Share codecs,
   field definitions, and canonical primitives. Do not share the calculation a
   verifier exists to recompute independently.
3. **Every expansion has a budget before allocation.** Stored size and expanded
   work are separate quantities. Count with checked arithmetic, reject first,
   iterate second.
4. **Public outputs are detached snapshots.** No returned artifact shares mutable
   nested state with inputs or another output.
5. **The wire format is explicit.** JSON shape, semantic validation, canonical
   hashing, unknown-field behavior, coercion, extensions, and version support are
   separate documented contracts.
6. **One concept has one authoritative production implementation.** Accidental
   copies go away; deliberately independent reference implementations are named
   and tested as such.
7. **Functional core, imperative shell.** Domain calculations are pure and
   deterministic; clocks, files, environment, subprocesses, progress, and CLI
   exit codes live at the edge.
8. **The runtime stays dependency-light.** Preserve the stdlib-only core unless
   an ADR demonstrates that a dependency materially reduces risk. Development,
   schema, and verification tools may be optional dependencies.
9. **CI runs what developers run.** One canonical verification command owns
   formatting, lint, types, tests, package checks, docs, shell, and artifact
   contract checks.
10. **Research scripts consume public engineering APIs.** They never reach into
    underscore-private compiler internals or redefine package policy silently.

## Target dependency architecture

The exact number of files is not a goal. The dependency direction and ownership
of concepts are.

```text
foundation
  errors · limits · clocks · canonical JSON · format IDs · small value types
      ↓
artifact contracts
  trace · canary · report · comparison parsers/validators/codecs/hashes
      ↓
sibling engines
  compile core        replay core        comparison policy
       \                  /                    /
        independent verification/reference paths
                         ↓
application services
  compile_verified · behavior_search · reduction · workflows
                         ↓
adapters and presentation
  capture · Kineto · PARAM · HTML · CLI
```

Required rules:

- Compile core does not call verification. A service composes compilation and
  verification; this avoids the current potential compile ↔ verification cycle.
- Comparison policy depends on report contracts, not on CLI or replay internals.
- Verification may depend on compile/replay public cores, but independent
  reference calculations live separately and never call the producer operation
  they attest.
- Adapters parse external data into canonical artifact/domain values; core code
  does not read environment variables or files.
- Experiments import only the public API or an explicitly experimental API.
- Compatibility modules may re-export old imports during migration, with tests
  and a dated deprecation policy.
- An import-boundary test (AST-based or `import-linter`) rejects upward imports,
  cycles, and cross-boundary underscore imports.

Use `TypedDict` for static typing of JSON wire objects, JSON Schema for portable
shape contracts, and small frozen value objects only where identity or behavior
belongs to the value (for example operation identity, replay configuration,
comparison thresholds, and resource budgets). Do not create a full dataclass
mirror of every JSON artifact.

## Findings ledger

Severity means implementation priority, not research importance.

| ID | Severity | Evidence | Risk | Planned phase |
|---|---|---|---|---|
| F-001 | P0 | `git status`; committed `cli.py` differs from live `cli.py`; required modules/workflows untracked | An incomplete commit can publish a release different from both HEAD and the tested tree | 0 |
| F-002 | P0 | `schema.py:812-831` makes hashes optional and never recomputes artifact provenance | Tampered artifacts validate despite the README's integrity claim | 1 |
| F-003 | P0 | Workload mutation after compile still returns `source_verified` | Verification ignores protected metadata/provenance correspondence | 1 |
| F-004 | P0 | `capture.py:393-405,418-423` puts raw rank environment text in a path | A workload can traverse outside `COMMCANARY_TRACE_DIR` | 1 |
| F-005 | P0 | `schema.py:62-80,120-163,305-365,632`; `replay.py:94-121`; `interop.py:328+` | Oversized JSON/repeats/motifs/exports can hang or exhaust memory before limits | 1 |
| F-006 | P0 | Compiler/replay/compare/baseline/reduce shallowly embed nested inputs | Later caller mutation silently changes artifacts and invalidates hashes | 1 |
| F-007 | P1 | Producer `compiler.py:1910-2027`; verifier `1076-1162` | Old plan's consolidation would destroy independent attestation | 3/4 |
| F-008 | P1 | Scheduler hash equals execution hash; source-normalized hash equals source-trace hash | Names imply distinct guarantees that do not exist | 3/6 |
| F-009 | P1 | `as_int`/`as_float` coerce strings; validators allow unknown/optional fields | Accepted wire shape and canonical representation are ambiguous | 3 |
| F-010 | P1 | Duplicate JSON keys accepted; recursion handling differs by loader | Cross-parser ambiguity and inconsistent failure behavior | 1/3 |
| F-011 | P1 | `schema.py` and `compiler.py` own multiple unrelated subsystems | Refactors and correctness changes have excessive blast radius | 5 |
| F-012 | P1 | `reduce.py` imports four compiler-private names; compiler lazy-imports replay | Boundaries are implicit and orchestration is in the wrong layer | 4/5 |
| F-013 | P1 | Canonical JSON, policy math, timing traversals, operation projections, stats, writers duplicated | Production behavior can drift across modules | 4 |
| F-014 | P1 | 126 tests in one class/file; CLI assertions are almost entirely success paths | Contract changes and negative CLI behavior are hard to audit | 3/6 |
| F-015 | P1 | Working CI uses `PYTHONPATH=src`; publish builds again without testing | Installed-package and release-only failures can escape | 2 |
| F-016 | P1 | Ruff 6 failures; mypy 3 default/15 strict; no configs | No enforceable static quality baseline | 2 |
| F-017 | P1 | `__init__.py` exports only three constants; capture proxy is `**Any` | Public API and compatibility expectations are undefined | 6 |
| F-018 | P1 | CLI application errors share argparse code 2; capture returns raw child codes | Automation cannot reliably classify results, usage, and failures | 6 |
| F-019 | P1 | `analyze.py` globs shared results and omits failed cells; no run/cell ledger | Stale, duplicate, or incomplete campaigns can yield polished output | 7 |
| F-020 | P1 | `setup.sh` upgrades tools, loosely installs deps, edits PARAM with `sed`, installs dirty editable tree | Experiment environments cannot be recreated exactly | 7 |
| F-021 | P1 | `overlap_replay.py:84-96` aliases every process group to WORLD | Non-world group traces are executed incorrectly or rejected too broadly | 7 |
| F-022 | P2 | Three JSON writers have different atomicity, modes, durability, and exception behavior | Partial files and inconsistent user-facing failures | 4/6 |
| F-023 | P2 | Capture has per-instance locks, unconditional `__exit__.save`, cached global recorder | Same-path races and exception masking are underspecified | 1/6 |
| F-024 | P2 | Behavior configurations accept `[]` as defaults, ignore unknown keys, need not have unique names | Surprising behavior and unbounded quadratic ranking work | 1/4 |
| F-025 | P2 | Workflows use floating tags and broad defaults; no exact-artifact publish or provenance policy | Supply-chain and release drift | 2/9 |
| F-026 | P2 | `context.md`, `.DS_Store`, and generated `repomix-output.xml` are in or near the release tree | Private/generated material can be published accidentally | 0 |
| F-027 | P2 | No architecture/decision/security/contribution docs or privacy inventory | Maintainers cannot preserve the intended contracts | 9 |
| F-028 | P2 | No scale benchmark; “long” fixture is only 56 events; capture domain check is O(E²) | Performance work is speculative and regressions are invisible | 8 |
| F-029 | P2 | HTML shows “No samples”; old plan proposed synthesizing a distribution | Default report is weak, but fabricated charts would be worse | 6 |
| F-030 | P2 | Verification mostly reuses compiler/simulator under test | “Proof” language overstates self-consistency/model recomputation | 1/9 |
| F-031 | P1 | Uniform behavior search uses `(bytes, events, limit)` while recorded `selection_metric` promises stored records too (`compiler.py:452-456,497`) | The artifact misstates the optimization objective and uniform/refinement stages rank candidates differently | 1/4 |
| F-032 | P2 | `_finalize_step` uses medians for scalar summaries but `arrival_offsets_us` from sample 0 (`compiler.py:1610-1623`) | One event summary mixes incompatible representatives | 3/4 |
| F-033 | P2 | `_update_size_metrics` silently stops after 12 self-referential iterations (`compiler.py:2412-2430`) | Stored byte metrics can be non-fixpoint without an error | 3/4 |
| F-034 | P2 | Replay aggregates full-precision values but emits independently rounded samples (`replay.py:637-652,704-715`) | Included samples and summaries can diverge at rounding boundaries | 3/4 |
| F-035 | P2 | Unused `merge_metadata`/`clean_private_keys`, a local shadows `arrival_skew_us`, and six imports are unused | Dead/ambiguous code raises maintenance cost and hides real lint signal | 2/5 |
| F-036 | P2 | Five SBATCH files duplicate setup/result scaffolding and `run_canary.sbatch` embeds large Python heredocs | Experiment fixes must be repeated and cannot be unit-tested cleanly | 7 |

## Phase 0 — Reconcile and freeze the intended release

**Priority:** P0. **Estimate:** 1-2 focused days. **No refactoring.**

The objective is a reviewable, recoverable repository state. Preserve all user
work before staging anything.

### Work

1. Create a safety branch or local recovery commit and record:
   - `HEAD`, index diff, worktree diff, untracked-file inventory, file hashes;
   - the passing test/package commands and environment;
   - which files belong to core 0.3, experiments, paper, private context, or
     generated output.
2. Assemble coherent commits rather than one snapshot:
   - repository hygiene and ignores;
   - core 0.3 behavior (`interop`, `reduce`, baselines, CLI/schema/tests/docs);
   - workflows, license, packaging metadata;
   - experiments/paper only after their publication boundary is decided;
   - release metadata last.
3. Remove `repomix-output.xml` from version control and ignore
   `repomix-output.*`, `.DS_Store`, caches, build outputs, and local result dirs.
4. Move `context.md` outside the public tree and ignore it. It contains session,
   commercial, cluster, and delegation context that is not product documentation.
5. Decide `paper/` and `experiments/` separately:
   - **recommended:** keep scripts/specs and the paper if they can meet Phase 7's
     reproducibility contract;
   - otherwise publish them in a deliberate artifact repository with immutable
     links. Do not silently delete or half-track them.
6. Verify the remote default branch before renaming `master` or changing URLs.
   Do not assume `main` merely because workflow/docs links say `main`.
7. Treat 0.3.0 as unreleased until its release gate passes. Do not create a tag
   just to match the changelog. Verify PyPI/GitHub/Pages external configuration
   separately when credentials and network state are in scope.

### Acceptance

- `git status` is clean on the intended release branch.
- A `git archive HEAD` (not the developer checkout) builds wheel and sdist,
  installs in a clean environment, passes the intended tests, and exposes every
  documented command.
- The archive contains every referenced license, example, schema, and doc—or the
  README links to a stable external location.
- No private context, generated aggregate, local result, secret, or `.DS_Store`
  is tracked.
- Version, changelog, and advertised commands describe the same tree. No tag is
  cut yet.

## Phase 1 — Seal correctness, integrity, ownership, and resource boundaries

**Priority:** P0. **Estimate:** 3-5 focused days. **Precedes structural refactors.**

### 1.1 Define an assurance ladder

Use distinct machine-readable states and documentation:

1. `structurally_valid` — shape/types/ranges are accepted;
2. `internally_consistent` — derived fields and stored commitments recompute;
3. `source_corresponding` — source commitments match a supplied source;
4. `model_recomputed` — a report matches a rerun of the declared model;
5. `behaviorally_verified` — declared behavior/ranking checks pass.

None of these is authenticity. SHA-256 detects changed content relative to a
known digest; it does not establish who produced the artifact. Use signatures or
attestations only if authenticity becomes a requirement.

### 1.2 Repair the fingerprint contract

- Write a field-coverage table for every digest:
  - raw input artifact bytes, if preserved;
  - normalized selected trace;
  - executable semantics;
  - calibration/observed semantics;
  - full nonvolatile artifact provenance.
- Require the commitments promised by each supported format/capability profile.
  If legacy v2 artifacts without them must remain readable, label them
  structurally valid but integrity-unverified; do not silently upgrade status.
- Recompute `artifact_provenance_sha256` in `validate_canary` and include it in
  fidelity verification. Define exactly why volatile/self-referential fields
  (`created_at`, digest fields, byte count) are excluded.
- Require or explicitly deprecate per-event source digests.
- Define workload/system metadata coverage. Recommended: full artifact provenance
  protects it; source correspondence additionally verifies source-derived fields.
- Build a tamper matrix: delete each commitment and mutate every protected field
  family. Each mutation must fail at the correct assurance level.
- Document both current alias pairs (`execution`/`scheduler` and
  `source_trace`/`source_normalized`). Keep compatibility fields in v2 if needed,
  but do not describe them as independent guarantees.

### 1.3 Make output ownership explicit

- Public compile, replay, compare, baseline, reduction, capture, and interop APIs
  return detached JSON snapshots.
- Normalize/copy inputs at the boundary; never retain caller-owned nested lists
  or mappings.
- Add bidirectional alias tests: mutating input after return and output after
  return must not affect the other object or a sibling artifact.
- Validate metadata is JSON-serializable at ingestion, not only at final write.

### 1.4 Put budgets before expansion

Introduce a typed `ResourceLimits` policy with documented safe defaults and
explicit override paths. Cover at least:

- input file bytes and JSON nesting depth;
- total object/list items, maximum string bytes, ranks, stored events, and stored
  timing records;
- expanded motif events, pattern repetitions, timing weights, replay events,
  verification work, and PARAM entries;
- behavior-search limits, configuration count, candidate count, retained ledger
  rows, and reduction oracle calls.

Implementation requirements:

- Reject duplicate JSON object keys and non-standard constants in all loaders.
- Catch recursion/decoder failures consistently as typed CommCanary errors.
- Preflight expansion with checked integer arithmetic; fail before building a
  list or entering a repeat loop.
- Pass one budget object through validation, hashing, replay, verification, and
  export. A path must not validate under one limit and expand under another.
- Stream/lazily iterate where possible. Never materialize motif expansion merely
  to count it.
- Replace capture's all-pairs rank-domain comparison with comparison against a
  representative/equivalence key (O(E)).
- Bound behavior search explicitly. Parallel evaluation must also bound workers,
  memory, retained diagnostics, and cancellation.

### 1.5 Close capture path and state hazards

- Accept only a numeric rank or encode it as a safe, length-bounded slug.
- Resolve every recorder shard path and prove it remains below the configured
  trace root before creating directories or files. Test `/`, `..`, separators,
  Unicode confusables, absolute paths, and oversized labels.
- Specify whether two recorder instances may target one path. If not, fail fast;
  if yes, coordinate across instances/processes rather than relying on per-object
  locks.
- Do not let a save failure in `__exit__` hide the workload's original exception;
  chain/report both.
- Re-evaluate or explicitly freeze environment-driven global-recorder settings.
- On a capture-child failure, optionally preserve partial shards and emit a
  failure manifest instead of always deleting the temporary directory.

### Acceptance

- Changing or deleting any required commitment fails validation; metadata
  tampering can no longer return `source_verified`.
- A seeded mutation in producer interval math is caught by the independent
  verifier.
- All public outputs pass bidirectional no-alias tests.
- Malicious duplicate-key, deep JSON, huge repeat/weight/motif/search/export, and
  rank-path fixtures fail with a stable error before expansion and within a
  deterministic time/memory envelope.
- Every computed shard path is contained by its configured trace root.
- Normal valid fixtures retain semantic hashes and replay metrics unless a
  format revision explicitly documents the change.

## Phase 2 — Install one enforceable quality and release gate

**Priority:** P1. **Estimate:** 2-4 focused days.

### Local gate

Create one cross-platform canonical entry point, for example
`python -m tools.verify`, with a thin `make check`/shell convenience alias. CI
must invoke the same implementation. It should support fast and full modes and
run, in order:

1. repository hygiene and generated-file checks;
2. Ruff format and lint;
3. mypy for the declared typed surface;
4. pytest with coverage and contract fixtures;
5. JSON/JSON-Schema, shell (`shellcheck`), and workflow (`actionlint`) checks;
6. README command/link/example verification;
7. wheel and sdist build plus metadata/content inspection;
8. clean installation of the built wheel outside the checkout with
   `PYTHONPATH` unset;
9. installed-package tests and CLI/module smoke tests;
10. reproducibility check under a fixed `SOURCE_DATE_EPOCH` for release mode.

### Static tooling

- Configure Ruff deliberately (target Python 3.9, line length/import policy,
  selected rules). Land formatting alone before semantic edits so blame and
  review stay useful.
- Fix the 6 current Ruff errors.
- Record mypy baselines: 3 default errors and 15 strict errors. Make the stable
  public API strict first, then ratchet modules. No new errors anywhere.
- Add `py.typed` only when the exported API and installed-wheel typing test pass.
- Define `test`, `dev`, and `experiment` optional dependency groups. Lock or
  constrain developer/release tooling; keep runtime dependencies empty unless an
  ADR changes that.

### Tests and coverage

- Preserve the measured statement baseline (~86%) and fail on regressions.
- Require full diff coverage for new/changed critical code where practical.
- Set branch targets for integrity, validators, resource limits, capture paths,
  and CLI error/exit handling. A global 90% number is secondary to those paths.
- Add a scheduled targeted mutation run for commitments and comparison policy;
  it need not burden every quick PR.

### CI and release supply chain

- Test supported Python versions through installed artifacts; include macOS and
  either add Windows or explicitly declare it unsupported.
- Pin every GitHub Action to a full commit SHA with a version comment.
- Use least-privilege `permissions`, timeouts, concurrency cancellation, and
  `persist-credentials: false` where appropriate.
- Build release artifacts once. Test, hash, inventory, attest, and publish those
  exact files—never rebuild inside the publish job.
- Gate publish on all required checks and a protected PyPI environment using
  trusted publishing.
- Check tag, project version, package metadata, changelog, format support, and
  release name for equality/compatibility.
- Generate an SBOM/provenance statement; add dependency/action update automation
  and vulnerability review appropriate to a near-stdlib project.

### Acceptance

- One documented local command reproduces the required CI gate.
- Ruff is clean and formatted; mypy has zero errors at the declared strict
  boundary and a checked ratchet elsewhere.
- Wheel and sdist contents match policy; the README never references missing
  packaged examples.
- Tests pass against the installed wheel with source imports made impossible.
- Two release builds with fixed source epoch and locked tooling are byte-identical.
- The publish job uploads the exact previously tested hashes.

## Phase 3 — Define and characterize wire contracts before moving code

**Priority:** P1. **Estimate:** 3-5 focused days.

### Contract decisions (ADRs)

1. **Validation vs parsing.** `parse_*` creates canonical values; `validate_*`
   checks an already canonical artifact. Do not coerce numeric strings while
   leaving the original string in a supposedly valid artifact.
2. **Unknown fields.** Recommended: reject unknown semantic fields, allow
   explicitly named extension/metadata namespaces, and preserve them in full
   provenance hashing. Specify forward-compatibility behavior.
3. **Format support.** List exact read/write versions per artifact. Migration is
   explicit, provenance-preserving, and opt-in; implement it only when real
   legacy fixtures justify it. Never mutate silently during load.
4. **Canonical JSON.** Specify encoding, key ordering, separators, Unicode,
   finite-number rules, negative zero/float handling, duplicate keys, and whether
   the contract is Python-specific or cross-language.
5. **Determinism.** Separate semantic determinism (same semantic hashes/metrics)
   from byte identity (`created_at` differs) and reproducible release builds.
6. **Integrity profiles.** Define mandatory fields and assurance levels per
   artifact/version.
7. **Privacy.** Inventory metadata propagation and redaction expectations for
   workload, system, host, process, topology, and cluster identifiers.

### Machine-readable contracts

- Publish versioned JSON Schemas for trace, canary, report, comparison, fidelity
  verification, behavior verification, and report verification.
- JSON Schema covers portable shape; runtime validators continue to enforce
  cross-field semantics. Test every golden fixture against both.
- Store cross-language canonical-JSON/hash vectors with expected bytes and
  digests.
- Add a capability/support matrix to docs and `commcanary --version` output.

### Characterization suite

Before module moves, lock down:

- parse/serialize round trips and unknown/coercion behavior;
- exact canonical bytes and all digest projections;
- flat vs motif and run/pattern semantic equivalence;
- deterministic replay by seed across supported Python versions;
- compiler/verifier disagreement mutation cases;
- comparison threshold boundary vectors and structured reason codes;
- malformed/adversarial budgets and duplicate-key cases;
- output ownership/no-alias behavior;
- installed CLI exit and stderr contracts;
- golden public API import/type snippets.

Keep independent reference implementations visibly named and test them with
differential/property tests. Do not refactor producer and verifier in the same
commit.

### Acceptance

- Every supported artifact version has valid, invalid, and tampered fixtures.
- Canonical bytes and hashes are stable across the supported Python matrix.
- Unknown-field, extension, numeric-coercion, duplicate-key, and version policies
  are executable tests, not prose only.
- A compatibility matrix states what reads, writes, migrates, and rejects.

## Phase 4 — Consolidate domain primitives without weakening checks

**Priority:** P1. **Estimate:** 3-5 focused days.

### Safe consolidation targets

- canonical JSON codec and JSON error mapping;
- format IDs and format parsing;
- resource budgets and checked arithmetic;
- atomic text/JSON writing policy;
- UTC clock abstraction only if it supplies an injectable test clock;
- numeric canonicalization and common structural traversal;
- statistics (median/percentiles/summary, plus experiment IQR if appropriate);
- a typed operation identity with named projections;
- comparison threshold policy and machine-readable reason codes;
- status/result enums serialized to stable strings;
- nonempty-trace loading and public configuration parsing.

Operation identity must not become one flag-heavy tuple helper. Define a core
identity, then named projections such as compression identity, scheduler
resource identity, capture-coalescing identity, and noise identity. Tests state
which fields each projection includes.

### Deliberately independent targets

Keep separate:

- bounded-interval evidence production and source-based recomputation;
- report production and model-recomputation comparison;
- production comparison policy and an external golden/reference vector set;
- semantic digest storage and recomputation checks.

They may share codecs and field definitions, but not the derived calculation
being attested.

### Atomic I/O policy

Unify implementation mechanics while making policy explicit per artifact:

- parent creation, temp location, overwrite semantics, flush/fsync and parent
  durability;
- sensitive trace permissions versus shareable report permissions;
- symlink/containment behavior;
- cleanup on `BaseException` without swallowing typed errors;
- Windows/POSIX behavior;
- uniform `CommCanaryIOError` wrapping with the original cause.

### Configuration contracts

- Parse behavior configurations into a typed immutable value.
- Distinguish `None` (use defaults) from `[]` (invalid empty set).
- Require unique, nonempty names; reject unknown keys; validate ranges.
- Precompute bounded ranking work and reject over-budget matrices.

### Compiler/replay semantic cleanups

Make these only after the Phase 3 golden vectors exist:

- Use one declared behavior-search size key in uniform selection, per-group
  refinement, metadata, and tests. If the objective changes, version/name it.
- Replace `_important_timing_indices`' magnitude-encoded priority bands
  (`1e9` through `1e12`) with named priority tiers plus explicit tie-breakers;
  preserve ordering with characterization vectors.
- Make grouped event summaries use one representative rule for scalars and
  arrival offsets, or remove redundant event-level summaries when timing records
  are authoritative.
- Remove self-reference from serialized-size accounting where possible. If a
  fixpoint remains necessary, prove convergence or fail rather than silently
  ending after an arbitrary iteration count.
- Define one rounding boundary. Aggregate and emitted sample reconciliation must
  derive from the same rounded or explicitly full-precision values.
- Delete truly dead helpers after reference/compatibility checks; fix shadowed
  names and unused imports in mechanical commits.

### Acceptance

- One production definition exists for every safe-consolidation concept.
- Mutation tests prove the independent verifier still detects producer defects.
- No flag soup for identity projections; each projection is named and tested.
- Writers have a documented permissions/durability matrix and consistent typed
  failures.
- Current Ruff/mypy/tests/package gates remain green after each small extraction.

## Phase 5 — Split by dependency boundary behind compatibility facades

**Priority:** P1. **Estimate:** 5-8 focused days.

Do this as a sequence of moves with no semantic change, using the Phase 3 golden
suite. Avoid a “big bang” rewrite.

### Suggested homes

- `artifacts/`: formats, codecs, limits, trace/canary/report/comparison contracts,
  semantic validators, hashing;
- `compilation/`: trace normalization, grouping, timing compression, sequence
  motifs, compile core;
- `replay/`: scheduler, timing expansion under budget, accumulator/report core;
- `verification/`: fidelity/source reference path, behavioral checks, report
  recomputation;
- `comparison/`: typed thresholds, evaluations, verdict/reasons, compare API;
- `services/`: behavior search, verified compile, reduction orchestration;
- `adapters/`: capture, Kineto input, PARAM output;
- `reporting/`: HTML presentation;
- `cli/` only if command handlers remain too large after application services
  are extracted.

Small cohesive files may stay combined. A module is split because it has
multiple reasons to change or violates dependency direction—not because it
crosses an arbitrary line count.

### Extraction order

1. foundation/artifact contracts and compatibility re-exports;
2. comparison policy (already close to pure);
3. replay scheduler/accumulator;
4. compile normalization and compression;
5. independent verification;
6. behavior search/reduction into application services;
7. adapters and presentation.

For each extraction:

- move one responsibility;
- preserve old import paths with a tested facade where public or documented;
- run contract, type, import-boundary, and installed-wheel tests;
- delete the old implementation only after callers use the new home;
- update the architecture doc/ADR in the same commit.

Replace `reduce.py`'s imports of compiler underscore names with a package-private
ranking/configuration abstraction. Do not make internals public merely to silence
an underscore-import rule.

### Acceptance

- The enforced dependency DAG has no cycles or upward imports.
- Compile core does not import verification; orchestration owns verified compile.
- No cross-boundary private imports exist; package-private shared modules are
  intentional and documented.
- Each module has one cohesive responsibility and a narrow tested interface.
- Old supported imports either work through a facade or fail with a documented
  versioned break.
- Golden semantic hashes, metrics, and CLI behavior remain unchanged unless a
  format/API decision explicitly says otherwise.

## Phase 6 — Design the stable API, CLI, and presentation contract

**Priority:** P1. **Estimate:** 3-5 focused days.

### Python API

- Keep the top-level stable API small: version/errors, format capability query,
  high-level compile, replay, compare, validate/load, and core result/config
  types.
- Put capture and ecosystem adapters in documented submodules. Put research
  baselines/reduction in an explicit `experimental` namespace until stability is
  intended.
- Define `__all__`, stability tiers, deprecation window, and import tests.
- Source `__version__` from installed metadata with a safe source-tree fallback,
  and keep build metadata the single release source of truth.
- Replace public `Mapping[str, Any]` where useful with wire `TypedDict`s and
  typed config/results. Runtime parsing remains authoritative.
- Give the module-level capture helper a real signature. Introduce a clearer
  `byte_count` name without breaking the documented `bytes=` keyword abruptly.
- Public APIs state ownership: inputs are read-only; outputs are detached.

### Hash/format evolution

- In the next intentional format revision, remove or redefine the two redundant
  digest aliases. Prefer a clear taxonomy over inventing fake independence.
- Preserve old values in a compatibility reader only as long as the version
  policy promises.
- Add producer package version, format/canonicalization version, source commit or
  build provenance, and resolved configuration to an artifact or attached
  manifest without contaminating semantic hashes.

### CLI

Define and test a stable exit table, for example:

| Code | Meaning |
|---:|---|
| 0 | Operation succeeded / positive verification |
| 1 | Valid negative verdict or verification failure |
| 2 | Argument/usage error (argparse) |
| 3 | CommCanary input/configuration/I/O error |
| 4 | Child/workload execution failure (original child code recorded separately) |
| 130 | Interrupted |

- Never let raw child return codes collide with the tool's contract; print and
  record the child code.
- Add `--version` with package, format, canonicalization, and model versions.
- Add optional structured JSON diagnostics/progress on stderr for automation;
  keep stdout suitable for requested machine output.
- Report progress, elapsed time, candidates/oracle calls, and cancellation for
  expensive behavior search and reduction.
- Normalize subcommand and option naming, seeds/default constants, and output
  semantics. Reject method-inapplicable baseline flags rather than ignoring them.
- Treat `capture -- command...` parsing and partial-failure output as a tested
  product contract.
- Prefer `render-html` or a uniform `--html` model over an ambiguous `report`
  noun, with a deprecation alias if already released.

### HTML

- Parse/test generated HTML structurally; keep escaping and CSP/self-contained
  behavior explicit.
- If samples are absent, show quantiles/count and “samples unavailable.” Do not
  synthesize a distribution.
- Factor repeated table rendering only where the data model is genuinely shared.
- Keep output writes on the unified atomic I/O layer.

### Acceptance

- A downstream typed example uses only documented imports and passes against the
  installed wheel.
- Every CLI command has success, negative-result, usage, application-error, and
  relevant child-error tests.
- `--version` and package metadata agree.
- No presentation output fabricates unavailable data; HTML escaping/injection
  fixtures pass.
- API and CLI deprecations have messages, tests, and removal versions.

## Phase 7 — Turn experiments into a reproducible subsystem

**Priority:** P1 for artifact credibility, but parallel-safe after Phase 3.
**Estimate:** 5-8 focused days. This phase is engineering infrastructure, not an
instruction to do more research.

The useful material from the legacy planning checkout is incorporated here:
manifest schemas, failure records, completeness validation, a single verifier,
and generated reports. Its HPX/LCI-specific code and commercial documents should
not be copied.

### Run and result model

- Every campaign has an immutable `run_id` and `run_manifest.json` containing:
  - expected matrix and stable cell IDs;
  - CommCanary commit, dirty flag and patch hash;
  - script/config/input hashes;
  - PARAM/PyTorch/NCCL/Python/tool versions and PARAM patch hash;
  - exact command, environment constraints/lock, and expected site constraints
    (modules, topology, binding, account/partition policy);
  - warmup/measurement/aggregation/tie/exclusion policies;
  - raw archive URI and SHA-256 when results are external.
- Observed scheduler/job/node/account/partition metadata belongs in an
  append-only submission ledger and immutable cell attempts, not in the frozen
  manifest whose hash is known before submission.
- Each attempt produces one atomic terminal record: success, failed,
  parse-failed, cancelled, or explicitly excluded. Retries preserve earlier
  attempts, while exactly one selected terminal attempt feeds completeness and
  analysis. Failures retain exit code, stdout/stderr locations, and reason; they
  are evidence, not invisible rows.
- Store results below `results/<run-id>/`; never mix campaigns by globbing a
  shared directory.

### Runner

- Replace duplicated `.sbatch` scaffolding with a small common shell layer and a
  tested Python `run_cell`/result-writer.
- Make SBATCH files thin site-specific wrappers.
- Support `--resume`, `--only-missing`, `--retry-failed`, `--dry-run`, and
  collision-safe writes. Reuse is allowed only when code/config/input hashes
  match the manifest.
- Precompute the expected matrix and reject duplicate cell ownership before
  submission.
- Preserve partial outputs and a failure manifest on interruption.

### Environment

- Stop unconstrained build-tool upgrades and loose `numpy<2`/`pydot` installs.
  Use a lock/constraints file with hashes where practical, or capture an
  immutable resolved environment plus artifact hashes.
- Install a built CommCanary wheel from a recorded commit, not an unknown dirty
  editable checkout.
- Represent the PARAM compatibility change as a committed patch with a verified
  preimage and patch hash; do not repeatedly mutate third-party code via an
  unverified `sed`.

### Analyzer and schemas

- Give distinct producers distinct stdout/result schema IDs, or define one
  honest common schema without placeholder fields.
- Validate every record and manifest. Fail closed on missing, duplicate, stale,
  failed, or unexpected cells by default. `--allow-incomplete` must be explicit
  and the incompleteness must appear prominently in every output.
- Generate aggregate JSON/CSV, paper tables, and claims from validated results;
  never hand-copy headline numbers.
- Import shared public stats/policy only when semantics match exactly; experiment
  policy can remain separate and named when it intentionally differs.
- Unit-test analyzer completeness, deduplication, retry selection, failure
  accounting, and golden campaign fixtures.

### Replay-specific repairs

- Build real process groups from `global_ranks`, with deterministic collective
  creation order across ranks, or fail early on unsupported layouts.
- Validate every request/wait pair, dtype, buffer size, rank membership, and
  pending operation before execution.
- Generalize beyond all-reduce only when each operation has contract fixtures;
  do not claim generic PARAM replay from an all-reduce-only reference runner.

### Artifact publication

- Commit schemas, manifests, scripts, tiny fixtures, aggregate outputs, and
  checksums. Large raw data may live in an immutable archive.
- Add one command that fetches/verifies raw hashes (when needed), validates the
  matrix, regenerates aggregates/tables, and checks that the tracked paper output
  is clean.
- If `paper/` remains here, CI checks that generated tables/claims match the
  validated aggregate. Otherwise link an immutable artifact release.

### Acceptance

- A golden mini-campaign runs locally without SLURM and exercises success,
  failure, resume, duplicate, stale, and incomplete cases.
- Analyzer output is impossible without an explicit completeness verdict.
- Every published aggregate traces to cell IDs, input hashes, code/environment
  provenance, and an exact regeneration command.
- Setup from the same lock/manifest yields the same toolchain artifacts.
- Paper tables regenerate byte-for-byte (or canonically) from validated data.

## Phase 8 — Optimize only against measured scale contracts

**Priority:** P2 after correctness budgets. **Estimate:** 3-6 initial days, then
data-driven.

### Establish benchmarks

- Generate deterministic 1K, 10K, and 100K stored-event fixtures plus compressed
  cases with large but allowed logical expansion.
- Measure wall time and peak RSS for load/validate, compile, hash, replay, verify,
  compare, capture merge, PARAM export, and behavior search.
- Store environment and semantic output hashes with results.
- Run a small smoke benchmark in PR CI and a fuller scheduled benchmark with
  reviewed regression thresholds.

### Likely work, gated by profiles

- Avoid materializing motif/timing expansion in validation and hashing.
- Reduce repeated whole-artifact canonical serialization and deep copies while
  preserving detached-output ownership.
- Make capture rank-domain checks linear.
- Cache invariant compile/verification work across behavior-search candidates.
- Parallelize independent candidates with deterministic result ordering and
  bounded workers/memory.
- Use dependency-aware suffix recomputation only after proving scheduler state
  equivalence; changing one group can affect downstream queue state.
- Never use binary search without a tested monotonicity theorem for the exact
  acceptance predicate.

### Acceptance

- Resource caps are checked before allocation and benchmarked adversarial cases
  fail within the stated envelope.
- Operations intended to be linear demonstrate near-linear scaling over the
  benchmark range.
- Every optimization preserves golden semantic hashes, metrics, verdicts, and
  independent verification results.
- Performance changes include before/after time, RSS, fixture, environment, and
  tradeoff evidence.

## Phase 9 — Maintainer documentation and release

**Priority:** P2, with docs updated incrementally earlier. **Estimate:** 2-4 days.

### Documentation

- `README.md`: engineering-first quick start, honest status, pipeline, support
  matrix, security/integrity language, and links to deeper material.
- `docs/architecture.md`: dependency DAG, component responsibilities, data flow,
  functional-core boundary, extension points.
- `docs/formats/`: schema/version/canonicalization/hash/compatibility contracts.
- `docs/cli.md` and API examples generated or tested against actual help/signatures.
- ADRs for assurance terminology, hash taxonomy, resource limits, API stability,
  format evolution, dependency policy, paper/experiment boundary, and platform
  support.
- `CONTRIBUTING.md`, `SECURITY.md`, privacy/redaction guidance, release runbook,
  and optionally `CITATION.cff`/artifact-evaluation instructions.
- Remove stale numeric claims such as a hard-coded test count; generate them or
  avoid them.

### Release gate

- Clean archived source passes the full canonical gate on supported platforms.
- Exact wheel/sdist artifacts have hashes, content inventory, SBOM/provenance,
  and reproducible-build evidence.
- README commands and Pages demo run against the installed wheel.
- Changelog documents API/format/security changes and migrations.
- Tag/version/release name match; only then create the release/tag and publish
  the already tested artifacts.
- Post-publish smoke installs from the public index in a clean environment and
  verifies `--version`, help, a tiny pipeline, and artifact hashes.

## Test architecture end state

Organize tests by contract/capability, not by the old file layout:

```text
tests/
  contracts/       # schemas, canonical bytes, hashes, compatibility fixtures
  security/        # path containment, resource budgets, duplicate/deep JSON
  compilation/     # normalization, compression, motifs, behavior search
  replay/          # scheduler, accumulator, determinism, budgets
  verification/    # independent fidelity, behavior, report mutation tests
  comparison/      # policy boundary/golden vectors and reason codes
  adapters/        # capture concurrency/fork, Kineto, PARAM
  cli/             # subprocess exit/stdout/stderr and installed-wheel tests
  experiments/     # manifest/runner/analyzer mini-campaign
  fixtures/
  builders.py
```

Split the current monolith incrementally after contract tests exist. Builders
remove repeated JSON literals, but important golden artifacts remain literal and
reviewable. Prefer parameterization for malformed-field matrices. Avoid brittle
prose-substring assertions when a stable reason code exists; parse HTML when
structure matters.

The most valuable property/metamorphic tests are:

- serialize(parse(x)) canonicalizes idempotently;
- flat/motif and run/pattern encodings replay equivalently;
- all protected field mutations are detected at the promised assurance level;
- compiler mutations fail the independent verifier;
- input/output mutation never crosses an API boundary;
- compare verdict equals policy evaluations and reason codes at every threshold
  edge;
- expansion never exceeds its preflight count/budget;
- repeated seeds reproduce semantic outputs across supported Python versions.

## Parallel workstreams and critical path

```text
Phase 0 repository truth
          ↓
Phase 1 correctness/security contracts
          ↓
Phase 2 quality gate ───────→ Phase 3 wire contracts/characterization
                                      ↓
                              Phase 4 safe consolidation
                                      ↓
                              Phase 5 architecture split
                                      ↓
                              Phase 6 public API/CLI

Phase 7 experiments can start after Phase 3 using only stable contracts.
Phase 8 profiling can establish baselines after Phase 2; optimizations wait for
Phase 5 boundaries. Phase 9 documentation is updated throughout and closes release.
```

Do not parallelize producer and verifier refactors in a way that lets one copy
the other's new logic. Do parallelize independent adapters, docs, contract
fixtures, experiment manifests, and package/release infrastructure once their
contracts are frozen.

## Release milestones

### R0 — Coherent development snapshot

- Phase 0 complete; archived HEAD represents the tested 0.3 feature set.
- No release tag.

### R1 — Trustworthy core beta

- Phases 1-3 complete.
- Integrity levels are honest, outputs detached, and all work budgeted.
- Installed-package CI and contract fixtures green.

### R2 — Maintainable public API

- Phases 4-6 complete.
- Dependency rules enforced, compatibility facade documented, CLI/API stable.

### R3 — Reproducible engineering artifact

- Phase 7 and release-critical parts of 8-9 complete.
- Campaigns/manifests regenerate published aggregates; exact artifacts are tested
  and attested.

## Definition of done

CommCanary is the “extremely elegant engineering artifact” targeted here when:

- a clean source archive, wheel, and sdist all describe and execute the same
  product;
- every integrity claim has mandatory fields, recomputation, mutation tests, and
  precise assurance language;
- untrusted artifacts and capture metadata cannot escape paths or exceed work
  budgets before rejection;
- public functions are typed, detached, deterministic at the semantic level,
  and compatibility-tested;
- the dependency DAG is enforced and no god module is needed to understand a
  single subsystem;
- producer and verifier remain genuinely independent where correctness depends
  on disagreement;
- one local command is the CI command and validates the installed artifact;
- CLI exit/error/progress behavior is stable enough for automation;
- experiment outputs are manifest-driven, complete, deduplicated, reproducible,
  and traceable to exact inputs/code/environment;
- docs, schemas, examples, and release metadata are executable contracts rather
  than parallel prose;
- performance work is justified by recorded time/RSS profiles and preserves
  semantic golden outputs.

## Effort and sequencing realism

This is roughly **4-7 focused engineering weeks** for one experienced engineer,
depending on compatibility promises and how far the experiment subsystem is
taken. A credible first release does not need every Phase 8 optimization, but it
does need Phases 0-3, the release-critical parts of Phase 6, and an explicit
decision about Phase 7 artifact claims.

The fastest safe slice is:

1. reconcile the tree;
2. fix provenance/path/alias/resource defects with regression tests;
3. install the wheel-based quality gate;
4. freeze wire/hash/API contracts;
5. then refactor in small, behavior-preserving commits.

That ordering produces elegance by making invariants visible first. The module
layout then becomes a consequence of the contracts instead of a cosmetic rewrite.
