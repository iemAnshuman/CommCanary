# CommCanary agent guide

## Purpose

CommCanary is a Python 3.9+ library for deterministic collective-trace
replay, comparison, and reduction. Version 0.3.0 is still unreleased. The
engineering priority is trustworthy, fail-closed behavior: inputs, ownership,
resource limits, provenance, and experiment evidence must be explicit and
independently verifiable.

This file contains stable operating rules. Read `HANDOFF.md` for the current
repository and Rostam campaign checkpoint. `ENGINEERING_PLAN.md` is the long
roadmap and contains useful history, but its top checkpoint can lag the live
handoff during cluster work. `experiments/rostam/DESIGN.md` includes historical
research context and must not be mistaken for current reproducible evidence.

## Working rules

- Preserve user changes and unrelated files. Inspect `git status` before an
  edit and never clean, reset, checkout, or delete work that is not yours.
- Use `apply_patch` for text edits. Use `rg`/`rg --files` for discovery.
- Do not commit or push unless the user authorizes it. `HANDOFF.md` may contain
  operational context; do not commit it unless the user explicitly requests
  that publication decision.
- Do not tag, publish, or change the 0.3.0 release identity without an explicit
  release decision.
- Do not spawn sub-agents unless the user explicitly requests delegation.
- Lead with verified outcomes. Distinguish observed evidence from assumptions
  and historical reports.

## GitHub issues & shared actions (hard rules)

- **Never** create, open, edit, close, or comment on a GitHub **issue**, **PR**,
  **discussion**, or review without the user's **explicit one-shot approval**
  for that specific action. "Fix this" or "look into X" is not approval to file
  or open anything.
- **Never** run `gh issue create`, `gh pr create`, `gh pr edit`,
  `gh issue comment`, or equivalent web/API actions unless the user clearly
  approved that exact action in the current turn.
- When you **find** a real bug, regression, missing test, docs hole, harness
  invariant violation, or upstream problem (including ones outside the current
  task):
  1. Report it clearly to the user: what, where, evidence, severity.
  2. **Prompt the user to raise a GitHub issue** (or to approve you creating
     one).
  3. Offer a ready-to-paste title + body if helpful.
  4. Do **not** file the issue or open a PR yourself unless they explicitly
     say to.
- Stay scoped to the assigned task. Surfacing extras is good; silently widening
  into unapproved PRs/issues is not.

## Verification

Use the smallest relevant gate while iterating, followed by a proportionate
broader gate:

```console
python -m pytest -q tests/experiments/rostam
python -m ruff check experiments/rostam tests/experiments/rostam
git diff --check
```

For library-wide changes, use the canonical verifier:

```console
python -m tools.verify --fast
```

Use `--reproducible` for artifact/reproducibility work and `--release` only at
the deliberate release boundary. Run the verifier from a supported dev
environment with the project dev extra installed. Never call a focused test
suite proof of release readiness.

## Rostam evidence invariants

The `experiments/rostam/` system is append-only and fail-closed:

1. A campaign is frozen before submission. Its manifest binds the repository
   state, catalog, inputs, execution scripts, matrix, and site contract.
2. A submission plan is immutable, canonical, and reviewed before `sbatch`.
   Submission requires the separate `--execute` acknowledgement.
3. Every cell has one explicit owner and every retry gets a new attempt ID.
   Never overwrite or delete failed, cancelled, or successful attempts.
4. A selection names exactly one terminal attempt for every expected cell.
   Analysis requires a persisted, fail-closed completeness verdict.
5. Raw archives and publications are post-run products bound to exact
   manifest, selection, and verdict hashes. Existing output must be verified
   and reused only when byte-identical.
6. A change to any manifest-bound input or execution script invalidates a
   frozen plan/run for future execution. Freeze a replacement run; preserve
   the old run as evidence.
7. Never bypass input, script, venv-wheel, PARAM-postimage, node, scheduler,
   selection, completeness, or trusted-join checks to rescue a campaign.

Generated physical results live on Rostam, not in the local checkout. Do not
fabricate cluster output or edit immutable result records by hand.

## Rostam access and operator boundary

Direct agent access to Rostam is permitted when the user explicitly authorizes
it and authentication is available through user-managed SSH configuration or
an SSH agent. Never request, print, copy, or store passwords, private keys,
tokens, or other credentials in the repository, workspace, command output, or
conversation.

- Guard the expected Git HEAD, tracked cleanliness, frozen hashes, and prior
  attempt inventory before scheduler mutation.
- Scope queue mutation to CommCanary jobs on partition `cuda-A100`. Jobs from
  other users or the user's unrelated partitions are read-only and must never
  be cancelled, held, reprioritized, or otherwise touched.
- `toranj0` is shared with Jenkins. A Jenkins allocation is not a reason to
  stop submission; CommCanary jobs may wait normally in the queue.
- **GPU fleet (2026-07-29):** After NVIDIA GPU reset failures, `toranj0` and
  `toranj1` were rebooted and report 4× A100 OK each. Also OK: `anvil` (8),
  `bahram` (2), `diablo` (4), `nasrin0` (2). `nasrin1` is UNREACHABLE /
  unstable—do not pin work there. Site-wide CUDA driver/toolkit upgrade is
  TBA (admin). Pre-execution `No CUDA GPUs are available` on an exclusive
  `cuda-A100` allocation is cluster/driver evidence, not a library bug—see
  `HANDOFF.md`. Re-probe with `nvidia-smi` inside the allocation before
  blaming the job.
- Maintain a low footprint: submit chunks of 16–24 cells unless a smaller
  canary is required first.
- Use literal job IDs in follow-up commands. Do not store or pass job IDs via
  environment variables. For completed jobs use `sacct -j LITERAL_ID`; do not
  depend on `squeue -j`, which can reject IDs after they leave the live queue.
- Treat the recurring Python `runpy` warning from the submission module as
  harmless only when the command otherwise completes and the frozen ledger is
  correct.
- Never run `git pull`, rebase, or merge casually on Rostam. Its branch has
  intentional local evidence-binding commits. Use exact guarded cherry-picks
  only when a reviewed fix is required, and freeze a new campaign afterward
  if a bound script changed.

## Key paths

| Path | Responsibility |
|---|---|
| `HANDOFF.md` | Live state, exact hashes, completed/remaining work |
| `ENGINEERING_PLAN.md` | Long engineering roadmap and historical checkpoints |
| `tools/verify.py` | Canonical local/release verification |
| `experiments/rostam/configs.json` | Strict physical campaign catalog |
| `experiments/rostam/lib/campaign.py` | Campaign construction and freezing |
| `experiments/rostam/lib/submission.py` | Canonical plan/freeze/submit boundary |
| `experiments/rostam/lib/cell_entrypoint.py` | Runtime ownership and evidence guards |
| `experiments/rostam/harness/` | Manifests, attempts, selections, completeness |
| `experiments/rostam/analysis/` | Trusted aggregation, archive, publication |
| `experiments/rostam/constraints/` | Reviewed environment and dependency evidence |
| `experiments/rostam/patches/` | Reviewed PARAM patch contract |
| `docs/artifact-evaluation.md` | Artifact and cluster-evidence procedure |

## Definition of done

A code change is done when its behavior is covered, relevant gates pass, docs
and contracts agree, and no generated/user state was disturbed. A physical
campaign is done only when all expected cells have verified selected success,
completeness is persisted with zero issues, all historical artifacts verify,
the raw archive descriptor verifies exact bytes, and publication regenerates
deterministically from the trusted evidence.
