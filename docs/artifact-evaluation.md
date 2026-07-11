# Artifact evaluation and pre-cluster handoff

This guide separates repository-local engineering verification from physical
execution on Rostam. Completing the first section does not imply that any
cluster command, GPU workload, PARAM replay, or SLURM submission has run.
The paper and legacy design document report an earlier narrow Rostam campaign;
its complete raw attempt archive is not tracked here, so this workflow does not
silently treat those reported numbers as a reproducible current campaign.

## Local reproducibility checks

From a clean source archive and a supported Python environment:

```console
python -m pip install ".[dev]"
python -m tools.verify --reproducible
python -m benchmarks smoke --output .benchmark-data/smoke.json
```

For a release candidate, first replace the changelog's `Unreleased` marker with
the reviewed ISO date, then run `python -m tools.verify --release
--expected-version 0.3.0`. Release mode deliberately fails while the version is
still marked unreleased.

The reproducible and release gates build a wheel and sdist, verify their exact members,
installs the wheel without a source-tree path override, executes the typed
public example and CLI smoke path, and emits inventory, SHA-256, SBOM, and
provenance metadata when output directories are requested. Release mode adds a
clean-HEAD and finalized-identity check. The benchmark smoke result contains
the environment and semantic hashes; its timing is not a portable regression
threshold.

The experiment subsystem has a local miniature campaign that covers success,
failure, retry selection, resume, collision, stale-input rejection, and
incomplete-matrix handling without SLURM. Analyzer output requires an explicit
completeness verdict and regenerates aggregate JSON, CSV, and Markdown from the
validated selected attempts.

The PARAM compatibility patch is also fully reviewed before the cluster
boundary. Its source archive is defined as the uncompressed bytes from
`git archive --format=tar
--prefix=param-a437fcebd3add1aee66fba880f28cec9fd744589/ COMMIT`, where `COMMIT` is
`a437fcebd3add1aee66fba880f28cec9fd744589`. This avoids claiming stability for
GitHub-generated download archives. Two Git 2.46.0 runs produced SHA-256
`d509a84fa3db007ab99be343b01f678d593628cda270af2ad571b15a2c06a7eb`.
The contract also binds target preimage
`68dfa9362b66d47a1203f95cc0f1484397f7052def3e0e124f2e12e8fa912f8d`,
contextual patch `59bf7dff99faf3d187a11424a641a9b2f0d190cf58794da2064d5542dc0141fc`,
and postimage
`219c95f65814d5db66762b96aa8ec5b34b7da4ca928b58abaaa48651880dd23a`.
The ordinary-context patch passes `git apply --check` on the clean commit;
neither setup nor PARAM execution was run.

## Inputs that remain site-observed

Before a Rostam submission, the operator must replace or record every unresolved
site value without changing the immutable expected cell matrix:

- account, partition/QoS, reservation, module set, node/GPU topology, CPU/GPU
  binding, and scheduler limits;
- exact Python, PyTorch, CUDA, NCCL, PARAM, compiler, driver, and tool versions;
- the resolved environment and every downloaded or built artifact hash;
- CommCanary wheel hash, source commit, dirty flag/patch hash, input/config and
  script hashes;
- the on-node bf16 GEMM duration used by both explicit-wait overlap exports;
- warmup, repetition, aggregation, tie, exclusion, timeout, and retry policies;
- observed job, node, scheduler, exit, stdout/stderr, and measurement records.

Observed values belong in the append-only submission ledger and immutable cell
attempts. They must not be guessed locally or edited into an already frozen
manifest.

The `overlap` and `shared-capture` catalog profiles already contain complete
profile/import/compile/export pipelines, named capture outputs, and replay
dependencies. Before freezing either campaign, replace
`PENDING_ROSTAM_GEMM_CALIBRATION_US` in its export command with the reviewed
on-node measurement and change that capture recipe's readiness value to
`ready`. The ordinary `core` profile remains independent of this calibration;
`shared-replay` separately binds the selected capture as the exact
`shared-param-trace` input.

## Physical execution boundary

Only after the local gate is green and the site values above have been reviewed
should an authorized operator run the setup resolver or submission commands on
Rostam. Preserve every terminal attempt, select exactly one attempt per expected
cell under the declared policy, verify raw archive hashes, and require the
default fail-closed completeness verdict before generating aggregates.

Published tables must be regenerated from that validated campaign. If a large
raw archive is stored externally, record an immutable URI and SHA-256 and use
the repository verifier before analysis. Never hand-copy headline values into
the paper.
