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

## Recorded pre-Rostam binding (2026-07-11)

The repository-local gate has been executed to completion and its exact
artifacts recorded. This is the handoff state a Rostam operator starts from;
nothing below implies any cluster execution.

- Source commit: `521b8fb7909933c29d73fb820bbae4015eb30ff4`
  (branch `codex/engineering-plan-implementation`, clean worktree).
- `python -m tools.verify --reproducible --artifact-dir dist --metadata-dir
  release-metadata` passed on macOS arm64 / CPython 3.10.14: 699 tests,
  90.47% statement coverage against the 86% floor, every per-responsibility
  branch floor met, strict mypy, Ruff, import boundaries, schema/shell/
  workflow/README checks, two fixed-epoch builds byte-identical, and the
  installed wheel retested outside the checkout.
- Exact artifacts (also in `release-metadata/SHA256SUMS`, inventory, and SPDX
  SBOM produced by the same run):
  - `commcanary-0.3.0-py3-none-any.whl`
    `sha256:416dbea60943cf5ff93282547b1c350edb8880e0cc3e719bf6d1aef794a6738e`
  - `commcanary-0.3.0.tar.gz`
    `sha256:49443d276dea934e06494a30159ff247a53d461bf048febe095da667e2058014`
- A `git archive HEAD` checkout of that commit passed all 699 tests in a fresh
  virtualenv against the exact tested wheel, and `commcanary --version`
  reports the full format capability matrix.
### Rostam rebuild and rebinding (2026-07-11, later the same day)

Running the same gate on rostam1 (CPython 3.12.3, linux x86_64, setuptools
80.10.2, source commit `1bc688f4898cfe1d4ab0e20d15086493bb61549a`) passed end
to end — 700 tests, all floors, two fixed-epoch builds byte-identical on that
machine — and produced a wheel whose zip container bytes differ from the
macOS build:

- `commcanary-0.3.0-py3-none-any.whl`
  `sha256:1fe7fa8e61731df41129ee012b8cb260ecfbee76448f83f08c2bf9cb5f4c484d`

Byte-identical rebuilds hold per machine and toolchain; the zip container is
not byte-stable across platforms/interpreters. The per-member contents are
what must agree. Sorted member content digest (sha256 over
`sha256(name + bytes)` per member) of both wheels:
`f70471f981614673bf34f1cdecb9f2955103d0dcc483fd7d92b8959f09e601f6`.

The environment contract's `commcanary_wheel` binds the **Rostam-built**
wheel — produced by the full canonical gate on the target platform, which is
the strongest provenance for the bytes that will actually be installed there.
The macOS wheel and sdist remain retained as the cross-platform cross-check.
The member content digest of the bound wheel must equal the recorded value
before any submission.

### Post-fix rebind (2026-07-12, after the import-budget fix)

The first physical campaign exposed a source fix (`import-kineto
--max-input-bytes`), so the wheel above is superseded. The current bindings,
each produced by a full green gate run on its platform:

- macOS reference: `commcanary-0.3.0-py3-none-any.whl`
  `sha256:0baa371773cd21674ff6e2ed1f2713d54a48a0cc953b6b4873c19f178bbbcc42`
- Rostam-built, contract-bound: `commcanary-0.3.0-py3-none-any.whl`
  `sha256:11c2aa5d2d505dcc6e2ceccc600a6b00949a0b83f55368ed4c1042b46b63563e`
- Sorted member content digest of both (the cross-platform equality check,
  iterated in sorted-name order, updating with
  `sha256(name + bytes)` raw digests per member):
  `c7bd941142e4a4617c1f85daa212cd9ede4905479f152945b39147b8b9a5ec48`

Rebinding lesson recorded for future wheel changes: `pip freeze --all`
records direct-URL installs as `commcanary @ file:///…whl#sha256=<wheel
hash>` (pip ≥ 24.x reads PEP 610 `direct_url.json`), so the contract's
per-environment `freeze_sha256` evidence **embeds the wheel hash**. Any wheel
rebind therefore requires re-capturing freeze evidence from disposable probe
venvs before `setup.sh` can certify the rebuilt environments — the freeze
check refusing after a rebind is the contract working, not an error.

### Post-item-budget rebind (2026-07-13, r6 pending)

The first r5 chunk exposed the independent structural JSON ceiling after the
byte-budget repair: every trace-build importer rejected its approximately
86.7 MiB Kineto profile at the default `max_json_items=2,000,000`. A
constant-memory structural scan of the largest retained profile measured
5,685,910 object members plus array elements. The reviewed repair adds
`import-kineto --max-json-items` as another explicit, per-invocation override
for trusted local profiles and binds 12,000,000 at the core, overlap, and
shared Rostam call sites. Defaults remain unchanged and fail closed.

Because this changes both wheel contents and the manifest-bound Rostam
catalog, r5 is retained as failed evidence and cannot be repaired in place.
Before r6 submission the operator must rebuild the target wheel, rebind its
digest, re-capture both wheel-embedding freeze hashes, rebuild/certify the two
venvs, and freeze a new manifest. The reproducible macOS reference package
gate produced:

- wheel SHA-256:
  `d1c2919af3157a6d76abe278ac76e7dbb6fdf03d005d418fcef54c1a423b44f4`
- sorted member-content digest:
  `2189f3dda9a484952a798a6ada156fc3c65abdb644f18829f72e28d953d4fedf`

The Rostam-built container digest remains pending; its member-content digest
must equal the reference above before the environment contract is rebound.

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
