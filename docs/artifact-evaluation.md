# Artifact evaluation and Rostam evidence

This guide separates repository-local engineering verification from physical
execution on Rostam. Historical pre-cluster sections below describe what had
not yet run at their recorded checkpoint; they are not claims about the current
repository state. The paper and legacy design document also report an earlier
narrow Rostam campaign whose complete raw attempt archive is not tracked here,
so this workflow does not silently treat those reported numbers as a
reproducible current campaign. The later verified-evidence section records the
new manifest-bound campaigns separately.

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

## Inputs that remained site-observed at the pre-cluster checkpoint

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

The `qualification-exact` profile is the deliberately narrow replacement
same-node gate. It contains one `nccl-2.20.5-default` cell pinned to `toranj1`,
the node that produced the preserved source capture. The cell collects twenty
maximum-rank whole-program samples internally; campaign repetitions remain
one. Freezing requires the exact request manifest, source trace, canary,
fidelity verification, materialization manifest, replay program, and preserved
source-capture evidence **and raw source-timing stdout** as separately named
inputs. The evidence file's committed stdout hash and timing summary must
recompute from those exact raw bytes. The physical producer revalidates the
complete request/materialization chain and source observation independently on
every rank before initializing `torch.distributed`, then executes only
asynchronous collective issue, the source-bound rank-local rectangular GEMM,
and immediate explicit wait.

The cell measurement retains both twenty-sample distributions, proves that
their maximum-rank timing semantics match, requires the observed execution
node to equal the source node, and deterministically reports signed and
relative median error. This is a byte-verifiable single-configuration
diagnostic, not an acceptance decision. Executor success and a same-node
timing comparison prove neither physical fidelity nor a qualification verdict.
Do not add an acceptance tolerance or issue a verdict until the later
multi-configuration ranking gate provides a reviewed basis for that policy.

### Predeclared decision-fidelity gate

The `decision-gate` profile is the product-claim experiment. Its immutable
policy is `experiments/rostam/policies/decision-fidelity-gate-v1.json`; the
policy file itself is a separately hashed campaign input. One cell per
configuration interleaves all six representations inside one allocation and
process group:

- direct source execution as ground truth;
- exact-work materialization as the product candidate;
- first-observed-per-collective-shape stratified sampling as the explicit kill
  baseline;
- the full blocking collective sequence without compute as the isolated
  incumbent;
- no-overlap and no-rank-skew causal ablations.

Each cell retains raw per-rank and maximum-rank CUDA-event samples, uses five
warmups and twenty measured passes in a rotated order, and requires a
deterministic SUM check for every collective shape. The frozen matrix covers
the two reviewed NCCL versions plus Ring/Tree by LL/LL128/Simple under NCCL
2.20.5. A producer or adapter change invalidates that campaign for future
execution: preserve every terminal attempt and freeze a replacement.

After exactly one successful terminal attempt is selected for every expected
cell, persist a zero-issue completeness verdict and regenerate the trusted
aggregate through `experiments.rostam.analyze`. Evaluate only that aggregate
against the exact bound policy bytes:

```console
python -m experiments.rostam.evaluate_decision_gate \
  PUBLICATION/aggregate.json \
  --policy experiments/rostam/policies/decision-fidelity-gate-v1.json \
  --output PUBLICATION/decision-fidelity-verdict.json
```

The evaluator recomputes the frozen policy ID and byte binding, campaign join,
configuration inventory, same-node execution, request/materialization/policy
identity, timing contract, bootstrap uncertainty, and stability limits. It
then reports ranking agreement, Kendall tau-b, false-positive and
false-negative counts, median/p95 relative error, execution-time ratios, all
predeclared criteria, and one of `pass`, `fail`, `inconclusive`, or
`incomparable`. The separate kill/reframe condition is evaluated only for
complete, comparable, stable evidence; noisy or incomplete evidence cannot
reframe the project.

Even a `pass` supports only the policy's declared single-node, four-A100,
single-inflight all-reduce/GEMM/wait domain. This reduced-source campaign does
not establish cost savings, importer generality, privacy acceptability, or
independent-operator usability.

## Historical physical execution boundary

Only after the local gate is green and the site values above have been reviewed
should an authorized operator run the setup resolver or submission commands on
Rostam. Preserve every terminal attempt, select exactly one attempt per expected
cell under the declared policy, verify raw archive hashes, and require the
default fail-closed completeness verdict before generating aggregates.

Published tables must be regenerated from that validated campaign. If a large
raw archive is stored externally, record an immutable URI and SHA-256 and use
the repository verifier before analysis. Never hand-copy headline values into
the paper.

### Cross-commit evidence extension

The normal trusted join still requires one repository identity. To extend
complete historical evidence with a campaign from a later repository state,
first prepare a non-executable compatibility candidate:

```console
python -m experiments.rostam.analyze prepare-compatibility \
  --ground-evidence OLD_RUN OLD_SELECTION OLD_VERDICT_SHA256 \
  --extension-evidence NEW_RUN NEW_SELECTION NEW_VERDICT_SHA256 \
  --output compatibility.candidate.json \
  --regeneration-command 'EXACT OLD REGENERATION COMMAND' \
  --golden-directory OLD_PUBLICATION
```

Repeat `--ground-evidence` for every campaign in the historical publication
and supply its archive descriptor/raw archive when that publication bound one.
The command revalidates completeness, regenerates all three publication files,
and byte-compares them before writing anything. Review the candidate's exact
campaigns, two repository identities, analyzer file inventory, and automatically
derived policy/input differences. Then repeat the same preparation with a new
output path and `--reviewed`.

The later trusted join supplies the reviewed contract and the immutable old
publication again:

```console
python -m experiments.rostam.analyze verify ... \
  --cross-commit-contract compatibility.reviewed.json \
  --compatibility-golden-directory OLD_PUBLICATION
```

If the old ground truth used a raw archive, also supply
`--compatibility-archive-descriptor` and `--compatibility-raw-archive`; the
ordinary archive flags bind the complete new join separately. The verifier
rejects candidate status, a third repository, omitted old campaigns, changed
analyzer/harness/schema bytes, an inexact evidence identity, policy changes
outside `script_hashes`, extra exemptions, and any non-byte-identical ground
truth. This contract proves consumer compatibility with exact historical
evidence; it does not claim that arbitrary code versions are semantically
equivalent.

## Verified physical evidence checkpoint (2026-07-26)

The following newly frozen campaigns were executed on Rostam at clean
repository commit `2855275288e67a1a2d0bbefff0740841fdf0ecf0`. Their generated
physical results, normalized raw archives, and publications are preserved under
[`experiments/rostam/results/`](../experiments/rostam/results/README.md). The
identities below were independently rehashed from the immutable manifests,
selections, completeness verdicts, archive descriptors, and byte-regenerated
publications rather than copied from scheduler output.

### Shared replay

`shared-replay-20260720-r2` has 40/40 selected successful cells and a
zero-issue completeness verdict:

- manifest SHA-256:
  `a402a4ec73ea3a182ab6bd5ec92e896600b16510dc4c1621c6defe9382ee149c`
- selection SHA-256:
  `a1e876861f7f9a315cb00a11a67414e6125078a68306c0b07a4a5d9f14b98d64`
- attempt-inventory SHA-256:
  `850deaf469170ae077a8d9d07656328a1da7375671deeb74a79c9200325057be`
- completeness-verdict SHA-256:
  `b6cd1aae4cfb2de020a840f941031d4a910d0d47d4c83aee1c09e0f5f6bc98db`
- raw archive: 1,057,573 bytes, SHA-256
  `3451ee540b634e1daad0aa49b6b95173fd601f5b6baea04deb6395d8d2c7b273`
- raw-archive descriptor SHA-256:
  `1a92668864f90071729b3d91d05d6e0838b72d42315fec8e444e17c8d7ee3d45`
- regenerated publication SHA-256 values: JSON
  `8d38b8f221c56e9aff6a3e9a91bc31123a55a864f998e46d76d8e16273a794f6`,
  CSV
  `7ac20fc5ca70d142d484e99a00db7e123a3a5caa9d2cc404446aad207012a91c`,
  Markdown
  `9a452cac23e4f9dba9a17b7b6fbf44b90d636583a578d19e4689a142471f5d96`

### Explicit overlap

`overlap-20260724-r1` has 80/80 selected successful cells, 400 verified
attempt artifacts totaling 4,399,697,532 bytes, and a zero-issue completeness
verdict:

- manifest SHA-256:
  `5d5eb9f4822e14f86023106742efdd48823b20b6a7cb866c3370367aced31d17`
- selection SHA-256:
  `e012e40613f57db8dc38d519dd85b85864f99dd3f95724b8cf0f7c62996239d9`
- attempt-inventory SHA-256:
  `6063afdeeb2d11eea0abd2b1f9b7fdf13a4f24e151d682cfa5c45a451611b2c2`
- completeness-verdict SHA-256:
  `5b153d0452df5033261d81f01db74211a0cb87015694b25190d8aa69650507fd`
- raw archive: 255,845,992 bytes, SHA-256
  `8c1f23012197fc55d59f1a5ec1c0d3518d4839403da7f15cc60498a801e02632`
- raw-archive descriptor SHA-256:
  `1b57eb8c1e3920563da37e95468c927b503fd68ac2f7601e4cdb00a3fda81afd`
- regenerated publication SHA-256 values: JSON
  `1de5266798ca361dce54107bcef993cfd403f28ee6c600784088feb3b3c391e9`,
  CSV
  `bb9acbce014d87621db61c596e0c7614b4ea96f5549d68aba9449abc011a6bfc`,
  Markdown
  `f3610943c71e07ce5e5c26b0eb6624f43d2b1faa8e0d7c04d1e805a25a0d8c65`

Each publication was regenerated a second time and compared byte-for-byte.
The overlap archive contains 1,164 normalized members and expands to
4,401,048,795 bytes; its published aggregate contains 16 rows and reports no
regression verdicts. These are observations from verified evidence, not
portable performance guarantees.

### Identity-aligned core

`core-20260724-r7` replaces `core-20260713-r6` for join purposes only; r6
remains valid evidence at its own older commit. It has 160/160 selected
successful cells, 640 verified attempt artifacts totalling 4,921,885,442
bytes, and a zero-issue completeness verdict:

- manifest SHA-256:
  `3bd321ec960c2bce0e6c0ff9aba8bb2be89cb19bdb99011a838cbd79068835b8`
- selection SHA-256:
  `d01db822c8e909fe2dbfac157bf62eeeb370a3eef45168f4eb1ff5023fbd4e3a`
- attempt-inventory SHA-256:
  `f6d1aa41199138347409fe4f791b49f46853154e7e74f54875656328f0425d40`
- completeness-verdict SHA-256:
  `46d8012b3594c18c96ee1a3f1e063ceeaf5642068c6a6aa895e655bcfb257bdd`
- raw archive: 272,033,323 bytes, 2,128 normalized members, SHA-256
  `c03e69fe41fb89431edbc9f8b95974d28d3c5055eee750bf07eb2c49625dd771`
- raw-archive descriptor SHA-256:
  `0a66df0863a6b359156bd43c28aa8086a8d5b7309ad5b2c20b5ffab0015e886d`
- regenerated publication SHA-256 values: JSON
  `6bb0f94637d40787f7f59fabce0627a5952d42936b1a90f30763db0a13b1fb65`,
  CSV
  `4e49c2cc91110c60ef1fc22ff20c00d14a4ccfb0a818b21d1cac800e45893d9f`,
  Markdown
  `2df1344cbaebbd767275a37fbca75251187ce83a611dabc7af270b436fb07c9d`

### Trusted three-campaign join

The join over core r7, shared replay r2, and overlap r1 covers all 280 selected
cells, reports `supported-by-complete-selected-evidence`, and was regenerated a
second time with the recorded regeneration command for a byte comparison:

- JSON `7a43e57ec4576f2f67e74d419bd793b6c639eccaec01fbdf769bdc0946220e2f`
- CSV `b5ff79a23abaf8bb5c5260c9c30ddef88858c82ce47c859d623d50f911af1257`
- Markdown `a53451c780b57b3d9fe3b49549f200c048a34e54b74ef99584490cb0c5e2c57d`

Two analyzer repairs were required and are covered by regression tests. The
join guard previously compared whole `campaign.policy` documents, so campaigns
of different catalog profiles could never join; it now compares the
analysis-relevant subset while input identity remains enforced by digest. The
publication serializer previously used the shared 1,000,000-item JSON budget;
the measured joined aggregate is 1,013,696 items in 8,101,263 bytes, so that
one call site now takes an explicit 4,000,000-item budget for output the
pipeline itself produced from validated evidence.

Reproducing the join requires the recorded regeneration command verbatim,
including argument order, because that string is embedded in the aggregate and
therefore in its hash.

## Exact-work same-node diagnostic (2026-07-31)

The duration-fitted qualification path was replaced with a source-bound
rank-local work recipe. Kineto evidence now binds each asynchronous collective
issue to the exact contiguous GEMMs before its explicit wait. Target
materialization reproduces that issue/work/wait program without fitting source
durations or applying a target calibration.

The reviewed target environment contract has SHA-256
`b51cffac1c66eeef636c034a40a80cd9b418969cf8c7b7a26a43f902e48f8d19`.
It binds wheel
`6511cca9c660992121e799a6aa999927d610b425794e428be9d85b35ef3be1b3`
and the complete CPython 3.12.3 / torch 2.4.1 environments with freeze hashes
`720014e3ea11675c48f6b7a2ba98814448e6d67d1ba5cfb7fd9244b1ab180dae`
for NCCL 2.19.3 and
`455bfa17975edee1da5e5328128608b46d42fdfda9de5450b6566381cdb8173b`
for NCCL 2.20.5. The resolver report, hash-complete locks, wheel, source
archive, checksums, inventory, and SBOM are retained with the results.

Two failed integration campaigns remain preserved. Job `178513` launched the
qualification file by path; job `178514` used module mode without exposing the
manifest-bound repository package to the isolated child. Their attempt-record
SHA-256 values are
`c07874af4d3be28e151560a5ebd8e8561670faf9b451aad5e9bf4b95c65cc636`
and
`1fffb8d92dcc5b55cdf700d2f117f227b45f931b3e8c4c477d4246d87e500240`.
Neither campaign was mutated after its bound script changed; r3 was frozen as a
replacement.

`qualification-exact-20260730-r3` selected one successful terminal attempt and
has a zero-issue completeness verdict:

- manifest SHA-256:
  `c0bc2e6a9ecb1da691a3caa083bd5842f054c770300a855a9ac69500a2834552`
- immutable plan SHA-256:
  `24b739986ad4c44bd70f581557b9929463ff861f07030bff6269c0fb61f1519e`
- attempt-record SHA-256:
  `43c41dac3423bfe5dfb7339842cb8cfd1b858559541565453e6302557ea55c79`
- selection SHA-256:
  `bf23480f39dd49794aa3d81b23a13d5576388563aeef12e8e77681877dd4ce9b`
- attempt-inventory SHA-256:
  `2099a1f3906867b3f70d2802d5fec4aec4c09fbb227e72e3d93414bfb2c4412c`
- completeness-verdict SHA-256:
  `593772c0902024443dd3e457f0e59003eea41a59014821417f629f5a34ff97b8`
- raw archive: 532,480 bytes, SHA-256
  `92ddc56f7ecbe5a7b162a7df749eeb01f4fd343eeab224be3c4f9d9996453bbe`
- raw-archive descriptor SHA-256:
  `b3e0293ded1f287a484d1883d2ba75c4b9cadc6082dcd14b761127db7ce68601`
- regenerated publication SHA-256 values: JSON
  `b42d4aea0451812cb1a7ff3d15a36cbcfaeadc47e0a8163725c7163b3bb80218`,
  CSV
  `853cb15b9ceafc96afaf315b29f719aef7a219256f534616aad77d7e99b0110d`,
  Markdown
  `c35446ce7c252c682bf83db8966ca6d4d521c3782eb192ba039037c04ca371cd`

Source job `177966` and replay job `178515` both ran on `toranj1` with twenty
whole-program samples. Source median/IQR were 1,434.112/9.216 us; replay
median/IQR were 1,541.0015/8.9725 us. The signed difference was +106.8895 us,
or +7.4533578967%, and all 32 deterministic data checks passed on four
A100-PCIE-40GB devices under NCCL 2.20.5.

This is a successful diagnostic execution, not an acceptance verdict. No
tolerance was declared before observation and no multi-configuration ranking
was measured. The trusted analyzer therefore reports
`not-applicable-no-full-workload`, and the preserved claims remain
`physical_fidelity: unproven`, `multi_configuration_ranking: not_measured`, and
`qualification_verdict: not_issued`.
