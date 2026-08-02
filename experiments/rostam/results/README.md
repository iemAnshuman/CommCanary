# Published Rostam evidence

This directory is an append-only publication of CommCanary's Rostam campaign
state through 2026-08-01. It contains immutable campaign manifests, submission
plans and ledgers, attempt records, scheduler records, selections, completeness
verdicts, source archives, normalized raw archives, and generated publications.
The exact-work owner-to-lab bundle additionally preserves the target wheel,
environment freezes, SBOM, request, materialization, raw observation, and a
second byte-identical publication.

The ordinary ignore rule remains in place so a local or cluster run cannot
accidentally stage new mutable state. These reviewed files were added
explicitly after their bytes were checked against Rostam.

## Nonredundant storage boundary

For the earlier large campaigns, selected raw attempt payloads are stored once
in normalized archives under `archives/`; their cluster staging `workspaces/`
are not duplicated. The exact-work diagnostic and all three decision-gate
campaigns are small and retained in full, including workspaces. r1 and r2 also
have deterministic preserved-campaign tarballs because their terminal failure
inventories correctly have no selection or completeness verdict and therefore
cannot claim a publication-grade raw-archive descriptor.

All normalized `.raw.tar.gz` archives use Git LFS consistently; three of them
exceed GitHub's ordinary 100 MiB object limit. A clone must run `git lfs pull`
before archive verification.

## Complete publication-grade campaigns

| Campaign | Selected cells | Verdict SHA-256 | Raw archive SHA-256 | Status |
| --- | ---: | --- | --- | --- |
| `core-20260724-r7` | 160/160 | `46d8012b3594c18c96ee1a3f1e063ceeaf5642068c6a6aa895e655bcfb257bdd` | `c03e69fe41fb89431edbc9f8b95974d28d3c5055eee750bf07eb2c49625dd771` | zero issues |
| `shared-replay-20260720-r2` | 40/40 | `b6cd1aae4cfb2de020a840f941031d4a910d0d47d4c83aee1c09e0f5f6bc98db` | `3451ee540b634e1daad0aa49b6b95173fd601f5b6baea04deb6395d8d2c7b273` | zero issues |
| `overlap-20260724-r1` | 80/80 | `5b153d0452df5033261d81f01db74211a0cb87015694b25190d8aa69650507fd` | `8c1f23012197fc55d59f1a5ec1c0d3518d4839403da7f15cc60498a801e02632` | zero issues |
| `qualification-exact-20260730-r3` | 1/1 | `593772c0902024443dd3e457f0e59003eea41a59014821417f629f5a34ff97b8` | `92ddc56f7ecbe5a7b162a7df749eeb01f4fd343eeab224be3c4f9d9996453bbe` | diagnostic, zero issues |
| `decision-gate-20260801-r3` | 8/8 | `6377a65bb05761427ae0f384f0ad07a14430285fc50e344155c44e4e3f11ce8d` | `285d1924e34b732467ed5918e407fe56dbde8e2e4226e1fb1279651d1fafb5f6` | zero issues; policy outcome `inconclusive` |

`core-20260713-r6` is retained as valid superseded evidence at its own
repository identity. Its archive SHA-256 is
`57e6e1a7054c6b64d67a7baa5957c5d6a837560580712a14199170c787e3529f`.
It is not an input to the current trusted join.

The trusted join over the 280 core/shared/overlap cells reports
`supported-by-complete-selected-evidence`. Its publication SHA-256 values are:

- JSON: `7a43e57ec4576f2f67e74d419bd793b6c639eccaec01fbdf769bdc0946220e2f`
- CSV: `b5ff79a23abaf8bb5c5260c9c30ddef88858c82ce47c859d623d50f911af1257`
- Markdown: `a53451c780b57b3d9fe3b49549f200c048a34e54b74ef99584490cb0c5e2c57d`

The exact-work diagnostic binds source job `177966` and replay job `178515`
on `toranj1`. Their twenty-sample medians are 1,434.112 us and 1,541.0015 us,
respectively: +106.8895 us, or +7.4533578967%. All 32 deterministic data checks
passed. No tolerance was declared and no multi-configuration ordering was
measured, so the evidence correctly issues no qualification verdict.

## Predeclared decision-fidelity gate

`decision-gate-20260801-r3` binds repository commit
`4585318ac244f218e438de06c0ccd38c0c88cbbf`, manifest
`1b1909b3e8561b035c0ee4f9e9c39e497c82d737f8a61e348b1cf6792d06e842`,
selection
`c675e42e34e44794a5dd8f12d3d4f4214e440c8b27643b9b4101edb1625391a0`,
and attempt inventory
`1a2ff04af7758fe23be1eb34f8dca9ba1ec6d23dcb22a3560ce139952119e27c`.
Jobs `178540` through `178547` ran one configuration each on `toranj1`; all
eight immutable attempts are successful. The NCCL 2.19.3 attempt reports the
actually mapped runtime library as `21903`; the other seven report `22005`.

Under the exact predeclared policy, exact-work replay observed:

- 26/28 pair agreement (92.86%) and Kendall tau-b 0.857;
- one false negative and one false positive;
- 1.55% median and 4.05% p95 absolute relative error;
- seven more agreeing pairs than the isolated baseline and sixteen more than
  stratified sampling.

All eight numeric criteria pass. The final outcome is nevertheless
`inconclusive`, verdict ID
`4084d4e4be96acfc5b68a4b259cc0a58be7a77bc85b8f722bcda2b5cb00dfaff`,
because pairwise bootstrap intervals cross decision boundaries and the
`nccl-2.20.5-tree-ll` source/exact-work/stratified cells exceed the frozen 20%
relative-IQR limit. The kill/reframe condition is not evaluated on noisy
evidence. The verdict file SHA-256 is
`3da7a5bf1af833276af84f448f9278f9e2bac1b0e7bdee5a1c365bb0a4dce29a`.

The evaluator-ready publication generated with analyzer source `1300d12` has
JSON/CSV/Markdown SHA-256 values
`5e153de3976cb913ae3321e1329adce2ea591277429522b4c859db44d35d4998`,
`08ec43c10b3573eec579443c7a1234a99c279f1520076fbaa88c0d9194d8ca26`,
and `af603c009ac1a2c65e090186489cb48663c36a124197a36f725782ca3e7c067e`.
A second regeneration and a second policy evaluation match byte-for-byte.
The earlier Rostam-side publication is retained separately: its older analyzer
did not project `decision_gate_runtime`, so it is reproducible historical
output but cannot feed the later fail-closed evaluator.

The source and exact-work representations execute the same closed event
program. This result is a positive control for exact qualification-capsule
reconstruction, not evidence that a reduced canary is smaller, faster, or
decision-faithful. The separately versioned replicated v2 design has not run
and contributes no rows or verdict to this directory.

r1 and r2 remain immutable integration evidence. r1 jobs `178523`–`178530`
all reached terminal `parse-failed` after PyTorch's local-version suffix was
compared literally. r2 jobs `178532`–`178538` succeeded, while 2.19.3 job
`178531` correctly failed because the producer reported PyTorch's compile-time
NCCL version instead of the selected mapped library. Their deterministic full
campaign archives have SHA-256 values
`78c20bccc9519371fe33f2ef0203b30624cde988636442bd1fac6ffcbe64033d` and
`244016097d74b5b5584479155cb3f5f81d206b53d0dd1045d1940d109fa2fd62`.

## Verification

Check the published archive bytes from the repository root:

```console
git lfs pull
shasum -a 256 experiments/rostam/results/archives/*.raw.tar.gz
shasum -a 256 experiments/rostam/results/archives/decision-gate-*.tar
shasum -a 256 experiments/rostam/results/exact-work-artifacts/raw/*
```

The descriptor adjacent to each archive binds its exact size and SHA-256 plus
the campaign manifest, selection, and completeness-verdict identities. The
analysis tooling validates the normalized member inventory; publications embed
their regeneration command. [`../../../docs/artifact-evaluation.md`](../../../docs/artifact-evaluation.md)
documents the fail-closed analysis and reproduction procedure.

Historical publication bytes must be regenerated with the analyzer source that
created them. `analyzer-sources/` preserves deterministic source-only
`git archive` exports for analyzer identities that should not depend on later
repository state:

- r6: commit `0911bbab808a3999b09b0a51a74c38db3ebf82a0`, archive SHA-256
  `1c308f69162cd6366a51f0eba05562a04020074ce543a8ba36281e6bfea0db65`
- trusted 280-cell join: commit
  `6757c8323056adda0e7df5b21c471abbed40590d`, archive SHA-256
  `93cf45cdea9c971a9f6553c74f40a61648514de441c38cea7472f29ff47d0f15`
- decision-fidelity aggregate/evaluator: commit
  `1300d12157926305ab7c2e61623cf52db1589a07`, source-only archive SHA-256
  `080bdbe4ea883ce2935bd2a5bdba2ff1f0f9a35462ce25c6c3fa19d5abd3132d`

These are analyzer-source conveniences, not replacements for a campaign's
manifest-bound submission source archive. In particular, Rostam no longer had
the original r6 submission tar whose SHA-256 is recorded as
`cd6dfc0fdec3604ee547e9316b1e0edbafefce26a6087c3f59e047e43ff7eccc`.
The exact r6 commit source, normalized selected evidence, descriptor, and all
three publication files are present, and the publication regenerates
byte-for-byte from them; the absent historical submission-tar bytes must not be
silently reconstructed or claimed as recovered.
