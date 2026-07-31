# Published Rostam evidence

This directory is an append-only publication of CommCanary's Rostam campaign
state through 2026-07-31. It contains immutable campaign manifests, submission
plans and ledgers, attempt records, scheduler records, selections, completeness
verdicts, source archives, normalized raw archives, and generated publications.
The exact-work owner-to-lab bundle additionally preserves the target wheel,
environment freezes, SBOM, request, materialization, raw observation, and a
second byte-identical publication.

The ordinary ignore rule remains in place so a local or cluster run cannot
accidentally stage new mutable state. These reviewed files were added
explicitly after their bytes were checked against Rostam.

## Nonredundant storage boundary

Selected raw attempt payloads are stored once, in the normalized archives under
`archives/`. The cluster's staging `workspaces/` directories are deliberately
not duplicated in Git: they contain the same selected payload bytes plus
ephemeral execution layout. All control-plane records, including failed and
superseded campaigns, remain present outside those workspaces. The exact-work
bundle is small and is retained in full.

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

## Verification

Check the published archive bytes from the repository root:

```console
git lfs pull
shasum -a 256 experiments/rostam/results/archives/*.raw.tar.gz
shasum -a 256 experiments/rostam/results/exact-work-artifacts/raw/*
```

The descriptor adjacent to each archive binds its exact size and SHA-256 plus
the campaign manifest, selection, and completeness-verdict identities. The
analysis tooling validates the normalized member inventory; publications embed
their regeneration command. [`../../../docs/artifact-evaluation.md`](../../../docs/artifact-evaluation.md)
documents the fail-closed analysis and reproduction procedure.

Historical publication bytes must be regenerated with the analyzer source that
created them. `analyzer-sources/` preserves deterministic `git archive` exports
for the two historical analyzer identities that are not otherwise recoverable
from a fresh checkout of the current branch:

- r6: commit `0911bbab808a3999b09b0a51a74c38db3ebf82a0`, archive SHA-256
  `1c308f69162cd6366a51f0eba05562a04020074ce543a8ba36281e6bfea0db65`
- trusted 280-cell join: commit
  `6757c8323056adda0e7df5b21c471abbed40590d`, archive SHA-256
  `93cf45cdea9c971a9f6553c74f40a61648514de441c38cea7472f29ff47d0f15`

These are analyzer-source conveniences, not replacements for a campaign's
manifest-bound submission source archive. In particular, Rostam no longer had
the original r6 submission tar whose SHA-256 is recorded as
`cd6dfc0fdec3604ee547e9316b1e0edbafefce26a6087c3f59e047e43ff7eccc`.
The exact r6 commit source, normalized selected evidence, descriptor, and all
three publication files are present, and the publication regenerates
byte-for-byte from them; the absent historical submission-tar bytes must not be
silently reconstructed or claimed as recovered.
