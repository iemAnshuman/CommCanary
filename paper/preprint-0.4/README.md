# CommCanary v0.4 preprint source reconstruction

This directory reconstructs the source and reproduction package for the
CommCanary v0.4 preprint published as
[`10.5281/zenodo.21939041`](https://doi.org/10.5281/zenodo.21939041). The
published record binds repository evidence inspection to commit
`c133ad3fb5883100cb8622b597d5b6233b5efca1`.

The reconstruction exists because the Zenodo record description says that a
complete source archive is included, while the public file inventory contains
only `commcanary-preprint.pdf`. These files were recovered from the published
paper and the immutable evidence retained at the bound commit. They are not
represented as the original authoring tree, and a rebuilt PDF is not expected
to match the published PDF byte for byte.

## Contents

- `main.tex`: complete reconstructed LaTeX manuscript.
- `references.bib`: bibliography used by the manuscript.
- `evidence/verified_metrics.json`: compact paper-facing values and exact
  source-file digests.
- `scripts/generate_figures.py`: deterministic generator for the four TikZ
  vector figures.
- `scripts/verify_sources.py`: checks the evidence snapshot against the bound
  repository artifacts.
- `scripts/package_source.py`: creates a deterministic source archive with a
  member manifest.
- `PROVENANCE.md`: publication identities, evidence locations, and claim
  limits.
- `WORKING_NOTES.md`: claim ledger and source-reconstruction notes.

## Build and verify

Run from the repository root at the bound commit or a descendant that retains
the same immutable evidence:

```console
python3 paper/preprint-0.4/scripts/verify_sources.py
paper/preprint-0.4/scripts/build.sh
python3 paper/preprint-0.4/scripts/package_source.py \
  output/preprint/commcanary-preprint-0.4-source.tar.gz
python3 paper/preprint-0.4/scripts/verify_archive.py \
  output/preprint/commcanary-preprint-0.4-source.tar.gz
```

The PDF is written to
`output/pdf/commcanary-preprint-0.4-source-reconstruction.pdf`. The source
verifier checks content identities and paper-facing metrics. It does not rerun
physical measurements.

## Publication boundary

This directory prepares files for review. It does not authorize a Git tag,
GitHub release, Zenodo modification, DOI reservation, package release, or
change to CommCanary's unreleased `0.3.0` software identity.
