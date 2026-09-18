# Provenance

This revision is derived from the v0.4 preprint reconstruction; the identities
below describe that published version, which this revision supersedes.

## Publication identities

- Version DOI: `10.5281/zenodo.21939041`
- Concept DOI: `10.5281/zenodo.21939040`
- Publication date: 2026-08-14
- Zenodo PDF MD5: `eb8888760b29a35cf30a22a307bca9fc`
- Zenodo PDF size: 185,647 bytes
- Repository evidence commit:
  `c133ad3fb5883100cb8622b597d5b6233b5efca1`

The downloaded Zenodo PDF inspected during source reconstruction had SHA-256
`71df67a0581e8037af3d5972a67906d744cbec587c828240323e1961c3e235f1`.

## Evidence inputs

The paper-facing snapshot is derived from these retained repository artifacts:

- `experiments/rostam/results/publications/trusted-join-core-shared-overlap-primary/`
- `experiments/rostam/results/publications/decision-gate-20260801-r3-analysis-1300d12-primary/`
- `experiments/rostam/results/exact-work-artifacts/publication/qualification-exact-20260730-r3-primary/`

`evidence/trusted_join_uncertainty.json` records the input digests and output of
the uncertainty and sensitivity analysis. `evidence/verified_metrics.json` records the SHA-256 identity of every file used
by the reconstruction. `scripts/verify_sources.py` rehashes those inputs and
checks the headline metrics against the retained aggregate and verdict data.

## Interpretation boundary

The trusted comparison contains 280 selected cells. The exact-work gate
contains eight selected cells, and the diagnostic contains one selected cell.
The persisted exact-work gate outcome is `inconclusive`. Nothing in this source
package changes those results or supports production qualification, a passing
decision gate, reduced-canary cost savings, held-out application safety,
multi-node behavior, or generalization beyond the measured single-node,
four-A100 scope.

The source tree regenerates manuscript figures and typesets the reconstructed
paper. It does not rerun the physical campaigns, replace immutable evidence,
or establish byte identity with the already-published PDF.
