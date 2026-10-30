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

## October 2026 implementation and retrospective analysis

The implementation description is bound separately to
`cefad5a086b4723755bbe33b33e96f9f3a2c8e36`. The inspected implementation files
are identical at the local starting HEAD `c87d8b9234e6`; that later commit
changes only paper typesetting. The supporting source paths are:

- `src/commcanary/services/fidelity.py` and
  `src/commcanary/artifacts/measurement_set.py` for decision regret;
- `src/commcanary/services/phase_reduction.py` and
  `src/commcanary/experimental/__init__.py` for experimental phase reduction;
- `src/commcanary/services/physical_gate.py` for the four-state product gate;
- `src/commcanary/product/serving_harness.py`,
  `src/commcanary/product/vllm_serving_engine.py`, and
  `src/commcanary/artifacts/serving_measurement.py` for serving measurements;
- `src/commcanary/schemas/commcanary.physical_execution_measurement.v1.schema.json`
  for cost accounting; and
- `docs/physical-canary.md` and `docs/product-status.md` for graph selection,
  signed exchange, and the physical qualification boundary. The code takes
  precedence over the older three-state gate description in the first file.

`scripts/decision_regret.py` reads the retained trusted-join aggregate directly,
requires its original CSV and JSON digests, verifies the complete 280-cell
selection, and checks the five repetitions for each of eight configurations
in the five compared workloads. It uses the current fidelity scorer with
4,000 resamples, 95% percentile intervals, base seed 20261004, and a five-
replicate minimum. It reproduces the published agreement counts before writing
`evidence/decision_regret.json` and `tables/decision-regret.tex`; `--check`
requires exact regeneration of both. The JSON retains the measurement sets,
input digests, and per-proxy derived seeds. This is a retrospective analysis,
not a new campaign or a revised frozen gate policy.

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
