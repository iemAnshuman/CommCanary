# CommCanary arXiv manuscript

This directory is the revision prepared for arXiv (cs.DC). It starts from the
v0.4 preprint source reconstruction in [`../preprint-0.4/`](../preprint-0.4/README.md),
which stays unchanged as the record of what was published on Zenodo, and
applies an endorser's review:

- The title no longer carries a software version.
- The trusted-join counts carry 95% repetition-bootstrap intervals, and a
  sensitivity analysis removes `nccl-2.20.5-tree-ll`, the one configuration
  behind the differences between proxies. Claims that do not survive either
  are withdrawn rather than softened in place.
- Stability is reported for the trusted join, and the 47.5% instability is
  attributed to the exact-work gate campaign it came from.
- W-micro and the exact-work gate's isolated control are stated to be
  different measurements.
- The gate's overlap and rank-skew ablations are reported, and a mechanism
  hypothesis for the misplaced low-latency-protocol configurations is stated
  with the measurement that would test it.
- The artifact section says where the raw evidence lives and how to restore it.

The October 2026 revision also describes the implementation at
`cefad5a086b4723755bbe33b33e96f9f3a2c8e36`: Chakra-bound candidate selection,
experimental phase reduction, execution semantics, four-state physical gating,
sustained-serving measurement, cost accounting, and signed exchange. These
software capabilities are separate from measured physical qualification.

The new retrospective decision-regret table uses the retained trusted-join
repetitions. It reports first-choice regret with bootstrap intervals, best-of-
shortlist regret, and the position assigned to the reference's worst
configuration. `scripts/decision_regret.py` checks the source hashes and
published pair counts, then generates both the JSON report and LaTeX table.
This revision adds no hardware measurements and preserves the inconclusive
exact-work verdict.

Every physical observation still comes from evidence present at commit
`c133ad3fb5883100cb8622b597d5b6233b5efca1`. The new values come from
`evidence/trusted_join_uncertainty.json`, which
`scripts/trusted_join_uncertainty.py` regenerates from the committed
trusted-join aggregate, and `evidence/decision_regret.json`. The test suite
checks exact regeneration of both analyses and the regret table.

## Build and verify

From the repository root:

```console
python3 paper/arxiv/scripts/verify_sources.py
python3 paper/arxiv/scripts/trusted_join_uncertainty.py --check
python3 paper/arxiv/scripts/decision_regret.py --check
paper/arxiv/scripts/build.sh
python3 paper/arxiv/scripts/package_arxiv.py output/preprint/commcanary-arxiv-upload.zip
python3 paper/arxiv/scripts/package_source.py output/preprint/commcanary-arxiv-source.tar.gz
python3 paper/arxiv/scripts/verify_archive.py output/preprint/commcanary-arxiv-source.tar.gz
```

The PDF is written to `output/pdf/commcanary-arxiv.pdf`. Building needs
[Tectonic](https://tectonic-typesetting.github.io/). The build keeps `main.bbl`
and fails on overfull boxes or unresolved citations and references.

The manuscript uses the standard LaTeX `article` class: 10-point type, two
columns, US letter paper, one-inch margins, and monochrome vector figures.
It uses ordinary TeX Live packages and needs no shell escape, external fonts,
or figure conversion. This is a chosen preprint layout, not an arXiv-mandated
template. See [arXiv's TeX submission guidance](https://info.arxiv.org/help/submit_tex.html).

`commcanary-arxiv-upload.zip` contains `main.tex` at its root, `main.bbl`,
`references.bib`, and only the included figure and table sources. Select
`main.tex` and pdfLaTeX when reviewing the arXiv compilation settings. Build
before packaging so the bibliography matches the manuscript. The larger
`commcanary-arxiv-source.tar.gz` is the review/reproduction archive and also
contains analysis scripts, evidence, and provenance notes; it is not the
minimal upload package.

[`submission-metadata.txt`](submission-metadata.txt) contains copyable title,
author, category, and comments fields plus an abridged plain-text abstract
under arXiv's 1,920-character metadata limit. The typeset abstract retains the
additional detail. See [arXiv's metadata rules](https://info.arxiv.org/help/prep.html).

## Publication boundary

This directory prepares files for review. Submitting to arXiv, and pointing
`CITATION.cff` at the arXiv entry once it exists, are separate decisions.
