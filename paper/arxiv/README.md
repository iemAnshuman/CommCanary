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

Every physical number still comes from evidence present at commit
`c133ad3fb5883100cb8622b597d5b6233b5efca1`. The new values come from
`evidence/trusted_join_uncertainty.json`, which
`scripts/trusted_join_uncertainty.py` regenerates from the committed
trusted-join aggregate; the test suite checks that it regenerates exactly.

## Build and verify

From the repository root:

```console
python3 paper/arxiv/scripts/verify_sources.py
python3 paper/arxiv/scripts/trusted_join_uncertainty.py --check
paper/arxiv/scripts/build.sh
python3 paper/arxiv/scripts/package_source.py output/preprint/commcanary-arxiv-source.tar.gz
python3 paper/arxiv/scripts/verify_archive.py output/preprint/commcanary-arxiv-source.tar.gz
```

The PDF is written to `output/pdf/commcanary-arxiv.pdf`. Building needs
[tectonic](https://tectonic-typesetting.github.io/).

## Publication boundary

This directory prepares files for review. Submitting to arXiv, and pointing
`CITATION.cff` at the arXiv entry once it exists, are separate decisions.
