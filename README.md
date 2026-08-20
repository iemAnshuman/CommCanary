# CommCanary

[![CI](https://github.com/iemAnshuman/commcanary/actions/workflows/ci.yml/badge.svg)](https://github.com/iemAnshuman/commcanary/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Preprint DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21939040.svg)](https://doi.org/10.5281/zenodo.21939040)

CommCanary aims to turn an expensive distributed-AI workload and a regression
policy into a short physical test that decides whether a stack change should
ship. Chakra carries the execution graph; CommCanary owns dependency-closed
selection, policy-conditioned minimization, asymmetric regression safety, and
the evidence bundle.

> **Research alpha.** Version 0.3.0 is unreleased. The implementation is ready
> for its first predeclared physical study. It is not a validated performance
> gate, a production CI product, or a source of measured cost savings.

## Supported domain

The first qualification study is deliberately narrow:

```text
single node
exactly four NVIDIA GPUs
tensor-parallel dense-decoder inference
vLLM first; SGLang as the planned replication
GEMM and all-reduce dependency structure
```

Multi-node communication, expert-parallel all-to-all, CUDA graphs,
multi-stream execution, KV-cache pressure, and network congestion are not
qualified.

## `capture → build → gate`

```bash
commcanary capture --output trace.json --chakra-output trace.et \
  --projection-output trace.projection.json --workload-name serving-decode \
  -- python examples/instrumented_decode.py

commcanary build trace.et --projection trace.projection.json \
  --policy study/policy.json --oracle-corpus study/oracle-corpus.json \
  --active-ledger study/active-study-ledger.json \
  --application-evidence study/application-evidence.json \
  --physical-evidence study/physical-evidence.json --runtime-budget 60s \
  --output serving.canary

commcanary gate serving.canary --baseline baseline.json \
  --candidate candidate.json --output gate.json --html gate.html \
  --junit gate.xml --sarif gate.sarif
```

The instrumented child must emit complete source-bound collective and GEMM
evidence. A frozen physical study produces the measured inputs to `build`.
Without a real application corpus, `build` writes a verified blocked bundle,
exits non-zero, and that bundle cannot be passed to `gate`. The
[operator quick start](docs/operator-quickstart.md) demonstrates this boundary
without fabricating evidence.

## Claim status

| Claim | Status |
|---|---|
| Chakra physical graph ingestion | Implemented and tested |
| Dependency-closed candidate construction | Implemented and tested |
| Training/holdout isolation | Implemented and tested |
| Exact-work configuration fidelity | Promising but inconclusive |
| Reduced physical runtime | Unmeasured |
| Held-out regression safety | Unproven |
| Production CI gate | Not validated |
| Multi-node support | Unsupported |

The exact-work study agreed on 26 of 28 configuration pairs and measured
Kendall tau-b 0.857. Its frozen result is still **inconclusive** because
confidence intervals crossed decision boundaries and one configuration was
unstable. It replayed the complete source-derived program, so it measured
neither physical reduction nor cost savings.

The evidence workflow and measured result are described in the
[CommCanary v0.4 preprint](https://doi.org/10.5281/zenodo.21939041). Use the
[concept DOI](https://doi.org/10.5281/zenodo.21939040) to cite the latest
preprint version and the version DOI above to identify the published v0.4
record. The manuscript source, evidence binding, and build procedure live in
[`paper/preprint-0.4/`](paper/preprint-0.4/README.md).

No real reduced-canary application corpus or held-out result is checked in.
Synthetic fixtures cannot qualify a bundle.

## Start here

- [Product status and evidence boundary](docs/product-status.md)
- [Preprint and reproducibility package](paper/preprint-0.4/README.md)
- [Machine-readable public claims](claims/public-claims.yaml)
- [Independent-operator quick start](docs/operator-quickstart.md)
- [Physical canary contract](docs/physical-canary.md)
- [Research history and detailed contracts](RESEARCH_HISTORY.md)
- [Research specification](RESEARCH_SPEC.md)
- [Integrity model](docs/integrity.md)
- [Privacy and exchange](docs/privacy.md)
- [Artifact evaluation](docs/artifact-evaluation.md)
- [Roadmap](ROADMAP.md)

CommCanary is not currently published on PyPI. Install only from a reviewed,
pinned source checkout or a wheel whose digest is recorded with the study.

```bash
python -m pip install .
python -m tools.verify --fast
```

Apache-2.0 covers the code. The CommCanary name and marks are reserved as
described in [NOTICE](NOTICE).
