# CommCanary

**Replaying a workload's communication exactly does not reproduce the
configuration ranking of the workload itself.** On four A100s, a faithful
replay of a tensor-parallel decode loop's collectives agreed with the real
workload on 16 of 28 NCCL configuration pairs, and ranked the real workload's
slowest configuration second.

CommCanary exists to make that kind of statement checkable. It turns a
workload into a shareable proxy that carries its own fidelity claim, measured
against the full workload and bound to hash-verified evidence, so that a team
qualifying new silicon, an interconnect or an NCCL version can give a vendor
something faithful without handing over weights or prompts.

The measurement, over 280 verified cells, against `W-full`: a four-rank decode
loop of 32 layers × 256 tokens, each layer a sharded GEMM sized to realistic
per-layer decode compute followed by a bf16 all-reduce. It is a synthetic
workload shaped like decode, not a served model — skew and overlap arise from
kernel jitter rather than injection.

| proxy | pair agreement | 95% interval | Kendall τ‑b | pairs reversed |
|---|---:|---:|---:|---:|
| `W-shared-overlap` — shared-trace overlap replay | **20/28** | 18–23 | **0.708** | 2 |
| `W-micro` — an isolated microbenchmark, the `nccl-tests` analogue | 18/28 | 17–20 | 0.490 | 5 |
| `W-canary` — faithful communication-only replay | **16/28** | 12–17 | **0.204** | **9** |
| `W-canary-overlap` — per-configuration overlap replay | 15/28 | 8–21 | 0.677 | 0 |

What the data supports:

- **Communication alone is not the load-bearing variable.** Communication-only
  replay puts the two low-latency-protocol configurations first and second;
  the real workload puts them fifth and last. Both overlap replays, which keep
  compute concurrent with communication, place them exactly where the real
  workload does, and shared-trace overlap replay beats communication-only
  replay in 99.9% of bootstrap draws.
- **The proxies differ from each other through one configuration.** Remove
  `nccl-2.20.5-tree-ll` and no proxy is distinguishable from another, so this
  does not show that communication-only replay is generally worse than a
  microbenchmark.
- **Low agreement is not always wrong order.** `W-canary-overlap` never
  reverses a pair; its 13 disagreements are all ties, because most of its
  medians sit within a few microseconds of each other while its spreads reach
  18 µs.
- **It is not a cost argument.** At this workload size the proxies are not
  cheaper: median per-cell wall time was 7.9 s for the full workload against
  17.2 s for communication-only replay and 11.4 s for overlap replay.

Chakra replay, ASTRA-sim, AICB/SimAI and PARAM comms-replay all replay
communication; this is why a proxy's fidelity has to be measured rather than
assumed.

The point values are read from
[the frozen evidence](experiments/rostam/results/publications/trusted-join-core-shared-overlap-primary/aggregate.json);
the intervals and the sensitivity analysis come from
[a script](paper/arxiv/scripts/trusted_join_uncertainty.py) that regenerates
[its output](paper/arxiv/evidence/trusted_join_uncertainty.json) from that
evidence under test. Regenerating the aggregate itself means restoring the
campaigns' raw workspaces first, about 8.7 GiB; see
[the results guide](experiments/rostam/results/README.md).

## Try it in sixty seconds

```console
GIT_LFS_SKIP_SMUDGE=1 pip install "git+https://github.com/iemAnshuman/CommCanary"
commcanary demo
```

CommCanary is not on PyPI yet, so install from the repository for now.
`GIT_LFS_SKIP_SMUDGE=1` keeps Git from downloading the experiment archives,
about 770 MB of Git LFS objects the package does not need.

That compiles a bundled example trace into a canary, replays a baseline and a
deliberately regressed candidate, compares them, and writes an HTML report,
printing its path. It needs no GPUs, no cluster, and no input files. The replay
is a deterministic simulator over a bundled example, so the demo's numbers
illustrate the workflow rather than measuring your hardware.

![The comparison report commcanary demo produces](docs/images/comparison-report.png)

That report is what the command above writes. The verdict, the median/p95/p99
deltas, and the per-phase and per-operation breakdown are exactly the values the
demo emits: +32.0%, +36.1% and +40.4%. It is also published from CI at
[iemanshuman.github.io/CommCanary](https://iemanshuman.github.io/CommCanary/),
so the site and the local run cannot diverge.

## What this is

CommCanary's goal is a workload proxy that can leave the building: a compact,
replayable artifact derived from a private distributed-AI workload, with a
measured fidelity claim against that workload and enough retained evidence for
a receiver to audit the claim. Chakra carries the execution graph. CommCanary
owns dependency-closed selection, policy-conditioned minimization, the
fidelity measurement, and the evidence bundle, including signed private
exchange.

That goal is **not yet achieved**, and this README will not pretend otherwise.
What is built, what is measured, and what remains unproven are listed below.

## Claim status

| Claim | Status |
|---|---|
| Proxy fidelity ranking (the table above) | **Measured** |
| Chakra physical graph ingestion | Implemented and tested |
| Dependency-closed candidate construction | Implemented and tested |
| Training/holdout isolation | Implemented and tested |
| Signed private exchange | Implemented and tested |
| Exact-work configuration fidelity | Promising but **inconclusive** |
| Reduced physical runtime | **Unmeasured** |
| Held-out regression safety | **Unproven** |
| Production CI gate | **Not validated** |
| Multi-node support | **Unsupported** |

The exact-work study agreed on 26 of 28 configuration pairs and measured Kendall
τ‑b 0.857. Its frozen result is still **inconclusive**, because confidence
intervals crossed decision boundaries and one configuration was unstable. It
replayed the complete source-derived program, so it measured neither physical
reduction nor cost savings.

[`claims/public-claims.yaml`](claims/public-claims.yaml) is the machine-readable
source for every statement here, and CI fails if this file disagrees with it.

## Supported domain

```text
single node
exactly four NVIDIA GPUs
tensor-parallel dense-decoder inference
vLLM first; SGLang as the planned replication
GEMM and all-reduce dependency structure
```

Multi-node communication, expert-parallel all-to-all, CUDA graphs, multi-stream
execution, KV-cache pressure, and network congestion are **not** qualified.

## The real workflow

```bash
commcanary capture --output trace.json --chakra-output trace.et \
  --projection-output trace.projection.json --workload-name serving-decode \
  -- python examples/instrumented_decode.py

commcanary build trace.et --projection trace.projection.json \
  --policy study/policy.json --oracle-corpus study/oracle-corpus.json \
  --output serving.canary

commcanary gate serving.canary --baseline baseline.json \
  --candidate candidate.json --output gate.json --html gate.html
```

`build` requires real measured evidence and refuses to proceed without it:
synthetic fixtures cannot qualify a canary, and a bundle built without a real
application corpus exits non-zero and cannot be gated. That refusal is
deliberate. The [operator quick start](docs/operator-quickstart.md) walks the
boundary without fabricating evidence.

## Start here

- [Product status and evidence boundary](docs/product-status.md)
- [Machine-readable public claims](claims/public-claims.yaml)
- [Independent-operator quick start](docs/operator-quickstart.md)
- [Physical canary contract](docs/physical-canary.md)
- [Integrity model](docs/integrity.md)
- [Privacy and exchange](docs/privacy.md)
- [Roadmap](ROADMAP.md)

## Status

**Research alpha, version 0.3.0.** Apache-2.0. The CommCanary name and marks are
reserved as described in [NOTICE](NOTICE).

```bash
python -m pip install -e ".[dev]"
python -m tools.verify --fast
```
