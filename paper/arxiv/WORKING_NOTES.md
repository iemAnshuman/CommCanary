# Working notes

## Supported statements

- The trusted join is complete at 280 of 280 selected successful cells.
- W-canary agrees with W-full on 16 of 28 pairs, or 57.1%, with Kendall tau-b
  0.204.
- W-micro agrees on 18 of 28 pairs, or 64.3%, with tau-b 0.490.
- W-shared-overlap is the strongest measured proxy at 20 of 28 pairs, or
  71.4%, with tau-b 0.708.
- Exact work has 26 of 28 point agreement, tau-b 0.857, one false negative,
  one false positive, 1.55% median absolute relative error, and 4.05% p95
  absolute relative error.
- Every exact-work point criterion passes, but the persisted outcome is
  `inconclusive` because uncertainty crosses decision boundaries and one
  configuration violates the frozen stability limit.
- The same-node diagnostic has a 1,434.112 microsecond source median and a
  1,541.0015 microsecond replay median; all 32 deterministic checks pass. It
  issues no qualification verdict.

- W-canary's 95% repetition-bootstrap interval is 12 to 17 agreeing pairs.
  W-shared-overlap agrees on more pairs than W-canary in 99.9% of draws.
- With nccl-2.20.5-tree-ll excluded, no proxy is distinguishable from
  another.
- W-canary-overlap reverses no pair; all 13 of its disagreements are ties.
- Every trusted-join cell is within the gate's 20% relative-IQR limit.
- The gate's no-overlap ablation agrees on 18 of 28 pairs and its
  no-rank-skew ablation on 24, against 26 for exact work.

## Unsupported statements

Do not claim that communication-only replay is generally worse than a
microbenchmark, or that per-configuration overlap replay is worse than plain
replay; both depend on one configuration or sit within noise.


Do not claim that CommCanary is production qualified, that the decision gate
passed, that a reduced physical canary saves time or money, that a held-out
vLLM or SGLang study succeeded, or that the result generalizes to other nodes,
accelerators, topologies, congestion conditions, or workloads.

## Reconstruction note

The public Zenodo file inventory contains only the PDF, although the record
description refers to a source archive. The manuscript source in this directory
was reconstructed from the published PDF and retained evidence. Preserve that
qualification in any correction note or new deposit version.
