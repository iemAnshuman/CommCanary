# CommCanary roadmap

CommCanary is aimed at one concrete exchange:

> A workload owner needs to qualify new hardware, an interconnect, or an NCCL
> version without sending the vendor model weights, prompts, or application
> code. The receiving lab needs to verify that the portable workload-shaped
> test corresponds to the owner's source evidence before it trusts a result.

The product is therefore the verifiable owner-to-lab handoff, not another
collective microbenchmark and not a new general workload-graph format.

## What exists

- Multi-rank Kineto profiles can be imported only when collective identity,
  message shape, dtype, reduction semantics, timing alignment, and linked
  overlap evidence are complete. Unknown or conflicting evidence refuses.
- `prepare-qualification` creates a model-free request that commits to the
  exact source profiles, trace, canary, fidelity proof, and rank-local compute
  recipes.
- A receiving lab can independently run `verify-qualification`, materialize a
  deterministic issue/work/wait program, reverify it, and execute the bounded
  PyTorch reference path.
- Immutable campaign manifests, attempt ownership, selections, completeness
  verdicts, archives, and publications make every retained physical result
  traceable to exact inputs and code.

The latest same-node diagnostic executed exact source-derived work on four
A100-PCIE-40GB GPUs. The source median was 1,434.112 us and the replay median
was 1,541.0015 us, a +7.4533578967% difference; all 32 deterministic data
checks passed. This is a measured diagnostic, not a qualification pass: no
acceptance tolerance was declared before observation and only one NCCL
configuration was tested.

The earlier physical matrix is equally important negative evidence. Against
the reference workload over 28 configuration pairs, communication-only replay
agreed on 57.1% of pairs, below the isolated microbenchmark at 64.3%.
Overlap-bearing replay reached 71.4%. A portable artifact must therefore carry
causal compute/communication structure; a faithful list of collectives is not
enough.

The predeclared exact-work decision gate has now run over eight configurations
on the same four-A100 node. All eight selected attempts passed correctness and
the persisted completeness verdict has zero issues. Exact-work replay observed
26/28 pair agreement (92.86%), Kendall tau-b 0.857, one false negative, one
false positive, 1.55% median absolute relative error, and 4.05% p95 error. The
isolated baseline observed 19/28 agreement (67.86%); stratified sampling
observed 10/28 (35.71%). Every numeric point-estimate criterion passed.

The policy outcome is still **`inconclusive`**. Bootstrap intervals crossed
required pair boundaries, and `nccl-2.20.5-tree-ll` exceeded the 20% relative
IQR stability limit for source, exact-work, and stratified measurements. The
kill/reframe condition is deliberately not evaluated on noisy evidence. These
results are promising evidence, not permission to relabel CommCanary a
validated qualification tool.

This gate reconstructed the complete source-derived program. It advances the
exact qualification capsule, not the separate reduced-canary claim. A reduced
physical canary must execute fewer or smaller artifacts and earn fidelity,
regression-sensitivity, runtime, and size results of its own.

See [`docs/artifact-evaluation.md`](docs/artifact-evaluation.md) for the exact
campaign identities, hashes, and reproduction contract.

## Product gates

1. **Replicated exact-capsule follow-up.** The first complete predeclared
   matrix is `inconclusive`, not failed and not passed. The versioned v2 design
   predeclares eight independent repetitions per configuration, 24
   carryover-balanced Williams passes, pre/post telemetry with a shared
   platform invariant, unpaired hierarchical resampling, and simultaneous
   uncertainty. It has not run and is freeze-blocked until a
   content-addressed Python/Torch runtime image is manifest-bound. Do not
   select a quieter successful attempt or tune its policy to an observed
   answer.
2. **Reduced decision canary.** Add and physically execute a genuinely reduced
   representation. Compare it with exact replay, stratified and random
   sampling, and ddmin on decision fidelity, false positives and negatives,
   runtime, and artifact size.
3. **Acceptance semantics — implemented for the first gate.** Statistic,
   thresholds, repetition policy, stability limit, bootstrap rule, and four
   outcomes were fixed and manifest-bound before execution. Preserve that
   policy/result separation in every later campaign.
4. **Independent exchange.** Have a party who did not create the source bundle
   verify, materialize, execute, and interpret it using only the repository and
   artifact.
5. **A second workload or topology.** Add an external trace, multi-node run, or
   materially different topology before making a general hardware-fidelity
   claim.
6. **Current ecosystem interop.** Consume a reviewed Chakra protobuf schema and
   preserve its dependency semantics end to end. The pinned legacy
   PARAM-derived encoding remains a narrow research bridge, not claimed
   current PARAM compatibility.

Until a stable decision-fidelity gate passes, CommCanary remains a deterministic
replay and evidence framework, not a hardware-qualification decision tool. A
future fail or kill-condition trigger would reframe it accordingly; the current
inconclusive result does neither. That boundary is part of the contract rather
than something a result may silently relax.

## Deliberate non-goals

- Replacing Chakra as a general workload graph.
- Claiming that a model-free artifact reveals no model structure; exact GEMM
  shapes and timing can be sensitive and are disclosed as such.
- Treating self-reported execution metadata as independent attestation.
- Promising cheaper CI until a representative full workload demonstrates a
  cost advantage.
- Generalizing single-node PCIe A100 measurements to other systems.

## Release boundary

Version 0.3.0 remains unreleased. A release requires the repository's explicit
release gate and a separate tag/publish decision; evidence publication does not
cross that boundary.
