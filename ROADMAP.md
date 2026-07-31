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

See [`docs/artifact-evaluation.md`](docs/artifact-evaluation.md) for the exact
campaign identities, hashes, and reproduction contract.

## Product gates

1. **Multi-configuration decision fidelity.** Compare the exact-work artifact,
   source workload, and isolated collective baseline over one predeclared
   configuration matrix. The artifact must preserve the decisions that an
   acceptance test would make, not merely approximate one latency.
2. **Acceptance semantics.** Define the statistic, tolerance, repetition
   policy, and inconclusive region before observing the gate campaign. Do not
   tune a threshold to the existing +7.45% result.
3. **Independent exchange.** Have a party who did not create the source bundle
   verify, materialize, execute, and interpret it using only the repository and
   artifact.
4. **A second workload or topology.** Add an external trace, multi-node run, or
   materially different topology before making a general hardware-fidelity
   claim.
5. **Current ecosystem interop.** Consume a reviewed Chakra protobuf schema and
   preserve its dependency semantics end to end. The pinned legacy
   PARAM-derived encoding remains a narrow research bridge, not claimed
   current PARAM compatibility.

If decision fidelity fails, CommCanary remains useful as a deterministic
replay and evidence framework, but it is not a hardware-qualification decision
tool. That boundary is part of the contract rather than something a result may
silently relax.

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
