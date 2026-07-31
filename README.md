# CommCanary

[![CI](https://github.com/iemAnshuman/commcanary/actions/workflows/ci.yml/badge.svg)](https://github.com/iemAnshuman/commcanary/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/commcanary)](https://pypi.org/project/commcanary/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Turn distributed workload profiles into a model-free, source-verifiable
hardware-qualification request—without shipping weights or prompts.**

Isolated collective microbenchmarks are known to mislead: `nccl-tests` can
report healthy numbers while the real workload ships a 20% regression
([NVIDIA/nccl#513](https://github.com/NVIDIA/nccl/issues/513)), because they
erase everything contextual — operation order, rank-arrival skew,
compute/communication overlap, queueing, and rare tail windows. Full
reference-workload runs preserve all of that but need model code, data, and a
cluster. CommCanary occupies the space between: a portable request distilled
from *your* workload's trace, carrying no weights or prompts, whose source
correspondence a receiving lab can recompute before target-specific replay.
Physical fidelity on imported profiles remains the deciding open experiment,
not an implied property of the file format.

![CommCanary comparison report: verdict FAIL, with median/p95/p99 deltas, a metrics
table, the threshold reasons that tripped, and per-phase and per-operation regression
breakdowns](docs/images/comparison-report.png)

*`commcanary compare` on the bundled example trace. Exits 1, names the phase and the
operation, and ships as standalone HTML next to the JSON. Reproduce it with the
[Quick start](#quick-start) below.*

## Why the gate is the whole product

A generic delta-debugging reducer, handed an oracle that only has to preserve
*the decision*, deletes 99 of 100 events from our adversarial trace in six
oracle calls — and every pairwise configuration ranking still holds:

```console
$ python examples/research_scaffolding.py          # writes out/research_scaffold/
$ commcanary reduce out/research_scaffold/adversarial_decode.trace.json \
    -o out/reduced.trace.json
ddmin reduced 100 -> 1 events in 6 oracle calls

$ commcanary compile out/reduced.trace.json -o out/reduced.canary.json
$ commcanary verify-behavior out/research_scaffold/adversarial_decode.trace.json \
    out/reduced.canary.json -o out/reduced.behavior.json
behavior verification: failed
- representation fidelity: lossless_timing
- source verified: failed
- behavioral fidelity: fail
- configuration ranking: pass        # <- the ranking survived. Nothing else did.
```

A ranking is a projection. Five backend configurations give ten pairs, scored
on four latency metrics — forty bits of agreement that one well-placed event
can carry on its own. Minimize against that alone and you get an artifact with
almost nothing in common with the workload it came from.

CommCanary ships that reducer as a baseline and builds everything else around
refusing its answer.

What makes it different:

- **Optional behavior-gated compilation.** Source/timing fidelity is always
  audited; callers can additionally require a canary to preserve declared
  simulator verdicts, pairwise rankings, and tail behavior. The distinction
  matters—a generic ddmin reducer with a ranking-only oracle happily collapses
  our adversarial 100-event trace to a **single event** (`commcanary reduce`,
  included as a baseline).
- **Auditable lossy compression.** Every approximation carries per-field
  max-error bounds and a SHA-256 commitment to the exact source segment it
  summarizes, so a third party holding the trace can recompute every claim.
- **Tamper-evident artifacts.** Report validation re-runs the scheduler model
  over embedded samples; `verify-report` recomputes bit-identically. Edited
  numbers fail validation. The exact digest coverage, assurance ladder, and
  authenticity limits are documented in [`docs/integrity.md`](docs/integrity.md).
- **Bounded untrusted input.** Loading, expansion, replay, behavior search,
  reduction, capture merging, and PARAM export share one immutable resource
  policy. Defaults and stricter service configurations are documented in
  [`docs/resource-limits.md`](docs/resource-limits.md).
- **Explicit ecosystem boundaries.** PyTorch Kineto profiler traces come in
  through `import-kineto`. The PARAM-basic-derived rank-aware encoding requires
  a CommCanary adapter; current upstream PARAM compatibility is not claimed.
- **Portable qualification requests.** `prepare-qualification` turns complete
  rank-local profiles into one source-verified owner-to-lab directory;
  `verify-qualification` independently rehashes every file and recomputes the
  trace-to-canary fidelity proof. The receiving lab can then run
  `materialize-qualification` and `verify-materialization` without fitting
  source durations to target timing. Materialization binds the exact
  source-derived rectangular GEMM recipe for every rank, its canonical
  projection hash, operation/FLOP counts, and the deterministic
  issue-work-wait program bytes. Timing-only mutations cannot change the
  executable work; missing or unsupported recipes refuse.

```
capture / import-kineto        compile                replay              compare
  workload trace  ────────▶  canary artifact  ────▶  report(s)  ────▶  pass / warn / fail
      (v1)          verified minimization     deterministic sim      CI exit code
                    + sha256 commitments

source + canary + fidelity ──▶ qualification request
                                      │ exact per-rank work recipes
                                      ▼
                           verified materialization
                                      │ bounded reference executor
                                      ▼
                          self-reported diagnostic
                                      │ GPU conformance + observation contract pending
                                      ▼
                           verifiable physical observation
```

The physical path is intentionally staged. A portable request contains the
source trace, canary, and fidelity proof. It does not contain a supposedly
portable executable: overlap-preserving GEMM counts depend on calibration for
the receiving device. The lab-side materialization binds its operator-supplied
calibration and exact generated program bytes, but explicitly does not claim
execution, measurement, current upstream PARAM compatibility, or a
qualification verdict.

### What this is not

- **Not yet a validated physical NCCL executor.** The bundled replay engine is
  a deterministic simulator. A separate `execute-materialization` reference
  runner now exercises torch.distributed, but its control flow has only local
  injected-runtime coverage; GPU conformance remains unproven.
- **Not yet a source of defensible hardware numbers.** The reference runner
  emits a request/materialization-bound diagnostic, not a versioned physical
  observation or qualification verdict. Claims about hardware still require
  GPU validation, retained raw measurements, and cross-system evaluation.
- **Not yet validated against a multi-node cluster.** That campaign is
  specified in [`docs/artifact-evaluation.md`](docs/artifact-evaluation.md)
  and has not run.

What the simulator *does* buy you is determinism, which is what makes the
verification story checkable at all: `verify-report` recomputes a report
bit-identically, so an edited number fails validation instead of surviving as
a screenshot. The research contract, including what is deliberately *not*
claimed, lives in [`RESEARCH_SPEC.md`](RESEARCH_SPEC.md).

## Quick start

```bash
pip install commcanary
```

Then, from a clone of this repository (for the bundled example traces):

```bash
commcanary compile examples/traces/llama70b_tp8_trace.json \
  --output out/workload.canary.json
commcanary replay out/workload.canary.json \
  --output out/baseline.report.json --html out/baseline.html --include-samples
commcanary replay out/workload.canary.json \
  --output out/candidate.report.json --html out/candidate.html \
  --latency-floor-us 12 --include-samples
commcanary compare out/baseline.report.json out/candidate.report.json \
  --output out/comparison.json --html out/comparison.html
```

What that prints:

```console
compiled 10 trace events into 5 canary events; event ratio=2.0x, byte ratio=0.474x, timing=lossless_timing
replayed 10 events: median=91.977 us p95=107.746 us p99=108.321 us hidden=16.27%
replayed 10 events: median=121.409 us p95=146.616 us p99=152.095 us hidden=13.25%
comparison verdict: fail
- p99 regression 40.4% exceeds 15.0%
- p95 regression 36.1% exceeds 10.0%
- median regression 32.0% exceeds 8.0%
- phase 'decode' p99 regression 41.5% exceeds 15.0%
- phase 'prefill' p99 regression 16.8% exceeds 15.0%
- operation 'all_reduce' p99 regression 40.4% exceeds 15.0%
```

The comparison command exits with status 1 when configured regression
thresholds are exceeded, which is the whole point of putting it in CI.

Note the two compression numbers in the first line. Event ratio and byte ratio
are reported separately because a canary with fewer events than its source can
still serialize to more bytes — as it does here on a ten-event toy trace.
Calling that "compression" would be a lie with units.

## Prepare a hardware-qualification request

For a model owner handing a workload-shaped artifact to a hardware lab:

```bash
commcanary prepare-qualification rank0.json rank1.json \
  --assume-shared-clock \
  --workload-name private-serving-decode \
  --output-directory out/qualification-request

# Run by the receiving party before trusting or materializing anything.
commcanary verify-qualification out/qualification-request

commcanary materialize-qualification out/qualification-request \
  --output-directory out/qualification-materialization

# Independently rederive and byte-compare the generated program.
commcanary verify-materialization out/qualification-request \
  out/qualification-materialization

# Reference execution requires target-compatible PyTorch installed separately.
# This diagnostic is not yet a qualification observation or verdict.
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module commcanary execute-materialization \
  out/qualification-request out/qualification-materialization \
  --device cuda --backend nccl \
  --distributed-timeout-seconds 300 \
  --output out/reference-execution.json
```

Preparation imports all profiles, requires known overlap, compiles with
lossless timing by default, recomputes source fidelity, and requires every
collective to carry a complete per-rank contiguous-GEMM recipe derived from the
same-thread region between collective issue and one explicit wait.
`all_reduce` and `reduce_scatter` must carry their source-bound reduction
operator; exact Kineto input/output element counts and split evidence must
match the encoded operation; and every broadcast must carry its source-bound
root rank. An unknown or incomplete compute recipe, reduction operator,
message shape, or broadcast root is a refusal, not permission to fit elapsed
time, assume SUM, reconstruct a convenient tensor shape, or choose the first
process-group rank.
The fixed inventory is:

- `qualification-request.json`;
- `source.trace.json`;
- `canary.json`; and
- `fidelity.json`.

The manifest binds the exact bytes of the other three files and the canary's
source, execution, calibration, and artifact-provenance commitments. Its claim
boundary is equally explicit: source correspondence is verified, but no
physical measurement is included, physical fidelity remains unproven, and no
qualification verdict is issued. The target contract also binds the
source-derived communication dtype and reduction-operator sets, the canonical
hash and count of exact per-rank compute work, source-validated per-event
message shapes, an equal-split-only `all_to_all` policy, issue/work/wait
structure, and disabled timestamp pacing. GEMM shapes and dtypes are explicitly
disclosed because they can reveal model structure even though weights and
prompts are absent.

Materialization writes exactly `replay-program.json` and
`materialization.json`, with the manifest installed last into another new,
non-overwriting directory. It takes no target timing calibration: each source
event becomes asynchronous collective issue, exact rank-local
`m×k @ k×n` work, and an immediate explicit wait. Start-to-start gaps, Python
overhead, exposed communication, and idle time never become synthetic compute.
The manifest binds the canonical work projection, per-rank operation counts,
source kernel observations, mathematical FLOP count, exact program bytes, and
the issue/work/wait semantics. It also records
`upstream_param_compatibility: not_claimed` and
`execution_adapter: conforming-adapter-required`; deterministic generation is
not evidence that a GPU run happened. Embedded digests identify content; they
do not authenticate who produced it.

`execute-materialization` is the candidate in-package adapter, not yet a claim
that the adapter conforms physically. Every rank revalidates the complete
request and materialization before importing PyTorch or initializing a process
group. Preflight checks request/wait lifetimes, rank membership, collective
shapes, rank-local rectangular GEMM dtype and dimensions, operation counts,
reduction operators, broadcast-root membership, and
per-rank/aggregate tensor allocation. Communication tensors are isolated by request identity, so two
overlapping same-shaped operations cannot race on a reused buffer or evade the
allocation budget. A positive, resource-bounded distributed timeout is applied to
default-group initialization and every encoded subgroup, replacing PyTorch's
backend-dependent long defaults and binding the chosen value into the
diagnostic. It also requires
the encoded process groups to cover exactly the launched dense rank domain, so
an operator cannot add unrepresented idle ranks and change contention. The
exact-work qualification generator supports `all_reduce`, `all_gather`,
`reduce_scatter`, `all_to_all`, and `broadcast`; other schedules refuse until
their causal work can be represented honestly. The executor runs each rank's
bound rectangular recipe, gathers rank-local issue-to-wait timings, and
also records whole-program wall time for every rank and measured iteration.
The decision-facing makespan is the maximum participating-rank duration, not a
median of individual collective calls. Output is written only on rank 0. Both
GEMM inputs and its reused output are preallocated
and included in the rank memory plan; the repeated loop performs no hidden
GEMM-output allocation. Before warmup or measurement, one separately bounded,
untimed pass uses rank-dependent zero/one patterns to check every collective
result under its bound reduction operator and every receive endpoint. The
executor passes the exact operator to PyTorch rather than relying on its SUM
default. All ranks exchange their exact check counts, and any mismatch fails
the run collectively instead of producing a timing-only success.
Its diagnostic remains deliberately outside the supported format table until
the GPU-backed conformance gate establishes which physical observation
semantics CommCanary can defend.

## Fidelity-first compilation

Exact run-length and periodic encodings are used whenever possible. Irregular
streams are represented by ordered bounded intervals that contain explicit
error bounds. Compilation can fail closed when approximation exceeds a chosen
budget:

```bash
commcanary compile trace.json -o canary.json \
  --timing-sample-limit 128 \
  --max-skew-error-us 2 \
  --max-overlap-error-us 3 \
  --max-prefix-gap-error-us 10
```

Require a completely lossless timing representation with:

```bash
commcanary compile trace.json -o canary.json --lossless-timing
```

Compiled canaries can also be behavior-gated. This is intentionally stricter
than field-level fidelity: compilation fails unless the generated canary passes
source verification, behavioral checks, and pairwise configuration-ranking
verification under the verifier's backend set.

```bash
commcanary compile trace.json -o canary.json --require-behavior-verification
```

For research minimization, use behavior-search mode. It compiles every timing
sample limit in the requested range, runs behavior verification for each
candidate, rejects failures, selects the smallest serialized passing artifact,
and then greedily lowers timing budgets for individual signature groups only
when the canary remains source-, behavior-, and ranking-verified:

```bash
commcanary compile trace.json -o canary.json \
  --behavior-search \
  --behavior-search-min-sample-limit 2 \
  --timing-sample-limit 128
```

The selected canary records every uniform-budget candidate, the per-group
refinement attempts, the accepted lower group budgets, and the selected timing
limit mode. It is still not a full per-window/Pareto optimizer, but it gives a
fail-closed behavioral minimization path for the current compiler and avoids
forcing quiet groups to carry the same sample budget as ranking-sensitive
windows.

The compiler reports both event compression and serialized-byte compression.
A smaller event count is not described as compression when the artifact is
actually larger.

## Sequence motifs and scheduler identity

CommCanary has a replay-equivalent `sequence_motif` representation for exact
repeated multi-event programs such as `A-B-A-B`, `A-B-C` loops, or
transformer-layer-like communication blocks. A motif is an artifact-level
wrapper around child event templates plus a repeat count; replay, validation,
source verification, and scheduler hashes expand it to the same ordered
simulator inputs as the flat encoding. Source/provenance fields may differ, but
flat and motif encodings that execute the same scheduler inputs share the same
`scheduler_execution_sha256`. Use `--disable-sequence-motifs` to emit only flat
events.

## Observed tail signal and calibration

A trace event may contain an optional measured value:

```json
{
  "observed_exposed_us": 73.2
}
```

This field must be present on every selected event or none. It is preserved as
part of each joint timing record, receives priority during bounded selection,
and produces a calibration section in replay reports: absolute error, bias,
and percentage error. Without this signal, tail selection is a structural
proxy based on skew, gaps, overlap, and change points; it is not claimed to
preserve measured p99 latency.

## Behavioral verification

`verify-fidelity` answers whether a canary's representation-level claims can be
recomputed from the source trace. `verify-behavior` answers a different
question: whether the compressed artifact preserves simulator-visible workload
behavior. It replays a lossless normalized source canary and the candidate
canary across multiple backend configurations, then reports four separate
statuses:

- `representation_fidelity_status`: the compiler-attested timing mode, such as
  `lossless_timing` or `bounded_approximate`;
- `source_verified_status`: whether source-to-canary commitments recompute;
- `source_coverage_status`: whether the candidate covers the full normalized
  source trace or only a prefix/subset;
- `behavioral_fidelity_status`: whether p50/p95/p99/max/mean, queue-wait
  distributions, hidden communication, phase metrics, operation metrics, and
  tail-event recall are within tolerance;
- `configuration_ranking_status`: whether pairwise backend rankings are
  preserved across latency metrics.

```bash
commcanary verify-behavior trace.json canary.json -o behavior.json \
  --relative-tolerance-pct 10 \
  --absolute-tolerance-us 1 \
  --hidden-tolerance-points 5 \
  --tail-recall-threshold 0.8 \
  --ranking-tie-tolerance-us 0.001
```

`compile --require-behavior-verification` uses this verifier as a fail-closed
compiler gate. This is meant for research claims, not for fastest iteration.
`verify-behavior` compares against the full normalized source trace by default.
Canaries generated from a prefix or subset of the trace are labelled
`partial_source_verified` and cannot receive a strong behavioral claim.

A canary with rank-local compute uncertainty can still be replayed, but strong
behavioral claims are downgraded to `behaviorally_unverified` rather than
`behaviorally_verified`.

## Replay ablations

Replay supports research ablations that deliberately remove one preservation
mechanism from the deterministic model:

```bash
commcanary replay canary.json -o out/ablation.report.json \
  --ablate arrival_skew \
  --ablate compute_overlap \
  --ablate rare_tail_windows
```

Supported ablations are `arrival_skew`, `compute_overlap`,
`operation_ordering`, `rare_tail_windows`, `queue_reset_gaps`, `pressure`, and
`observed_exposed_us`. Ablations are recorded in the replay protocol and are
therefore covered by `verify-report`. They are not a physical intervention;
they are simulator controls for paper ablations.

## Point-to-point messages

Point-to-point traffic is represented as `point_to_point` rather than as a fake
collective. Merged send/recv observations preserve `sender_rank`,
`receiver_rank`, `tag`, `channel`, `message_sequence`, and rank-local send/recv
observation metadata. Scheduler identity and resource labelling include these
fields so reversing sender/receiver or changing a channel is not treated as the
same execution.

## Ranking-inversion scaffold

The repository includes a synthetic adversarial experiment that demonstrates why
field-level compression is not enough. It constructs an isolated collective
baseline, random-sampling, frequency-representative, and clustering controls,
and a full decode-like workload whose queue-reset gaps and high-overlap tail
windows change configuration ranking. A canary that is too small is labelled
unverified; behavior-search finds the smallest verified timing budget in the
declared range, and a lossless compact canary preserves the workload ranking.

```bash
python examples/research_scaffolding.py
```

The script writes traces, canaries, and behavior-verification outputs under
`out/research_scaffold/`.

## Research baselines

Baseline traces are generated explicitly so they can be compiled, replayed, and
verified under the same simulator contract as CommCanary artifacts:

```bash
commcanary baseline trace.json -o out/isolated.trace.json --method isolated
commcanary baseline trace.json -o out/random.trace.json --method random --sample-count 16 --seed 7
commcanary baseline trace.json -o out/frequency.trace.json --method frequency
commcanary baseline trace.json -o out/cluster.trace.json --method cluster --cluster-count 8
commcanary baseline trace.json -o out/stratified.trace.json --method stratified --strata-per-group 4 --seed 7
```

`isolated` removes workload order, skew, queue-reset gaps, and overlap, matching
the spirit of an isolated collective microbenchmark. `random` samples source
events and tiles them to the original event count by default for count-fair
behavioral comparison. `frequency` preserves operation frequency and order but
replaces each signature by one representative, removing within-signature tails.
`cluster` is a stronger negative control: it preserves event count, operation
order, operation signatures, and several deterministic timing medoids per
signature, while still discarding exact burst/tail correlations and source
commitments. `stratified` is the kill-condition control named in
`RESEARCH_SPEC.md`: events are grouped by operation signature, each group is
cut into deterministic timing strata, and one seeded random member is drawn
per stratum; every event is replaced by its stratum's sample. These baselines
are intentionally not source-verified against the original trace;
`verify-behavior` should label them unverified unless they actually pass the
full source, behavioral, and ranking gates.

## Decision-preserving reduction baseline

`commcanary reduce` is a ddmin-style generic reducer for comparing against
behavior-search compilation. Its oracle preserves only the decision: a
candidate event subset is accepted when compiling and replaying it across the
configuration set reproduces the full trace's pairwise latency-metric
rankings. It deliberately does not enforce behavioral fidelity, so it shows
what decision-only reduction gives up: on the synthetic ranking-inversion
scaffold it happily collapses 100 events to a single event while keeping the
ranking, which is precisely why the fail-closed behavioral verifier gates on
tail recall, queue waits, hidden communication, and distribution agreement in
addition to rankings.

```bash
commcanary reduce trace.json -o out/reduced.trace.json \
  --ranking-tie-tolerance-us 0.001 \
  --max-oracle-calls 256
```

The reduced trace records the oracle-call ledger under
`workload.reduction` and is labelled not source-verified.

## Ecosystem interop: Kineto import and legacy PARAM-derived export

Chakra already provides the ecosystem's general workload graph: its official
schema represents compute, memory, communication, dependencies, timing, and
resource constraints. CommCanary should not replace that interchange layer.
Its intended boundary is the smaller one Chakra does not supply: reduce a
shared workload description to a decision-preserving acceptance case, bind it
to exact source and calibration evidence, and let the receiving party verify
the result. A future Chakra adapter must consume the official length-delimited
protobuf execution trace and preserve its dependency/attribute semantics; a
JSON object with similar names is not compatibility. Until that adapter is
implemented and tested, Kineto remains the only supported real-profile input.

CommCanary can ingest real collective metadata from a PyTorch profiler
(Kineto) trace and can emit the historical PARAM comms-replay “basic” JSON
encoding used by the pinned Rostam research harness:

```bash
commcanary import-kineto profiler_trace.json -o imported.trace.json \
  --workload-name llama70b-serve --phase decode
# Multiple rank profiles retain cross-rank arrivals and rank-local overlap.
# This explicit assertion is appropriate only when the profiles share a clock.
commcanary import-kineto rank0.json rank1.json -o imported.trace.json \
  --assume-shared-clock
# The next step succeeds only when every collective linked to complete CUDA
# kernel evidence and multi-rank arrival calibration; otherwise it refuses.
commcanary compile imported.trace.json -o imported.canary.json
commcanary export-param imported.canary.json -o param_comms_trace.json
```

This is not current upstream PARAM interoperability. Upstream removed `basic`
and Kineto trace parsing in
[PARAM PR #155](https://github.com/facebookresearch/param/pull/155); its
current communication replayer accepts Chakra host execution traces. The
legacy commit pinned by `experiments/rostam/patches/param-patch-contract.json`
accepts basic JSON but hardwires blocking replay, so CommCanary's overlap
campaign uses a separate explicit-wait reference replayer. For that reason,
qualification manifests call the output
`commcanary.source-bound-compute-recipe.v2`, require a conforming adapter, and
make no upstream PARAM compatibility claim. The encoding does not convert
inter-communication gaps into work. It binds the exact per-rank contiguous
GEMMs observed between asynchronous issue and the corresponding explicit wait,
then reproduces issue, work, and wait in that order. Pipelined or unsupported
compute schedules need a dependency graph and refuse rather than being
serialized into a different program. This is deliberately narrower than
Chakra's graph semantics.
[Chakra import/current-replay interop](https://github.com/mlcommons/chakra) is
an open product requirement, not something inferred from similar field names.

The Kineto import reads `record_param_comms` events (torch >= 2.2): collective
name, dtype, element counts, process-group name and ranks, and single-rank
timestamps rebased to the trace start. Raw monotonic and wall-clock origins are
used transiently for explicit multi-rank alignment but are not retained in the
shareable trace. One profile remains an observational single-rank import: it
does not invent cross-rank arrival skew or measured exposed latency.
With multiple rank profiles, records are matched by invocation ordinal inside
the exact process-group rank domain. Every participating rank must contribute
one compatible record. `--assume-shared-clock` is an explicit zero-offset
claim; profiles from separate clocks instead require one additive
`--clock-offset-us RANK=OFFSET` value per imported rank. With neither claim the
merged trace remains inspectable, marks arrival skew unknown, and cannot be
compiled.

`record_param_comms` omits the broadcast root. For a broadcast, the importer
recovers `root_rank` only from an interval-containing `c10d::broadcast_` CPU
event on the same thread, using the concrete dispatcher input corresponding to
`BroadcastOptions.rootRank`. The root must be a member of the recorded process
group and every rank-local contribution must agree. Missing, malformed,
out-of-group, or conflicting evidence remains unknown or fails; owner-side
qualification and legacy PARAM-derived export then refuse instead of guessing.
Both broadcasts in the public issue #131462 pair resolve to rank 0 and retain
that value through trace, canary, materialization, and reference-executor
preflight.

`record_param_comms` also does not carry a canonical reduction operator. For
`all_reduce` and `reduce_scatter`, the importer derives `reduction_op` only when
every uniquely linked NCCL communication kernel has one recognized, consistent
operator token (`sum`, `product`, `min`, `max`, or `avg`). Generic,
unrecognized, ambiguously linked, or incomplete kernel evidence leaves the
operator unknown. A general trace remains inspectable without this optional
field, but owner-side qualification refuses to silently execute an unknown
operator as SUM. Multi-rank merging preserves the operator only when every
participant derives the same value. The public issue #131462 pair derives SUM
for all five reduction events and preserves it through materialization.

Kineto's input/output element counts and split lists are also retained as
normalized source evidence. Multi-rank merging requires exact agreement before
coalescing. Qualification independently checks the counts against the operation
and process-group size, checks the stored byte count against dtype width, and
refuses any skipped zero/missing-size event or explicit split vector. The
current reference executor supports only source-verified equal-split
`all_to_all`; observational import can retain an unsupported shape without
turning it into a different executable. All seven events in the public issue
#131462 pair have exact materializable shapes.

Kineto dtype names are normalized to an explicit canonical dtype on every
trace event. Dtype remains part of compilation grouping, source fidelity, and
execution commitments, then legacy PARAM-derived export uses it per event to
derive element counts. `export-param --dtype ...` is an explicit whole-trace override for
legacy or deliberately transformed inputs; omission no longer silently turns
typed Kineto events into float32.

The CLI reads and hashes each bounded profile in one pass. The imported trace
records only its exact byte SHA-256, byte size, and distributed rank (when
present) under `system.kineto_source_profiles`; it never records the source
path or filename. Multi-rank identities are sorted by rank. Retain the original
profiles separately if another party must verify those commitments: the trace
binds the bytes but does not embed them, and a hash is identification rather
than authenticity.

For a
collective whose external correlation id links to complete NCCL kernel
activities, the importer measures `compute_overlap_us` as the union of time
where non-communication kernels run concurrently on another stream of the same
device. Same-stream work, another device's kernels, and other communication
kernels do not count; overlapping compute intervals are unioned rather than
double-counted. Missing, malformed, or non-unique linkage leaves that event at
`compute_overlap_unknown: true`, and `compile` fails closed unless every event
is known. The output records derived/unknown counts plus a reason on every
event. This derivation is unit-tested against Kineto-shaped events and has been
exercised by the single-configuration same-node diagnostic described in
[`ROADMAP.md`](ROADMAP.md); that run does not establish cross-configuration
decision fidelity. As a
format reality check, both ranks in the public 2-GPU ResNet-50 profile attached
to [PyTorch issue #131462](https://github.com/pytorch/pytorch/issues/131462)
merge from 14 rank-local records into 7 logical events with 7/7 known overlap
and compile losslessly under the explicit shared-clock assumption. The merge
retains different per-rank overlap values and exposes cross-rank arrival
offsets from 0.28–5.26 ms. That proves the adapter preserves real profiler
evidence; it does not prove that replay ranks hardware correctly.

Truncated rank lists from non-uniform process groups are only reconstructed
from an explicit global rank start/stride; otherwise the import fails closed
rather than fabricate group membership. Collectives without a CommCanary op
mapping (for example `reduce`, `gather`) are imported as `custom_op` events
rather than dropped or mislabelled.

The legacy PARAM-derived export expands the canary's full event program (motifs, patterns,
and run-length weights included) into one entry per logical occurrence with
element counts, process-group ids, and cumulative `startTime_ns` timestamps,
so `--use-timestamp` replay reproduces inter-op gaps. Sharded collectives use
PARAM's size conventions (`all_gather` gathers `world_size` shards of
`in_msg_size`; `reduce_scatter` scatters into `out_msg_size` shards;
`all_to_all` requires equal input/output shards divisible by `world_size`).
Every point-to-point transfer exports as a matched send/recv entry pair carrying
`src_rank`/`dst_rank`, because PARAM executes each side only on its own rank.
Ops with no PARAM equivalent — including `send`/`recv` events without peer
ranks — fail closed unless `--skip-unsupported` is passed.
Qualification preparation never permits that escape hatch: missing dtype,
unsupported dtype/operation, incompatible process-group membership, or
byte counts not exactly divisible by dtype width or sharded-collective sizing
fails before a bundle directory is created.

## Trace timing semantics

A trace must use one unambiguous ordering mode:

1. all events have `start_us`; events are chronologically sorted and gaps are
   derived from timestamps;
2. no events have `start_us`; input order is retained and `gap_us`, or
   `compute_before_us` as a fallback, defines readiness;
3. mixed timestamp availability is accepted only when **every** event supplies
   an explicit `gap_us`, making input order authoritative.

Conflicting `start_us` and `gap_us` values are rejected rather than guessed.
Sub-microsecond gaps are stored to nanosecond decimal precision, and pattern
records preserve their exact total duration.

## Capture API

```python
from commcanary.capture import record_collective

record_collective(
    op="all_reduce",
    bytes=128 * 1024,
    ranks=list(range(8)),
    dtype="bfloat16",
    phase="decode",
    collective_id="decode-token-42-tp-allreduce",
    rank_arrival_us={str(rank): rank * 2.5 for rank in range(8)},
    compute_overlap_us=18.0,
    observed_exposed_us=67.4,
)
```

For distributed capture, each logical occurrence needs a globally stable,
unique `collective_id`. Per-process shards include rank, PID, and recorder UUID,
so independent recorders cannot overwrite one another. Merging is fail-closed:
it rejects mixed sessions, duplicate or missing rank contributions, conflicting
collective metadata, incompatible clock calibration, and partially measured
observed latency.

Cross-process arrival timestamps are combined only when an explicit clock
offset/calibration is supplied. Otherwise the merged trace marks cross-rank
skew unknown, and compilation refuses to turn that uncertainty into zero skew.

```bash
commcanary capture --output trace.json --workload-name llama70b -- \
  python examples/instrumented_decode.py
```

Direct, non-sharded output files are single-owner across processes. For a child
that may fail after writing useful partial shards, preserve a bounded checksum
bundle without recording its environment or raw command line:

```bash
commcanary capture --output trace.json --preserve-on-failure failed-capture -- \
  python examples/instrumented_decode.py
```

## Reports and comparison

Reports contain:

- median, p95, p99, maximum, and mean exposed latency;
- arrival-skew, queue-wait, and average-rank-wait statistics;
- communication hidden by modeled overlap;
- phase and operation breakdowns;
- source-normalized, scheduler-execution, calibration-evaluation, artifact, and
  replay-protocol fingerprints;
- compiler fidelity metadata, source commitments for approximate intervals, and
  sequence-motif metadata;
- model calibration when observed latency is available.

Report validation reconciles metrics and breakdowns with included samples. Even
without samples, breakdown counts, weighted means, maxima, names, and quantile
ordering are checked. `verify-report` goes further: it replays the declared
canary with the declared backend and protocol, then compares canary identity,
replay protocol, backend settings, workload, canary-summary metadata, metrics,
breakdowns, calibration, and samples when present. Comparison output localises
the largest phase- and operation-level regressions in addition to applying
global thresholds.

## Formats

- `commcanary.trace.v1`
- `commcanary.canary.v2`
- `commcanary.report.v2`
- `commcanary.compare.v2`
- `commcanary.fidelity_verification.v1`
- `commcanary.behavior_verification.v1`
- `commcanary.report_verification.v1`
- `commcanary.qualification_request.v1`
- `commcanary.qualification_materialization.v1`

Exact read/write/validation/migration support and the published JSON Schemas are
listed in [`docs/formats/compatibility.md`](docs/formats/compatibility.md).

Replay bandwidth is interpreted as **Gbit/s**.

## Development and verification

```bash
python -m pip install -e ".[dev]"
python -m tools.verify --fast
```

`python -m tools.verify` additionally builds wheel and sdist artifacts, installs
the exact wheel outside the checkout with source-path overrides removed, and
runs installed-package tests. `python -m tools.verify --reproducible` checks two
fixed-epoch builds byte-for-byte without pretending an unreleased changelog is
final; `--release` additionally checks release identity. CI invokes the same
gate; `PYTHONPATH=src` is not part of the supported verification path.

Contributor workflow, platform guarantees, security reporting, and artifact
redaction guidance are documented in [CONTRIBUTING.md](CONTRIBUTING.md),
[docs/platform-support.md](docs/platform-support.md),
[SECURITY.md](SECURITY.md), and [docs/privacy.md](docs/privacy.md). The stable
[Python API](docs/api.md) and [CLI contract](docs/cli.md) are documented
separately; maintainers use the [release runbook](docs/release.md). Durable
engineering choices live in the [ADR index](docs/adr/README.md), and the
[artifact-evaluation guide](docs/artifact-evaluation.md) names the exact
handoff between local verification and authorized cluster execution.
The enforced package DAG, component ownership, data flow, and extension points
are in [docs/architecture.md](docs/architecture.md).

## Important limitations

The paper and Rostam design notes report a narrow historical physical campaign
on one 4×A100 PCIe node. The legacy raw attempt archive was not committed, so
the repository-local engineering gate cannot independently regenerate or
revalidate those numbers. A new publication-grade claim requires the
manifest-bound, hash-verified campaign described in the artifact-evaluation
guide. Multi-node, NVLink-class, multi-model, and multi-generation-hardware
evaluation; Chakra ET/current PARAM or Nsight ingestion; dependency-aware compute-kernel
synthesis; and full per-window/per-motif Pareto minimisation remain open.
“Model-free” means the artifact omits weights and application code; it does not
by itself prove privacy or absence of trace leakage.
