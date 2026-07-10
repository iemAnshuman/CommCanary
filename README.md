# CommCanary

[![CI](https://github.com/iemAnshuman/commcanary/actions/workflows/ci.yml/badge.svg)](https://github.com/iemAnshuman/commcanary/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/commcanary)](https://pypi.org/project/commcanary/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Distill a distributed-LLM communication trace into a small, model-free
regression canary — and prove the distillation didn't change the answer.**

Isolated collective microbenchmarks are known to mislead: `nccl-tests` can
report healthy numbers while the real workload ships a 20% regression
([NVIDIA/nccl#513](https://github.com/NVIDIA/nccl/issues/513)), because they
erase everything contextual — operation order, rank-arrival skew,
compute/communication overlap, queueing, and rare tail windows. Full
reference-workload runs preserve all of that but need model code, data, and a
cluster. CommCanary occupies the space between: a minutes-scale artifact
distilled from *your* workload's trace, carrying no weights and no prompts,
replayable against a candidate config before rollout.

What makes it different:

- **Decision-preserving reduction.** Minimization is gated on a fail-closed
  verifier: a canary is only emitted if it provably preserves the source
  workload's regression verdicts, pairwise configuration rankings, and tail
  behavior. The gate matters — a generic ddmin reducer with a ranking-only
  oracle happily collapses our adversarial 100-event trace to a **single
  event** (`commcanary reduce`, included as a baseline).
- **Auditable lossy compression.** Every approximation carries per-field
  max-error bounds and a SHA-256 commitment to the exact source segment it
  summarizes, so a third party holding the trace can recompute every claim.
- **Tamper-evident artifacts.** Report validation re-runs the scheduler model
  over embedded samples; `verify-report` recomputes bit-identically. Edited
  numbers fail validation.
- **Ecosystem-native.** PyTorch Kineto profiler traces in
  (`import-kineto`), PARAM comms-replay traces out (`export-param`) for
  physical NCCL execution.

```
capture / import-kineto        compile                replay              compare
  workload trace  ────────▶  canary artifact  ────▶  report(s)  ────▶  pass / warn / fail
      (v1)          verified minimization     deterministic sim      CI exit code
                    + sha256 commitments            │
                                              export-param ────▶ physical NCCL replay
```

The bundled replay engine is a **deterministic simulator**, not a physical
NCCL executor — useful for validating trace compression, testing regression
logic, and designing experiments. Physical execution goes through the PARAM
export; claims about real hardware require that path plus cross-system
evaluation. The research contract, including what is deliberately *not*
claimed, lives in [`RESEARCH_SPEC.md`](RESEARCH_SPEC.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

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

The comparison command exits with status 1 when configured regression
thresholds are exceeded.

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
PYTHONPATH=src python examples/research_scaffolding.py
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

## Ecosystem interop: Kineto import and PARAM export

CommCanary can ingest real collective metadata from a PyTorch profiler
(Kineto) trace and can emit a compiled canary as a PARAM comms-replay
"basic" trace, giving the minimized artifact a physical NCCL execution path
via `facebookresearch/param`:

```bash
commcanary import-kineto profiler_trace.json -o imported.trace.json \
  --workload-name llama70b-serve --phase decode
commcanary compile imported.trace.json -o imported.canary.json
commcanary export-param imported.canary.json -o param_comms_trace.json --dtype float32
```

The Kineto import reads `record_param_comms` events (torch >= 2.2): collective
name, dtype, element counts, process-group name and ranks, and single-rank
timestamps rebased to the trace start (the raw monotonic-clock values are
preserved via `workload.kineto_trace_start_us` and
`system.kineto_base_time_ns`). It is an observational single-rank import — it
does not invent cross-rank arrival skew, compute overlap, or measured exposed
latency, and the imported workload notes say so. Truncated rank lists from
non-uniform process groups are only reconstructed from an explicit global
rank start/stride; otherwise the import fails closed rather than fabricate
group membership. Collectives without a CommCanary op mapping (for example
`reduce`, `gather`) are imported as `custom_op` events rather than dropped or
mislabelled.

The PARAM export expands the canary's full event program (motifs, patterns,
and run-length weights included) into one entry per logical occurrence with
element counts, process-group ids, and cumulative `startTime_ns` timestamps,
so `--use-timestamp` replay reproduces inter-op gaps. Sharded collectives use
PARAM's size conventions (`all_gather` gathers `world_size` shards of
`in_msg_size`; `reduce_scatter` scatters into `out_msg_size` shards). Every
point-to-point transfer exports as a matched send/recv entry pair carrying
`src_rank`/`dst_rank`, because PARAM executes each side only on its own rank.
Ops with no PARAM equivalent — including `send`/`recv` events without peer
ranks — fail closed unless `--skip-unsupported` is passed.

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

Replay bandwidth is interpreted as **Gbit/s**.

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Important limitations

Physical NCCL replay evidence now exists through the PARAM export path and
the overlap-aware reference replayer in [`experiments/rostam/`](experiments/rostam/).
That evidence is deliberately narrow: one workload, one `cuda-A100` node,
PCIe, and 4 GPUs. The repository still does not provide multi-node,
NVLink-class, multi-model, or multi-generation-hardware evaluation; Chakra ET
or Nsight ingestion, dependency-aware compute-kernel synthesis, and full
per-window/per-motif Pareto minimisation also remain open. “Model-free”
means the artifact omits weights and application code; it does not by itself
prove privacy or absence of trace leakage.
