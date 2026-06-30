# CommCanary

CommCanary distils a distributed-LLM communication trace into a small,
model-free regression canary. It preserves details that isolated collective
microbenchmarks usually erase: operation order, rank-arrival skew,
compute/communication overlap, queueing, rare timing discontinuities, and—when
captured—observed exposed communication latency.

The current replay engine is a **deterministic simulator**, not a physical NCCL
executor. It is useful for validating trace compression, testing regression
logic, and designing experiments. Claims about real hardware still require an
executable GPU replay backend and cross-system evaluation; see
[`RESEARCH_SPEC.md`](RESEARCH_SPEC.md).

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
baseline and a full decode-like workload whose queue-reset gaps and high-overlap
tail windows change configuration ranking. A canary that is too small is
labelled unverified; a lossless compact canary preserves the workload ranking.

```bash
PYTHONPATH=src python examples/research_scaffolding.py
```

The script writes traces, canaries, and behavior-verification outputs under
`out/research_scaffold/`.

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
ordering are checked. Comparison output localises the largest phase- and
operation-level regressions in addition to applying global thresholds.

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

The repository does not yet provide physical CUDA/NCCL replay, automatic
Chakra/Nsight ingestion, dependency-aware compute-kernel synthesis,
ranking-aware canary minimisation, or evidence across real serving engines and
GPU generations. “Model-free” means the artifact omits weights and application
code; it does not by itself prove privacy or absence of trace leakage.
