# Local scale benchmarks

CommCanary's benchmark fixtures are generated on demand. The standard fixture
manifest contains traces with exactly 1,000, 10,000, and 100,000 stored events,
plus compressed canaries whose short stored programs expand to 10,000 and
100,000 logical events. Generation is deterministic: manifests contain raw
SHA-256 hashes, semantic counts, no timestamps, and compressed fixtures use a
fixed descriptive `created_at` value.

Generate the reviewed scale inputs without running a benchmark:

```console
python -m benchmarks fixtures .benchmark-data/fixtures
```

Run the full local operation matrix and keep the machine-readable result:

```console
python -m benchmarks run .benchmark-data/fixtures/manifest.json \
  --output .benchmark-data/results.json
```

Each operation runs in a fresh subprocess by default. This makes the portable
`resource.getrusage` high-water RSS reading meaningful and prevents an earlier
case from setting the process peak for later cases. The runner also records wall
time, Python allocation peak, platform/Python/package environment, stored and
logical sizes, the fixture SHA-256, and a semantic output hash. Operations with
prepared multi-input state also record a prepared-input semantic hash. Semantic
hashes exclude descriptive timestamps and host labels, so correctness can be
compared across machines even though performance measurements cannot.

Preparation is deliberately outside `wall_time_seconds`: compare prepares two
reports, capture merge prepares two valid rank-local shards, and behavior search
prepares a compressible two-record motif trace. Preparation remains resident,
and `peak_rss_baseline_bytes` is sampled after it, so the memory envelope still
accounts for that state. `python_peak_allocated_bytes` covers only the registered
operation. PARAM export consumes the manifest canary directly.

The default registry now measures all Phase 8 families:

- trace: load, validate, hash, compile, capture merge, and behavior search;
- canary: load, validate, hash, replay, independent report verification,
  compare, and PARAM export.

Use repeated `--operation` flags to select a bounded campaign. For example:

```console
python -m benchmarks run .benchmark-data/fixtures/manifest.json \
  --output .benchmark-data/new-families.json \
  --operation compare \
  --operation capture_merge \
  --operation param_export \
  --operation behavior_search
```

For a quick local or future PR-CI check:

```console
python -m benchmarks smoke --output .benchmark-data/smoke.json
```

The smoke profile uses 64-event fixtures and covers every operation family. It
also times three adversarial checks: capture shard count, PARAM entry expansion,
and behavior candidate count must reject at preflight under deliberately tiny
resource policies. These return stable rejection records through the same
operation registry and measurement path. The smoke is a functional benchmark,
not a regression threshold.

The reviewed local observation is
[`benchmarks/baselines/local-arm64-macos-cpython310-20260711.json`](../benchmarks/baselines/local-arm64-macos-cpython310-20260711.json).
It stores only compact results and environment metadata, not the generated 25
MB fixture set. It explicitly contains no regression thresholds. A single local
run is evidence for investigation, not a portable performance promise.

That campaign completed capture merge at 1K, 10K, and 100K stored events and
PARAM export and compare at 1K, 10K, and 100K logical events. Capture merge took
approximately 0.177 s, 1.808 s, and 18.260 s, while peak RSS reached about 675 MB
at 100K. PARAM export took approximately 0.254 s, 2.704 s, and 27.916 s and
reached about 451 MB peak RSS at 100K. Timed compare stayed near 1.4-1.8 ms because
report preparation is intentionally excluded; its post-preparation RSS baseline
grew with replay size.

## Measured adapter optimization review

The same baseline file now also contains an immediate before/after review of
capture merge and PARAM export. Each value below is the median of three fresh,
isolated processes on the recorded CPython 3.10/macOS arm64 environment. The
fixture-manifest SHA-256 is
`af34b767c9c3b4900e8ff8f5b1f536abf77c7767f8209fdfefc97936a24d2357`.
Before and after suites produced identical semantic-set hashes.

| Operation | Scale | Wall before | Wall after | Wall change | RSS change | Python peak change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Capture merge | 1K stored | 0.0745 s | 0.0714 s | -4.1% | -1.8% | -0.0% |
| Capture merge | 10K stored | 0.8066 s | 0.7515 s | -6.8% | -14.5% | -0.1% |
| Capture merge | 100K stored | 8.3064 s | 7.8794 s | -5.1% | -5.9% | -13.7% |
| PARAM export | 10K logical | 1.2479 s | 1.2166 s | -2.5% | +5.8% | +0.0% |
| PARAM export | 100K logical | 13.0461 s | 12.4790 s | -4.3% | -0.9% | +0.0% |

Capture merge previously allocated a list for every identity even when it had
only one contribution, then copied every singleton event a second time. It now
stores singleton buckets directly, promotes only real collisions to lists,
finalizes the merge-owned copy in place, and releases the bucket index before
ordering and final validation. The 100K semantic output remains
`2ffd5032396e1c6b0fad80e4c16108dc119a3e7c66b308a21d35b8a6125b884c`.

PARAM's compressed motif expansion repeatedly normalized the same stored rank
and timing templates. Production expansion now caches those immutable derived
tuples while injected iterators retain uncached dynamic behavior; every output
entry still owns independent mutable lists. Profiling at 10K attributed about
90% of end-to-end cumulative time to mandatory `validate_canary`, which remains
unchanged and explains why end-to-end gains remain modest; the 10K RSS movement
is within high-water-mark noise. The standard full manifest contains compressed
10K and 100K canaries; its separate smoke evidence continues to cover 1K-scale
PARAM export.

These are observational engineering measurements, not regression thresholds.
In particular, RSS is a process high-water mark and the small PARAM changes are
close enough to noise that they must not be generalized to another runner.

Behavior search took approximately 19.776 s at 1K on this runner. The 10K and
100K cases are recorded as skipped rather than allowed to monopolize the local
campaign or used to invent a threshold. This is a profiling target: candidate
compile/replay/verification work should be optimized only with semantic-hash and
verification equivalence evidence.

Peak RSS is recorded on macOS and platforms where `ru_maxrss` units are known
(Linux and FreeBSD). It is `null` with an explicit method label on unknown
platforms. Verification prepares its replay report before the timed region;
the report remains resident and therefore correctly contributes to the
operation's memory envelope.

No command in this benchmark package connects to Rostam, SLURM, or any other
cluster. Large 100K generation and the full matrix are intentionally excluded
from fast unit tests; the checked-in baseline records the explicit local run.

The pinned weekly `benchmark.yml` workflow runs the 14-operation smoke and
three isolated repetitions of the scalable 1K/10K/100K families, then retains
the manifest and machine-readable observations for 90 days. It deliberately
does not fail a build on wall time or RSS yet. A threshold becomes enforceable
only after repeated observations on a stable named runner class are reviewed;
the threshold, statistic, warmup policy, and allowed variance must then be
committed here and in the workflow together.
