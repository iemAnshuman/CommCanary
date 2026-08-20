# Local benchmark evidence: 2026-08-04

This directory contains a fresh benchmark smoke observation generated after
local executable-path redaction was added. It is local engineering evidence,
not a portable performance claim or regression threshold.

The `executable` field records only `python`; it contains no user or absolute
path. Of 24 requested cells, 15 completed without failure and 9 unsupported
cells were recorded as explicit skips.

| File | SHA-256 |
| --- | --- |
| `smoke.json` | `c3906c85e3b8ae54c52923de544b6190f6b9f8d375126f099e6b430b05d688f7` |

Regenerate from the repository root:

```console
python -m benchmarks smoke --output benchmarks/evidence/local-arm64-macos-cpython314-20260804/smoke.json
```

Benchmark timing bytes vary across runs. Treat a regenerated observation as a
new content-addressed artifact instead of overwriting this file.
