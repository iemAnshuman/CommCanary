# Local benchmark evidence: 2026-07-11

This directory preserves the raw machine-readable observations behind the
reviewed compact baseline in
[`../../baselines/local-arm64-macos-cpython310-20260711.json`](../../baselines/local-arm64-macos-cpython310-20260711.json)
and the measurements discussed in [`../../../docs/benchmarks.md`](../../../docs/benchmarks.md).
They are observational engineering evidence, not portable performance claims
or regression thresholds.

The JSON records intentionally retain the measured host's Python executable
path as provenance. The generated 1K, 10K, and 100K fixture payloads are not
duplicated here: `python -m benchmarks fixtures .benchmark-data/fixtures`
recreates them deterministically. `manifest.json` binds their exact SHA-256
identities and semantic sizes.

| File | SHA-256 |
| --- | --- |
| `capture-after-repeats3.json` | `8d45e904c3be8f912ad5a85a6a4f0cc154a26c27155d7d35006ffe3dfe741e20` |
| `capture-after.json` | `c27ba52c79d7db599651ecd04dd9cd9541db3b9420b069f68ce9bee9000fce31` |
| `capture-before-repeats3.json` | `2d7054d62b7afbfdcdf1974b83747ea66c247333dd013ac78bde27101b6a1a9e` |
| `param-after-repeats3-v2.json` | `92eb0f00f2f4f8fda67d695ff8586ee4565ce26b4a0ad8b7413a1abf10f58023` |
| `param-after-repeats3.json` | `3a99dc209effd9951e1c2db20d071d47bb674e0067c229dee8dee98fd818c9f1` |
| `param-after.json` | `c7504e6be1767f7f3c32999fd7ce8711ec9e2680f5db9ea8c149f599b543b723` |
| `param-before-repeats3.json` | `aa87d0749d2d826da805425e708b453e00d8876cfa39ce0faf38fa399243f7c2` |
| `phase8-before.json` | `02162a9afe6b58d1cc7344e7f1f996a1cdaf1aa346e55725cfa8bf5a06229afa` |
| `manifest.json` | `af34b767c9c3b4900e8ff8f5b1f536abf77c7767f8209fdfefc97936a24d2357` |

Verify the preserved bytes from the repository root:

```console
shasum -a 256 benchmarks/evidence/local-arm64-macos-cpython310-20260711/*.json
```
