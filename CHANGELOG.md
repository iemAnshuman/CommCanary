# Changelog

## Unreleased

- Added `--overlap-structure` to `export-param`: collectives are emitted for asynchronous issue with explicit `wait` entries placed after the next gap's gemm entries, reconstructing compute/communication concurrency; issue entries carry an `issue` marker so parsers separate issue lines from completion-bearing wait lines.
- Added compute-fill mode to `export-param` (`--compute-fill-us-per-gemm`, `--compute-fill-gemm-dim`): inter-collective gaps export as PARAM `{"compute": "gemm"}` entries instead of idle timestamps, so physical replay reproduces compute/communication interference. Replay compute-filled traces without `--use-timestamp`.

## 0.3.0 - 2026-07-03

### Research fidelity

- Strengthened `verify-behavior` so it separately reports representation fidelity, source verification, behavioral fidelity, and configuration-ranking status.
- Added queue-wait distribution checks, phase/op behavior checks, tail-event recall, and pairwise backend ranking agreement across latency metrics.
- Added source commitments to bounded timing intervals: source count, source segment SHA-256, source gap sum, representative-selection method, representative index, and a complete error vector.
- Added separate source-normalized, scheduler-execution, calibration-evaluation, and artifact/provenance fingerprints.
- Added an adversarial ranking-inversion scaffold and tests that label too-small behaviorally lossy canaries as unverified.
- Added replay-equivalent sequence motif compression for exact repeated multi-event programs, with scheduler-hash equivalence to flat encodings.
- Added fail-closed behavior-gated compilation via `--require-behavior-verification`.
- Added behavior-search compilation that exhaustively searches timing sample limits and selects the smallest source-, behavior-, and ranking-verified canary.
- Added greedy per-group behavior-search refinement so quiet signature groups can use lower timing budgets while ranking-sensitive groups retain detail.
- Strengthened `verify-report` so forged canary identity, replay protocol, backend, workload, or canary-summary metadata fails model recomputation.
- Added research baseline trace generators for isolated-collective, random-sampling, frequency-representative, and clustering-representative controls, plus a `commcanary baseline` CLI.
- Added a stratified sampling baseline generator (`baseline --method stratified`), the kill-condition control named by RESEARCH_SPEC.md.
- Added a ddmin-style decision-preserving reducer (`commcanary reduce`) that minimizes a trace under a pairwise configuration-ranking oracle, as a generic property-preserving reduction baseline for behavior-search comparisons.
- Closed a canary validator gap: every timing sample now needs a weight that matches its declared source interval (single-index records are weight one), and sample intervals must tile the repeat range contiguously, so occurrences can no longer be silently double-counted or dropped.
- Rejected non-ASCII digit strings in integer parsing.
- `examples/make_synthetic_trace.py` now writes `llama70b_tp8_trace_long.json` instead of silently overwriting the small checked-in fixture.

### Ecosystem interop

- Added `commcanary import-kineto`: single-rank observational import of `record_param_comms` collective metadata (op, dtype, element counts, process-group ranks, timestamps) from PyTorch profiler traces (torch >= 2.2); timestamps are rebased to the trace start, truncated non-uniform rank lists fail closed instead of fabricating membership, unmapped collectives become `custom_op` events, control ops are skipped and counted, and no cross-rank skew or overlap is invented.
- Added `commcanary export-param`: expands a canary's full event program (motifs, patterns, run-length weights) into a PARAM comms-replay "basic" JSON trace with element counts, PARAM's asymmetric size conventions for `all_gather`/`reduce_scatter`, process-group ids, matched send/recv entry pairs (with `src_rank`/`dst_rank`) per point-to-point transfer, and cumulative `startTime_ns` timestamps for `--use-timestamp` replay — a physical NCCL execution path for minimized canaries.
- Tightened `verify-behavior` so it replays the full normalized source trace by default and marks prefix/subset canaries as partial-source rather than behaviorally verified.
- Added simulator ablation controls for skew, overlap, ordering, rare tails, queue-reset gaps, pressure, and observed exposed latency.
- Strengthened point-to-point semantics with sender/receiver, tag, channel, message sequence, and send/recv observations.

## 0.2.0

### Correctness

- Preserved exact sub-microsecond timing through periodic compression and replay.
- Rejected ambiguous mixed timestamp/gap traces and conflicting timing fields.
- Added queue-aware deterministic scheduler model v4 and counter-based randomness.
- Reconciled report metrics, breakdowns, samples, and calibration data.
- Hardened one-rank skew, pattern sums, interval coverage, integer, and finite-number validation.

### Research fidelity

- Added optional measured `observed_exposed_us` as a joint timing feature.
- Added model calibration error to reports.
- Added lossless compilation mode and explicit approximation budgets.
- Added prefix-cumulative-gap error and serialized-byte compression metrics.
- Added phase- and operation-level regression localisation.

### Capture

- Added UUID-qualified shard names and generation-ordered saves.
- Rejects mixed capture sessions, conflicting workload/session metadata, missing or duplicate ranks, and partially observed latency.
- Preserves clock uncertainty instead of inventing cross-rank skew.
- Validates recorder inputs before storage.

### Performance

- Uses incremental source hashing, direct timing comparisons, compact replay arrays, and event-local counter-based random generation.
