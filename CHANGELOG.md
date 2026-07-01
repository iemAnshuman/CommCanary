# Changelog

## Unreleased

### Research fidelity

- Strengthened `verify-behavior` so it separately reports representation fidelity, source verification, behavioral fidelity, and configuration-ranking status.
- Added queue-wait distribution checks, phase/op behavior checks, tail-event recall, and pairwise backend ranking agreement across latency metrics.
- Added source commitments to bounded timing intervals: source count, source segment SHA-256, source gap sum, representative-selection method, representative index, and a complete error vector.
- Added separate source-normalized, scheduler-execution, calibration-evaluation, and artifact/provenance fingerprints.
- Added an adversarial ranking-inversion scaffold and tests that label too-small behaviorally lossy canaries as unverified.
- Added replay-equivalent sequence motif compression for exact repeated multi-event programs, with scheduler-hash equivalence to flat encodings.
- Added fail-closed behavior-gated compilation via `--require-behavior-verification`.
- Added behavior-search compilation that exhaustively searches timing sample limits and selects the smallest source-, behavior-, and ranking-verified canary.
- Strengthened `verify-report` so forged canary identity, replay protocol, backend, workload, or canary-summary metadata fails model recomputation.
- Added research baseline trace generators for isolated-collective, random-sampling, and frequency-representative controls, plus a `commcanary baseline` CLI.
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
