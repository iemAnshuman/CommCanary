# Metadata privacy and redaction

CommCanary removes model weights and prompts from its intended artifact model,
but communication traces are not automatically anonymous. Operation order,
message sizes, rank topology, timing, workload labels, host/runtime metadata,
and experiment provenance can reveal architecture, scale, deployment layout,
performance characteristics, or internal naming.

## Data inventory

Review these field families before sharing an artifact:

| Family | Examples | Typical risk |
|---|---|---|
| Workload identity | names, phases, user metadata, input IDs | model/product or customer disclosure |
| System identity | hostnames, cluster/site, runtime versions, device/topology | infrastructure fingerprinting |
| Process/scheduler identity | rank, PID, job, account, partition, node | user and cluster attribution |
| Communication semantics | operation sequence, dtypes, bytes, groups, peers, channels | precision, model parallelism, and architecture inference |
| Timing/calibration | gaps, arrival skew, overlap, observed latency | capacity and performance disclosure |
| Provenance | commit, dirty patch hash, script/config/input hashes | internal development-state disclosure |
| Logs and commands | argv, stdout/stderr, environment | secrets, paths, tokens, or tenant data |

The artifact-provenance digest protects included metadata from unnoticed
editing; it does not make that metadata safe to disclose. Hashing a low-entropy
identifier such as a hostname is often reversible by guessing and is not
redaction.

Kineto CLI import deliberately records no local source path or filename. It
does record each source profile's exact byte SHA-256 and size so a hardware
owner can verify the private input later. The size is a direct disclosure, and
the digest lets a recipient confirm a guessed or already-known profile; review
both before sharing even though the profile contents are not embedded. Raw
Kineto monotonic starts and wall-clock base times are used only transiently for
rebasing/alignment and are omitted from imported traces.

A qualification request deliberately includes `source.trace.json` so the
receiving party can independently recompute trace-to-canary fidelity. This is
more revealing than sending the canary alone: operation order, collective
dtypes, exact message sizes, rank topology, arrivals, and overlap remain
visible. Review or redact at the source boundary before preparing the request;
post-hoc removal invalidates its byte and semantic bindings.

A materialization contains the expanded event program and exact per-rank GEMM
dtypes and `m`, `n`, and `k` dimensions. Expansion can reveal repeated
operation order, hidden dimensions, token/batch structure, and rank-dependent
work more directly than the compact canary. These shapes are necessary to
reproduce causal work and are an explicit privacy cost, not anonymous
metadata. Review both files before sharing. The materialization deliberately
omits source kernel durations from executable entries and omits hostname,
device name, and execution output; adding those ad hoc would bypass the
documented evidence and privacy contracts.

The optional reference-execution diagnostic is more revealing still. It binds
request/materialization identities, runtime PyTorch/CUDA/NCCL versions,
per-rank planned tensor bytes and GEMM counts, and per-rank physical timing
samples. Those values can identify a software stack and expose target
performance. The diagnostic is not embedded into or required by the portable
request; review it as target-owned sensitive output and do not treat its
current unsupported diagnostic contract as a safe external exchange format.

## Redaction policy

Redact at the source boundary, before compilation and hashing. Use stable
campaign-local pseudonyms only when correlation is required. Remove unnecessary
free-form metadata rather than replacing it with opaque hashes. Do not change
protected fields after compilation and then recompute commitments while calling
the result source-corresponding; compile and verify the redacted source as a new
artifact instead.

At minimum, a public artifact should:

- use synthetic workload/system names;
- omit host, user, account, absolute-path, job, node, cluster, and partition
  identifiers unless scientifically necessary;
- include only allowlisted environment/version fields;
- use synthetic input IDs and confirm that metadata contains no prompts, data
  paths, tokens, or customer identifiers;
- inspect stdout/stderr and failure bundles separately from structured results;
- document any topology/timing detail intentionally retained; and
- validate and recompute all commitments after redaction at the source boundary.

CommCanary's capture failure bundle deliberately records the workload label,
session ID, return code, bounded shard paths/sizes/hashes, and timestamp; it does
not copy the raw command line or environment. Shard contents can still contain
workload/system metadata and must be reviewed.

## Experiment publication

Frozen run manifests contain declared reproducibility inputs and expected site
constraints. Observed scheduler/job/node/account metadata belongs in immutable
attempt/submission records and is private by default. Publish aggregates and
checksums from a reviewed export; do not expose a shared results directory or
hand-copy rows around completeness validation.

Large raw data should live in an access-controlled immutable archive. Its URI
can itself be sensitive. A public manifest may record a redacted archive ID and
trusted digest while the access mapping remains private.
