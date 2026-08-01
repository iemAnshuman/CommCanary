# Command-line contract

The `commcanary` console script and `python -m commcanary` use the same entry
point. Command output requested as JSON is written to the specified file;
human summaries stay on stdout and diagnostics stay on stderr.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Operation succeeded or verification/comparison produced a positive result |
| 1 | Valid negative comparison or verification result |
| 2 | Argument or usage error reported by `argparse` |
| 3 | CommCanary input, configuration, validation, or I/O error |
| 4 | Captured workload/child execution failure |
| 130 | Interrupted |

Code 1 is evidence, not an application crash. Automation should still retain
the comparison/verification output file. Code 4 deliberately does not return a
raw child status that could collide with this table. The original child code is
printed on stderr and, when `capture --preserve-on-failure` is used, stored as
`child_returncode` in the immutable failure manifest.

## Version and capability output

```console
commcanary --version
```

The stable multi-line output includes the package version, canonicalization ID,
replay-model version, and all eight exact artifact format IDs. Package metadata,
`commcanary.__version__`, and this output must agree.

## JSON diagnostics

Place the global option before the subcommand:

```console
commcanary --diagnostics-json compile trace.json -o canary.json
```

Stderr becomes JSON Lines using `commcanary.diagnostic.v1`. A dispatched command
emits `started` and one terminal `completed`, `error`, or `interrupted` row;
terminal rows record elapsed seconds. Behavior search and reduction additionally
emit bounded-work `progress` rows with planned/evaluated candidates or oracle
calls and budget exhaustion. Child-failure rows carry the original child return
code. Human-requested stdout and output artifacts are unchanged, so a caller can
parse stderr without scraping prose. Argument-parser failures occur before
command dispatch and retain argparse's standard text/exit-2 contract.

Without JSON diagnostics, behavior search and reduction print a short progress
line to stderr and retain their final counts in the output artifact. `Ctrl-C`
maps to 130 and a structured `interrupted` row records the elapsed time.

`import-kineto` prints derived/unknown totals for overlap, input/output message
shape, reduction operators, and broadcast roots. The output trace retains those
totals and a per-event reason. Import itself succeeds for an inspectable
mixed/unknown trace, but canary-producing commands fail until every event has
measured overlap; qualification applies the stronger semantic requirements
described below.

Pass two or more rank-local profiles to merge one logical distributed trace:

```console
commcanary import-kineto rank0.json rank1.json -o trace.json --assume-shared-clock
```

`--assume-shared-clock` is an explicit zero-offset assertion, not an inference
from profile metadata. For different clock domains, repeat
`--clock-offset-us RANK=OFFSET` exactly once per imported rank; offsets are
additive microseconds into the chosen reference clock. Without either option,
the merge retains rank-local evidence but marks cross-rank arrival skew unknown,
so `compile` refuses it. Missing rank contributions, duplicate ranks, and
operation/signature disagreement fail during import.

Every successfully imported source profile is committed by exact byte SHA-256
and byte size under `system.kineto_source_profiles`, with a distributed rank
when present. The records are path-free and sorted by rank, so input argument
order and local filenames do not affect their ordering. Equivalent JSON with
different whitespace has a different commitment. CommCanary does not copy the
profiles into the output; retain them separately for later byte verification.

## Portable qualification request

Diagnose readiness before attempting to create an immutable request:

```console
commcanary doctor rank0.json rank1.json \
  --assume-shared-clock \
  --output readiness.json
```

The command exits zero only when every required exact-work gate passes. Exit
status 1 is a diagnostic negative result, not an application crash. The JSON
report records stable reason codes, coverage, offending event/rank locations,
next capture actions, a structural privacy disclosure, and conservative size
and memory estimates. Physical runtime remains `not_estimable` until target
measurements exist.

Request preparation is a separate command:

```console
# The checked-in policy is illustrative. Review and freeze thresholds for the
# actual qualification decision before execution.
commcanary verify-policy examples/qualification-policy.json

commcanary prepare-qualification rank0.json rank1.json \
  --assume-shared-clock \
  --policy examples/qualification-policy.json \
  --output-directory qualification-request
```

It shares the Kineto filters, clock options, and bounded-input overrides of
`import-kineto`. It additionally compiles, independently verifies source
fidelity, binds the exact predeclared decision policy, and writes a fixed
five-file v2 request directory. The directory must not already exist and is
never overwritten. Its manifest is installed last; an interrupted or failed
preparation therefore cannot look complete. Historical four-file v1 requests
remain read/verify compatible, but the current writer never emits them.

Preparation preserves source-bound communication dtype and requires a complete
per-rank contiguous-GEMM recipe derived between asynchronous collective issue
and one same-thread explicit wait. It fails before creating the directory if a
recipe is missing or ambiguous, an operation or dtype is unsupported, an
`all_reduce` or `reduce_scatter` lacks an explicit source-bound reduction
operator, a broadcast lacks an explicit source-bound root rank, a communicator
name has conflicting membership, or message shapes cannot be sized exactly.
For Kineto inputs it independently verifies exact input/output element counts,
normalized split lists, and dtype-derived bytes. It never converts elapsed
gaps to compute, substitutes SUM, reconstructs a convenient message shape, or
chooses a broadcast root from group order.

Timing is lossless by default. `--allow-bounded-timing` is an explicit
acknowledgement that the included fidelity proof may describe bounded
approximation. Unknown overlap or uncalibrated multi-rank arrival timing still
fails before the output directory is created.

The receiving party runs:

```console
commcanary verify-qualification qualification-request

commcanary materialize-qualification qualification-request \
  --output-directory qualification-materialization

commcanary verify-materialization qualification-request \
  qualification-materialization

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module commcanary execute-materialization \
  qualification-request qualification-materialization \
  --device cuda --backend nccl \
  --iterations 1 --warmup 1 \
  --distributed-timeout-seconds 300 \
  --output reference-execution.json
```

Verification requires the exact inventory, rejects symlinks and non-regular
files, rehashes every artifact, validates trace and canary semantics,
recomputes the fidelity document exactly, and checks all manifest-to-canary
bindings. It does not claim authenticity. The request contains no physical
measurement or verdict and deliberately contains no PARAM trace. The manifest
binds the communication dtype and reduction-operator sets, source-validated
message shapes, the exact compute-recipe projection hash and counts, the
equal-split-only `all_to_all` policy, single-inflight issue/work/wait structure,
and disabled timestamp pacing. It also discloses that GEMM shapes and dtypes
are shared and may reveal model structure.

Materialization creates a new, non-overwriting two-file directory and installs
`materialization.json` last. `replay-program.json` is rederived byte-for-byte
by `verify-materialization`; the manifest binds the exact request-manifest
bytes, canonical source-work projection, per-rank operation counts, source
kernel observations, mathematical FLOPs, program bytes, entry count, and
async-issue/exact-rank-work/explicit-wait contract. No target calibration or
duration quantization is accepted. Materialization does not run the program:
`execution_adapter` remains `conforming-adapter-required`, current upstream
PARAM compatibility is `not_claimed`, and physical measurement and verdict
remain absent.

`execute-materialization` is a bounded torch.distributed reference runner,
not a completed qualification workflow. Install the PyTorch build appropriate
for the target CUDA/NCCL or CPU/Gloo environment separately; CommCanary does
not select a CUDA wheel. Every torchrun rank verifies both directories and
preflights the complete program before importing PyTorch or initializing
distributed state. The exact-work generator supports the five materialized
collectives, requires the encoded groups to cover exactly the launched
rank domain, creates a distinct runtime process group for every encoded group
identity, and aggregates rank-local issue-to-explicit-wait timings after the
measured pass. It separately records wall-clock duration for each rank and
iteration and reports the maximum-rank whole-program makespan; per-operation
latency is diagnostic and is not substituted for workload latency. The
owner-side materializability check rejects uneven or explicit-split
`all_to_all` source evidence because this runner uses PyTorch's equal-split
`all_to_all_single` form. Each rank executes only its exact source-bound
rectangular GEMM recipe; rank timing differences emerge from different work
rather than synthetic delays. Both GEMM inputs and one reused output
are allocated and budgeted before the measured loop. Communication buffers are
request-isolated and fully counted, including repeated same-shaped operations,
so asynchronous lifetimes never alias one tensor.
Before warmup and measurement, the runner performs one resource-counted but
untimed correctness pass. Deterministic rank-dependent patterns validate every
collective result under its bound reduction operator and every receive
endpoint; the runtime maps `sum`, `product`, `min`, `max`, and `avg` to the
matching PyTorch `ReduceOp` instead of relying on the default. All ranks
exchange the check inventory and any wrong result fails the run collectively.
The pass is not included in the reported timing samples.
Default-group initialization and every encoded subgroup share the explicit
`--distributed-timeout-seconds` duration. The default is 300 seconds and the
shared resource policy caps it at 3,600 seconds, avoiding PyTorch's otherwise
backend-dependent 10- or 30-minute failure delay. The diagnostic records the
selected value.

The output is intentionally
`commcanary.reference-execution.stdout.v1`, a diagnostic rather than a
published artifact format. It binds the request ID, materialization ID, program
digest, runtime versions, tensor allocation, rank samples, and max-rank
per-operation aggregate, max-rank whole-program makespans, plus the passed
per-rank correctness-check inventory. Its claims
remain `physical_fidelity: unproven` and
`qualification_verdict: not_issued`; local fake-runtime tests do not establish
CUDA/NCCL conformance.

Once independently produced baseline and candidate observations exist, apply
the same explicit policy without re-running either measurement:

```console
commcanary evaluate-qualification examples/qualification-policy.json \
  baseline.observation.json candidate.observation.json \
  --output qualification-verdict.json
```

The stable verdict is exactly one of `pass`, `fail`, `inconclusive`, or
`incomparable`. Confidence-boundary crossings and unstable measurements are
not coerced into pass/fail.

`export-param` is a low-level legacy encoding command for pinned integrations.
Current upstream PARAM removed its basic and Kineto parsers and accepts Chakra
host execution traces; do not interpret successful JSON export as proof that
an installed upstream PARAM checkout can execute it.

## Capture parsing and failure evidence

The workload command starts after `--`; it is passed as an argument vector and
is not interpreted by a shell:

```console
commcanary capture --output trace.json --workload-name decode -- \
  python examples/instrumented_decode.py
```

An empty command is an application error. A successful child that produced no
trace is also an application error unless `--allow-empty` is explicit. A stale
pre-existing output is never accepted as evidence from the child.

On failure, `--preserve-on-failure DIRECTORY` copies only bounded regular shard
files into a new collision-resistant bundle and records their sizes and SHA-256
digests. It does not record the raw command line or environment. An existing
destination is never overwritten.

## Baseline option applicability

Method-specific baseline flags fail when they do not apply. `--sample-count`
and `--partial` belong only to `random`; `--cluster-count` belongs only to
`cluster`; `--strata-per-group` belongs only to `stratified`; and `--seed`
belongs to `random` and `stratified`. Defaults are applied after the method is
selected, so an irrelevant explicit flag is never silently ignored.

## HTML command compatibility

Replay and compare accept `--html` alongside their JSON output. The primary
standalone command is `render-html REPORT --output HTML`. The older `report`
spelling remains a deprecated compatibility alias through 0.4 and emits a
replacement/removal diagnostic.
Generated HTML is self-contained, escapes untrusted values, declares a strict
content-security policy, and says samples are unavailable when a report has only
summary quantiles; it never synthesizes a distribution.

## Implementation boundary

`commcanary.cli:main` remains the console-script and compatibility entry point.
Its implementation is dependency-directed under `commcanary.command_line`:

- `parser` declares arguments and injects handlers without importing engines;
- `lifecycle` owns parse/dispatch/error/interrupt completion semantics;
- `diagnostics` owns version text, JSON Lines records, and elapsed-time rounding;
- `commands` adapts parsed arguments to public domain services;
- `capture` owns child-process argv/environment orchestration; and
- `capture_failure` owns bounded immutable failure evidence.

The compatibility module wires these boundaries and retains the characterized
private handler seams used by tests. Domain calculations remain below the CLI;
the parser and lifecycle do not import them, and command-line modules never
import the compatibility module back upward.
