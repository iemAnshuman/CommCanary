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
replay-model version, and all seven exact artifact format IDs. Package metadata,
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
