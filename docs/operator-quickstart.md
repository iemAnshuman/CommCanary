# Operator quick start

CommCanary 0.3.0 is an unreleased research alpha. This guide is for an engineer
who did not build the product path. The operator trial is a release gate; no
independent operator has completed it yet.

The guide has two boundaries:

- local orientation is read-only with respect to Rostam and ends in an expected
  blocked bundle;
- runner construction, study freeze, and scheduler execution require the live
  checkpoint in `HANDOFF.md` and separate authorization for any mutation.

Do not reuse paths, hashes, job IDs, or fleet state from this document.

## 1. Install a pinned artifact

The trial owner supplies two things through independent channels:

1. a reviewed source checkout; and
2. the exact wheel plus its SHA-256 digest.

Verify the wheel before installing it:

```console
sha256sum /path/to/commcanary-0.3.0-py3-none-any.whl
```

On macOS, use `shasum -a 256`. The result must equal the independently supplied
digest. Do not substitute a wheel digest copied from an old handoff.

```console
python3 -m venv .venv-operator
. .venv-operator/bin/activate
python -m pip install /path/to/commcanary-0.3.0-py3-none-any.whl
commcanary --version
```

The checkout's canonical development gate is:

```console
python -m pip install -e ".[dev]"
python -m tools.verify --fast
```

`--fast` is a source-tree gate, not release readiness. The reproducible package
gate builds twice in temporary directories and compares exact bytes:

```console
python -m tools.verify --reproducible
```

Pass `--artifact-dir NEW_EMPTY_DIRECTORY` only when the exact tested wheel and
sdist should be copied out. `--release` also enforces release-source and
changelog rules; tagging and publication remain separate decisions.

## 2. Run the local blocked orientation

The checked-in example is synthetic and four-rank. It exists to exercise
instrumented capture, not to supply application ground truth.

Choose a new output directory. The commands refuse to overwrite evidence.

```console
mkdir -p out/operator-local

commcanary capture \
  --output out/operator-local/trace.json \
  --chakra-output out/operator-local/trace.et \
  --projection-output out/operator-local/trace.projection.json \
  --workload-name operator-orientation \
  -- python examples/instrumented_decode.py
```

The child records an explicit sum all-reduce and a complete GEMM recipe. Capture
must produce all three files. It does not infer reduction semantics or GEMM
shapes from an opaque process.

Build without an oracle corpus:

```console
commcanary build out/operator-local/trace.et \
  --projection out/operator-local/trace.projection.json \
  --policy examples/physical-canary-policy.json \
  --runtime-budget 60s \
  --output out/operator-local/canary.bundle
```

Exit code 1 is expected. The bundle is written, immediately verified, and
marked `blocked_missing_physical_oracle_corpus`. Confirm that independently:

```console
python -c "from commcanary.workflows.physical_canary import verify_physical_canary_bundle as v; print(v('out/operator-local/canary.bundle')['status'])"
```

Do not create fake baseline and candidate observations merely to make `gate`
run. `gate` accepts only a `qualified_physical_decision_canary` bundle.

The example policy contains a placeholder runner digest and illustrative
thresholds. It must never be used for a physical study.

## 3. Understand the block

`physical_decision_canary.v1` has seven terminal bundle statuses:

| Status | Meaning |
|---|---|
| `blocked_missing_physical_oracle_corpus` | No real application decisions were supplied. |
| `blocked_synthetic_evidence` | Only synthetic search evidence was supplied. |
| `blocked_incomplete_baseline_corpus` | Required baseline coverage is missing. |
| `blocked_no_decision_preserving_training_candidate` | No reduced candidate passed the training gates. |
| `failed_held_out_validation` | The frozen candidate failed held-out safety. |
| `failed_privacy_policy` | The bundle exceeded the disclosure policy. |
| `qualified_physical_decision_canary` | Every predeclared gate passed. |

The first four are blocked, the next two are measured failures, and the last is
the only status accepted by `gate`. A blocked result is useful evidence that
the workflow refused to promote missing or synthetic facts.

## 4. Build the digest-pinned runner

This step is Rostam-only. Read `AGENTS.md` and the complete live `HANDOFF.md`
before doing anything on the cluster. Confirm the expected Git HEAD, tracked
cleanliness, wheel identity, runner lock, and current GPU status.

Prepare a new build context with exactly:

```text
build-context/
├── Containerfile
├── runner-lock.json
└── commcanary.whl
```

Use the vLLM files under `experiments/rostam/product_canary/` and the verified
wheel from step 1. Then run:

```console
python3 -I experiments/rostam/product_canary/build_runner.py \
  --context build-context \
  --output out/vllm-runner-build
```

The output directory must not exist. The builder:

- builds from the digest-pinned base;
- exports and verifies a content-addressed OCI archive;
- verifies the OCI configuration is Linux on the locked architecture;
- creates an Apptainer SIF;
- probes installed versions inside the SIF; and
- binds the archive, SIF, log, inputs, and versions in `descriptor.json`.

Keep every output byte together. A study rehashes the external SIF and the
frozen inputs. Changing any bound byte requires a new build or study identity.

## 5. Review the study inputs

Do not freeze until these inputs have been reviewed:

- a complete four-rank source Chakra ET and matching projection;
- a physical-canary policy whose runner OCI digest matches the descriptor;
- `experiments/rostam/product_canary/perturbations.json` or a reviewed
  replacement;
- the checked-in model configuration directory used with dummy-initialized
  weights;
- the runner descriptor and SIF; and
- the exact wheel installed in the runner.

The current vLLM plan contains one baseline, seven training perturbations, and
three held-out perturbations. The policy fixes the 10× runtime gate, severe
false-negative ceiling, false-positive ceiling, pair-agreement floor, runtime
budget, privacy limit, runner digest, protocol, and world size before evidence
exists.

The model input for the main product study is
`experiments/rostam/product_canary/model`; it contains configuration, not real
weights. The driver uses `load_format="dummy"`. The separate historical issue
study has a different contract and must not be substituted.

## 6. Freeze and verify

Use new output paths and the reviewed physical inputs:

```console
python3 -I -m experiments.rostam.product_canary.study freeze \
  --study-id vllm-product-v1 \
  --output out/vllm-product-v1 \
  --source-et out/capture/trace.et \
  --projection out/capture/trace.projection.json \
  --policy out/capture/regression-policy.json \
  --perturbations experiments/rostam/product_canary/perturbations.json \
  --model experiments/rostam/product_canary/model \
  --runner-descriptor out/vllm-runner-build/descriptor.json \
  --runner-sif out/vllm-runner-build/runner.sif \
  --wheel /path/to/commcanary-0.3.0-py3-none-any.whl

python3 -I -m experiments.rostam.product_canary.study verify \
  --study out/vllm-product-v1
```

Record the printed manifest ID. Freeze copies repository-controlled inputs,
binds the external runner, closes the inventory, and leaves only `state/` and
`publication/` writable. `verify` is read-only and submits nothing.

Changing a policy, script, wheel, SIF, model file, projection, or trace after
freeze invalidates future execution. Preserve the study and freeze a new one.

## 7. Inspect, execute, or hand off

These commands are read-only:

```console
python3 -I -m experiments.rostam.product_canary.study status \
  --study out/vllm-product-v1

python3 -I -m experiments.rostam.product_canary.study run \
  --study out/vllm-product-v1
```

With missing application allocations, the second command reports
`awaiting_training_application_allocations`. It does not submit work.

Scheduler mutation requires a separate explicit authorization for `--execute`:

```console
python3 -I -m experiments.rostam.product_canary.study run \
  --study out/vllm-product-v1 \
  --execute \
  --no-wait
```

Before that command, recheck the live handoff, exact Git HEAD, frozen manifest,
runner and wheel hashes, prior attempt inventory, and `cuda-A100` queue. Scope
all mutation to CommCanary jobs. Other users and unrelated partitions are
read-only. The workflow creates a new append-only attempt ID for every
submission.

An operator without scheduler authority stops here and hands over the verified
study path, manifest ID, descriptor ID, SIF identity, wheel identity, and a
written note of every check performed.

## 8. Interpret status

Each attempt has one of two terminal record statuses:

| Attempt status | Meaning |
|---|---|
| `success` | Scheduler completion and the expected batch result were recorded. |
| `failed` | The job failed, was cancelled, timed out, or failed a runtime guard. |

No `terminal.json` means the attempt is unsubmitted or still being observed;
inspect `submission.json` and scheduler accounting before deciding which.

The active synthesis result has exactly two statuses:

| Study result | Exit | Meaning |
|---|---:|---|
| `qualified_active_candidate` | 0 | Training and held-out statistics satisfy the policy. |
| `failed_active_candidate` | 2 | At least one policy gate failed. Read the ledger checks. |

While work is incomplete, `status` returns `result: null`. The final summary is
`state/result/result.json`; detailed reasons live in
`state/result/active-study-ledger.json`. Do not invent a third terminal study
status such as “inconclusive.” Describe uncertainty from the recorded checks.

## 9. Recover without rewriting history

Never delete or overwrite an attempt. For a reviewed failed logical cell, append
a retry:

```console
python3 -I -m experiments.rostam.product_canary.study run \
  --study out/vllm-product-v1 \
  --execute \
  --retry-failed
```

- A cancelled or timed-out attempt remains failed evidence.
- `--no-wait` may disconnect safely after submission; resume with `status`, then
  the same `run` command.
- A failed batch is retried with a new attempt ID.
- A changed manifest-bound input requires a new frozen study, not an in-place
  repair.
- Use literal job IDs with `sacct -j JOB_ID` after jobs leave the live queue.

## 10. Verify and interpret the result

First re-run frozen-input verification and the read-only aggregation path. Then
verify the generated bundle from exact bytes:

```console
python3 -I -m experiments.rostam.product_canary.study verify \
  --study out/vllm-product-v1

python -c "from commcanary.workflows.physical_canary import verify_physical_canary_bundle as v; print(v('out/vllm-product-v1/state/result/bundle')['bundle_id'])"
```

Cross-check that bundle ID against `state/result/result.json`. The result also
binds the corpus, ledger, application-evidence, physical-evidence, selection,
and canary ET identities. A mismatch is an inconsistency to investigate, not a
reason to edit either file.

Only a qualified bundle can be used with:

```console
commcanary gate out/vllm-product-v1/state/result/bundle \
  --baseline baseline-observation.json \
  --candidate candidate-observation.json \
  --output gate.json \
  --html gate.html \
  --junit gate.xml \
  --sarif gate.sarif
```

A passed study still does not make the project release-ready. The independent
operator report, second-engine or materially different replication, claims
review, and deliberate release gate remain.

## 11. Report friction

Return a plain-text or Markdown report with:

```text
operator:
date:
reviewed wheel sha256:
runner descriptor id:
frozen manifest id:
outcome:

step durations:
- install:
- local orientation:
- runner build:
- input review:
- freeze and verify:
- execution or handoff:
- result verification:

obstacles:
- step:
  command or action:
  observed result:
  expected result:
  resolution or blocker:
  severity: blocker | delay | confusion_only

missing or incorrect documentation:
suggested changes:
would you repeat the workflow without the author: yes | no | yes_with_changes
```

Record blocked and failed attempts. The trial evaluates whether the workflow is
independently usable; it is not a request to manufacture a successful result.
