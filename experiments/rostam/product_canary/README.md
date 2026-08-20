# Rostam product-canary studies

This directory contains the cluster-only evidence workflow behind the installed
`capture -> build -> gate` surface. It does not contain a completed product
result. The checked-in perturbation plans, runner locks, site contract, and
study code are inputs; successful measurements remain append-only on Rostam.

## What runs

The vLLM and SGLang studies use the same experimental design:

- one baseline plus seven training and three held-out perturbations;
- eight independent application allocations over at least two days;
- position-balanced Williams schedules inside fresh child processes;
- application truth from the real engine, with dummy-initialized model weights;
- active candidate execution through the four-rank Chakra runner;
- independent exact replay, random, stratified, ddmin, and
  communication-only baselines;
- complete correctness, environment, and cycle-telemetry evidence; and
- selection frozen before the held-out application callback can read evidence.

The supported domain is single-node, four-GPU tensor-parallel dense-decoder
inference with all-reduce and rank-local GEMM overlap. Other operations fail
with an explicit unsupported reason.

## Build a runner

Build the wheel first. Create a new context containing exactly
`Containerfile`, `runner-lock.json`, and the wheel renamed to
`commcanary.whl`. For vLLM, use the files in this directory. For SGLang, use
the files under `sglang/`.

Run the builder on Rostam:

```console
python3 -I experiments/rostam/product_canary/build_runner.py \
  --context out/vllm-runner-context \
  --output out/vllm-runner-build
```

The output directory must not exist. The builder pulls the locked base manifest,
creates an OCI archive and Apptainer SIF, probes exact package versions, and
writes `descriptor.json`. Keep the archive, SIF, log, descriptor, context,
and wheel together. The study verifier rehashes the external SIF and every
frozen input before scheduler work.

## Freeze a study

Capture a complete source trace and its physical projection, then prepare a
reviewed `commcanary.physical_canary_policy.v1` whose runner digest equals the
descriptor's OCI manifest digest. Thresholds must be fixed before measurement.

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
  --wheel dist/commcanary-0.3.0-py3-none-any.whl

python3 -I -m experiments.rostam.product_canary.study verify \
  --study out/vllm-product-v1
```

Freezing copies every repository-controlled input, records external runner
bytes, closes the member inventory, and makes the study root read-only while
leaving private `state/` and `publication/` directories writable.

## Submit or resume

A run without `--execute` is read-only. It can verify and aggregate existing
attempts but refuses when a missing measurement would require submission.

```console
python3 -I -m experiments.rostam.product_canary.study status \
  --study out/vllm-product-v1

python3 -I -m experiments.rostam.product_canary.study run \
  --study out/vllm-product-v1 \
  --execute
```

`--execute` is the scheduler acknowledgement. Jobs use partition
`cuda-A100`, one exclusive node, four GPUs, exact wrapper bytes passed through
`sbatch` stdin, and fresh append-only attempt IDs. The workflow does not cancel
jobs on timeout. Use `--retry-failed` only after reviewing the preserved
terminal attempt.

The run first completes training application allocations, then searches and
measures candidates. It freezes the selection before it submits held-out
application allocations. A successful terminal run writes the corpus, active
ledger, application evidence, physical evidence, audit bundle, and result under
`state/result/`.

## Historical version-sensitivity study

`historical/` is a separate study for the version pattern reported in vLLM
issue 2971. It locks official vLLM 0.2.7, 0.3.2, and 0.3.3 image manifests and
the issue-date OpenHermes 2.5 Mistral 7B revision. Build images with
`historical/build_images.py`, stage the locked model files with
`historical/stage_model.py`, then freeze and run through
`historical/study.py`.

Its claim boundary is fixed: four A100 GPUs and deterministic offline batches
differ from the reported single RTX 4090 and unknown server request stream.
The result can report whether the version and eager-mode pattern appears in
this scope. It never issues a causal claim or calls itself an exact
reproduction.

