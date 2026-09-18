# Physical decision canaries

`physical_decision_canary.v1` is CommCanary's physically reduced product
artifact. It contains a dependency-closed Chakra ET subgraph and the evidence
needed to say whether that subgraph has earned a decision-preservation claim.
It is separate from `commcanary.canary.v2`, which compresses representation but
can still expand to the complete logical workload.

## Assurance states

The bundle manifest records one exact status. The main states are:

| Status | Meaning |
|---|---|
| `blocked_missing_physical_oracle_corpus` | A dependency-closed measurement candidate exists, but no application/candidate measurements were supplied. |
| `blocked_synthetic_evidence` | The algorithm ran on test data; no physical claim is allowed. |
| `blocked_incomplete_baseline_corpus` | A policy-required comparison method is absent. |
| `blocked_no_decision_preserving_training_candidate` | No recorded candidate satisfies the training policy. |
| `failed_held_out_validation` | The frozen candidate missed a held-out safety, agreement, runtime, or cost gate. |
| `failed_privacy_policy` | Declared disclosure exceeds the bound policy. |
| `qualified_physical_decision_canary` | Measured application ground truth and a candidate frozen without using holdout decisions satisfy the held-out policy. |

Only the last state can be used by `gate` as a comparable performance test.
Every other state retains a useful artifact while leaving decision preservation
and runtime reduction unproven.

## Inputs

`commcanary build` consumes a Chakra ET, its semantic projection, and a
policy. A qualified active-study bundle also supplies the oracle corpus,
active-search ledger, complete application evidence, and complete physical
execution evidence.

### Chakra ET

The reader follows Chakra's on-disk framing: one varint-length-delimited
`GlobalMetadata` protobuf message followed by length-delimited `Node` messages.
CommCanary reads metadata version, node ID and type, control dependencies, data
dependencies, and the standard integer `comm_type` and `comm_size` attributes.
It validates node uniqueness, dependency existence, and acyclicity. Complete
messages, including unknown fields and opaque attributes, are retained
byte-for-byte when a node is selected.

Plain and gzip-compressed inputs are accepted. Exact input bytes and expanded
protobuf bytes are bounded separately. Output `canary.et` is deterministic and
uncompressed.

### Chakra projection

`commcanary.chakra_projection.v1` binds the exact ET SHA-256, size, and metadata
version. It assigns the workload semantics that cannot be safely inferred from
opaque protobuf attributes:

- one supported operation and phase per node;
- executed FLOPs and tensor bytes;
- feature tags such as rare-tail, overlap, or rank-skew coverage;
- disclosure categories; and
- dependency-closed candidate regions.

The projection must cover every source node in source order. In the current
domain, Chakra compute nodes may be declared only as `gemm`; that GEMM label
remains owner-supplied. Collective nodes may be declared only as `all_reduce`,
and that declaration and projected tensor size must agree with Chakra's
standard `comm_type` and `comm_size` attributes. Each region must already
contain the transitive dependencies of its nodes. Region feature and
disclosure summaries are recomputed from their node rows.

### Policy

`commcanary.physical_canary_policy.v1` binds:

- the supported domain and maximum runtime;
- the severe-regression boundary;
- required feature tags and baseline methods;
- minimum runtime reduction;
- maximum severe false negatives and false-positive rate;
- minimum pair-decision agreement;
- deployment gate metrics and directions; and
- the allowed disclosure score; and
- the declared OCI runner digest, execution protocol, and world size.

The digest is a content commitment. `commcanary build` does not fetch an image.
The Rostam product-study boundary separately builds an OCI archive and
Apptainer SIF from a digest-pinned base, records their byte identities and
version probes, and refuses a study whose policy, descriptor, or SIF differs.
The bundle verifier then binds every supplied application and physical
measurement to that runner digest.

The current CLI accepts strict JSON. YAML parsing is not included in 0.3; a
future YAML surface must normalize to the exact JSON contract before policy
identity is computed.

### Physical oracle corpus

`commcanary.physical_oracle_corpus.v1` contains actual application decisions
and recorded candidate decisions for predeclared perturbations. Every
perturbation belongs to `training` or `holdout`. Every candidate row covers
every perturbation and records:

```text
physical runtime
GPU-seconds
GPU count
executed collectives
executed FLOPs
peak memory
decision: pass or fail
```

The corpus binds the baseline and perturbed stack identities, application
artifact, runner digest, environments, raw-evidence commitments, and each
candidate executable commitment. For reduced candidates, CommCanary rebuilds
the exact Chakra subgraph and recomputes its digest, collective count, and FLOP
count. It also requires GPU-seconds to equal runtime multiplied by the declared
world-size GPU count. The application row is explicitly named
`application_ground_truth`; a trace-derived executor cannot fill that role.

For active studies, `commcanary.application_evidence_set.v1` retains the
training and held-out baseline allocations plus every perturbation allocation.
`commcanary.physical_execution_evidence_set.v1` retains every candidate run,
including correctness commitments, per-rank samples, environment records, and
cycle telemetry. Full-audit verification revalidates these records and
recomputes the corpus rows. The active ledger binds both evidence-set IDs and
the selection-before-holdout boundary.

The corpus can also contain independent exact replay, random, stratified,
ddmin, and communication-only results. A policy decides which are mandatory.
The validator checks every corpus row before search, including held-out rows.
Selection statistics use only training decisions; held-out decision values
enter synthesis statistics after the selection commitment is computed.

## Build

```bash
commcanary build trace.et \
  --projection trace.projection.json \
  --policy regression-policy.json \
  --oracle-corpus physical-corpus.json \
  --active-ledger active-study-ledger.json \
  --application-evidence application-evidence.json \
  --physical-evidence physical-evidence.json \
  --runtime-budget 60s \
  --mode internal \
  --output serving.canary
```

`--runtime-budget` is an acknowledgement, not an override. It must equal the
value already bound into the policy. Omitting `--oracle-corpus` writes a
blocked measurement candidate and exits 1.

The output directory must not exist. Publication writes a fixed inventory,
then independent verification reloads the exact source, policy, projection,
corpus, search ledger, and raw evidence; reruns synthesis; and byte-compares
the canary, ledgers, certificate, leakage assessment, and manifest.

## Capture and active synthesis

An instrumented `torchrun` command can write the source trace, executable
Chakra ET, and source-bound projection in one invocation:

```bash
commcanary capture \
  --output trace.json \
  --chakra-output trace.et \
  --projection-output trace.projection.json \
  --workload-name serving-decode \
  -- torchrun application.py
```

The child must emit complete CommCanary collective shards, including all-rank
arrival evidence and rank-local GEMM recipes. `import-kineto` accepts the same
Chakra output options for supported complete profiles. Arbitrary uninstrumented
Python or opaque engine kernels are not inferred.

The active synthesis service starts from feature coverage, executes candidate
subgraphs against training perturbations, adds regions that remove severe
false negatives, and prunes passing candidates. It writes the exact selection
before its holdout callback can load held-out application evidence. The Rostam
site executor gives each missing candidate a fresh append-only SLURM attempt
and passes the frozen wrapper through `sbatch` stdin.

The bundle retains these scientific roles:

```text
application_ground_truth       actual application measurement only
trace_derived_reference        trace executor; never application ground truth
exact_materialization_control  conformance and measurement-floor control
reduced_decision_canary        selected physical subgraph
```

## Gate

Baseline and candidate observations identify the exact bundle and environment,
then provide a non-empty sample array for every policy metric:

```json
{
  "format": "commcanary.physical_gate_observation.v1",
  "bundle_id": "<64 lowercase hex characters>",
  "role": "baseline",
  "subject_sha256": "<64 lowercase hex characters>",
  "environment_sha256": "<64 lowercase hex characters>",
  "runner_oci_digest": "sha256:<64 lowercase hex characters>",
  "executable_sha256": "<64 lowercase hex characters>",
  "evidence_sha256": "<64 lowercase hex characters>",
  "samples": {
    "p99_latency_ms": [41.2, 41.0, 41.3],
    "throughput_per_second": [1012.0, 1007.0, 1010.0]
  }
}
```

Run the gate with:

```bash
commcanary gate serving.canary \
  --baseline baseline.json \
  --candidate candidate.json \
  --output gate.json \
  --html gate.html \
  --junit gate.xml \
  --sarif gate.sarif
```

The command re-verifies the bundle first. Both observations must name the
bundle's exact reduced executable and policy-bound runner. The baseline subject
must equal the baseline certified by the bundle, and the candidate must
identify a different subject and evidence commitment. A different environment
commitment or a non-qualified bundle is `incomparable`. A mandatory metric
beyond its threshold is `fail`. Otherwise the outcome is `pass`; advisory
regressions are retained as warnings. Exit 0 means pass and exit 1 means a
valid negative or incomparable result.

## Privacy modes

`internal` and `full_audit` retain the source ET, source projection, corpus,
selection ledgers, application allocations, and physical measurements. Their
inventories are intentionally identical; the mode names record the intended
audience.

`private_exchange` requires an Ed25519 private/public key pair. It signs the
exact manifest, omits the source ET, source projection, corpus, selection
ledger, and raw application and physical evidence, and retains their signed
content commitments. Verification requires the owner's independently supplied
public key. Signing uses the local OpenSSL command and refuses encrypted,
non-Ed25519, or mismatched keys. It needs OpenSSL 3.0 or newer on `PATH`. The
`openssl` that ships with macOS is LibreSSL, which has no Ed25519; signing
names that and stops rather than failing inside it.

The leakage score covers five declared inference categories: model family,
hidden dimension, parallelism degree, batch/token geometry, and topology. It
does not prove privacy. Opaque Chakra attributes require a separate owner
review, recorded as its own boolean rather than hidden inside the score.
