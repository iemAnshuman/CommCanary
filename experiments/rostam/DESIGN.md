# Rostam physical experiment — design

> **Evidence status (2026-07-11):** this document records the design and
> observations from a historical campaign. The complete raw attempt archive is
> absent from the repository, so its numeric results are not independently
> regenerable from this checkout. Treat the site facts below as prior
> observations to re-verify, and use the immutable manifest/attempt/completeness
> workflow for the next publication-grade run.

Goal: physical evidence for the paper's central claim. Does a minimized,
behavior-verified canary preserve the *decision* (configuration ranking and
regression verdict) that the full workload exhibits on real hardware — in a
setting where an isolated collective microbenchmark misleads?

## July 2026 reproducibility boundary and pre-cluster handoff

The measurements described later in this document are **historical reported
evidence** from earlier Rostam work. This repository does not contain the
complete frozen manifest, immutable attempt tree, submission ledger, and raw
archive needed to revalidate those measurements. The Phase 7 work does not
claim to have reproduced them. Any new paper/release claim must come from a new
manifest-bound campaign and pass the strict completeness gate in `analyze.py`.

The local, non-cluster preparation is now implemented:

- `configs.json` is a strict declarative catalog for site constraints, eight
  configurations, physical producer/result schemas, eight workload recipes,
  and four core/overlap/shared profiles. Catalog values are copied into an
  immutable campaign; the catalog bytes are also a hashed manifest input.
- `lib/campaign.py` creates stable cell ownership before submission.
  `lib/submission.py` validates the entire matrix and all bound inputs, freezes
  hashes every physical producer/adapter/wrapper used by the run, freezes an
  exact argv plan, and implements `--resume`, `--only-missing`,
  `--retry-failed`, and `--dry-run`. Submission is a separate action requiring
  `submit --plan ... --execute`; observed job IDs go into an append-only ledger.
- The five `.sbatch` files contain only wrapper identity and delegation through
  `lib/common.sh`. Partition, node, exclusivity, account, time, output path, and
  interpreter are supplied from the frozen plan. No wrapper searches a shared
  results directory.
- `lib/cell_entrypoint.py` owns one cell/attempt, validates the SLURM allocation,
  explicit dependency attempt IDs, inputs, runtime, operation/layout, and PARAM
  request/wait structure before launch, runs argv arrays without a shell, and
  writes one immutable terminal attempt plus a canonical `CellResult`.
- Distinct micro/full/PARAM/overlap/capture measurement schemas are committed
  under `schemas/`; the analyzer validates those physical envelopes before any
  aggregate can be generated. The three JSON-emitting runners also use strict,
  distinct raw stdout schema IDs: micro contains only dtype/message-size timing
  fields, full contains the real workload shape, and overlap contains only its
  completion timings. No runner invents placeholder fields for another
  producer's contract.
- `overlap_replay.py` strictly parses and validates the complete PARAM trace
  before importing torch or initializing `torch.distributed`. It aliases a
  process group to `WORLD` only after proving exact dense-world membership and
  rejects missing/duplicate requests, unknown waits, unsupported operations,
  invalid sizes/dtypes, and traces without one explicit wait per collective.
  The standalone preflight contract mirrors CommCanary's default resource
  policy: at most 64 MiB, JSON depth 64, two million total JSON items, and two
  million top-level PARAM entries. These limits are scanned before decoding;
  the validated decoded list is then reused without a normalized-tree copy.
- `setup.sh` is fail-closed. It has no editable install, loose dependency,
  build-tool upgrade, clone, or `sed` mutation. It accepts only a reviewed
  CommCanary wheel, complete `--require-hashes` locks, a verified PARAM commit
  and preimage, the committed patch hash, and the verified postimage.

PARAM patch evidence is reviewed locally. The review fetched only the exact
public commit `a437fcebd3add1aee66fba880f28cec9fd744589` into a read-only temporary
checkout; it did not run PARAM or its setup. The source archive is deliberately
defined as the uncompressed output of this Git-object operation, not as a
GitHub-generated download whose transport bytes are outside this contract:

```console
COMMIT=a437fcebd3add1aee66fba880f28cec9fd744589
git archive --format=tar --prefix=param-a437fcebd3add1aee66fba880f28cec9fd744589/ "$COMMIT" | shasum -a 256
```

Equivalently, the method is `git archive --format=tar
--prefix=param-a437fcebd3add1aee66fba880f28cec9fd744589/ COMMIT`. Two consecutive
Git 2.46.0 invocations produced the same 5,867,520-byte archive with SHA-256
`d509a84fa3db007ab99be343b01f678d593628cda270af2ad571b15a2c06a7eb`.
The target preimage is
`68dfa9362b66d47a1203f95cc0f1484397f7052def3e0e124f2e12e8fa912f8d`;
the ordinary-context patch is
`59bf7dff99faf3d187a11424a641a9b2f0d190cf58794da2064d5542dc0141fc`;
and its independently applied postimage is
`219c95f65814d5db66762b96aa8ec5b34b7da4ca928b58abaaa48651880dd23a`.
`git apply --check` succeeds against the clean pinned checkout without
`--unidiff-zero`.

No setup, scheduler, GPU, SSH, remote shell, or Rostam command was run while
building this layer. The sole network action was the read-only fetch of that
exact public PARAM object. The following target-observed values remain
deliberately unresolved; their contracts contain `null`/pending state rather
than invented hashes:

| Gate before any new experiment submission | Current state | Evidence to collect on Rostam |
|---|---|---|
| Python/platform resolver | Blocked: `constraints/environment-contract.json` is `pending-rostam-resolution` | Exact CPython ABI/platform tags, resolver argv/report, every wheel filename/size/SHA-256, two complete hash locks, and reviewed `pip freeze --all` hashes |
| CommCanary artifact | Blocked | Build the wheel from the exact manifest commit; record filename, repository commit, size, and SHA-256; verify both venvs install those bytes |
| PARAM source and patch | Reviewed locally: exact commit, deterministic source archive, target preimage/postimage, and contextual patch are hash-bound | No Rostam collection; `setup.sh` re-verifies the checkout commit and pre/postimages before and after applying the committed patch |
| Site/runtime fingerprint | Unconfirmed for a new campaign | Reconfirm module `python/3.12.3`, partition/account policy, `toranj0` topology/driver/CUDA visibility, torch version, and runtime `ncclGetVersion` for both environments |
| Overlap capture | Blocked by catalog readiness gates | Measure and review the bf16 GEMM calibration used by `--compute-fill-us-per-gemm`; replace the explicit pending calibration token in the per-config and shared capture recipes, mark those recipes ready, and bind the shared trace path/size/SHA-256 as `shared-param-trace` |
| New reproducible claims | Not available | Run a newly frozen campaign, preserve every terminal attempt, select exactly one attempt per cell, obtain a complete verdict, and regenerate aggregates/tables from that verdict |

The submit boundary is therefore clear: local static work is complete, while
`setup.sh` and the submission planner intentionally refuse to proceed until the
rows above marked blocked or unconfirmed have reviewed values. The `core`
catalog profile covers micro, full, trace build, and timestamp-paced PARAM
replay. Unresolved overlap work is excluded from `core`. The dedicated
`overlap` and `shared-capture` profiles already contain their complete profile,
import, compile, explicit-wait export, named-output, and replay dependency
graphs; they remain explicitly blocked only by the
`PENDING_ROSTAM_GEMM_CALIBRATION_US` token and readiness gate until that
target-observed value is reviewed. `shared-replay` requires an exact external
`shared-param-trace` input instead of discovering one by glob.

Hardware (actual): one Rostam `cuda-A100` node — 4× A100-PCIE-40GB,
tensor-parallel width 4, single node, all user-space, no root. Chosen over
the 8-GPU `DGX-A100` node (`anvil`) because that node is a single unit shared
with jenkins CI (persistent multi-hour jobs), whereas `cuda-A100` has idle
nodes. The PCIe interconnect (vs the DGX's NVSwitch) is a feature here, not a
compromise: PCIe exposes communication more than NVLink, so arrival skew and
overlap are less hidden — exactly the regime where an isolated microbenchmark
diverges from the real workload. GPUs are not SLURM-managed on Rostam (no
GRES); a node is claimed with `-N 1 --exclusive`. Optional later extension:
the `cuda` partition's 4× V100-SXM2 (NVLink) as a second interconnect point,
or a multi-node `cuda-A100` run over InfiniBand.

## Configurations under test (the things we rank)

Two axes, chosen because both have documented ranking crossovers in the
decode message-size regime (32–256 KB):

| Axis | Values | Mechanism |
|---|---|---|
| NCCL version | 2.19.3 vs 2.20.5 | separate venvs pinning `nvidia-nccl-cu12==<v>` under one identical `torch==2.4.1` binary, so the A/B isolates NCCL alone. torch 2.4.1 chosen on Rostam-verified evidence: it is the first torch whose exported Kineto traces carry the named collective args over NCCL (probe: 2.2.2 and 2.3.1 emit none, 2.4.1 emits all incl. Process Group Ranks); its native NCCL is 2.20.5 and 2.19.3 link-loads under it (maps + ncclGetVersion verified); torch 2.8 cannot load either version (`undefined symbol: ncclMemFree` / `ncclGroupSimulateEnd`). torch >= 2.4 emits nested frontend/backend record_param_comms pairs — the importer drops the inner duplicates (`skipped_nested_events`). This pair is the documented all_reduce regression window of NVIDIA/nccl#1222. |
| NCCL_ALGO × NCCL_PROTO | (Ring, Tree) × (LL, LL128, Simple) | env vars per run. LL/LL128/Simple crossovers sit exactly in the decode size regime. NVLS excluded: Hopper-only. |

Config set for ranking: 6 ALGO×PROTO combos on the newer NCCL, plus the two
versions at default ALGO/PROTO = 8 configurations. Pairwise rankings over
these 8 are the decision object.

Every cell is pinned to one node (`#SBATCH -w toranj0`): the `cuda-A100`
partition is heterogeneous (nasrin0 carries only 2 GPUs and a different
driver), and a single node removes cross-node variance from the rankings.

## Workloads (the four verdicts we compare)

1. **W-micro** — `microbench_tp8.py` under `torchrun --nproc_per_node=4`:
   back-to-back `dist.all_reduce` over the same 64/128/256 KB message sizes,
   no interleaved compute, no injected skew, CUDA-event timed. Its per-config
   latency table = the microbenchmark verdict. We deliberately use a pure
   `torch.distributed` loop rather than the `nccl-tests` C++ binary: it holds
   the NCCL build **identical** to W-full (nccl-tests would link its own NCCL,
   confounding the comparison), removes the CUDA-toolkit/`nvcc` build
   dependency (torch pip wheels bundle their runtime and need only the
   driver), and isolates workload structure — interleaved compute, arrival
   skew, operation order — as the sole variable separating W-micro from W-full.
2. **W-full** — `workload_tp8.py` under `torchrun --nproc_per_node=4`: a
   decode-like loop. Per token, per layer: a sharded GEMM sized to
   ~200–400 µs (realistic decode per-layer compute) followed by a bf16
   all_reduce alternating 64/128/256 KB, 32 layers × 256 tokens. Skew and
   overlap arise naturally from kernel jitter and the compute/comm stream
   structure — no injection in the primary condition. A secondary,
   clearly-labeled condition injects per-rank compute imbalance (±10% GEMM
   size by rank) to widen skew. Median per-token decode latency per config
   = the workload verdict.
3. **W-canary** — rank 0 of one W-full run records a `torch.profiler`
   Kineto trace → `commcanary import-kineto` → `compile
   --require-behavior-verification` (not `--behavior-search`: size-cycling
   makes every event its own signature group, so the search's candidates are
   byte-identical while costing ~30 min/cell — measured; the fail-closed
   behavioral gate is kept)
   → `export-param` → PARAM comms-replay (`--trace-type basic`,
   timestamp-paced) per config. Replay wall time and exposed-latency
   distribution per config = the canary verdict.
4. **W-baselines** — the same export path applied to `baseline --method
   stratified` and `--method random` traces, plus `reduce` (ddmin) output:
   do the negative controls reach the same verdict, or does verification
   earn its keep?
5. **W-canary-compute** — the canary exported with *compute fill*
   (`export-param --compute-fill-us-per-gemm`): inter-collective gaps become
   PARAM gemm entries (bf16, per-gemm duration CUDA-event-calibrated on the
   node), replayed WITHOUT `--use-timestamp`. Motivated by the first sweep
   (Jul 2026): comm-only replay agreed with W-full on only 60.7% of pairs —
   no better than the microbenchmark's 64.3% — because idle-sleep gaps cannot
   reproduce compute/communication interference (tree-ll: micro rank 6,
   canary rank 2, full rank 8 at +45%). Hypothesis: interference-bearing
   replay recovers the workload ranking.
6. **W-baseline-stratified-compute** — compute fill applied to the
   timing-destroying stratified baseline, as the control: if fill rescues
   this too, the gemms do all the work; if it rescues only the faithful
   canary, fidelity and interference are jointly necessary.
7. **W-canary-overlap / W-baseline-stratified-overlap** — sweep 3's
   hypothesis. Sweep 2 falsified serialized compute fill: every config got
   uniformly ~5-10% faster (GEMMs ramped boost clocks — thermal state, not
   contention state) and tree-ll stayed top-3 (57.1% agreement, below even
   comm-only). Diagnosis: PARAM replays each entry synchronously, so compute
   next to a collective is still, from the collective's view, an idle GPU.
   The overlap export (`--overlap-structure`) issues collectives async with
   explicit waits placed after the NEXT gap's GEMMs — collective k executes
   while gap k+1's compute occupies the SMs. PARAM's replayer is hardwired
   blocking (`self.is_blocking = True`), so these variants run under
   `overlap_replay.py`, a ~180-line overlap-aware reference replayer
   consuming the same trace format (locally verified end to end on gloo).
   Latency = issue-to-completion via CUDA events, one sync per pass so host
   run-ahead is preserved. Same stratified control logic applies.

## Measurement discipline

- ≥5 repetitions per (workload, config) cell, interleaved run order
  (config round-robin, not blocks) to absorb thermal/clock drift; report
  median and IQR. We cannot pin clocks without root — repetitions and
  interleaving are the mitigation, and this is stated as a limitation.
- One warmup iteration per process before timing.
- Every cell emits one JSON result file: config, workload, rep, timing
  distribution, environment fingerprint (NCCL version string, torch
  version, hostname, SLURM job id, GPU clocks via `nvidia-smi -q`).

## Analysis (`analyze.py`)

- Per workload: rank the 8 configs by median latency; all 28 pairwise
  relations with a tie tolerance = max(IQR_i, IQR_j) — a pair is a tie if
  medians differ by less than either config's spread.
- Primary result: pairwise agreement (and Kendall tau) of W-micro vs
  W-full, and W-canary vs W-full. Claim shape: canary agrees with workload
  where microbenchmark disagrees.
- Regression 2×2: for the NCCL version pair specifically — does each
  workload flag 2.20.5 as a regression vs 2.19.3 (using commcanary
  compare's dual thresholds on the canary path)?
- Cost table: wall time and bytes for full workload vs canary replay vs
  microbenchmark; canary artifact bytes vs raw trace bytes.

## Success and honest-failure criteria

- **Success:** ≥1 config pair where W-micro and W-full disagree on order
  (outside both tie tolerances) AND W-canary matches W-full; plus the 2×2
  showing the canary catches any version regression the workload shows.
- **Honest failure:** no inversion found → the paper reports the agreement
  study (how well canary rankings track workload rankings vs baselines)
  and says so plainly. Per RESEARCH_SPEC kill conditions, if the stratified
  baseline tracks the workload as well as the verified canary does, that is
  reported too.

## Historical kit layout (superseded)

The block below describes the legacy campaign scripts whose observations are
reported earlier. It is retained as historical context; the current
manifest-bound implementation and handoff follow it.

```
experiments/rostam/
  DESIGN.md            this file
  setup.sh             venvs (nccl-2.19.3, nccl-2.20.5), PARAM checkout/patches
  workload_tp8.py      instrumented decode loop (torchrun), --profile emits Kineto JSON
  configs.json         the 8 configurations (env vars + venv selector)
  run_micro.sbatch     W-micro sweep
  run_full.sbatch      W-full sweep
  run_canary.sbatch    trace → canary → PARAM / overlap replay sweep
  run_matrix.sh        submits/monitors the interleaved matrix
  analyze.py           rankings, agreement, 2x2, cost table → results.md + results.json
  results/             one JSON per cell (gitignored except results.md)
```

SLURM specifics: `-p cuda-A100 -w toranj0 -N 1 --exclusive`; Rostam does
not expose GPU GRES, so no `--gres` is requested. The node is 4×
A100-PCIE-40GB, tensor-parallel width is 4, and launches use `torchrun
--standalone --nproc_per_node=4`. No `cuda` module is required because the
torch pip wheels carry the user-space runtime. All scripts must be
re-runnable and must never require root. PARAM is used through the
`export-param` path with the legacy `train/comms/pt/commsTraceReplay.py`
(`--trace-type basic`, `--use-one-trace`) for synchronous replay, and
`overlap_replay.py` is the overlap-aware reference replayer for the physical
NCCL overlap variants.

## Current kit layout and ownership

The v2 catalog is declarative. It owns the expected site, eight configurations,
eight workload recipes, exact argument vectors, dependencies, timeouts,
producer/result schema IDs, and four named profiles. It does not execute a
workload while loading or planning.

```text
experiments/rostam/
  configs.json                     strict catalog (expected inputs only)
  constraints/
    direct-*.txt                   reviewed direct requirements
    environment-contract.json      unresolved target resolver/lock evidence
    locks/README.md                 lock collection protocol; no fake locks
  patches/
    param-patch-contract.json       upstream/preimage/postimage contract
    param-use-triton-default.patch  committed patch; no unverified sed
  schemas/                         distinct physical producer result schemas
  harness/                         immutable manifest, attempts, selection,
                                   completeness, bounded local runner
  analysis/                        completeness-gated aggregate pipeline
  lib/
    campaign.py                    bind hashed inputs and freeze run manifest
    submission.py                  immutable plan and explicit submission ledger
    cell_entrypoint.py             one terminal record per physical attempt
    physical_results.py            fail-early PARAM/result adapters
    environment_contract.py        setup/patch/lock/wheel verification
    common.sh                      common body for thin SLURM wrappers
  run_*.sbatch                     wrapper identity only; no embedded scaffolding
  run_matrix.sh                    plans by default; submission is explicit
  setup.sh                         fail-closed install after reviewed contracts
  analyze.py                       validated aggregate regeneration CLI
```

Observed scheduler/job/node/account/partition values never mutate the frozen
run manifest. They enter the append-only submission ledger and immutable cell
attempts. Every retry keeps its predecessor; exactly one selected terminal
attempt feeds completeness and analysis. The analyzer rejects missing,
duplicate, stale, failed, or unexpected cells unless incomplete analysis is
explicit, in which case every output is prominently marked incomplete.

## Trusted analysis and publication contract

Publication analysis follows one hash-bound chain. The campaign manifest is
frozen before execution and binds the repository, catalog, inputs, workload
matrix, configurations, runtime contract, and exact command plan. Execution
then creates immutable terminal attempts without changing that manifest. An
immutable selection names exactly one terminal attempt for every expected
cell, and a persisted completeness verdict binds that selection and is checked
again immediately before outputs are written. Physical measurements are also
validated against the selected attempt, manifest workload and runtime fields,
declared dependencies, and the exact trace or capture artifact that supplied
the replay. Rankings, pairwise relations, Kendall agreement, regression 2x2,
and cost claims are generated only from this selected evidence. If any joined
campaign is incomplete, those claims are withheld even when
`--allow-incomplete` was explicitly used to produce a diagnostic aggregate.

The raw archive descriptor is deliberately a **post-run** record, not a
campaign input. Each entry in its `campaigns` array binds the exact `run_id`,
`campaign_id`, repository commit, manifest SHA-256, selection ID, selection
SHA-256, and verdict SHA-256 used by the analysis. The descriptor additionally
records an immutable-capable URI label (`https`, `s3`, `gs`, `doi`, `urn`, or
`ipfs`) and the archive's SHA-256 and byte size. Trusted regeneration accepts
the descriptor and local archive bytes only as a pair, verifies the bounded
descriptor against every joined campaign identity, and verifies the archive
bytes against the declared hash and size. Local paths and a mutable `file` or
plain `http` URI are not publication identities.

`--join-evidence` may combine core and shared campaigns without weakening that
chain. Joined manifests must be distinct and must agree on repository identity
and expected-site contract; configurations with the same ID must be identical,
and inputs with the same ID must have identical hashes and sizes. Completeness,
attempt accounting, selected-cell provenance, and claim generation cover the
whole trusted join rather than treating a later dataset as an unverified
append. Public JSON, CSV, and Markdown retain the evidence hashes needed for
verification but exclude hostnames, job IDs, scheduler/account details,
physical execution commands, executor metadata, and measurement artifact or
workspace paths. The exact user-supplied regeneration command remains part of
the publication provenance.

The historical glob-based analyzer is outside this trust boundary. It runs
only after the operator supplies `--unsafe-legacy-glob-analysis`; its JSON and
Markdown are watermarked as unsafe, unverified, and unsuitable for publication.

## Pre-cluster handoff: deliberately unresolved evidence

The repository-local work stops here. The following values require observation
or artifact collection on Rostam and are intentionally `null`, pending, or
absent rather than guessed from the historical campaign:

| Evidence to collect/review | Current fail-closed representation |
|---|---|
| SLURM account and confirmation of partition, node constraint, exclusivity, GPU count/model, GRES policy, topology, and binding | `configs.json` expected site; `site.account` is `null` and all historical values must be reconfirmed |
| Python module, implementation, exact version, platform tags, and ABI | `constraints/environment-contract.json` with empty platform/ABI evidence |
| Resolver version/commands/report, every wheel filename/size/SHA-256, complete hashed locks, and post-install freezes for both NCCL environments | contract status `pending-rostam-resolution`; lock files intentionally absent |
| Exact clean CommCanary commit, built wheel filename/SHA-256, source archive hash, and dirty-patch hash if applicable | unresolved `commcanary_wheel` plus immutable campaign repository/input fields |
| Loaded driver/CUDA/PyTorch/NCCL runtime mapping and `ncclGetVersion` fingerprint | expected runtime in the catalog; observed values belong in attempts |
| Warmup/runtime observations, job/node IDs, clocks, stdout/stderr, failures, exclusions, and retry reasons | append-only submission ledger and terminal cell attempts |
| On-node GEMM calibration and the resulting overlap-export artifact hash | `overlap-trace-build` and `shared-trace-capture` have complete pipelines but retain `PENDING_ROSTAM_GEMM_CALIBRATION_US` plus a readiness gate |

The PARAM archive, preimage, contextual patch, and postimage are already
reviewed in `patches/param-patch-contract.json`; they are not site-observed
inputs. `environment_contract verify-ready` still fails until all environment,
lock, freeze, and wheel evidence is reviewed. `setup.sh` calls that verifier
before it creates a venv or modifies PARAM. The `core` profile excludes overlap
work; the dedicated `overlap` and `shared-capture` profiles stay blocked until
their calibration token and readiness state are replaced with the reviewed
target value. The separate `shared-replay` profile stays blocked until the
selected capture artifact is bound as its fixed input.

## Authorized execution sequence and exact boundary

After an authorized operator has collected and reviewed the target evidence:

1. Complete the environment contract, generate the full
   `--require-hashes` locks, and record the clean CommCanary wheel.
2. Record the reviewed on-node GEMM duration in both overlap export commands,
   replace their pending readiness state with `ready`, and review the resulting
   catalog hash. Do not reuse the historical calibration without measuring it.
3. Run the static contract audit and `verify-ready`; only then run `setup.sh`.
4. Use `lib.campaign` to bind the reviewed catalog, wheel, environment lock,
   PARAM contract, source commit/archive, repetitions, and requested profile
   into `results/<run-id>/run_manifest.json`.
5. Run `run_matrix.sh` (or `submission plan`) to freeze and review a unique
   ownership plan. `--resume`, `--only-missing`, `--retry-failed`, and
   `--dry-run` affect the plan but never submit anything.
6. **Cluster mutation begins only at** `submission submit --plan PLAN
   --execute`. That is the first command permitted to invoke `sbatch`; it
   records exact argv, outcome, job ID, stdout, and stderr in the append-only
   ledger.
7. After all terminal attempts exist, persist the immutable selection and
   require a fail-closed completeness verdict. Build the raw archive and its
   post-run descriptor bound to the exact manifest, selection, and verdict
   identities described above; then regenerate aggregate JSON/CSV/Markdown and
   paper fragments through `analyze.py` (joining additional trusted campaign
   evidence explicitly when required).

No command in the completed local verification for this engineering plan ran
step 3's setup, step 6, `sbatch`, `srun`, `torchrun`, a GPU probe, or a Rostam
login. The historical numeric results above remain reported observations; a
new publication-grade claim begins with this manifest-bound sequence.
