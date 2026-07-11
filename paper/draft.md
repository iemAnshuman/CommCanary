# Idle Gaps Are Not Compute: What a Communication Trace Must Preserve to Rank GPU Collective Configurations

**Anshuman Agrawal** — Louisiana State University *(affiliation to confirm)*

*Draft v0.1 — July 2026. The numbers below report a historical Rostam
campaign. Its complete raw attempt archive is not tracked in this checkout, so
the current open artifact cannot yet regenerate them. Publication requires a
new manifest-bound, hash-verified campaign and byte-for-byte table
regeneration.*

## Abstract

Operators tune GPU collective-communication stacks — NCCL versions,
algorithms, protocols — using isolated microbenchmarks, and production
incident reports show these microbenchmarks can pass while real workloads
regress. The assumed fix is trace replay: record the workload's
communication and replay it faithfully. We test that assumption physically.
On a 4×A100 node we rank eight NCCL configurations under a tensor-parallel
decode-style workload and compare the ranking against nine cheaper proxies:
a microbenchmark, faithful communication-trace replay, replay with
serialized compute insertion, replay with reconstructed compute/
communication *concurrency*, and sampling baselines — each with controls.
The results decompose replay fidelity into its load-bearing parts. The
microbenchmark inverts rankings (64.3% pairwise agreement with the
workload; its top-ranked config is mid-pack in situ, and a config it calls
acceptable is catastrophic, +45%). Faithful communication-only replay is
*equally blind* (60.7%): timestamp-paced idle gaps cannot reproduce
interference. Inserting the trace's compute serially makes agreement worse
(50.0%) while uniformly *speeding up* every collective — the GEMMs ramp
boost clocks without contending, changing thermal state rather than
contention state. Only replay that reconstructs concurrency — collectives
in flight while compute occupies the SMs — beats the microbenchmark
(67.9%, Kendall τ 0.54), and it is the first proxy to expose the
catastrophic config's pathology, reproducing not just its rank but its
instability signature. A control inverts the final assumption: a
stratified-sampling baseline that destroys exact burst timing performs
equivalently under the same concurrency-bearing replay (71.4%, within one
tie-boundary pair). On this workload, per-event timing precision — the
quantity trace-compression fidelity metrics optimize — contributed nothing;
operation/size/order structure plus a concurrency-bearing replayer carried
all of it. The operational form works: a single trace, captured once and
replayed under all eight configurations, ranks them at 71.4% — as well as
per-configuration capture and above the microbenchmark — so the finding is
a cheap pre-rollout tuner, not only a diagnostic principle. We release the
full pipeline, including a PyTorch-profiler
importer, a PARAM-format exporter with compute-fill and overlap-structure
modes, and a 180-line overlap-aware reference replayer built because
PARAM's own replayer is architecturally unable to express overlap.

## 1 Introduction

Distributed LLM serving splits a model across GPUs that synchronize
constantly through collective operations. How those collectives are
configured — library version, algorithm (ring vs. tree), wire protocol
(LL, LL128, Simple) — moves end-to-end latency by tens of percent, so
operators tune. The industry-standard instrument is the isolated
microbenchmark (`nccl-tests` and equivalents): run collectives
back-to-back on idle GPUs, read the latency table, pick the winner.

Production evidence says this instrument misleads. NVIDIA's issue tracker
records upgrades where single-node microbenchmarks looked healthy while
real training ran ~20% slower (nccl#513); operators document validation
suites that pass on degraded fabrics. The intuitive diagnosis is that
microbenchmarks lack *workload context* — the ordering, timing, and
co-execution structure of real communication — and the intuitive remedy is
trace replay: capture the workload's communication stream and replay it
faithfully against candidate configurations. A substantial tooling
ecosystem (PARAM, Chakra, and their kin) embodies this remedy.

This paper measures whether the remedy works, and more precisely, *which
parts of a communication trace actually carry* the information needed to
rank configurations. We stage the question on real hardware as a chain of
controlled eliminations, each adding one ingredient:

1. **No structure** (microbenchmark): 64.3% pairwise ranking agreement
   with the real workload — and concretely dangerous: its rank-1 config is
   mid-pack in situ, while a config it ranks 6-of-8 is the workload's
   worst by a large margin (+45%, with 25× the run-to-run variance of its
   neighbors).
2. **Full communication structure, no concurrency** (faithful trace
   replay, timestamp-paced): 60.7% — *no better than the microbenchmark*.
   The catastrophic config ranks second. Idle gaps are not compute.
3. **Serialized compute** (the trace's inter-collective gaps filled with
   calibrated GEMMs, executed sequentially): 50.0% — worse, with a
   diagnosable mechanism: every collective got uniformly ~5–10% *faster*
   because the GEMMs ramped GPU boost clocks. Sequential compute changes
   the chip's thermal/clock state, not its contention state.
4. **Concurrency** (collectives issued asynchronously, in flight while
   the gap's GEMMs occupy the SMs): 67.9%, τ 0.54 — the first proxy to
   beat the microbenchmark, the first to demote the catastrophic config,
   and the first to reproduce that config's *instability signature*
   (5.5–16× the IQR of every other config, echoing the workload's 25×).
5. **The control**: the same concurrency-bearing replay applied to a
   stratified-sampling baseline — which preserves operation/size/order
   structure but destroys exact per-event timing — performs equivalently
   (71.4%; one tie-boundary pair from the faithful trace). Exact timing
   fidelity, the property trace-compression research measures itself
   against, added nothing here.

The decomposition yields a design rule we have not found stated, let alone
measured, in prior work: **for configuration ranking, the trace must
preserve operation/size/order structure; the replayer must supply
concurrency; per-event timing precision is (at least on this workload)
irrelevant.** Communication-trace replay tools inherit the microbenchmark's
blindness not because their traces are unfaithful but because their
*replayers are synchronous* — we show PARAM's replayer hardwires blocking
execution and cannot express overlap regardless of trace content, and we
contribute a 180-line overlap-aware reference replayer for its trace
format.

Contributions:

- **C1 — Physical ranking inversions, characterized** (§5.1): reproducible
  microbenchmark-vs-workload inversions on commodity A100 hardware,
  including a config whose failure mode is invisible to every
  synchronous proxy.
- **C2 — A measured decomposition of replay fidelity** (§5.2–5.5): the
  five-step elimination chain above, each step with 5 repetitions,
  dispersion, and controls, isolating concurrency as the load-bearing
  ingredient and falsifying per-event timing fidelity as one.
- **C3 — Tooling findings and instruments** (§4): PARAM's replayer cannot
  express overlap (hardwired blocking); PyTorch's profiler emits usable
  collective metadata only from torch 2.4 (2.2/2.3 record the events but
  export no named args); an open importer/exporter/replayer chain that
  turns one profiler trace into all nine verdicts.
- **C4 — An honest negative** (§5.5): under concurrency-bearing replay,
  a timing-destroying stratified baseline ties the faithful trace,
  triggering our pre-registered kill condition for the
  timing-fidelity-matters sub-claim and bounding what "faithful replay"
  is worth.
- **C5 — The operational form works** (§5.6): a single trace captured once
  and replayed under all eight configurations ranks them at 71.4% — as well
  as per-configuration capture and above the microbenchmark — establishing
  the finding as a cheap pre-rollout tuner (one capture, N replays), not
  only a principle for replay-tool builders.

## 2 Background and related work

**Microbenchmark practice.** `nccl-tests`-style tools measure collectives
back-to-back on otherwise idle GPUs. They are the de facto acceptance and
tuning instrument for GPU fleets; their unrepresentativeness under real
workloads is documented anecdotally in vendor issue trackers (e.g.,
nccl#513: single-node tests "good," training 20% slower) and, recently,
systematically: AICB/SimAI (NSDI'25) built workload-aligned communication
benchmarks precisely because isolated ones mislead. Our contribution is
orthogonal: not another benchmark, but a measurement of *which trace and
replay properties* close the gap.

**Trace replay.** PARAM (Meta) replays recorded collective streams on real
hardware; Mystique (ISCA'23) generates production benchmarks from
execution traces; ScalaTrace (JPDC'08) pioneered compressed,
deterministically replayable MPI communication traces; ATLAHS (SC'25)
replays NCCL traces in simulation. All treat the communication stream as
the object of fidelity. None, to our knowledge, measures configuration-
ranking preservation as the acceptance criterion, and none reconstructs
compute/communication concurrency from a communication-only trace — we
show the reference implementation (PARAM) is architecturally serialized
(§4.4).

**Simulators.** ASTRA-sim, SimAI, and kin rank configurations in
simulation from traces. Their fidelity question is the model's; ours is
the *trace's and replay architecture's*, on physical silicon.

**Workload reduction with validity criteria.** SimPoint and the SPEC
subsetting literature select representative program slices validated
post-hoc against full-run metrics, including cross-machine rank
preservation. Our stratified baseline imports exactly that spirit as a
control — and the control's success under overlap replay (tie with the
full trace) is a SimPoint-flavored result for communication traces:
structure-preserving sampling suffices *given the right replayer*.

**The CommCanary pipeline.** This study is built on CommCanary, an
open-source tool for distilling communication traces into verified
regression canaries with fail-closed behavioral verification. Here it
serves as the instrument: its importer, exporter, and baseline generators
produce every artifact in the elimination chain from a single profiler
trace. Two of its prior simulator-side findings foreshadowed this paper's
physical results: a decision-only ddmin reducer collapses traces to
single events while "preserving rankings" (degenerate reduction, §5.6),
and adjacency-based compression achieves nothing on size-cycling
workloads (its canary is honestly reported as 2.9× *larger* than the raw
trace, §5.7).

## 3 Experimental design

**Hardware.** One node (`toranj0`, LSU Rostam cluster): 4× NVIDIA
A100-PCIE-40GB, driver 580.82.07, no NVLink — PCIe deliberately, because
it exposes communication rather than hiding it. Every cell of every sweep
runs on this one node (`SLURM -w`), removing cross-node variance. GPUs
have default application clocks; we cannot pin clocks without root, so we
use ≥5 repetitions per cell with config-interleaved (round-robin) run
order to absorb thermal drift, reporting median and IQR.

**Configurations (the decision object).** Eight: NCCL 2.19.3 and 2.20.5
at default settings (one identical `torch==2.4.1` binary; the pinned
`nvidia-nccl-cu12` wheel is authoritative and runtime-verified via
`/proc/self/maps` + `ncclGetVersion` in every result's fingerprint), plus
NCCL 2.20.5 under forced `NCCL_ALGO ∈ {Ring, Tree}` × `NCCL_PROTO ∈
{LL, LL128, Simple}`. The 28 pairwise order relations among these eight
are what every proxy is scored on.

**Ground truth (W-full).** A decode-style tensor-parallel loop: per token,
per layer, a bf16 GEMM (~300 µs, calibrated) then a bf16 `all_reduce`
cycling 64/128/256 KB messages; 32 layers × 256 tokens, natural skew and
overlap (no injection). Median per-token latency per config is the
workload verdict; 8,192 collectives per run.

**The proxies (nine verdicts).**

| Verdict | Carries | Replayer |
|---|---|---|
| W-micro | nothing (back-to-back all_reduce, same sizes) | torch.distributed loop |
| W-canary | full comm structure + timestamps | PARAM (timestamp-paced) |
| W-canary-compute | + serialized GEMM fill in gaps | PARAM (blocking) |
| W-canary-overlap | + concurrency (async issue, wait after next gap's GEMMs) | reference replayer |
| W-baseline-stratified | structure, medoid timing (timing destroyed) | PARAM |
| W-baseline-stratified-compute | + serialized GEMMs | PARAM |
| W-baseline-stratified-overlap | + concurrency (control) | reference replayer |
| W-baseline-random | one sampled event tiled | PARAM |
| W-baseline-ddmin | decision-preserving reduction output | PARAM |

All canary-family artifacts derive from a Kineto profiler trace of W-full
(torch 2.4.1; §4.1), through CommCanary's import → verified-compile → export
pipeline. The GEMM fill quantum is CUDA-event-calibrated per job on the node
(44.03 µs per bf16 1024³ multiply — measured, not assumed).

**Two trace-capture regimes, and why both.** In the primary sweeps, each
config's canary is compiled from a profiler trace of W-full *run under that
same config*. This is the honest way to ask the *decomposition* question —
"given a faithful trace, which replay semantics preserve the ranking?" —
because it holds trace fidelity maximal and varies only replay semantics. It
is **not** the operational "capture once, replay across candidates" scenario:
if you have already run W-full under every config to obtain each trace, the
replay has saved you nothing. We therefore add a second regime (§5.6): a
**single** canary, captured once under a reference config, replayed
(overlap grammar) across all eight — same trace, same gaps, only the NCCL
execution varies. The decomposition sweeps answer "what must a trace and
replayer preserve"; the shared-trace sweep answers "does one captured trace
rank candidates." We report both and are explicit about which claim each
supports.

**Metric.** For each proxy, all 28 pairwise relations
(better/worse/tie, tie iff |Δmedian| < max(IQR_i, IQR_j)) are compared
with W-full's; we report exact-agreement percentage and Kendall τ.
Comparing tie-tolerant *relations* rather than raw orderings is
deliberate: it refuses credit for coin-flip orderings of statistically
indistinguishable configs.

**Fixed goalposts.** W-micro and W-full were measured once (sweep 1) and
never re-run; each replay innovation was evaluated against frozen ground
truth, pre-registered in the experiment's design document before its
sweep ran.

## 4 The instrument: from one profiler trace to nine verdicts

*(Condensed; the repository's `experiments/rostam/DESIGN.md` and setup
scripts encode every detail with the failure that motivated it.)*

### 4.1 Importing reality: the profiler metadata cliff

CommCanary's `import-kineto` reads collective metadata (`Collective
name`, element counts, dtype, process-group ranks) from PyTorch profiler
traces. A finding practitioners should know: **this metadata reaches the
exported Chrome trace only from torch 2.4**. torch 2.2 and 2.3 create the
`record_param_comms` events and even contain the full serialization
plumbing (`saveNcclMeta` → `extra_meta_` → `addMetadata`), but the
debug-info payload never arrives; we verified empirically on-cluster that
2.2.2 and 2.3.1 export zero named args while 2.4.1 exports all of them.
torch ≥ 2.4 additionally emits *nested* frontend/backend event pairs,
both carrying sizes; the importer deduplicates by time-interval
containment. The import is honestly scoped: single-rank, observational,
no invented cross-rank skew.

### 4.2 Export: three replay grammars from one canary

The exporter emits PARAM "basic" traces in three modes. *Timestamped*:
entries carry cumulative `startTime_ns`; gaps are idle. *Compute-fill*:
each gap becomes `⌈gap/quantum⌉` GEMM entries (`{"compute": "gemm"}`),
executed where the workload computed. *Overlap-structure*: collectives
are emitted for asynchronous issue, and each one's explicit `wait` entry
is placed *after the following gap's GEMMs* — collective *k* is in
flight while gap *k+1*'s compute occupies the SMs, which is the
workload's actual shape. Process groups are declared via explicit `init`
entries (required by PARAM's parser; undocumented outside its source).

### 4.3 Replaying with PARAM — and PARAM's ceiling

The serialized modes run under Meta's PARAM `commsTraceReplay` (pinned to
its last torch-2.4-compatible, internally consistent commit; the pin
archaeology — three distinct internal drift bugs — is documented in the
kit). The finding that shaped this paper: **PARAM's replayer hardwires
`self.is_blocking = True`**; it synchronizes after every operation to time
it. No trace content can make it express concurrency. This is not a bug —
it is a measurement-architecture choice — but it means the entire
PARAM-format ecosystem, as executed by its reference replayer, is
structurally confined to steps 1–3 of our elimination chain.

### 4.4 The overlap-aware reference replayer

`overlap_replay.py` (~180 lines, torch.distributed only) consumes the
same trace format and honors the overlap grammar: `async_op=True` issue,
GEMMs between issue and wait, per-collective issue-to-completion latency
via CUDA events, collected after a *single* synchronize per pass so host
run-ahead — part of the concurrency being measured — is never blocked.
Serialized replayers degrade the overlap grammar gracefully (waits become
immediate); the two grammars are one artifact. The reference replayer
implements the `all_reduce` path this study exercises; it is a research
instrument for these experiments, not a general-purpose replacement for
PARAM across arbitrary collective workloads (reduce_scatter, all_to_all,
point-to-point would each need adding).

## 5 Results

Table 1 consolidates the campaign: what each proxy's trace and replayer
carry, its pairwise ranking agreement with the real workload, Kendall τ, and
where it places tree-ll — the config whose in-situ pathology (workload rank
8, +45%, 25× IQR) is the campaign's exhibit.

**Table 1 — Proxy fidelity decomposition (8 configs, 28 pairwise relations,
5 reps each; W-full is ground truth).**

| Proxy | Trace carries | Replayer | Agreement | τ | tree-ll rank |
|---|---|---|---:|---:|:--:|
| W-micro | nothing | back-to-back loop | 64.3% | 0.49 | 6 |
| W-canary | full timing | PARAM (timestamp) | 60.7% | 0.36 | 2 |
| W-canary-compute | + serialized GEMMs | PARAM (blocking) | 50.0% | 0.33 | 4 |
| **W-canary-overlap** | **+ concurrency** | **reference (async)** | **67.9%** | **0.54** | **7** |
| W-baseline-stratified | structure only | PARAM | 60.7% | 0.36 | 2 |
| W-baseline-stratified-compute | + serialized GEMMs | PARAM | 57.1% | 0.32 | 2 |
| **W-canary-shared-overlap** | **structure + concurrency (1 trace, all configs)** | **reference (async)** | **71.4%** | **0.55** | **7** |
| W-baseline-stratified-overlap | + concurrency | reference (async) | 71.4% | 0.55 | 7 |
| W-baseline-random | one event tiled | PARAM | 42.9% | 0.58* | 7 |
| W-baseline-ddmin | 1-event reduction | PARAM | 46.4% | 0.53* | 4 |
| *W-full (ground truth)* | — | real workload | — | — | *8* |

*Random and ddmin τ are inflated by degenerate orderings of near-identical
medians; their low agreement is the honest signal (see §5.7). The two
overlap rows are the only proxies that place tree-ll near the bottom and
clear the microbenchmark on agreement — and they are statistically
indistinguishable from each other (§5.5), the paper's central negative.*

### 5.1 The microbenchmark inverts rankings (and how)

| config | W-micro rank (µs) | W-full rank (µs/token) |
|---|---|---|
| ring-ll | **1** (107.5) | 5 (8,466) |
| ring-ll128 | 2 (110.6) | 3 (8,258) |
| 2.20.5-default | 3 (132.1) | 2 (8,246) |
| ring-simple | 4 (132.1) | 4 (8,453) |
| 2.19.3-default | 5 (133.1) | **1** (8,238) |
| tree-ll | 6 (150.5) | **8 (11,969, IQR 578)** |
| tree-ll128 | 7 (157.7) | 6 (9,006) |
| tree-simple | 8 (203.8) | 7 (9,768) |

Agreement 64.3% (10/28 pairs inverted). Two operationally meaningful
failures: the config the microbenchmark would deploy (ring-ll, 19% ahead
in isolation) is 2.8% *slower* than the in-situ winner; and tree-ll,
ranked a tolerable 6th, is the workload's disaster — 45% slower than the
winner, 23% slower than tree-simple, with 25× the IQR of neighboring
configs. The workload's preferred configs are the *defaults* (NCCL's
internal tuner), which no forced setting beats in situ.

### 5.2 Faithful communication replay is equally blind

W-canary (full structure and timestamps, PARAM-replayed): 60.7%
agreement. It ranks tree-ll **second**. The mechanism is definitional:
timestamp pacing renders gaps as idle sleeps; from the collective's
perspective an idle GPU is an idle GPU, and LL's pathology needs busy SMs
to express. The trace was faithful; the replay semantics discarded the
property that mattered.

### 5.3 Serialized compute makes it worse, diagnosably

W-canary-compute (gaps filled with calibrated GEMMs, executed
sequentially by PARAM): 50.0% agreement, tree-ll still top-4 — and every
config's collectives got uniformly ~5–10% *faster* than under idle-gap
replay. Sequential GEMMs cannot contend with collectives; what they can
do is ramp boost clocks (these GPUs idle at 210 MHz and boost to 1410).
The insertion changed the chip's clock state, not its contention state —
a warning for any replay methodology that adds compute "for realism"
without concurrency.

### 5.4 Concurrency is the ingredient

W-canary-overlap: **67.9% agreement, τ 0.540** — above the
microbenchmark on both measures, the only proxy family to get there.
tree-ll falls to rank 7, below tree-ll128, matching the workload's
ordering of the tree family. Most tellingly, tree-ll is the *only*
config whose overlap-replay IQR blows out (8.2 µs against ≤1.5 µs for
all others — 5.5–16×), reproducing in miniature the workload's
instability signature (578 µs against ~23–68 µs, 25×). The proxy doesn't
just rank the pathology; it *exhibits* it.

### 5.5 The control: timing fidelity bought nothing

W-baseline-stratified-overlap — the stratified-sampling baseline
(operation/size/order preserved; exact per-event timing replaced by
per-signature medoids) under the same overlap replayer — agrees at
**71.4%**, one tie-boundary pair from the faithful trace's 67.9%.
Statistically, they tie. Under the serialized replayers the same
baseline also tied its faithful counterpart (57.1% vs 60.7%;
compute-filled 57.1% vs 50.0%). Across all three replay semantics, exact
burst timing never separated from structure-preserving sampling.

We pre-registered this as a kill condition and report it as one: **on
this workload, per-event timing precision is not what a communication
trace needs to preserve for configuration ranking.** What the elimination
chain leaves standing is the pair (structure, concurrency): destroy
structure (random tiling: 42.9%; single-event ddmin: 46.4%) and no
replayer saves you; supply structure without concurrency and you match a
microbenchmark at best; supply both and you beat it.

### 5.6 One captured trace, many candidates: the operational test

The decomposition sweeps (§5.1–5.5) profile each config's own W-full run, so
they answer "which replay semantics preserve the ranking," not "can one
captured trace rank candidates." The latter is the operational question, and
we test it directly: capture a single canary under a reference config
(2.20.5-default), replay it (overlap grammar) across all eight configs'
NCCL settings — identical trace, identical gaps, only the collective
execution varying (`capture_shared_trace.sbatch` + `run_shared.sbatch`).

The result holds: **W-canary-shared-overlap agrees with the workload on
71.4% of pairs (τ 0.550), matching the per-config overlap canary's 67.9%
(one pairwise relation apart — statistically indistinguishable) and
exceeding the microbenchmark's 64.3%.** A single trace, captured once under
2.20.5-default and replayed under all eight configurations' NCCL settings,
ranks them as well as traces captured per-config. Critically, it reproduces
the exhibit: tree-ll falls to rank 7 (below tree-ll128, matching the
workload's tree ordering), and it again carries the largest run-to-run
dispersion of any config in the table — the instability signature persists
through a config-agnostic capture. This confirms mechanistically what the
decomposition implied: the ranking signal is in how each configuration's
NCCL *executes* the collectives under compute concurrency, not in the
per-config timing of the captured trace. The gaps were held identical
across all eight; only the collective execution varied; the ranking
followed the execution.

Two consequences. First, the operational scenario is viable: **one capture,
N cheap replays** ranks candidates as well as N full captures — CommCanary
can function as a pre-rollout tuner, not merely a diagnostic principle.
Second, and consistent with §5.5, the shared canary (71.4%) also ties the
stratified-overlap baseline (71.4%): what carried the operational result is
again *structure + concurrency*, not the faithful trace's exact timing. The
cheap tuner works; it works because of concurrency-bearing replay over
structure-preserving traces, and the "faithful" qualifier earns no
additional agreement on this workload.

### 5.7 Reduction degenerates without behavioral gates

The ddmin baseline — decision-preserving reduction with a ranking-only
oracle — collapsed the 8,192-event trace to **one event** in 13 oracle
calls (as it collapsed a 100-event adversarial trace to 1 in
simulation). Its physical agreement: 46.4%. Rankings alone are a
degenerate acceptance criterion for trace reduction; this is the
physical argument for gating reduction on behavioral distributions, not
decisions only.

### 5.8 Compression honesty

The baselines "compress" spectacularly — random tiling 175×, stratified
130× — by destroying exactly the information under test; the faithful
canary, whose compiler refuses to fabricate redundancy, is honestly
reported at 0.35× (2.9× *larger* than the raw trace, its adjacency
grouping defeated by size-cycling). Compression ratios quoted without a
downstream-decision metric are, on this evidence, advertising.

### 5.9 What no replay captured

All proxies place the two *default* configurations mid-pack; the
workload puts them first and second. NCCL's runtime tuner, choosing
algorithm/protocol per message size in situ, beats every forced setting
— and no replay mode reproduced whatever contextual signal the tuner
exploits. This bounds the achievable agreement of the entire proxy
family studied here (~70–75% on this config set) and is the sharpest
open question the data leaves.

## 6 Implications

**For operators**: microbenchmark rankings of ALGO/PROTO settings are
unreliable in the decode message-size regime; forced settings lost to
defaults in situ anyway; and if you must proxy, the proxy needs busy
SMs, not just your trace.

**For trace-format and replay-tool builders** (PARAM, Chakra, and kin):
faithful streams replayed synchronously inherit microbenchmark
blindness. Overlap semantics — async issue with explicit dependency
waits — is a small grammar (three entry types) and a small replayer
(180 lines), and it moved agreement more than every fidelity feature
combined.

**For trace-compression research**: fidelity metrics that bound
per-event timing error measure a property that, in the one physical
decomposition we know of, carried zero decision weight. Decision-level
acceptance criteria (does the reduced artifact rank configurations as
the workload does?) are measurable, and stricter behavioral gates are
needed to keep reduction from degenerating (§5.6).

## 7 Limitations

One workload (TP-4 decode-style loop), one node, one interconnect
(PCIe), 4 GPUs, single-rank observational traces, two NCCL versions that
happened to tie at defaults, and a 44 µs compute quantum. The
elimination chain's *ordering* is mechanistically explained and we
expect it to generalize; the *magnitudes* (64/61/50/68/71%) are
point estimates for this setting. The concurrency reconstruction uses
uniform GEMMs, not the workload's kernel mix; the ~30% residual
disagreement (§5.8) is unexplained by construction. Multi-node,
NVLink-class, MoE (all_to_all), and injected-skew conditions are natural
extensions the released kit already parameterizes.

## 8 Artifact

The repository preserves the CommCanary importer/compiler/exporter/verifier
pipeline, the manifest-driven Rostam experiment subsystem, the overlap-aware
reference replayer, and the historical analysis source. It does **not** contain
the historical campaign's complete raw attempts, frozen manifest, selection,
completeness verdict, or all sweep result JSONs. Consequently this checkout
cannot regenerate the numeric tables above, and neither those tables nor a
historical test count are current release evidence.

A future publication artifact must use the current immutable campaign workflow:
retain every terminal attempt, select exactly one attempt per expected cell,
obtain a complete fail-closed verdict, regenerate the aggregate and paper
fragment from the hash-bound evidence, and byte-compare the tracked outputs.
Until that evidence exists, this document remains the historical draft described
in `paper/README.md` and is excluded from CommCanary release distributions.

---

*Acknowledgments: experiments ran on the Rostam cluster (LSU CCT).*
*Draft notes for co-author review: (1) affiliation line, (2) whether to
name Claude/AI assistance in acknowledgments per venue policy, (3)
related-work citations are named in-text but need BibTeX entries — list
compiled, conversion pending, (4) candidate venues: arXiv immediately;
then HotInfra/MLArchSys-class workshop or MLSys benchmarks track.*
