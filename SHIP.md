# SHIP.md — the only plan

Written 2026-08-20. Revised same day after reading `experiments/rostam/workload_tp8.py`.
**Supersedes** `TODO.md`, `STRATEGY.md`, `future_plan.md`, `plan-github-stars.md`,
and `plan.md` for all forward decisions. `HANDOFF.md` stays authoritative for
Rostam state. `claims/public-claims.yaml` stays authoritative for public language.

If this file disagrees with another planning document, this file wins. If a new
planning document appears before 2026-11-08, that is the failure mode this file
exists to stop.

**The goal is unchanged and it is the right goal:**

> Given a workload and a regression policy, produce the shortest physical test
> that safely predicts whether a stack change should ship.

This plan exists to make that claim **true and verified**, not to retreat from it.

---

## 0. Diagnosis

| | |
|---|---:|
| Source / tests | 31,341 / 29,308 lines, 104 modules, 1,158 tests passing |
| CLI subcommands | 23 |
| Commits | 225, Jun 23 → **Aug 4**, then stopped |
| Uncommitted since Aug 5 | 141 files, +5,089 / −1,467 |
| Public since | Jun 30 (51 days) |
| **Stars / forks / PyPI** | **0 / 0 / not published** |
| Positioning changes | 3 in 6 weeks |

The engineering is strong — zero runtime dependencies, strict mypy over 167
files, an enforced import DAG, content-addressed artifacts, byte-reproducible
reports, fail-closed refusals. That is not the problem.

### The finding that reframes everything

`experiments/rostam/workload_tp8.py` is the reference workload `W-full` that the
entire August campaign was measured against. It is:

```python
def token_step(token_index):
    for layer in range(args.layers):
        _run_layer(torch, dist, activation, weight, comm_buffer)   # GEMM 1024³ + all_reduce
```

with `--msg-sizes 64K,128K,256K`, `--warmup 1`, and a token loop. The physical
runner defaults to `--warmups 3 --iterations 12`. `examples/instrumented_decode.py`
emits 24 hand-written events.

**The reference workload is a synthetic microbenchmark.** The August campaign
compared four microbenchmarks against a fifth one. So:

- **7.9 s per cell is not evidence against reduction.** It is the runtime of a
  12-iteration synthetic loop. There is nothing in it to reduce. A canary cannot
  be 10× shorter than something that is already 8 seconds and mostly
  `init_process_group`.
- **17.2 s for the canary is a fixed-cost artifact.** CUDA context, NCCL init,
  allocation and warmups are paid identically by both, and with a 12-iteration
  payload they dominate completely. The canary then serialized, without overlap,
  work that was concurrent in the source. Of course it lost.
- **The cost claim has therefore never been tested.** It was measured against a
  reference that had no cost, using the one artifact variant (comm-only, no
  overlap) that the same campaign proved is the losing configuration.

This is an experimental-design error, not a feasibility result. It is fixable on
the hardware already available, and fixing it is the whole of §3.

### The second structural problem

The rigor apparatus points the wrong way. `docs/operator-quickstart.md` asks a
new user to obtain a reviewed checkout through one channel and a SHA-256-verified
wheel through another, then declares success when the run **ends in a blocked
bundle**. `build` requires six artifacts nobody possesses. Every refusal is
individually correct and collectively fatal: the front door is locked from the
inside, and the observed adoption is the arithmetic consequence.

Launch pre-flight completed 2026-07-26, all nine items checked, with the note
*"Nothing is blocking the push."* It was then gated on `TODO.md §0`, an internal
documentation-consistency task. Twenty-five days later: 5,089 uncommitted lines,
a fourth positioning, three strategy documents, zero external users.

---

## 1. The claim, made precise

The pitch is right. It has been vague in exactly one place, and that vagueness is
why it could never be verified. Sharpen it:

> **Given a workload W and a class of substrate changes S, produce a canary C such
> that C runs at least 10× faster than W on the same hardware, and for every
> change s ∈ S held out from C's selection, C's ship/don't-ship verdict matches
> W's, with zero severe false negatives.**

Two definitions do the work.

**Substrate changes.** The canary replays a *frozen schedule*, so it predicts the
effect of changes to what runs underneath the schedule — NCCL version, NCCL
algo/proto, CUDA, driver, firmware, GPU model, interconnect topology, clock caps,
cloud instance type, container image. It **cannot** predict the effect of changes
to the engine's own scheduling logic; a vLLM release that alters batching is
outside the boundary, because the canary froze the batching. State this boundary
first, in every venue. It is what converts a mushy claim into a testable one, and
the entire fleet/vendor-qualification market lives inside it.

**Severe false negative.** The canary says ship; the full workload regresses past
the policy threshold. This is the only asymmetry that matters commercially, and
the machinery to enforce selection-before-holdout already exists and is the best
code in the repository.

### Why this is provable on 4×A100 and needs no multi-node

Because the expense comes from **duration**, not node count. A 30-minute vLLM
serving benchmark on four A100s is a genuinely expensive test that a real team
genuinely runs before a driver or NCCL upgrade. Compressing it to 90 seconds is
the product. Multi-node would broaden the result; it is not required to prove it.

---

## 2. The asset you already have, unpublished for 19 days

280 verified cells, 4×A100, byte-reproducible, against 28 configuration pairs:

| Proxy | Pair agreement | Kendall τ |
|---|---:|---:|
| `W-shared-overlap` | **71.4%** | **0.708** |
| `W-micro` (`nccl-tests` analogue) | 64.3% | 0.490 |
| `W-canary` (communication-only) | **57.1%** | **0.204** |

Faithful communication-only replay ranks configurations *worse than the
microbenchmark it was built to replace*. Only overlap-carrying replay wins, and
only it ranks `nccl-2.20.5-tree-ll` last — the original motivating regression,
reproducing on real hardware.

**Read this as a specification, not an obituary.** It says precisely what a
reduced canary must preserve to be worth anything: not the communication, but the
*concurrency structure around* the communication. Every design decision in §3
follows from it. It is also a warning that generalizes to every trace-replay
system in the neighbourhood — Chakra replay, ASTRA-sim, AICB/SimAI, PARAM
comms-replay all replay communication, and this data says communication alone is
not the load-bearing variable.

It is finished, frozen, and nobody has read it.

---

## 3. The engineering program

Three gaps stand between the current repository and a verified claim. None of
them is research risk; all three are build work.

### G1 — There is no real reference workload. Build one.

`W-real`: vLLM, Llama-3.1-8B-Instruct (or Qwen2.5-7B), TP4 on 4×A100-40GB,
ShareGPT-shaped request trace with a real arrival process, sustained 20–40
minutes, decision metric = **sustained throughput at a fixed p99 TTFT/ITL
budget** — the number an actual team uses to decide an upgrade.

`product/application_driver.py` is most of the way there but currently generates
synthetic token ids in a fixed batch loop with `--iterations 12`. It needs a
request trace, an arrival process, a sustained duration, and a serving-level
metric instead of per-iteration latency.

- [ ] Extend the vLLM driver: trace-driven requests, Poisson/replay arrivals,
      duration-bounded run, serving metrics with p50/p95/p99.
- [ ] Freeze the traffic trace as a content-addressed artifact like everything
      else. Same discipline, new input.

### G2 — Reduction has one axis, and the big one is missing.

Existing: ddmin over the event list (`services/reduction.py`) and behavior search
over timing budgets (`services/behavior_search.py`). Both operate *within* a
program's structure.

Missing: **repetition and phase reduction**, which is where 100–1000× lives.
A 30-minute serving run is 10⁵–10⁶ collective events drawn from a handful of
distinct structural phases — prefill at varying lengths, decode steady state,
preemption and recompute. Cluster iterations by structural signature (operation
sequence, message sizes, overlap profile, concurrency degree), select *k*
weighted representatives, replay *k* instead of *N*.

This is SimPoint applied to communication phases. The literature to cite is
already listed in `context.md`: SimPoint, Pac-Sim, SPEC subsetting, ScalaTrace.

- [ ] Implement phase signature extraction and clustering over a captured trace.
- [ ] Implement weighted representative selection with a declared, deterministic
      total ordering, matching the existing `BehaviorSearchSizeKey` discipline.
- [ ] Compose the axes: phase reduction first (cheap, huge, low risk), then
      existing structural reduction *within* representatives (fine, risky, already
      verified fail-closed).
- [ ] **Preserve overlap through both axes.** This is §2 as a hard constraint:
      a representative carries its `compute_overlap_us`, `compute_before_us`,
      `compute_pressure`, `concurrent_groups`, and `rank_arrival_us`, or it is
      rejected. Reject, do not default to zero — see the `baselines.py:372` bug
      below.

This axis is pure algorithm work, needs no GPU, and is testable locally against
the existing fixtures. **It can start today, in parallel with everything else.**

#### Status 2026-08-20 — first implementation and what it measured

`services/phase_reduction.py` and `experiments/phase_reduction_local.py` exist.
Ruff, strict mypy, and the full suite (1,183 passed) are green. Four results,
all simulator-only:

1. **Segmentation cannot be inferred from event structure.** Every compression
   objective over the collective stream is maximised by shredding it to single
   events, because a ranking is a projection that survives almost any subset —
   the τ=0.204 mechanism restated as an optimiser. The repetition unit is a
   property of the workload, so it is now taken from a declared
   `iteration_index`, an explicit period, or an opt-in compute-gap heuristic,
   and an undeclared segmentation is **refused**. All three paths agree on the
   probe. *Consequence for G1: the vLLM driver must emit step boundaries into
   the capture. That is a capture-contract change and it belongs in this phase.*
2. **Ranking preservation is worthless as a fidelity signal.** 40/40 pairwise
   decisions survive at every level tested, up to **400×** compression on an
   eight-event artifact. The behavior verifier fails all of them
   (`configuration_ranking_status: pass`, `model_behavior_preservation_status:
   fail`). This independently reproduces `one hundred events became one` on a
   different reduction method, and it is why the frontier in §4 must be measured
   against the verifier and physical evidence, never against rankings.
3. **A reduced trace needs its timeline rebased.** Dropping iterations leaves
   holes in absolute `start_us`; replay models queueing from inter-arrival time,
   so the holed artifact presents fictitious idle. Weight-materialising the
   representatives without rebasing produced a **100,000% error**. Rebasing to
   the source's median cadence is implemented.
4. **Representative selection preserves the median and destroys the tail.**
   With rebasing, median error is 2–17%, but **p95 and p99 are 58–80% under**
   at every reduction level. This is structural, not a tuning problem: the
   medoid is by construction the typical member of its cluster, so tails are
   systematically discarded. For a gate whose decision metric is p99 latency —
   which is what serving teams actually gate on — that is disqualifying.

#### Update — the tail problem is arithmetic, and it bounds the product claim

Both obvious fixes were implemented, measured, and **rejected by their own
numbers** before the right one was found:

| selection method | median err | p95 err | p99 err |
|---|---:|---:|---:|
| medoid only | +2 to +17% | **-58 to -63%** | **-73 to -80%** |
| medoid + reserved tail clusters | +13 to +237% | **+46 to +81%** | +3 to +17% |
| proportional sampling (shipped) | **+4%** | **+16%** | **+7%** at 2x |

The mechanism: **percentiles are proportions.** The probe's tail is carried
entirely by prefill -- decode-only p99 is 84 us against prefill-only 332 us --
and prefill is **3.2% of events**. A medoid is by construction not a tail, so
medoid-only selection deletes it. Reserving budget for the longest clusters
recovers p99 but inflates prefill to a quarter of the artifact, so the median
and p95 blow out instead. Only preserving the *mix* preserves the percentiles.

Retention is now proportional (largest remainder, with one slot floored for any
cluster holding a tail iteration), and cadence is per cluster.

**The consequence is the important part, and it is unwelcome:**

| retained | compression | median | p95 | p99 |
|---|---:|---:|---:|---:|
| 200 | 2x | +4.0% | +15.5% | +6.6% |
| 100 | 4x | +12.1% | +43.9% | +12.5% |
| 64 | 6x | +26.9% | +69.2% | +16.7% |
| 32 | 12x | +252% | +88% | +24.9% |

Reduction floors out around **2-4x** on this workload, not 10x, and the artifact
stops shrinking below ~16 retained iterations because the tail-carrying clusters
occupy the whole budget. This is not a tuning failure: estimating a q-quantile
requires enough retained iterations that a phase occupying `1-q` of the workload
still appears in proportion. **A p99 gate over a 3%-rare tail phase cannot be
served by a 10x-reduced canary, however the iterations are chosen.**

If that survives physical measurement, `minimum_runtime_reduction_ratio: 10.0`
in the policy is unreachable for p99-gated serving workloads, and §1's claim
needs its ratio restated against the decision metric's quantile. That is exactly
the frontier §4 Step 5 exists to measure -- it has now been measured in the
simulator, and it says the honest number is single-digit.

Caveat: synthetic trace, deterministic simulator. The *mechanism* is
workload-independent arithmetic; the *numbers* are not evidence.

- [ ] Re-measure against the behavior verifier and then physically.
- [ ] Restate the reduction ratio in §1 against the decision metric's quantile.

### G3 — Cost accounting is dominated by fixed setup, and the canary never amortizes.

`execution/physical_runner.py` does `init_process_group` → allocate → 3 warmups
→ 12 iterations. At that payload size the setup *is* the measurement.

- [ ] Report `setup_s` and `measured_s` separately in every physical measurement.
- [ ] Run the canary in one process with one process-group init, reusing the
      allocation, so reduction shows up as wall-clock rather than being eaten.
- [ ] Publish the reduction ratio **both ways** — total wall-clock including
      setup, and steady-state only. Lead with total, because that is what a
      customer pays. Disclose both, because that is the house style.

### G4 — The product gate cannot express uncertainty. **FIXED 2026-08-20.**

Audited 2026-08-20. Rigor in this repository is distributed inversely to
exposure: the research path is stronger than the product path.

| | Research path (`qualification_decision`) | Product path (`commcanary gate`) |
|---|---|---|
| Outcome states | `pass`/`fail`/**`inconclusive`**/`incomparable` | `pass`/`fail`/`incomparable` |
| Uncertainty | percentile bootstrap, CI-crosses-threshold → inconclusive | none |
| Noise gate | `max_relative_iqr_pct` → unstable → inconclusive | none |
| Comparison | interval on median difference | median vs median, one `%` threshold |
| Minimum samples | enforced | schema `minItems: 1` |

The r3 campaign was declared `inconclusive` on 4×A100 because bootstrap
intervals crossed decision boundaries and one configuration breached the IQR
stability limit. **Run that same evidence through `commcanary gate` and it
returns `pass`** — the surface a customer would put in CI has no state in which
to say "I could not tell," so unresolvable differences collapse into `pass`.
That is a severe-false-negative generator by construction, and "zero severe
false negatives" cannot be claimed until it is fixed.

- [x] `inconclusive` added to `physical_gate_result.v1` and to `evaluate_physical_gate`.
- [x] Percentile bootstrap on the regression-oriented median difference, seeded
      from the frozen policy, plus the `max_relative_iqr_pct` stability gate and
      a `minimum_samples` replicate floor — all predeclared in the policy and
      bound to its content identity.
- [x] The result validator independently rechecks the decision rule from the
      reported interval, so a forged status is refused.
- [x] Regression test proving the severe-false-negative case: a candidate whose
      median sits under the boundary but whose interval reaches past it now
      returns `inconclusive` instead of `pass`.

### G5 — The physical runner corrupts the variable the product depends on. **FIXED AND VALIDATED ON HARDWARE 2026-08-20.**

Validated at **four ranks** by job `186094` on `toranj1` (4x A100-PCIE-40GB,
torch 2.4.1+cu121, NCCL 2.20.5) after the node was rebooted 2026-08-20 11:00,
and independently at two ranks by `186047`/`186048` on `nasrin0`. Twelve
same-dtype collectives per region over 4 slots, so the pool wraps 3x:
buffer pool is distinct memory; the correctness oracle passes with overlap
**enabled** over 48 probes; the commitment hash is byte-identical whether the
collectives ran concurrently or serially (`4bb77b1f...` both ways), which is the
strongest available evidence that nothing reduced into overlapping memory; probe
order is stable across modes; overlap measured **1.60x** faster than serialised
execution at four ranks (1.32x at two). Two defects were found by the run itself and are recorded in the
changelog: an unrunnable GEMM correctness probe, and an over-optimistic skew
floor.

- [x] **Concurrent all-reduces aliased one buffer.** `_allocate` keeps a single
      collective tensor **per dtype**, sized to the largest `numel`; every
      all-reduce in a region then takes `collective_buffers[dtype][:numel]` and
      is issued `async_op=True` with `work.wait()` deferred to the end of the
      region. Two same-dtype collectives in flight therefore reduce into
      overlapping memory. Results are undefined, and NCCL may serialise them —
      destroying the very overlap being measured. Worse, correctness validation
      runs with `disable_overlap=args.disable_overlap`, so in the default
      overlap-enabled mode the `buffer.fill_(signature)` probe races an
      in-flight collective. Allocate per outstanding operation, or bound
      concurrency to one collective per buffer. *Fixed with a bounded
      four-slot pool per dtype: a slot's previous collective is settled before
      the buffer is reused, so concurrency is bounded rather than aliased.
      Workspace accounting and correctness-probe ordering both preserved.*
- [x] **Rank skew was injected with `time.sleep()` at microsecond scale.**
      `time.sleep(rank_skew_us * rank / 1e6)` for skews of 1–20 µs, against an
      OS granularity of tens to hundreds of microseconds — systematically
      inflated and jittery. It is also a *host-side* sleep before an async
      enqueue, so it does not reliably produce GPU-side arrival skew at all.
      This project refuses to default skew to zero because zero skew is a strong
      physical claim; injecting skew wrong by an order of magnitude is a
      stronger and worse one. *Replaced with a monotonic spin accurate at this
      scale; skew below `MINIMUM_RESOLVABLE_SKEW_US` is now refused rather than
      silently inflated, and the mechanism is recorded in the evidence as
      `host_issue_monotonic_spin` so it cannot be read as a GPU-side arrival
      guarantee. A CUDA-side delay kernel remains the stronger mechanism.*

### Supporting fixes, same phase

- [x] ~~**Overlap import path (`TODO.md §0c`).**~~ **Already implemented in the
      uncommitted working tree** — `adapters/kineto.py:772-833` links
      `args["External id"]` through to kernel intervals, unions the linked
      communication-kernel spans, intersects them against non-communication
      kernels on other streams, and records explicit `unavailable_*` statuses
      instead of fabricating zeros. `TODO.md §0c` describes the pre-pivot state
      and is stale. **This is a further argument for Phase 0's first item: the
      most valuable work in this repository is sitting uncommitted, and the
      tracked planning documents actively misdescribe it.**
- [x] **`baselines.py` fabricated concurrency defaults.** `_features` defaulted
      `compute_overlap_us` and `compute_before_us` to `0.0` and
      `compute_pressure` to **`0.5`** — a midpoint asserting the event was half
      loaded, so an undeclared field clustered with events measured at exactly
      that. Each field now carries a known-indicator, so unknown can only match
      unknown. *Note `require_known_overlap` covers only `compute_overlap_us`;
      neither `compute_before_us` nor `compute_pressure` is required anywhere.*
- [x] **Chakra capture tagged unknown overlap as measured-zero overlap.**
      `chakra_capture.py` set the `overlap` tag only when the value was `> 0.0`
      with missing coerced to `0.0`, so an undeclared trace was tagged as
      genuinely non-overlapping — the τ=0.204 configuration, propagated into the
      tagging that drives windowing. Now emits a distinct `overlap-unknown` tag.
- [x] **A vacuous false-positive rate satisfied the qualification gate.**
      `false_positives / passing if passing else 0.0` let a corpus containing no
      passing perturbation report 0% and clear `maximum_false_positive_rate`.
      The denominator is now published as `passing_perturbations` and the gate
      refuses to treat a vacuous zero as measured. This caught a real defect in
      the project's own fixture: the CEGIS holdout contained only regressions,
      so qualification against it asserted a false-positive bound that the
      evidence could not falsify. The fixture now carries a passing holdout.
- [ ] **Trace volume.** A 30-minute captured run is large. Bounded windowed
      capture with declared sampling, not best-effort truncation.

---

## 4. The verification protocol

This is what makes the pitch *verified* rather than asserted, and it is why the
existing audit machinery finally earns its cost.

**Step 1 — Sensitivity precheck. Run this before spending anything.**
Does `W-real` discriminate the change set beyond noise? 8 configurations × 1
short replicate, ~1 node-hour.

This is a genuine risk and must be checked first: on TP4 with an 8B model, the
all-reduce may be a small enough fraction of step time that NCCL differences
vanish into noise. The August campaign saw large effects, but its message sizes
were 64K–256K in a comm-bound synthetic loop.

If the effect size is below noise, do not proceed — widen the change class
instead. Note that the canary is a *mixed* GEMM+collective program, so driver,
CUDA, clock-cap and GPU-model changes are inside the boundary and produce larger,
more robust effects than NCCL versions alone. That widening is a feature, and it
strengthens the fleet-qualification pitch rather than weakening it.

**Step 2 — Freeze.** Predeclare the change set, the policy thresholds, the
selection/holdout split, replicate count, and the stability limit. Content-address
the lot. Machinery exists.

**Step 3 — Select.** Build the canary using only the selection-set configurations.
The existing selection-before-holdout enforcement is what makes this credible.

**Step 4 — Measure held-out.** Only then unlock the held-out configurations.

**Predeclared thresholds** (already correct in `docs/product-status.md`):

| Metric | Threshold |
|---|---|
| Severe false negatives, held out | **0** |
| False positives | ≤ 10% |
| Pair-decision agreement | ≥ 90% |
| Wall-clock speedup vs `W-real` | ≥ 10× |
| Replication | independent allocations, ≥ 2 days |

**Step 5 — The frontier.** Sweep reduction aggressiveness and plot speedup
against decision agreement. Find the knee; ship the default there. This is the
paper *and* the product's default setting, and it is publishable whatever shape
the curve has.

---

## 5. Schedule

Hard anchor: the master's-application artifact freezes **2026-11-08**. GSoC
wrap-up owns the next ~2 weeks; Phase 1 is writing, which fits alongside it.

### Phase 0 — Unblock. Thu Aug 20 → Sun Aug 23. ~6 hours.

- [ ] **Commit and push the tree.** 141 files, +5,089 lines, 15 days, on a project
      whose entire thesis is provenance. The "intentionally uncommitted" note expired.
- [ ] **Send the Hartmut email.** `email.md` is drafted and unsent. You used 11.9
      node-hours outside GSoC and are about to publish a paper acknowledging CCT.
      Sending it after an HN front page is a catastrophe; sending it today is a
      courtesy. **Add the ask that §3 depends on: does Rostam access continue past
      the GSoC coding period, and can you have ~30 node-hours on `cuda-A100`
      through October?** The study does not exist without that answer.
- [ ] **Close `TODO.md §0` by decision, not reconciliation.** Adopt §1 as the
      single story, rewrite the README opening once, two hours. It has cost 25 days.
- [ ] **Publish 0.3.0 to PyPI.** The release gate requires an independent-operator
      trial that will never happen for a package nobody can install. Invert it:
      ship the wheel, let the public be the operator trial. Keep the alpha banner.

### Phase 1 — Publish the finding. Mon Aug 24 → Fri Sep 4.

Highest-value two weeks available, and 90% of the work is done. Reframed: this is
not a post-mortem, it is the **design constraint** post that sets up the product.

- [ ] One post, ~3,000 words. Titles: *Faithful replay ranked our GPU configs
      worse than a microbenchmark* / *I replayed the communication exactly and got
      the wrong answer*. Arc: motivating NCCL regression → why microbenchmarks were
      suspected → building the faithful comm-only replay → **the table** → the
      confession that the faithful thing lost → the mechanism → what it implies for
      Chakra/ASTRA-sim/AICB-class replay → **what we are building because of it**.
      Use the `blogger` skill.
- [ ] **arXiv, cs.DC — not Zenodo.** Two Zenodo DOIs carry near-zero external
      signal. This is also the master's-application artifact and must be arXiv.
- [ ] HN Tue Sep 1 or Wed Sep 2, 08:00 ET. Submit the post, never the repo.
      Top-level comment within five minutes naming the scope limit yourself.
      Three hours in-thread. `plan-github-stars.md` §Venues has the mechanics.
- [ ] r/MachineLearning `[P]` +2h. r/HPC day 2. X thread, repo link last tweet only.
- [ ] Post into NVIDIA/nccl#513 and the Chakra WG as a contribution. Small, expert,
      and the actual buyer pool.

### Phase 2 — Build, in parallel from Aug 24.

Local, no GPU needed, therefore not blocked on Rostam or on Phase 1:

- [ ] G2 phase/repetition reduction, tested against existing fixtures. **Start here.**
- [ ] G3 cost accounting split.
- [ ] `baselines.py:372` bug.
- [ ] Kineto overlap import path.

Needs GPU, from ~Sep 7:

- [ ] G1 real vLLM serving reference.
- [ ] Sensitivity precheck (~1 node-hour). **Gate: do not proceed if the effect
      size is below noise — widen the change class instead.**

### Phase 3 — The study. Mon Sep 14 → Fri Oct 10. ~15–20 node-hours.

- [ ] Freeze, select, measure held-out, replicate across days and allocations.
- [ ] Then the frontier sweep.

### Phase 4 — Cut the surface. Sep, low intensity, after Phase 1 tells you who came.

- [ ] **23 subcommands → 3**: `capture`, `check`, `compare`. Everything else moves
      behind `commcanary internal <cmd>`. Facade change, not a rewrite. A stranger
      must reach a real result in one command with no pre-existing artifacts.
- [ ] 1.0.0 to PyPI once that path works on a stranger's profile.

### Phase 5 — Lock. Mon Oct 26 → Sun Nov 8.

Freeze README, arXiv paper, PyPI release, blog series, star count. This is what
goes into the master's applications. Stop adding after Nov 8.

---

## 6. What to stop doing

- **Stop writing planning documents**, including replacements for this one.
- **Stop gating public action on internal consistency.** The July 26 gate cost 25 days.
- **Stop Zenodo.** Two DOIs, zero readers.
- **Stop adding surface area** to a 60,000-line repository with zero users.
  Every line before the claim is verified delays the only thing that matters.

### Repository hygiene, Phase 0, one hour

Move to `docs/history/`: `STRATEGY.md`, `future_plan.md`, `plan.md`,
`plan-github-stars.md`, `TODO.md`, `RESEARCH_HISTORY.md`, `ENGINEERING_PLAN.md`,
and `context.md` (stale — it describes an 11k-line MIT project with 123 tests;
reality is 31k lines, Apache-2.0, 1,158 tests).

Top level keeps `README.md`, `SHIP.md`, `ROADMAP.md`, `CHANGELOG.md`,
`CONTRIBUTING.md`, `LICENSE`, `SECURITY.md`, `CITATION.cff`, `HANDOFF.md`.
`docs/` 29 files → ~8.

---

## 7. The startup

With §1 verified, the pitch is a sentence and the evidence is yours:

> *You qualify your fleet with `nccl-tests`. Here is a published measurement,
> made against my own tool including the variant that lost, showing `nccl-tests`
> agrees with real workload rankings 64% of the time. I can give you a test that
> runs in ninety seconds instead of thirty minutes, predicts your real workload's
> verdict on substrate changes with zero severe false negatives, and that your
> customer can hand you without shipping you their weights.*

Nobody else in this space publishes a fidelity number about themselves. That is
the moat, and it is already built.

**Shape, honestly.** The buyer set is ~40 organisations — silicon vendors,
neoclouds, hyperscaler procurement, one tier of HPC labs. Cost-center spend,
6–12 month cycles, no self-serve motion. That is a good business and a poor
venture bet; pitching it as the latter gets read instantly by anyone who knows
the space. Build it as `A-Square Systems`: paid qualification engagements first,
software as the instrument, product only where design partners pull.

**Sequence. Do not start before November.**

| When | Move |
|---|---|
| Nov 2026 – Jan 2027 | 3–5 paid qualification engagements, $15–40k. Targets: TensorWave, Lambda, Crusoe, Together, Nebius, plus AMD and Keysight contacts. HN + arXiv are the entire top of funnel — this is why Phase 1 precedes it. |
| Feb – Jun 2027 | Convert 2 into design partners. Productize only what all of them needed. Incorporate when the first contract requires it. |
| Jul 2027+ | Licensed artifact and hosted verification, if and only if partners pull. |

Two constraints to plan for now: contracting from India to US vendors needs an
entity and probably a US wrapper before the first invoice; and **every engagement
needs compute you do not own** — Rostam cannot host customer work, which is
exactly the boundary the Hartmut email is about. Price rented H100 time into the
first quote.

---

## 8. Kill conditions

Written now, while honesty is cheap.

- **Sensitivity precheck shows no separation beyond noise, even after widening the
  change class.** The workload does not carry a decision. Change the workload;
  do not weaken the thresholds.
- **Held-out study produces a severe false negative.** The canary is unsafe at
  that reduction level. Move up the frontier and re-run; do not relabel the metric.
- **Frontier knee lands below 10× at ≥90% agreement.** Publish the frontier —
  it is a real result — and sell the *fidelity* proposition (§2 positioning B)
  rather than the cost proposition. This is the honourable fallback, not a failure.
- **HN + Reddit yields <50 stars and no external runs by Sep 15.** Positioning is
  wrong, not execution. Do not rewrite the tool.
- **Rostam access ends with GSoC.** Phase 3 becomes a methodology paper on frozen
  evidence, still publishable. Do not buy GPU time personally to rescue a campaign.
- **No paid engagement by Feb 2027.** §7 is dead as a business; the work was still
  correct as a credential. Take the job.

Not a kill condition: a disappointing frontier curve. A measured, honest negative
curve is the same class of result as the August finding, and it is publishable.

---

## 9. This week

1. Commit and push the tree. *(today)*
2. Send the Hartmut email, **with the 30 node-hour ask**. *(today)*
3. Start G2 phase reduction — local, no GPU, no dependencies, biggest lever. *(today)*
4. Rewrite the README opening around §1. Two hours, once. *(Fri)*
5. `python -m tools.verify --release`, publish 0.3.0 to PyPI. *(Fri)*
6. Draft the post. *(weekend)*

The claim is provable. It has never been tested. Everything above is the shortest
path from here to testing it honestly.
