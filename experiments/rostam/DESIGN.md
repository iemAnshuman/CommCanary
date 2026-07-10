# Rostam physical experiment — design

Goal: physical evidence for the paper's central claim. Does a minimized,
behavior-verified canary preserve the *decision* (configuration ranking and
regression verdict) that the full workload exhibits on real hardware — in a
setting where an isolated collective microbenchmark misleads?

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

## Kit layout (implementation spec)

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
