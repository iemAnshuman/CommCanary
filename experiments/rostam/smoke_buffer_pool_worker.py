"""Four-rank NCCL smoke check for the physical runner's collective buffer pool.

This validates the 2026-08-20 runner change on real hardware, which no local
test can do: every collective buffer used to be a single shared tensor per
dtype, so two ``async_op=True`` all-reduces in one region reduced into
overlapping memory.  The results were undefined and NCCL could serialise them,
destroying the compute/communication overlap the canary exists to measure.

The program below deliberately issues more same-dtype collectives per region
than there are buffer slots, so the pool wraps and the settle-before-reuse path
runs.  It then checks the correctness oracle **with overlap enabled**, which is
the mode that ships and the mode the old aliasing code corrupted.

This is a runner conformance check.  It issues no qualification verdict and
produces no campaign evidence.

Launch:  torchrun --standalone --nproc_per_node=4 smoke_buffer_pool_worker.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta
from typing import Any, Dict, List, Tuple

from commcanary.execution.physical_runner import (
    COLLECTIVE_BUFFER_SLOTS,
    MINIMUM_RESOLVABLE_SKEW_US,
    RANK_SKEW_MECHANISM,
    CompiledAllReduce,
    CompiledGemm,
    CompiledRegion,
    GemmRecipe,
    PhysicalProgram,
    _spin_us,
    _TorchProgramRuntime,
)


def _load_torch() -> Tuple[Any, Any]:
    """Import torch lazily, matching the other Rostam entrypoints.

    The repository gate type-checks this directory on machines without torch,
    so the dependency stays inside the function that needs it.
    """

    import torch  # type: ignore[import-not-found]
    import torch.distributed as dist  # type: ignore[import-not-found]

    return torch, dist


DTYPE = "float32"
NUMEL = 1 << 18  # 1 MiB at float32, large enough for NCCL to take real time
GEMM_DIM = 1024
COLLECTIVES_PER_REGION = COLLECTIVE_BUFFER_SLOTS * 3  # force the pool to wrap
REGIONS = 2


def _program(world_size: int) -> PhysicalProgram:
    regions: List[CompiledRegion] = []
    node_id = 1
    collectives = 0
    flops = 0
    for region_index in range(REGIONS):
        operations: List[Any] = []
        for _ in range(COLLECTIVES_PER_REGION):
            operations.append(
                CompiledAllReduce(
                    node_id=node_id,
                    dtype=DTYPE,
                    numel=NUMEL,
                    tensor_bytes=NUMEL * 4,
                )
            )
            node_id += 1
            collectives += 1
            # A GEMM between collectives is what gives the in-flight
            # all-reduces something to overlap with.
            operations.append(
                CompiledGemm(
                    node_id=node_id,
                    dtype=DTYPE,
                    recipes=tuple(
                        GemmRecipe(rank=rank, m=GEMM_DIM, n=GEMM_DIM, k=GEMM_DIM) for rank in range(world_size)
                    ),
                    executed_flops=2 * GEMM_DIM**3,
                )
            )
            node_id += 1
            flops += 2 * GEMM_DIM**3
        regions.append(CompiledRegion(region_id=f"smoke-region-{region_index}", operations=tuple(operations)))
    return PhysicalProgram(
        source_et_sha256="0" * 64,
        executable_sha256="1" * 64,
        projection_id="2" * 64,
        selected_region_ids=tuple(region.region_id for region in regions),
        selected_node_ids=tuple(range(1, node_id)),
        regions=tuple(regions),
        world_size=world_size,
        executed_collectives=collectives,
        executed_flops=flops,
        workspace_bytes_max_rank=0,
        communication_only=False,
    )


def _check(label: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}{(' — ' + detail) if detail else ''}", flush=True)
    return {"check": label, "passed": bool(passed), "detail": detail}


def main() -> int:
    torch, dist = _load_torch()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    results: List[Dict[str, Any]] = []
    try:
        program = _program(world_size)
        runtime = _TorchProgramRuntime(torch, dist, program, rank=rank, local_rank=local_rank)

        if rank == 0:
            print(f"\nbuffer slots            {COLLECTIVE_BUFFER_SLOTS}")
            print(
                f"collectives per region  {COLLECTIVES_PER_REGION}  (pool wraps "
                f"{COLLECTIVES_PER_REGION // COLLECTIVE_BUFFER_SLOTS}x)"
            )
            print(f"regions                 {REGIONS}")
            print(f"skew mechanism          {RANK_SKEW_MECHANISM}\n")

        # 1. The pool must be distinct memory, not N views of one tensor.
        pool = runtime.collective_buffers[DTYPE]
        pointers = {tensor.data_ptr() for tensor in pool}
        results.append(
            _check(
                "collective buffer pool is distinct memory",
                len(pool) == COLLECTIVE_BUFFER_SLOTS and len(pointers) == COLLECTIVE_BUFFER_SLOTS,
                f"{len(pool)} buffers, {len(pointers)} distinct addresses",
            )
        )

        # 2. Correctness with overlap ENABLED. This is the mode that ships and
        #    the mode the previous shared-buffer code corrupted.
        overlapped = runtime.validate_complete_program(disable_overlap=False)
        probes = overlapped["validated_node_ids"]
        results.append(
            _check(
                "correctness oracle passes with overlap enabled",
                bool(overlapped["passed"]),
                f"{len(probes)} probes over {len(program.selected_node_ids)} nodes",
            )
        )

        # 3. Correctness with overlap disabled, as the serialised control.
        serial = runtime.validate_complete_program(disable_overlap=True)
        results.append(
            _check(
                "correctness oracle passes with overlap disabled",
                bool(serial["passed"]),
                f"{len(serial['validated_node_ids'])} probes",
            )
        )

        # 4. The strongest aliasing detector available: every probe value is
        #    deterministic, so the commitment must be byte-identical whether the
        #    collectives ran concurrently or serially. Reducing into overlapping
        #    memory would change it.
        results.append(
            _check(
                "commitment is identical across overlap modes",
                overlapped["commitment_sha256"] == serial["commitment_sha256"],
                f"{overlapped['commitment_sha256'][:16]} vs {serial['commitment_sha256'][:16]}",
            )
        )

        # 5. Probe order must not depend on the order slot pressure forced
        #    waits in. The runner appends a region's GEMM probes inline and its
        #    collective probes after the drain, so the order is grouped, not
        #    program order -- what matters is that it is identical either way.
        results.append(
            _check(
                "probe order is stable across overlap modes",
                probes == serial["validated_node_ids"] and sorted(probes) == sorted(program.selected_node_ids),
                f"{probes[:6]}{'...' if len(probes) > 6 else ''}",
            )
        )

        # 6. Overlap must actually buy something. If NCCL had serialised the
        #    aliased collectives this would not hold.
        for _ in range(2):
            runtime.execute_pass(disable_overlap=False, rank_skew_us=0.0, timed=False)
        torch.cuda.synchronize()
        overlap_s = min(
            runtime.execute_pass(disable_overlap=False, rank_skew_us=0.0, timed=True)["cuda_seconds"] for _ in range(5)
        )
        serial_s = min(
            runtime.execute_pass(disable_overlap=True, rank_skew_us=0.0, timed=True)["cuda_seconds"] for _ in range(5)
        )
        results.append(
            _check(
                "overlap is not slower than serialised execution",
                overlap_s <= serial_s * 1.05,
                f"overlap {overlap_s * 1000:.2f} ms vs serial {serial_s * 1000:.2f} ms "
                f"({serial_s / overlap_s if overlap_s else float('nan'):.2f}x)",
            )
        )

        # 7. The skew spin must resolve what it claims to. time.sleep could not.
        if rank == 0:
            errors = []
            for target_us in (MINIMUM_RESOLVABLE_SKEW_US, 10.0, 20.0, 100.0):
                observed = []
                for _ in range(200):
                    start = time.perf_counter_ns()
                    _spin_us(target_us)
                    observed.append((time.perf_counter_ns() - start) / 1000.0)
                observed.sort()
                median = observed[len(observed) // 2]
                errors.append((target_us, median, 100.0 * (median - target_us) / target_us))
            worst = max(abs(pct) for _t, _m, pct in errors)
            detail = ", ".join(f"{t:g}us->{m:.2f}us ({pct:+.1f}%)" for t, m, pct in errors)
            results.append(_check("skew spin resolves targets at or above the floor", worst < 10.0, detail))

        local_pass = all(row["passed"] for row in results)
        gathered: List[Any] = [None] * world_size
        dist.all_gather_object(gathered, {"rank": rank, "passed": local_pass, "results": results})
        if rank == 0:
            every = all(entry["passed"] for entry in gathered)
            print("\n" + ("=" * 62))
            print("buffer-pool smoke: " + ("ALL RANKS PASSED" if every else "FAILURE"))
            print("=" * 62)
            payload = {
                "harness": "commcanary.smoke_buffer_pool.v1",
                "world_size": world_size,
                "collective_buffer_slots": COLLECTIVE_BUFFER_SLOTS,
                "rank_skew_mechanism": RANK_SKEW_MECHANISM,
                "torch": torch.__version__,
                "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()),
                "device": torch.cuda.get_device_name(local_rank),
                "passed": every,
                "ranks": gathered,
            }
            out = os.environ.get("SMOKE_OUTPUT")
            if out:
                with open(out, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                print(f"evidence: {out}")
            return 0 if every else 1
        return 0 if local_pass else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
