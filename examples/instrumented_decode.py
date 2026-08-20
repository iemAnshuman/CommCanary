from __future__ import annotations

import random
import time

from commcanary.capture import record_collective

RANKS = list(range(4))
GEMM_RECIPE = {
    "op": "gemm",
    "dtype": "bfloat16",
    "m": 1024,
    "n": 1024,
    "k": 1024,
    "source_kernel_count": 1,
    "source_kernel_duration_us": 19.08992,
}


def main() -> None:
    """Emit a deterministic synthetic trace for local workflow orientation."""

    rng = random.Random(8)
    for token in range(24):
        time.sleep(0.001)
        skew = 8.0 + (token % 5) * 3.2 + rng.uniform(-0.8, 1.0)
        record_collective(
            op="all_reduce",
            bytes=128 * 1024 if token % 2 else 64 * 1024,
            ranks=RANKS,
            dtype="bfloat16",
            reduction_op="sum",
            phase="decode",
            group="tp0",
            rank_arrival_us={str(rank): skew * rank / (len(RANKS) - 1) for rank in RANKS},
            compute_before_us=26.0 + rng.uniform(-2.0, 2.0),
            compute_overlap_us=17.0 + rng.uniform(-3.0, 4.0),
            compute_pressure=0.64,
            compute_recipe=[GEMM_RECIPE],
        )


if __name__ == "__main__":
    main()
