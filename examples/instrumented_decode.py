from __future__ import annotations

import random
import time

from commcanary.capture import record_collective


def main() -> None:
    rng = random.Random(8)
    ranks = list(range(8))
    for token in range(24):
        time.sleep(0.001)
        skew = 8.0 + (token % 5) * 3.2 + rng.uniform(-0.8, 1.0)
        record_collective(
            op="all_reduce",
            bytes=128 * 1024 if token % 2 else 64 * 1024,
            ranks=ranks,
            phase="decode",
            group="tp0",
            rank_arrival_us={str(rank): skew * rank / 7.0 for rank in ranks},
            compute_before_us=26.0 + rng.uniform(-2.0, 2.0),
            compute_overlap_us=17.0 + rng.uniform(-3.0, 4.0),
            compute_pressure=0.64,
        )


if __name__ == "__main__":
    main()