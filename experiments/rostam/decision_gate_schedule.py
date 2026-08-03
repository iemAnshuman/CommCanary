"""Frozen representation schedule for the replicated decision gate."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

REPRESENTATION_IDS = (
    "source",
    "exact_work",
    "stratified",
    "isolated",
    "no_overlap",
    "no_rank_skew",
)
REPLICATED_ORDER_METHOD = "configuration-repetition-williams-design.v3"
WILLIAMS_CYCLE_LENGTH = len(REPRESENTATION_IDS)

_WILLIAMS_BASE = (0, 1, 5, 2, 4, 3)
WILLIAMS_ROWS: Tuple[Tuple[str, ...], ...] = tuple(
    tuple(REPRESENTATION_IDS[(index + shift) % WILLIAMS_CYCLE_LENGTH] for index in _WILLIAMS_BASE)
    for shift in range(WILLIAMS_CYCLE_LENGTH)
)


def representation_order(
    iteration: int,
    *,
    configuration_repetition: Optional[int] = None,
) -> Tuple[str, ...]:
    """Return the legacy rotation or one row of the replicated Williams design."""

    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    if configuration_repetition is None:
        offset = iteration % len(REPRESENTATION_IDS)
        return REPRESENTATION_IDS[offset:] + REPRESENTATION_IDS[:offset]
    if (
        isinstance(configuration_repetition, bool)
        or not isinstance(configuration_repetition, int)
        or configuration_repetition < 0
    ):
        raise ValueError("configuration_repetition must be a non-negative integer")
    return WILLIAMS_ROWS[(configuration_repetition + iteration) % WILLIAMS_CYCLE_LENGTH]


def replicated_schedule(
    configuration_repetition: int,
    *,
    iterations: int,
) -> Tuple[Tuple[str, ...], ...]:
    """Build one complete, position- and first-order-carryover-balanced schedule."""

    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
        or iterations % WILLIAMS_CYCLE_LENGTH
    ):
        raise ValueError(f"iterations must be a positive multiple of {WILLIAMS_CYCLE_LENGTH}")
    return tuple(
        representation_order(iteration, configuration_repetition=configuration_repetition)
        for iteration in range(iterations)
    )


def frozen_schedule_inventory(
    *,
    configuration_repetitions: int,
    iterations: int,
) -> Dict[str, Any]:
    """Return the compact policy inventory that determines every schedule row."""

    if (
        isinstance(configuration_repetitions, bool)
        or not isinstance(configuration_repetitions, int)
        or configuration_repetitions <= 0
    ):
        raise ValueError("configuration_repetitions must be a positive integer")
    replicated_schedule(0, iterations=iterations)
    row_indices: List[List[int]] = []
    for repetition in range(configuration_repetitions):
        row_indices.append(
            [
                (repetition + iteration) % WILLIAMS_CYCLE_LENGTH
                for iteration in range(iterations)
            ]
        )
    return {
        "rows": [list(row) for row in WILLIAMS_ROWS],
        "row_index_by_configuration_repetition": row_indices,
    }


__all__ = [
    "REPRESENTATION_IDS",
    "REPLICATED_ORDER_METHOD",
    "WILLIAMS_CYCLE_LENGTH",
    "WILLIAMS_ROWS",
    "frozen_schedule_inventory",
    "replicated_schedule",
    "representation_order",
]
