"""Frozen representation schedule for the replicated decision gate."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

REPRESENTATION_IDS = (
    "source",
    "exact_work",
    "stratified",
    "isolated",
    "no_overlap",
    "no_rank_skew",
)
REPLICATED_ORDER_METHOD = "configuration-repetition-frozen-global-carryover-design.v4"
CONFIGURATION_ORDER_METHOD = "eight-configuration-williams-design.v1"
WILLIAMS_CYCLE_LENGTH = len(REPRESENTATION_IDS)

_WILLIAMS_BASE = (0, 1, 5, 2, 4, 3)
WILLIAMS_ROWS: Tuple[Tuple[str, ...], ...] = tuple(
    tuple(REPRESENTATION_IDS[(index + shift) % WILLIAMS_CYCLE_LENGTH] for index in _WILLIAMS_BASE)
    for shift in range(WILLIAMS_CYCLE_LENGTH)
)

# This exact 24-row stream retains four complete Williams cycles while also
# balancing transitions across row boundaries.  Every directed non-self
# transition in the flattened representation stream occurs four or five times.
MEASURED_ROW_INDEX_SEQUENCE = (
    0,
    1,
    2,
    3,
    4,
    5,
    0,
    2,
    1,
    3,
    5,
    4,
    2,
    0,
    4,
    3,
    1,
    5,
    1,
    0,
    5,
    3,
    2,
    4,
)
WARMUP_ROW_INDEX_SEQUENCE = tuple(range(WILLIAMS_CYCLE_LENGTH))
_CONFIGURATION_WILLIAMS_BASE = (0, 1, 7, 2, 6, 3, 5, 4)


def _rotated_row_index(base_index: int, configuration_repetition: int) -> int:
    return (base_index + configuration_repetition) % WILLIAMS_CYCLE_LENGTH


def configuration_order_by_repetition(configuration_ids: Sequence[str]) -> Tuple[Tuple[str, ...], ...]:
    """Freeze the eight-row position- and carryover-balanced configuration order."""

    if len(configuration_ids) != len(_CONFIGURATION_WILLIAMS_BASE) or list(configuration_ids) != sorted(
        set(configuration_ids)
    ):
        raise ValueError("configuration schedule requires eight sorted unique configuration IDs")
    return tuple(
        tuple(configuration_ids[(index + shift) % len(configuration_ids)] for index in _CONFIGURATION_WILLIAMS_BASE)
        for shift in range(len(configuration_ids))
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
    base_index = MEASURED_ROW_INDEX_SEQUENCE[iteration % len(MEASURED_ROW_INDEX_SEQUENCE)]
    return WILLIAMS_ROWS[_rotated_row_index(base_index, configuration_repetition)]


def warmup_representation_order(
    warmup_iteration: int,
    *,
    configuration_repetition: int,
) -> Tuple[str, ...]:
    """Return one row of the repetition-rotated full warmup cycle."""

    if isinstance(warmup_iteration, bool) or not isinstance(warmup_iteration, int) or warmup_iteration < 0:
        raise ValueError("warmup_iteration must be a non-negative integer")
    if (
        isinstance(configuration_repetition, bool)
        or not isinstance(configuration_repetition, int)
        or configuration_repetition < 0
    ):
        raise ValueError("configuration_repetition must be a non-negative integer")
    base_index = WARMUP_ROW_INDEX_SEQUENCE[warmup_iteration % len(WARMUP_ROW_INDEX_SEQUENCE)]
    return WILLIAMS_ROWS[_rotated_row_index(base_index, configuration_repetition)]


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
    warmup: int = WILLIAMS_CYCLE_LENGTH,
) -> Dict[str, Any]:
    """Return the complete frozen measured and warmup row matrices."""

    if (
        isinstance(configuration_repetitions, bool)
        or not isinstance(configuration_repetitions, int)
        or configuration_repetitions <= 0
    ):
        raise ValueError("configuration_repetitions must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup <= 0 or warmup % WILLIAMS_CYCLE_LENGTH:
        raise ValueError(f"warmup must be a positive multiple of {WILLIAMS_CYCLE_LENGTH}")
    replicated_schedule(0, iterations=iterations)
    row_indices: List[List[int]] = []
    warmup_row_indices: List[List[int]] = []
    for repetition in range(configuration_repetitions):
        row_indices.append(
            [
                _rotated_row_index(
                    MEASURED_ROW_INDEX_SEQUENCE[iteration % len(MEASURED_ROW_INDEX_SEQUENCE)],
                    repetition,
                )
                for iteration in range(iterations)
            ]
        )
        warmup_row_indices.append(
            [
                _rotated_row_index(
                    WARMUP_ROW_INDEX_SEQUENCE[iteration % len(WARMUP_ROW_INDEX_SEQUENCE)],
                    repetition,
                )
                for iteration in range(warmup)
            ]
        )
    return {
        "rows": [list(row) for row in WILLIAMS_ROWS],
        "row_index_by_configuration_repetition": row_indices,
        "warmup_row_index_by_configuration_repetition": warmup_row_indices,
    }


__all__ = [
    "CONFIGURATION_ORDER_METHOD",
    "REPRESENTATION_IDS",
    "REPLICATED_ORDER_METHOD",
    "WILLIAMS_CYCLE_LENGTH",
    "WILLIAMS_ROWS",
    "MEASURED_ROW_INDEX_SEQUENCE",
    "WARMUP_ROW_INDEX_SEQUENCE",
    "configuration_order_by_repetition",
    "frozen_schedule_inventory",
    "replicated_schedule",
    "representation_order",
    "warmup_representation_order",
]
