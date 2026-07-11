"""Typed operation identity with named, purpose-specific projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple

from .artifacts import JsonDict, as_int, normalize_ranks

CompressionKey = Tuple[
    str,
    str,
    int,
    Tuple[int, ...],
    str,
    Optional[int],
    Optional[int],
    Optional[str],
    Optional[str],
    Optional[int],
    int,
    bool,
]
BaselineShapeKey = Tuple[
    str,
    str,
    int,
    Tuple[int, ...],
    str,
    Optional[int],
    Optional[int],
    Optional[str],
    Optional[str],
]
SchedulerOrderingKey = Tuple[str, str, int, str, Tuple[int, ...], int, int, str, str]
CaptureCoalescingKey = Tuple[Any, ...]


def _optional_int(data: Mapping[str, Any], key: str) -> Optional[int]:
    return as_int(data.get(key)) if key in data else None


def _optional_text(data: Mapping[str, Any], key: str) -> Optional[str]:
    return str(data.get(key)) if key in data else None


@dataclass(frozen=True)
class NoiseIdentity:
    """Replay-noise identity projected from an operation occurrence."""

    phase: str
    op: str
    ranks: Tuple[int, ...]
    group: str
    arrival_offsets_us: Tuple[float, ...]
    occurrence: int
    sender_rank: Optional[int]
    receiver_rank: Optional[int]
    message_sequence: Optional[int]
    tag: Optional[str]
    channel: Optional[str]

    def to_wire(self) -> JsonDict:
        """Return the exact canonical mapping hashed by replay noise."""

        identity: JsonDict = {
            "phase": self.phase,
            "op": self.op,
            "ranks": list(self.ranks),
            "group": self.group,
            "arrival_offsets_us": list(self.arrival_offsets_us),
            "occurrence": self.occurrence,
        }
        for key, numeric_value in (
            ("sender_rank", self.sender_rank),
            ("receiver_rank", self.receiver_rank),
            ("message_sequence", self.message_sequence),
        ):
            if numeric_value is not None:
                identity[key] = numeric_value
        for key, text_value in (("tag", self.tag), ("channel", self.channel)):
            if text_value is not None:
                identity[key] = text_value
        return identity


@dataclass(frozen=True)
class OperationIdentity:
    """Normalized operation fields shared by named identity projections.

    Each consumer has a named projection so adding a field to compression does
    not silently change scheduling, capture grouping, or deterministic noise.
    """

    phase: str
    op: str
    byte_count: int
    ranks: Tuple[int, ...]
    group: str
    sender_rank: Optional[int]
    receiver_rank: Optional[int]
    tag: Optional[str]
    channel: Optional[str]
    message_sequence: Optional[int]
    concurrent_groups: int
    custom_op: bool
    capture_session_id: Any
    collective_id: Any
    shard: Any
    event_id: Any

    @classmethod
    def from_mapping(cls, operation: Mapping[str, Any]) -> "OperationIdentity":
        """Normalize a validated trace/canary operation without retaining it."""

        ranks = tuple(normalize_ranks(operation.get("ranks"))) if "ranks" in operation else ()
        return cls(
            phase=str(operation.get("phase", "unknown")),
            op=str(operation.get("op", "unknown")),
            byte_count=as_int(operation.get("bytes"), 0),
            ranks=ranks,
            group=str(operation.get("group", "default")),
            sender_rank=_optional_int(operation, "sender_rank"),
            receiver_rank=_optional_int(operation, "receiver_rank"),
            tag=_optional_text(operation, "tag"),
            channel=_optional_text(operation, "channel"),
            message_sequence=_optional_int(operation, "message_sequence"),
            concurrent_groups=as_int(operation.get("concurrent_groups"), 1),
            custom_op=operation.get("custom_op") is True,
            capture_session_id=operation.get("capture_session_id"),
            collective_id=operation.get("collective_id"),
            shard=operation.get("shard"),
            event_id=operation.get("id"),
        )

    def compression_key(self) -> CompressionKey:
        """Fields that permit adjacent compiler grouping and motif stride."""

        return (
            self.phase,
            self.op,
            self.byte_count,
            self.ranks,
            self.group,
            self.sender_rank,
            self.receiver_rank,
            self.tag,
            self.channel,
            self.message_sequence,
            self.concurrent_groups,
            self.custom_op,
        )

    def scheduler_ordering_key(self) -> SchedulerOrderingKey:
        """Stable operation-ordering ablation key; occurrence is excluded."""

        return (
            self.phase,
            self.op,
            self.byte_count,
            self.group,
            self.ranks,
            self.sender_rank if self.sender_rank is not None else -1,
            self.receiver_rank if self.receiver_rank is not None else -1,
            self.tag or "",
            self.channel or "",
        )

    def scheduler_resource_label(self) -> str:
        """Scheduler serialization resource used by replay."""

        if self.op == "point_to_point" and self.sender_rank is not None and self.receiver_rank is not None:
            channel = self.channel or "default"
            tag = self.tag or "default"
            sequence = str(self.message_sequence) if self.message_sequence is not None else "unknown"
            return (
                f"p2p:{self.group}:{self.sender_rank}->{self.receiver_rank}:channel={channel}:tag={tag}:seq={sequence}"
            )
        if self.group and self.group != "default":
            return self.group
        if self.ranks:
            return "default:ranks=" + ",".join(str(rank) for rank in self.ranks)
        return self.group or "default"

    def capture_coalescing_key(self) -> CaptureCoalescingKey:
        """Capture occurrence key; operation-shape fields are intentionally excluded."""

        if self.capture_session_id is None or self.collective_id is None:
            return "uncoalesced", self.shard, self.event_id
        return self.capture_session_id, self.collective_id

    def noise_identity(self, offsets: Iterable[float], *, occurrence: int) -> NoiseIdentity:
        """Fields that seed deterministic replay noise for one occurrence."""

        return NoiseIdentity(
            phase=self.phase,
            op=self.op,
            ranks=self.ranks,
            group=self.group,
            arrival_offsets_us=tuple(round(value, 9) for value in offsets),
            occurrence=occurrence,
            sender_rank=self.sender_rank,
            receiver_rank=self.receiver_rank,
            message_sequence=self.message_sequence,
            tag=self.tag,
            channel=self.channel,
        )

    def baseline_shape_key(self) -> BaselineShapeKey:
        """Research-baseline grouping identity including phase."""

        return self._baseline_shape_key(self.phase)

    def isolated_baseline_shape_key(self) -> BaselineShapeKey:
        """Isolated-collective grouping identity deliberately excluding phase."""

        return self._baseline_shape_key("*")

    def _baseline_shape_key(self, phase: str) -> BaselineShapeKey:
        return (
            phase,
            self.op,
            self.byte_count,
            self.ranks,
            self.group,
            self.sender_rank,
            self.receiver_rank,
            self.tag,
            self.channel,
        )


__all__ = ["NoiseIdentity", "OperationIdentity"]
