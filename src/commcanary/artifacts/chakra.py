"""Bounded, byte-preserving access to MLCommons Chakra execution traces.

Chakra ET files are a sequence of varint-length-delimited protobuf messages:
one ``GlobalMetadata`` message followed by ``Node`` messages. CommCanary does
not vendor generated Chakra bindings. It reads only the stable graph fields
needed to select a dependency-closed subgraph plus the standard ``comm_type``
and ``comm_size`` collective attributes, and retains each complete protobuf
message verbatim.

This boundary intentionally does not interpret ``Node.inputs``, ``outputs``,
or other attributes. Workload-specific semantics live in a separately
committed CommCanary projection, so an unknown protobuf field cannot be
silently lost or rewritten when a physical canary is emitted.
"""

from __future__ import annotations

import gzip
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union

from ..errors import SchemaError
from ..resources import DEFAULT_RESOURCE_LIMITS, ResourceLimits

Pathish = Union[str, Path]
MAX_PROTOBUF_VARINT_BYTES = 10


@dataclass(frozen=True)
class ChakraNode:
    """The graph fields CommCanary reads plus the exact protobuf bytes."""

    node_id: int
    node_type: int
    control_dependencies: Tuple[int, ...]
    data_dependencies: Tuple[int, ...]
    attributes: Tuple["ChakraAttribute", ...]
    raw_message: bytes

    @property
    def dependencies(self) -> Tuple[int, ...]:
        """Return stable de-duplicated dependencies from both edge classes."""

        return tuple(dict.fromkeys((*self.control_dependencies, *self.data_dependencies)))


@dataclass(frozen=True)
class ChakraAttribute:
    """One named Chakra attribute retained with its parsed wire fields."""

    name: str
    fields: Tuple[Tuple[int, int, Union[int, bytes]], ...]
    raw_message: bytes


@dataclass(frozen=True)
class ChakraExecutionTrace:
    """One validated, bounded Chakra ET byte snapshot."""

    metadata_version: str
    metadata_raw: bytes
    nodes: Tuple[ChakraNode, ...]
    source_sha256: str
    source_bytes: int
    source_compression: str
    source_raw: bytes

    @property
    def node_ids(self) -> Tuple[int, ...]:
        return tuple(node.node_id for node in self.nodes)


@dataclass(frozen=True)
class ChakraNodeRecord:
    """Minimal Chakra node fields emitted by CommCanary's capture adapter."""

    node_id: int
    name: str
    node_type: int
    control_dependencies: Tuple[int, ...] = ()
    data_dependencies: Tuple[int, ...] = ()
    int64_attributes: Tuple[Tuple[str, int], ...] = ()


def encode_chakra_execution_trace(
    metadata_version: str,
    nodes: Iterable[ChakraNodeRecord],
) -> bytes:
    """Encode the strict Chakra protobuf subset used by the physical runner.

    The result uses Chakra's length-delimited on-disk framing. It is decoded
    again before return so duplicate IDs, missing dependencies, and cycles
    cannot leave this boundary.
    """

    if not isinstance(metadata_version, str) or not metadata_version:
        raise SchemaError("Chakra metadata version must be a non-empty string")
    records = tuple(nodes)
    if not records:
        raise SchemaError("Chakra execution trace must contain at least one node")
    metadata = _protobuf_bytes_field(1, metadata_version.encode("utf-8"))
    frames = [metadata]
    for index, record in enumerate(records):
        if not isinstance(record, ChakraNodeRecord):
            raise TypeError(f"Chakra node record {index} has an unsupported type")
        frames.append(_encode_node_record(record, index=index))
    encoded = b"".join(_encode_varint(len(frame)) + frame for frame in frames)
    decode_chakra_execution_trace(encoded)
    return encoded


def load_chakra_execution_trace(
    path: Pathish,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> ChakraExecutionTrace:
    """Read and validate one bounded Chakra ET file.

    Gzip input is accepted because Chakra tooling commonly supports it.  The
    identity is always over the exact supplied bytes; the expanded protobuf
    stream is independently limited by ``max_input_bytes``.
    """

    source = Path(path)
    try:
        with source.open("rb") as handle:
            raw = handle.read(limits.max_input_bytes + 1)
    except OSError as exc:
        raise SchemaError(f"cannot read Chakra ET {source}: {exc}") from exc
    if len(raw) > limits.max_input_bytes:
        raise SchemaError(f"Chakra ET input exceeds max_input_bytes={limits.max_input_bytes}")
    return decode_chakra_execution_trace(raw, limits=limits)


def decode_chakra_execution_trace(
    raw: bytes,
    *,
    limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS,
) -> ChakraExecutionTrace:
    """Validate an exact Chakra ET byte snapshot without protobuf bindings."""

    if not isinstance(raw, bytes):
        raise TypeError("raw Chakra ET input must be bytes")
    if len(raw) > limits.max_input_bytes:
        raise SchemaError(f"Chakra ET input exceeds max_input_bytes={limits.max_input_bytes}")
    exact_sha256 = hashlib.sha256(raw).hexdigest()
    compression = "gzip" if raw.startswith(b"\x1f\x8b") else "none"
    stream_bytes = _bounded_gzip_decode(raw, limits=limits) if compression == "gzip" else raw
    frames = _decode_frames(stream_bytes, limits=limits)
    if len(frames) < 2:
        raise SchemaError("Chakra ET must contain GlobalMetadata and at least one Node")

    metadata_raw = frames[0]
    metadata_version = _required_utf8_string(
        metadata_raw,
        field_number=1,
        label="GlobalMetadata.version",
    )

    nodes = tuple(_decode_node(frame, index=index) for index, frame in enumerate(frames[1:]))
    _validate_graph(nodes)
    return ChakraExecutionTrace(
        metadata_version=metadata_version,
        metadata_raw=metadata_raw,
        nodes=nodes,
        source_sha256=exact_sha256,
        source_bytes=len(raw),
        source_compression=compression,
        source_raw=raw,
    )


def chakra_dependency_closure(
    trace: ChakraExecutionTrace,
    selected_node_ids: Iterable[int],
) -> Tuple[int, ...]:
    """Return the source-order transitive dependency closure of selected IDs."""

    requested: Set[int] = set()
    for node_id in selected_node_ids:
        if not isinstance(node_id, int) or isinstance(node_id, bool) or node_id < 1:
            raise SchemaError("selected Chakra node IDs must be positive integers")
        requested.add(node_id)
    if not requested:
        raise SchemaError("a physical canary must select at least one Chakra node")

    by_id = {node.node_id: node for node in trace.nodes}
    unknown = sorted(requested.difference(by_id))
    if unknown:
        raise SchemaError(f"selected Chakra node IDs are absent from the source ET: {unknown[:10]}")

    closure = set(requested)
    pending = list(requested)
    while pending:
        node = by_id[pending.pop()]
        for dependency in node.dependencies:
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    return tuple(node.node_id for node in trace.nodes if node.node_id in closure)


def encode_chakra_subgraph(
    trace: ChakraExecutionTrace,
    selected_node_ids: Iterable[int],
) -> Tuple[bytes, Tuple[int, ...]]:
    """Emit an uncompressed, byte-preserving dependency-closed Chakra ET."""

    closure = chakra_dependency_closure(trace, selected_node_ids)
    selected = set(closure)
    frames = [trace.metadata_raw]
    frames.extend(node.raw_message for node in trace.nodes if node.node_id in selected)
    return b"".join(_encode_varint(len(frame)) + frame for frame in frames), closure


def chakra_int64_attribute(node: ChakraNode, name: str) -> int:
    """Read one exact signed-int64 Chakra attribute by name."""

    matches = [attribute for attribute in node.attributes if attribute.name == name]
    if len(matches) != 1:
        raise SchemaError(f"Chakra Node.id={node.node_id} attribute {name!r} must occur exactly once")
    value_fields = [(number, wire_type, value) for number, wire_type, value in matches[0].fields if 3 <= number <= 32]
    if len(value_fields) != 1:
        raise SchemaError(f"Chakra Node.id={node.node_id} attribute {name!r} must contain exactly one value")
    number, wire_type, value = value_fields[0]
    if number != 9 or wire_type != 0 or not isinstance(value, int):
        raise SchemaError(f"Chakra Node.id={node.node_id} attribute {name!r} must use int64_val")
    return value - (1 << 64) if value >= (1 << 63) else value


def _bounded_gzip_decode(raw: bytes, *, limits: ResourceLimits) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
            expanded = stream.read(limits.max_input_bytes + 1)
    except (OSError, EOFError) as exc:
        raise SchemaError(f"invalid gzip-compressed Chakra ET: {exc}") from exc
    if len(expanded) > limits.max_input_bytes:
        raise SchemaError(f"expanded Chakra ET exceeds max_input_bytes={limits.max_input_bytes}")
    return expanded


def _decode_frames(raw: bytes, *, limits: ResourceLimits) -> Tuple[bytes, ...]:
    frames: List[bytes] = []
    offset = 0
    while offset < len(raw):
        if len(frames) >= limits.max_chakra_messages:
            raise SchemaError(f"Chakra ET messages exceed max_chakra_messages={limits.max_chakra_messages}")
        length, offset = _decode_varint(raw, offset, label="Chakra frame length")
        if length > limits.max_chakra_message_bytes:
            raise SchemaError(
                f"Chakra ET message bytes={length} exceeds max_chakra_message_bytes={limits.max_chakra_message_bytes}"
            )
        end = offset + length
        if end > len(raw):
            raise SchemaError("truncated Chakra ET message")
        frames.append(raw[offset:end])
        offset = end
    if not frames:
        raise SchemaError("Chakra ET is empty")
    return tuple(frames)


def _encode_node_record(record: ChakraNodeRecord, *, index: int) -> bytes:
    if not isinstance(record.node_id, int) or isinstance(record.node_id, bool) or record.node_id < 1:
        raise SchemaError(f"Chakra node record {index} ID must be a positive integer")
    if not isinstance(record.name, str) or not record.name:
        raise SchemaError(f"Chakra node record {index} name must be a non-empty string")
    if not isinstance(record.node_type, int) or isinstance(record.node_type, bool) or record.node_type < 1:
        raise SchemaError(f"Chakra node record {index} type must be a positive integer")
    fields = [
        _protobuf_varint_field(1, record.node_id),
        _protobuf_bytes_field(2, record.name.encode("utf-8")),
        _protobuf_varint_field(3, record.node_type),
    ]
    if record.control_dependencies:
        fields.append(_protobuf_packed_varints(4, record.control_dependencies, label=f"Chakra node {record.node_id}"))
    if record.data_dependencies:
        fields.append(_protobuf_packed_varints(5, record.data_dependencies, label=f"Chakra node {record.node_id}"))
    seen_attributes: Set[str] = set()
    for name, value in record.int64_attributes:
        if not isinstance(name, str) or not name or name in seen_attributes:
            raise SchemaError(f"Chakra node {record.node_id} attributes must have unique non-empty names")
        seen_attributes.add(name)
        if not isinstance(value, int) or isinstance(value, bool) or not -(1 << 63) <= value < (1 << 63):
            raise SchemaError(f"Chakra node {record.node_id} attribute {name!r} is outside signed int64")
        encoded_value = value if value >= 0 else value + (1 << 64)
        attribute = _protobuf_bytes_field(1, name.encode("utf-8")) + _protobuf_varint_field(9, encoded_value)
        fields.append(_protobuf_bytes_field(10, attribute))
    return b"".join(fields)


def _protobuf_varint_field(field_number: int, value: int) -> bytes:
    return _encode_varint((field_number << 3) | 0) + _encode_varint(value)


def _protobuf_bytes_field(field_number: int, value: bytes) -> bytes:
    return _encode_varint((field_number << 3) | 2) + _encode_varint(len(value)) + value


def _protobuf_packed_varints(field_number: int, values: Iterable[int], *, label: str) -> bytes:
    encoded_values = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SchemaError(f"{label} dependencies must be positive integers")
        encoded_values.append(_encode_varint(value))
    payload = b"".join(encoded_values)
    return _protobuf_bytes_field(field_number, payload)


def _decode_node(raw: bytes, *, index: int) -> ChakraNode:
    fields = _protobuf_fields(raw, label=f"Node[{index}]")
    node_id = _required_scalar_from_fields(fields, field_number=1, label=f"Node[{index}].id")
    node_type = _required_scalar_from_fields(fields, field_number=3, label=f"Node[{index}].type")
    if node_id < 1:
        raise SchemaError(f"Node[{index}].id must be positive")
    if node_type < 1:
        raise SchemaError(f"Node[{index}].type must be a known positive enum value")
    control = _repeated_varints(fields, field_number=4, label=f"Node[{index}].ctrl_deps")
    data = _repeated_varints(fields, field_number=5, label=f"Node[{index}].data_deps")
    _require_unique(control, label=f"Node[{index}].ctrl_deps")
    _require_unique(data, label=f"Node[{index}].data_deps")
    attributes = _decode_attributes(fields, node_index=index)
    return ChakraNode(
        node_id=node_id,
        node_type=node_type,
        control_dependencies=control,
        data_dependencies=data,
        attributes=attributes,
        raw_message=raw,
    )


def _decode_attributes(
    fields: Tuple[Tuple[int, int, Union[int, bytes]], ...],
    *,
    node_index: int,
) -> Tuple[ChakraAttribute, ...]:
    attributes: List[ChakraAttribute] = []
    names: Set[str] = set()
    for field_number, wire_type, value in fields:
        if field_number != 10:
            continue
        if wire_type != 2 or not isinstance(value, bytes):
            raise SchemaError(f"Node[{node_index}].attr must use length-delimited protobuf messages")
        attribute_fields = _protobuf_fields(value, label=f"Node[{node_index}].attr")
        name = _required_utf8_string_from_fields(
            attribute_fields,
            field_number=1,
            label=f"Node[{node_index}].attr.name",
        )
        if name in names:
            raise SchemaError(f"Node[{node_index}].attr repeats name {name!r}")
        names.add(name)
        attributes.append(
            ChakraAttribute(
                name=name,
                fields=attribute_fields,
                raw_message=value,
            )
        )
    return tuple(attributes)


def _validate_graph(nodes: Tuple[ChakraNode, ...]) -> None:
    by_id: Dict[int, ChakraNode] = {}
    for node in nodes:
        if node.node_id in by_id:
            raise SchemaError(f"Chakra ET repeats Node.id={node.node_id}")
        by_id[node.node_id] = node
    for node in nodes:
        for dependency in node.dependencies:
            if dependency == node.node_id:
                raise SchemaError(f"Chakra Node.id={node.node_id} depends on itself")
            if dependency not in by_id:
                raise SchemaError(f"Chakra Node.id={node.node_id} references missing dependency Node.id={dependency}")

    # Use Kahn's algorithm instead of recursive DFS.  A valid bounded ET may
    # contain far more nodes than Python's recursion limit.
    remaining_dependencies = {node.node_id: len(node.dependencies) for node in nodes}
    dependents: Dict[int, List[int]] = {node.node_id: [] for node in nodes}
    for node in nodes:
        for dependency in node.dependencies:
            dependents[dependency].append(node.node_id)
    ready = [node.node_id for node in reversed(nodes) if remaining_dependencies[node.node_id] == 0]
    visited_count = 0
    while ready:
        node_id = ready.pop()
        visited_count += 1
        for dependent in dependents[node_id]:
            remaining_dependencies[dependent] -= 1
            if remaining_dependencies[dependent] == 0:
                ready.append(dependent)
    if visited_count != len(nodes):
        cycle_node = next(node.node_id for node in nodes if remaining_dependencies[node.node_id] > 0)
        raise SchemaError(f"Chakra dependency graph contains a cycle at Node.id={cycle_node}")


def _protobuf_fields(raw: bytes, *, label: str) -> Tuple[Tuple[int, int, Union[int, bytes]], ...]:
    fields: List[Tuple[int, int, Union[int, bytes]]] = []
    offset = 0
    while offset < len(raw):
        key, offset = _decode_varint(raw, offset, label=f"{label} field key")
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number < 1:
            raise SchemaError(f"{label} contains protobuf field number zero")
        value: Union[int, bytes]
        if wire_type == 0:
            value, offset = _decode_varint(raw, offset, label=f"{label} field {field_number}")
        elif wire_type == 1:
            end = offset + 8
            if end > len(raw):
                raise SchemaError(f"{label} has a truncated fixed64 field")
            value = raw[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _decode_varint(raw, offset, label=f"{label} length")
            end = offset + length
            if end > len(raw):
                raise SchemaError(f"{label} has a truncated length-delimited field")
            value = raw[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(raw):
                raise SchemaError(f"{label} has a truncated fixed32 field")
            value = raw[offset:end]
            offset = end
        else:
            raise SchemaError(f"{label} uses unsupported protobuf wire type {wire_type}")
        fields.append((field_number, wire_type, value))
    return tuple(fields)


def _required_utf8_string(raw: bytes, *, field_number: int, label: str) -> str:
    return _required_utf8_string_from_fields(
        _protobuf_fields(raw, label=label),
        field_number=field_number,
        label=label,
    )


def _required_utf8_string_from_fields(
    fields: Tuple[Tuple[int, int, Union[int, bytes]], ...],
    *,
    field_number: int,
    label: str,
) -> str:
    matches = [(wire_type, value) for number, wire_type, value in fields if number == field_number]
    if len(matches) != 1:
        raise SchemaError(f"{label} must occur exactly once")
    wire_type, value = matches[0]
    if wire_type != 2 or not isinstance(value, bytes):
        raise SchemaError(f"{label} must use protobuf string encoding")
    try:
        result = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{label} must contain valid UTF-8") from exc
    if not result:
        raise SchemaError(f"{label} must be non-empty")
    return result


def _required_scalar_from_fields(
    fields: Tuple[Tuple[int, int, Union[int, bytes]], ...],
    *,
    field_number: int,
    label: str,
) -> int:
    matches = [(wire_type, value) for number, wire_type, value in fields if number == field_number]
    if len(matches) != 1:
        raise SchemaError(f"{label} must occur exactly once")
    wire_type, value = matches[0]
    if wire_type != 0 or not isinstance(value, int):
        raise SchemaError(f"{label} must use protobuf varint encoding")
    return value


def _repeated_varints(
    fields: Tuple[Tuple[int, int, Union[int, bytes]], ...],
    *,
    field_number: int,
    label: str,
) -> Tuple[int, ...]:
    values: List[int] = []
    for number, wire_type, value in fields:
        if number != field_number:
            continue
        if wire_type == 0 and isinstance(value, int):
            values.append(value)
        elif wire_type == 2 and isinstance(value, bytes):
            offset = 0
            while offset < len(value):
                item, offset = _decode_varint(value, offset, label=label)
                values.append(item)
        else:
            raise SchemaError(f"{label} must use packed or repeated protobuf varints")
    for value in values:
        if value < 1:
            raise SchemaError(f"{label} values must be positive Node IDs")
    return tuple(values)


def _require_unique(values: Tuple[int, ...], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise SchemaError(f"{label} contains duplicate Node IDs")


def _decode_varint(raw: bytes, offset: int, *, label: str) -> Tuple[int, int]:
    value = 0
    for index in range(MAX_PROTOBUF_VARINT_BYTES):
        if offset >= len(raw):
            raise SchemaError(f"truncated {label}")
        byte = raw[offset]
        offset += 1
        if index == MAX_PROTOBUF_VARINT_BYTES - 1 and byte > 1:
            raise SchemaError(f"{label} exceeds uint64")
        value |= (byte & 0x7F) << (7 * index)
        if byte < 0x80:
            return value, offset
    raise SchemaError(f"{label} uses an overlong varint")


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint value must be non-negative")
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


__all__ = [
    "ChakraAttribute",
    "ChakraExecutionTrace",
    "ChakraNode",
    "chakra_dependency_closure",
    "chakra_int64_attribute",
    "decode_chakra_execution_trace",
    "encode_chakra_subgraph",
    "load_chakra_execution_trace",
]
