from __future__ import annotations

import copy
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pytest

from commcanary.adapters.chakra_capture import commcanary_trace_to_chakra
from commcanary.artifacts.chakra import (
    ChakraNodeRecord,
    chakra_dependency_closure,
    chakra_int64_attribute,
    decode_chakra_execution_trace,
    encode_chakra_execution_trace,
    encode_chakra_subgraph,
    load_chakra_execution_trace,
)
from commcanary.artifacts.physical_canary import (
    SUPPORTED_PHYSICAL_DOMAIN,
    leakage_assessment,
    validate_chakra_projection,
    validate_physical_gate_result,
    validate_physical_oracle_corpus,
    with_content_identity,
)
from commcanary.cli import main
from commcanary.errors import CommCanaryIOError, SchemaError
from commcanary.formats import (
    CHAKRA_PROJECTION_FORMAT,
    PHYSICAL_CANARY_POLICY_FORMAT,
    PHYSICAL_GATE_OBSERVATION_FORMAT,
    PHYSICAL_ORACLE_CORPUS_FORMAT,
    TRACE_FORMAT,
)
from commcanary.reporting import (
    physical_gate_junit_bytes,
    physical_gate_sarif,
    render_physical_gate_html,
)
from commcanary.resources import ResourceLimits
from commcanary.services import evaluate_physical_gate, synthesize_physical_decision_canary
from commcanary.workflows import build_physical_canary_bundle, verify_physical_canary_bundle


def _varint(value: int) -> bytes:
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _packed(number: int, values: Iterable[int]) -> bytes:
    payload = b"".join(_varint(value) for value in values)
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _bytes_field(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _attribute(name: str, value: int) -> bytes:
    payload = _bytes_field(1, name.encode("utf-8")) + _field(9, value)
    return _bytes_field(10, payload)


def _frame(payload: bytes) -> bytes:
    return _varint(len(payload)) + payload


def chakra_bytes(*, first_comm_type: int = 0, first_comm_size: int = 16) -> bytes:
    metadata = _bytes_field(1, b"0.0.4")
    nodes = [
        _field(1, 1) + _field(3, 4) + _field(99, 17),
        _field(1, 2)
        + _field(3, 7)
        + _packed(4, [1])
        + _attribute("comm_type", first_comm_type)
        + _attribute("comm_size", first_comm_size),
        _field(1, 3) + _field(3, 4) + _packed(4, [2]),
        _field(1, 4) + _field(3, 7) + _packed(5, [3]) + _attribute("comm_type", 0) + _attribute("comm_size", 32),
        _field(1, 5) + _field(3, 4) + _packed(4, [4]),
        _field(1, 6) + _field(3, 4) + _field(4, 5),
    ]
    return b"".join([_frame(metadata), *(_frame(node) for node in nodes)])


def captured_trace() -> Dict[str, Any]:
    recipe = {
        "op": "gemm",
        "dtype": "bfloat16",
        "m": 2,
        "n": 4,
        "k": 3,
        "source_kernel_count": 1,
        "source_kernel_duration_us": 2.0,
    }
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "captured-vllm"},
        "system": {"source_format": "pytorch-kineto"},
        "events": [
            {
                "id": "collective-0",
                "op": "all_reduce",
                "dtype": "bfloat16",
                "bytes": 32,
                "ranks": [0, 1, 2, 3],
                "group": "default",
                "start_us": 10.0,
                "phase": "decode",
                "reduction_op": "sum",
                "compute_overlap_us": 2.0,
                "rank_arrival_us": {"0": 0.0, "1": 0.5, "2": 1.0, "3": 1.5},
                "compute_recipe_by_rank": {str(rank): [copy.deepcopy(recipe)] for rank in range(4)},
            }
        ],
    }


def policy() -> Dict[str, Any]:
    return with_content_identity(
        {
            "format": PHYSICAL_CANARY_POLICY_FORMAT,
            "supported_domain": SUPPORTED_PHYSICAL_DOMAIN,
            "runtime_budget_seconds": 10.0,
            "severe_regression_threshold_pct": 5.0,
            "required_feature_tags": ["rare_tail"],
            "required_baselines": [],
            "runner": {
                "oci_digest": f"sha256:{hashlib.sha256(b'fixture-runner').hexdigest()}",
                "execution_protocol": "chakra-et-collective-graph.v1",
                "world_size": 4,
            },
            "qualification_gates": {
                "minimum_runtime_reduction_ratio": 10.0,
                "maximum_severe_false_negatives": 0,
                "maximum_false_positive_rate": 0.1,
                "minimum_pairwise_decision_agreement": 0.9,
            },
            "gate_metrics": [
                {
                    "name": "p99_latency_ms",
                    "direction": "lower_is_better",
                    "regression_threshold_pct": 5.0,
                    "mandatory": True,
                },
                {
                    "name": "throughput_per_second",
                    "direction": "higher_is_better",
                    "regression_threshold_pct": 5.0,
                    "mandatory": True,
                },
            ],
            "minimum_samples": 5,
            "uncertainty": {
                "method": "percentile_bootstrap_median_difference",
                "confidence": 0.95,
                "bootstrap_resamples": 1000,
                "seed": 0,
            },
            "noise": {"max_relative_iqr_pct": 20.0},
            "privacy": {
                "maximum_leakage_score": 1.0,
                "private_exchange_requires_reviewed_opaque_attributes": True,
            },
        },
        "policy_id",
    )


def projection(raw: bytes) -> Dict[str, Any]:
    trace = decode_chakra_execution_trace(raw)

    def gemm_node(
        node_id: int,
        m: int,
        *,
        feature_tags: Sequence[str],
        disclosures: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "node_id": node_id,
            "chakra_node_type": 4,
            "operation": "gemm",
            "executed_flops": 8 * m,
            "tensor_bytes": 16 * m + 8,
            "phase": "decode",
            "feature_tags": list(feature_tags),
            "disclosures": list(disclosures),
            "execution": {
                "dtype": "float16",
                "rank_recipes": [{"rank": rank, "m": m, "n": 1, "k": 1} for rank in range(4)],
            },
        }

    node_rows = [
        gemm_node(1, 1, feature_tags=[], disclosures=["hidden_dimension"]),
        {
            "node_id": 2,
            "chakra_node_type": 7,
            "operation": "all_reduce",
            "executed_flops": 0,
            "tensor_bytes": 16,
            "phase": "decode",
            "feature_tags": ["rare_tail"],
            "disclosures": [],
            "execution": {"dtype": "float16", "ranks": [0, 1, 2, 3], "reduction": "sum"},
        },
        gemm_node(3, 2, feature_tags=["overlap"], disclosures=[]),
        {
            "node_id": 4,
            "chakra_node_type": 7,
            "operation": "all_reduce",
            "executed_flops": 0,
            "tensor_bytes": 32,
            "phase": "decode",
            "feature_tags": [],
            "disclosures": [],
            "execution": {"dtype": "float16", "ranks": [0, 1, 2, 3], "reduction": "sum"},
        },
        gemm_node(5, 3, feature_tags=["rank_skew"], disclosures=[]),
        gemm_node(6, 4, feature_tags=[], disclosures=[]),
    ]
    return with_content_identity(
        {
            "format": CHAKRA_PROJECTION_FORMAT,
            "source_et": {
                "sha256": trace.source_sha256,
                "bytes": trace.source_bytes,
                "metadata_version": trace.metadata_version,
            },
            "capture_provenance": {
                "adapter": "fixture-adapter.v1",
                "source_format": "fixture.trace.v1",
                "source_sha256": hashlib.sha256(b"fixture-source").hexdigest(),
            },
            "supported_domain": SUPPORTED_PHYSICAL_DOMAIN,
            "nodes": node_rows,
            "regions": [
                {
                    "region_id": "r0",
                    "node_ids": [1, 2],
                    "feature_tags": ["rare_tail"],
                    "disclosures": ["hidden_dimension"],
                },
                {
                    "region_id": "r1",
                    "node_ids": [1, 2, 3, 4],
                    "feature_tags": ["overlap", "rare_tail"],
                    "disclosures": ["hidden_dimension"],
                },
                {
                    "region_id": "r2",
                    "node_ids": [1, 2, 3, 4, 5],
                    "feature_tags": ["overlap", "rank_skew", "rare_tail"],
                    "disclosures": ["hidden_dimension"],
                },
                {
                    "region_id": "r3",
                    "node_ids": [1, 2, 3, 4, 5, 6],
                    "feature_tags": ["overlap", "rank_skew", "rare_tail"],
                    "disclosures": ["hidden_dimension"],
                },
            ],
            "privacy_review": {
                "opaque_chakra_attributes_reviewed": True,
                "reviewed_disclosures": ["hidden_dimension"],
            },
        },
        "projection_id",
    )


def _metrics(runtime: float, *, collectives: int, flops: int) -> Dict[str, Any]:
    return {
        "physical_runtime_seconds": runtime,
        "gpu_seconds": runtime * 4,
        "gpu_count": 4,
        "executed_collectives": collectives,
        "executed_flops": flops,
        "peak_memory_bytes": 1024,
    }


def corpus(
    raw: bytes,
    projection_document: Mapping[str, Any],
    policy_document: Mapping[str, Any],
    *,
    evidence_kind: str = "measured",
) -> Dict[str, Any]:
    perturbations = [
        ("train-ok", "training", "pass", 0.0),
        ("train-regression", "training", "fail", 8.0),
        ("holdout-ok", "holdout", "pass", 0.0),
        ("holdout-regression", "holdout", "fail", 12.0),
    ]
    truth = [
        {
            "perturbation_id": perturbation_id,
            "split": split,
            "application_decision": decision,
            "regression_magnitude_pct": magnitude,
            "severe": decision == "fail" and magnitude >= 5.0,
            "application_evidence_sha256": hashlib.sha256(f"application:{perturbation_id}".encode("utf-8")).hexdigest(),
            "environment_sha256": hashlib.sha256(f"environment:{perturbation_id}".encode("utf-8")).hexdigest(),
            "candidate_subject_sha256": hashlib.sha256(
                f"candidate-stack:{perturbation_id}".encode("utf-8")
            ).hexdigest(),
            "application_metrics": _metrics(100.0, collectives=20, flops=10_000),
        }
        for perturbation_id, split, decision, magnitude in perturbations
    ]

    def evaluation(
        candidate_id: str,
        selected: Sequence[str],
        decisions: Sequence[str],
        runtime: float,
        collectives: int,
        flops: int,
    ) -> Dict[str, Any]:
        region_map = {row["region_id"]: row for row in projection_document["regions"]}
        node_ids = {int(node_id) for region_id in selected for node_id in region_map[region_id]["node_ids"]}
        executable, _closure = encode_chakra_subgraph(trace, node_ids)
        return {
            "candidate_id": candidate_id,
            "method": "reduced_decision_canary",
            "selected_region_ids": list(selected),
            "executable_sha256": hashlib.sha256(executable).hexdigest(),
            "observations": [
                {
                    "perturbation_id": row[0],
                    "decision": decision,
                    "evidence_sha256": hashlib.sha256(f"{candidate_id}:{row[0]}".encode("utf-8")).hexdigest(),
                    "physical_metrics": _metrics(runtime, collectives=collectives, flops=flops),
                }
                for row, decision in zip(perturbations, decisions)
            ],
        }

    trace = decode_chakra_execution_trace(raw)
    return with_content_identity(
        {
            "format": PHYSICAL_ORACLE_CORPUS_FORMAT,
            "source_et_sha256": trace.source_sha256,
            "projection_id": projection_document["projection_id"],
            "policy_id": policy_document["policy_id"],
            "runner_oci_digest": policy_document["runner"]["oci_digest"],
            "baseline_subject_sha256": hashlib.sha256(b"baseline-subject").hexdigest(),
            "evidence_kind": evidence_kind,
            "application": {
                "ground_truth_kind": "application_ground_truth",
                "name": "dense-decoder-fixture",
                "engine": "fixture-engine",
                "artifact_sha256": hashlib.sha256(b"fixture-application").hexdigest(),
            },
            "perturbations": truth,
            "evaluations": [
                evaluation(
                    "candidate-small",
                    ["r0"],
                    ["pass", "pass", "pass", "pass"],
                    3.0,
                    1,
                    8,
                ),
                evaluation(
                    "candidate-counterexample-complete",
                    ["r1"],
                    ["pass", "fail", "pass", "fail"],
                    5.0,
                    2,
                    24,
                ),
                evaluation(
                    "candidate-larger",
                    ["r2"],
                    ["pass", "fail", "pass", "fail"],
                    8.0,
                    2,
                    48,
                ),
            ],
        },
        "corpus_id",
    )


def observation(
    bundle_id: str,
    role: str,
    *,
    p99: Sequence[float],
    throughput: Sequence[float],
    executable_sha256: str | None = None,
) -> Dict[str, Any]:
    return {
        "format": PHYSICAL_GATE_OBSERVATION_FORMAT,
        "bundle_id": bundle_id,
        "role": role,
        "subject_sha256": hashlib.sha256(f"{role}-subject".encode("utf-8")).hexdigest(),
        "environment_sha256": "a" * 64,
        "runner_oci_digest": policy()["runner"]["oci_digest"],
        "executable_sha256": executable_sha256 or hashlib.sha256(b"fixture-canary").hexdigest(),
        "evidence_sha256": hashlib.sha256(f"{role}-evidence".encode("utf-8")).hexdigest(),
        "samples": {
            "p99_latency_ms": list(p99),
            "throughput_per_second": list(throughput),
        },
    }


def test_chakra_reader_preserves_unknown_fields_and_emits_dependency_closure() -> None:
    raw = chakra_bytes()
    trace = decode_chakra_execution_trace(raw)

    assert trace.metadata_version == "0.0.4"
    assert trace.node_ids == (1, 2, 3, 4, 5, 6)
    assert trace.nodes[0].raw_message.endswith(_field(99, 17))
    reduced, closure = encode_chakra_subgraph(trace, [4])
    decoded = decode_chakra_execution_trace(reduced)

    assert closure == (1, 2, 3, 4)
    assert decoded.node_ids == closure
    assert decoded.nodes[0].raw_message == trace.nodes[0].raw_message


def test_leakage_probes_infer_exact_geometry_even_when_disclosure_labels_omit_it() -> None:
    raw = chakra_bytes()
    projection_document = projection(raw)

    assessment = leakage_assessment(projection_document, ["r1"])

    assert assessment["method"] == "deterministic-leakage-probes.v1"
    assert assessment["inferred_disclosed_categories"] == [
        "batch_token_geometry",
        "hidden_dimension",
        "parallelism_degree",
    ]
    assert assessment["inference_probes"]["parallelism_degree"]["status"] == "exact_value_exposed"
    assert assessment["inference_probes"]["topology"]["status"] == "not_inferred"


def test_chakra_writer_and_automatic_capture_emit_executable_projection() -> None:
    encoded = encode_chakra_execution_trace(
        "0.0.4",
        [
            ChakraNodeRecord(node_id=1, name="collective", node_type=7, int64_attributes=(("comm_type", 0),)),
            ChakraNodeRecord(node_id=2, name="compute", node_type=4, control_dependencies=(1,)),
        ],
    )
    decoded = decode_chakra_execution_trace(encoded)
    assert decoded.node_ids == (1, 2)
    assert chakra_int64_attribute(decoded.nodes[0], "comm_type") == 0

    result = commcanary_trace_to_chakra(captured_trace(), opaque_attributes_reviewed=True)
    trace = decode_chakra_execution_trace(result.chakra_et)
    validate_chakra_projection(result.projection, trace)

    assert trace.node_ids == (1, 2)
    assert trace.nodes[1].dependencies == ()
    assert result.projection["capture_provenance"]["adapter"].endswith(".v1")
    assert result.projection["regions"][0]["node_ids"] == [1, 2]
    assert set(result.projection["regions"][0]["feature_tags"]) >= {"overlap", "rank-skew", "rare-tail"}
    assert result.projection["nodes"][0]["execution"]["ranks"] == [0, 1, 2, 3]
    assert result.projection["nodes"][1]["execution"]["rank_recipes"][3]["rank"] == 3


def test_automatic_chakra_capture_rejects_incomplete_rank_programs() -> None:
    document = captured_trace()
    document["events"][0]["compute_recipe_by_rank"]["3"] = []
    with pytest.raises(SchemaError, match="has no executable GEMM recipe"):
        commcanary_trace_to_chakra(document)


def test_chakra_reader_bounds_compressed_expansion_and_rejects_cycles() -> None:
    padding = _varint((99 << 3) | 2) + _varint(1_000) + (b"\x00" * 1_000)
    padded = _frame(_bytes_field(1, b"0.0.4")) + _frame(_field(1, 1) + _field(3, 4) + padding)
    compressed = gzip.compress(padded, mtime=0)
    assert len(compressed) < 200
    with pytest.raises(SchemaError, match="expanded Chakra ET"):
        decode_chakra_execution_trace(compressed, limits=ResourceLimits(max_input_bytes=200))

    metadata = _frame(_bytes_field(1, b"0.0.4"))
    node1 = _frame(_field(1, 1) + _field(3, 4) + _packed(4, [2]))
    node2 = _frame(_field(1, 2) + _field(3, 7) + _packed(4, [1]))
    with pytest.raises(SchemaError, match="cycle"):
        decode_chakra_execution_trace(metadata + node1 + node2)


def test_chakra_reader_handles_a_graph_deeper_than_python_recursion() -> None:
    frames = [_frame(_bytes_field(1, b"0.0.4"))]
    for node_id in range(1, 1_501):
        dependencies = b"" if node_id == 1 else _packed(4, [node_id - 1])
        frames.append(_frame(_field(1, node_id) + _field(3, 4) + dependencies))

    trace = decode_chakra_execution_trace(b"".join(frames))

    assert len(trace.nodes) == 1_500
    assert chakra_dependency_closure(trace, [1_500]) == tuple(range(1, 1_501))


def test_chakra_reader_rejects_bounded_framing_and_selection_failures(tmp_path: Path) -> None:
    raw = chakra_bytes()
    with pytest.raises(TypeError, match="must be bytes"):
        decode_chakra_execution_trace(bytearray(raw))  # type: ignore[arg-type]
    with pytest.raises(SchemaError, match="input exceeds"):
        decode_chakra_execution_trace(raw, limits=ResourceLimits(max_input_bytes=1))
    with pytest.raises(SchemaError, match="GlobalMetadata and at least one Node"):
        decode_chakra_execution_trace(_frame(_bytes_field(1, b"0.0.4")))
    with pytest.raises(SchemaError, match="invalid gzip"):
        decode_chakra_execution_trace(b"\x1f\x8bnot-a-gzip-stream")
    with pytest.raises(SchemaError, match="truncated Chakra ET message"):
        decode_chakra_execution_trace(_varint(5) + b"x")
    with pytest.raises(SchemaError, match="messages exceed"):
        decode_chakra_execution_trace(raw, limits=ResourceLimits(max_chakra_messages=1))
    with pytest.raises(SchemaError, match="message bytes"):
        decode_chakra_execution_trace(raw, limits=ResourceLimits(max_chakra_message_bytes=1))

    missing = tmp_path / "missing.et"
    with pytest.raises(SchemaError, match="cannot read Chakra ET"):
        load_chakra_execution_trace(missing)
    source = tmp_path / "large.et"
    source.write_bytes(raw)
    with pytest.raises(SchemaError, match="input exceeds"):
        load_chakra_execution_trace(source, limits=ResourceLimits(max_input_bytes=1))

    trace = decode_chakra_execution_trace(raw)
    with pytest.raises(SchemaError, match="at least one Chakra node"):
        chakra_dependency_closure(trace, [])
    with pytest.raises(SchemaError, match="positive integers"):
        chakra_dependency_closure(trace, [0])
    with pytest.raises(SchemaError, match="absent from the source"):
        chakra_dependency_closure(trace, [999])


def _chakra_with_nodes(*nodes: bytes, metadata: bytes | None = None) -> bytes:
    metadata_message = _bytes_field(1, b"0.0.4") if metadata is None else metadata
    return b"".join([_frame(metadata_message), *(_frame(node) for node in nodes)])


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_chakra_with_nodes(_field(1, 1) + _field(3, 4), metadata=b""), "version must occur"),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4), metadata=_field(1, 1)),
            "string encoding",
        ),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4), metadata=_bytes_field(1, b"\xff")),
            "valid UTF-8",
        ),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4), metadata=_bytes_field(1, b"")),
            "must be non-empty",
        ),
        (_chakra_with_nodes(_field(3, 4)), "Node\\[0\\]\\.id must occur"),
        (_chakra_with_nodes(_bytes_field(1, b"x") + _field(3, 4)), "varint encoding"),
        (_chakra_with_nodes(_field(1, 0) + _field(3, 4)), "id must be positive"),
        (_chakra_with_nodes(_field(1, 1) + _field(3, 0)), "known positive enum"),
        (_chakra_with_nodes(_field(1, 1) + _field(3, 4) + _packed(4, [0])), "positive Node IDs"),
        (
            _chakra_with_nodes(
                _field(1, 1) + _field(3, 4),
                _field(1, 2) + _field(3, 4) + _packed(4, [1, 1]),
            ),
            "duplicate Node IDs",
        ),
        (_chakra_with_nodes(_field(1, 1) + _field(3, 4) + _field(10, 1)), "attr must use"),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4) + _attribute("comm_type", 0) + _attribute("comm_type", 0)),
            "repeats name",
        ),
        (
            _chakra_with_nodes(
                _field(1, 1) + _field(3, 4),
                _field(1, 1) + _field(3, 4),
            ),
            "repeats Node.id",
        ),
        (_chakra_with_nodes(_field(1, 1) + _field(3, 4) + _packed(4, [1])), "depends on itself"),
        (_chakra_with_nodes(_field(1, 1) + _field(3, 4) + _packed(4, [2])), "missing dependency"),
        (_chakra_with_nodes(_field(1, 1) + _field(3, 4) + b"\x00"), "field number zero"),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4) + _varint((99 << 3) | 3)),
            "unsupported protobuf wire type",
        ),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4) + _varint((99 << 3) | 1) + b"\x00"),
            "truncated fixed64",
        ),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4) + _varint((99 << 3) | 2) + _varint(5) + b"x"),
            "truncated length-delimited",
        ),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4) + _varint((99 << 3) | 5) + b"\x00"),
            "truncated fixed32",
        ),
        (
            _chakra_with_nodes(_field(1, 1) + _field(3, 4) + _varint((4 << 3) | 5) + (b"\x00" * 4)),
            "packed or repeated protobuf varints",
        ),
    ],
)
def test_chakra_reader_rejects_malformed_protobuf_fields(raw: bytes, message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        decode_chakra_execution_trace(raw)


def test_chakra_attribute_reader_rejects_ambiguous_types_and_decodes_signed_int64() -> None:
    trace = decode_chakra_execution_trace(chakra_bytes())
    with pytest.raises(SchemaError, match="must occur exactly once"):
        chakra_int64_attribute(trace.nodes[0], "comm_type")

    two_values = _bytes_field(1, b"comm_type") + _field(9, 0) + _field(8, 0)
    raw = _chakra_with_nodes(_field(1, 1) + _field(3, 7) + _bytes_field(10, two_values))
    with pytest.raises(SchemaError, match="exactly one value"):
        chakra_int64_attribute(decode_chakra_execution_trace(raw).nodes[0], "comm_type")

    wrong_type = _bytes_field(1, b"comm_type") + _bytes_field(3, b"all_reduce")
    raw = _chakra_with_nodes(_field(1, 1) + _field(3, 7) + _bytes_field(10, wrong_type))
    with pytest.raises(SchemaError, match="must use int64_val"):
        chakra_int64_attribute(decode_chakra_execution_trace(raw).nodes[0], "comm_type")

    signed = _bytes_field(1, b"comm_type") + _field(9, (1 << 64) - 1)
    raw = _chakra_with_nodes(_field(1, 1) + _field(3, 7) + _bytes_field(10, signed))
    assert chakra_int64_attribute(decode_chakra_execution_trace(raw).nodes[0], "comm_type") == -1


def test_projection_rejects_unsupported_operation_and_nonclosed_region() -> None:
    raw = chakra_bytes()
    trace = decode_chakra_execution_trace(raw)
    document = projection(raw)
    document["nodes"][1]["operation"] = "all_to_all"
    document = with_content_identity(document, "projection_id")
    with pytest.raises(SchemaError, match="all_to_all_not_qualified"):
        validate_chakra_projection(document, trace)

    document = projection(raw)
    document["regions"][1]["node_ids"] = [3, 4]
    document["regions"][1]["feature_tags"] = ["overlap"]
    document["regions"][1]["disclosures"] = []
    document = with_content_identity(document, "projection_id")
    with pytest.raises(SchemaError, match="not dependency closed"):
        validate_chakra_projection(document, trace)

    broadcast_raw = chakra_bytes(first_comm_type=5)
    with pytest.raises(SchemaError, match="collective_type_5_not_qualified"):
        validate_chakra_projection(
            projection(broadcast_raw),
            decode_chakra_execution_trace(broadcast_raw),
        )

    wrong_size_raw = chakra_bytes(first_comm_size=17)
    with pytest.raises(SchemaError, match="does not match Chakra comm_size"):
        validate_chakra_projection(
            projection(wrong_size_raw),
            decode_chakra_execution_trace(wrong_size_raw),
        )


def test_counterexample_guided_synthesis_selects_measured_holdout_passing_candidate() -> None:
    raw = chakra_bytes()
    trace = decode_chakra_execution_trace(raw)
    projection_document = projection(raw)
    policy_document = policy()
    corpus_document = corpus(raw, projection_document, policy_document)

    result = synthesize_physical_decision_canary(
        trace,
        projection_document,
        policy_document,
        corpus_document,
    )

    assert result.status == "qualified_physical_decision_canary"
    assert result.selected_candidate_id == "candidate-counterexample-complete"
    assert result.selected_node_ids == (1, 2, 3, 4)
    assert result.ledger["counterexample_steps"][0]["counterexample_perturbation_ids"] == ["train-regression"]
    assert (
        result.ledger["holdout_evaluation"]["read_after_selection_sha256"]
        == result.ledger["selection"]["selection_sha256"]
    )
    assert result.certificate["claims"]["application_ground_truth"] == "measured"
    assert result.certificate["claims"]["decision_preservation"] == "held_out_measured"


def test_synthetic_corpus_and_missing_corpus_cannot_issue_fidelity_claims() -> None:
    raw = chakra_bytes()
    trace = decode_chakra_execution_trace(raw)
    projection_document = projection(raw)
    policy_document = policy()
    synthetic = corpus(raw, projection_document, policy_document, evidence_kind="synthetic")

    synthetic_result = synthesize_physical_decision_canary(
        trace,
        projection_document,
        policy_document,
        synthetic,
    )
    missing_result = synthesize_physical_decision_canary(trace, projection_document, policy_document)

    assert synthetic_result.status == "blocked_synthetic_evidence"
    assert synthetic_result.certificate["claims"]["decision_preservation"] == "unproven"
    assert missing_result.status == "blocked_missing_physical_oracle_corpus"
    assert missing_result.ledger["holdout_evaluation"] is None


def test_oracle_corpus_recomputes_identity_and_executed_work() -> None:
    raw = chakra_bytes()
    projection_document = projection(raw)
    policy_document = policy()
    document = corpus(raw, projection_document, policy_document)
    document["evaluations"][1]["observations"][0]["physical_metrics"]["executed_flops"] = 301
    document = with_content_identity(document, "corpus_id")

    with pytest.raises(SchemaError, match="executed work"):
        validate_physical_oracle_corpus(
            document,
            decode_chakra_execution_trace(raw),
            projection_document,
            policy_document,
        )

    document = corpus(raw, projection_document, policy_document)
    document["evaluations"][1]["executable_sha256"] = "a" * 64
    document = with_content_identity(document, "corpus_id")
    with pytest.raises(SchemaError, match="executable_sha256 does not match"):
        validate_physical_oracle_corpus(
            document,
            decode_chakra_execution_trace(raw),
            projection_document,
            policy_document,
        )

    document = corpus(raw, projection_document, policy_document)
    document["evaluations"][1]["observations"][0]["physical_metrics"]["gpu_seconds"] += 1
    document = with_content_identity(document, "corpus_id")
    with pytest.raises(SchemaError, match="gpu_seconds does not match"):
        validate_physical_oracle_corpus(
            document,
            decode_chakra_execution_trace(raw),
            projection_document,
            policy_document,
        )

    trace = decode_chakra_execution_trace(raw)
    document = corpus(raw, projection_document, policy_document)
    document["evaluations"][2]["selected_region_ids"] = ["r3"]
    executable, _closure = encode_chakra_subgraph(trace, [1, 2, 3, 4, 5, 6])
    document["evaluations"][2]["executable_sha256"] = hashlib.sha256(executable).hexdigest()
    for row in document["evaluations"][2]["observations"]:
        row["physical_metrics"]["executed_flops"] = 1_000
    document = with_content_identity(document, "corpus_id")
    with pytest.raises(SchemaError, match="does not execute fewer Chakra nodes"):
        validate_physical_oracle_corpus(
            document,
            trace,
            projection_document,
            policy_document,
        )


def test_bundle_is_immutable_recomputable_and_private_exchange_is_owner_signed(tmp_path: Path) -> None:
    raw = chakra_bytes()
    source = tmp_path / "source.et"
    source.write_bytes(raw)
    projection_document = projection(raw)
    policy_document = policy()
    corpus_document = corpus(raw, projection_document, policy_document)
    bundle = tmp_path / "serving.canary"

    manifest = build_physical_canary_bundle(
        str(source),
        projection_document,
        policy_document,
        str(bundle),
        corpus=corpus_document,
    )

    assert manifest["status"] == "qualified_physical_decision_canary"
    assert verify_physical_canary_bundle(str(bundle)) == manifest
    original_manifest = (bundle / "manifest.json").read_bytes()
    with pytest.raises(CommCanaryIOError, match="cannot prepare physical decision canary"):
        build_physical_canary_bundle(
            str(source),
            projection_document,
            policy_document,
            str(bundle),
            corpus=corpus_document,
        )
    assert (bundle / "manifest.json").read_bytes() == original_manifest
    with pytest.raises(SchemaError, match="owner-private-key"):
        build_physical_canary_bundle(
            str(source),
            projection_document,
            policy_document,
            str(tmp_path / "private.canary"),
            corpus=corpus_document,
            mode="private_exchange",
        )
    if shutil.which("openssl") is None:
        pytest.skip("OpenSSL is not installed")
    private_key = tmp_path / "owner-private.pem"
    public_key = tmp_path / "owner-public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(private_key)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)],
        check=True,
        capture_output=True,
    )
    private_bundle = tmp_path / "private.canary"
    private_manifest = build_physical_canary_bundle(
        str(source),
        projection_document,
        policy_document,
        str(private_bundle),
        corpus=corpus_document,
        mode="private_exchange",
        owner_private_key=str(private_key),
        owner_public_key=str(public_key),
    )
    assert private_manifest["publication_mode"] == "private_exchange"
    assert {path.name for path in private_bundle.iterdir()} == {
        "canary.et",
        "canary-projection.json",
        "fidelity-certificate.json",
        "leakage-assessment.json",
        "manifest.json",
        "owner-signature.json",
        "policy.json",
    }
    with pytest.raises(SchemaError, match="trusted owner"):
        verify_physical_canary_bundle(str(private_bundle))
    assert (
        verify_physical_canary_bundle(
            str(private_bundle),
            trusted_owner_public_key=str(public_key),
        )
        == private_manifest
    )
    (bundle / "canary.et").write_bytes((bundle / "canary.et").read_bytes() + b"\x00")
    with pytest.raises(SchemaError, match="identity mismatch"):
        verify_physical_canary_bundle(str(bundle))


def test_bundle_manifest_requires_the_complete_fixed_inventory(tmp_path: Path) -> None:
    raw = chakra_bytes()
    source = tmp_path / "source.et"
    source.write_bytes(raw)
    projection_document = projection(raw)
    policy_document = policy()
    corpus_document = corpus(raw, projection_document, policy_document)
    bundle = tmp_path / "serving.canary"
    build_physical_canary_bundle(
        str(source),
        projection_document,
        policy_document,
        str(bundle),
        corpus=corpus_document,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"].pop("source.et")
    manifest = with_content_identity(manifest, "bundle_id")
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SchemaError, match="artifact inventory is not exact"):
        verify_physical_canary_bundle(str(bundle))


def test_gate_has_fail_precedence_and_deterministic_ci_presentations() -> None:
    policy_document = policy()
    bundle_id = hashlib.sha256(b"bundle").hexdigest()
    baseline = observation(bundle_id, "baseline", p99=[100.0] * 6, throughput=[1000.0] * 6)
    candidate = observation(bundle_id, "candidate", p99=[107.0] * 6, throughput=[1001.0] * 6)

    result = evaluate_physical_gate(
        bundle_id=bundle_id,
        bundle_status="qualified_physical_decision_canary",
        certified_baseline_subject_sha256=baseline["subject_sha256"],
        certified_canary_et_sha256=baseline["executable_sha256"],
        policy=policy_document,
        baseline=baseline,
        candidate=candidate,
    )

    assert result["outcome"] == "fail"
    assert result["issues"] == ["mandatory_metric_regression:p99_latency_ms"]
    assert "Physical gate: FAIL" in render_physical_gate_html(result)
    assert b"mandatory performance regression" in physical_gate_junit_bytes(result)
    assert physical_gate_sarif(result)["runs"][0]["results"][0]["level"] == "error"
    malformed = copy.deepcopy(result)
    malformed["metric_results"][0]["status"] = "pass"
    with pytest.raises(SchemaError, match="status is inconsistent"):
        validate_physical_gate_result(malformed)

    wrong_baseline = copy.deepcopy(baseline)
    wrong_baseline["subject_sha256"] = "c" * 64
    with pytest.raises(SchemaError, match="certified baseline subject"):
        evaluate_physical_gate(
            bundle_id=bundle_id,
            bundle_status="qualified_physical_decision_canary",
            certified_baseline_subject_sha256=baseline["subject_sha256"],
            certified_canary_et_sha256=baseline["executable_sha256"],
            policy=policy_document,
            baseline=wrong_baseline,
            candidate=candidate,
        )

    wrong_executable = copy.deepcopy(candidate)
    wrong_executable["executable_sha256"] = "d" * 64
    with pytest.raises(SchemaError, match="canary executable"):
        evaluate_physical_gate(
            bundle_id=bundle_id,
            bundle_status="qualified_physical_decision_canary",
            certified_baseline_subject_sha256=baseline["subject_sha256"],
            certified_canary_et_sha256=baseline["executable_sha256"],
            policy=policy_document,
            baseline=baseline,
            candidate=wrong_executable,
        )

    wrong_runner = copy.deepcopy(candidate)
    wrong_runner["runner_oci_digest"] = f"sha256:{'e' * 64}"
    with pytest.raises(SchemaError, match="runner_oci_digest does not match"):
        evaluate_physical_gate(
            bundle_id=bundle_id,
            bundle_status="qualified_physical_decision_canary",
            certified_baseline_subject_sha256=baseline["subject_sha256"],
            certified_canary_et_sha256=baseline["executable_sha256"],
            policy=policy_document,
            baseline=baseline,
            candidate=wrong_runner,
        )

    incomparable = copy.deepcopy(candidate)
    incomparable["environment_sha256"] = "b" * 64
    result = evaluate_physical_gate(
        bundle_id=bundle_id,
        bundle_status="qualified_physical_decision_canary",
        certified_baseline_subject_sha256=baseline["subject_sha256"],
        certified_canary_et_sha256=baseline["executable_sha256"],
        policy=policy_document,
        baseline=baseline,
        candidate=incomparable,
    )
    assert result["outcome"] == "incomparable"
    assert "environment_mismatch" in result["issues"]


def test_capture_build_gate_surface_builds_and_gates_a_qualified_bundle(tmp_path: Path) -> None:
    raw = chakra_bytes()
    source = tmp_path / "trace.et"
    projection_path = tmp_path / "projection.json"
    policy_path = tmp_path / "policy.json"
    corpus_path = tmp_path / "corpus.json"
    bundle = tmp_path / "serving.canary"
    source.write_bytes(raw)
    projection_document = projection(raw)
    policy_document = policy()
    corpus_document = corpus(raw, projection_document, policy_document)
    for path, document in (
        (projection_path, projection_document),
        (policy_path, policy_document),
        (corpus_path, corpus_document),
    ):
        path.write_text(json.dumps(document), encoding="utf-8")

    assert (
        main(
            [
                "build",
                str(source),
                "--projection",
                str(projection_path),
                "--policy",
                str(policy_path),
                "--oracle-corpus",
                str(corpus_path),
                "--runtime-budget",
                "10s",
                "--output",
                str(bundle),
            ]
        )
        == 0
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    baseline = observation(
        manifest["bundle_id"],
        "baseline",
        p99=[100.0] * 6,
        throughput=[1000.0] * 6,
        executable_sha256=manifest["artifacts"]["canary.et"]["sha256"],
    )
    candidate = observation(
        manifest["bundle_id"],
        "candidate",
        p99=[104.0] * 6,
        throughput=[960.0] * 6,
        executable_sha256=manifest["artifacts"]["canary.et"]["sha256"],
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    result_path = tmp_path / "gate.json"
    html_path = tmp_path / "gate.html"
    junit_path = tmp_path / "gate.xml"
    sarif_path = tmp_path / "gate.sarif"

    assert (
        main(
            [
                "gate",
                str(bundle),
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(result_path),
                "--html",
                str(html_path),
                "--junit",
                str(junit_path),
                "--sarif",
                str(sarif_path),
            ]
        )
        == 0
    )
    assert json.loads(result_path.read_text(encoding="utf-8"))["outcome"] == "pass"
    assert html_path.is_file()
    assert junit_path.is_file()
    assert sarif_path.is_file()


def _gate(policy_document: Dict[str, Any], baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_physical_gate(
        bundle_id=baseline["bundle_id"],
        bundle_status="qualified_physical_decision_canary",
        certified_baseline_subject_sha256=baseline["subject_sha256"],
        certified_canary_et_sha256=baseline["executable_sha256"],
        policy=policy_document,
        baseline=baseline,
        candidate=candidate,
    )


def test_gate_reports_inconclusive_below_the_policy_replicate_floor() -> None:
    policy_document = policy()
    bundle_id = hashlib.sha256(b"floor").hexdigest()
    baseline = observation(bundle_id, "baseline", p99=[100.0, 100.0], throughput=[1000.0, 1000.0])
    candidate = observation(bundle_id, "candidate", p99=[107.0, 107.0], throughput=[1001.0, 1001.0])

    result = _gate(policy_document, baseline, candidate)

    # Two repetitions cannot resolve a 7% difference against a 5% boundary. The
    # point estimate alone would have called this a confident failure.
    assert result["outcome"] == "inconclusive"
    assert "insufficient_samples:p99_latency_ms" in result["issues"]
    row = next(item for item in result["metric_results"] if item["name"] == "p99_latency_ms")
    assert row["status"] == "inconclusive"
    assert row["confidence_interval_lower"] is None
    assert row["bootstrap_resamples"] == 0
    validate_physical_gate_result(result)


def test_gate_reports_inconclusive_when_the_interval_straddles_the_boundary() -> None:
    """The r3 situation: a median past the threshold that the evidence cannot resolve."""

    policy_document = policy()
    bundle_id = hashlib.sha256(b"straddle").hexdigest()
    baseline = observation(
        bundle_id,
        "baseline",
        p99=[100.0, 101.0, 99.0, 100.5, 99.5, 100.0, 100.2, 99.8],
        throughput=[1000.0] * 8,
    )
    candidate = observation(
        bundle_id,
        "candidate",
        p99=[104.0, 108.0, 103.0, 107.0, 105.0, 106.0, 104.5, 107.5],
        throughput=[1000.0] * 8,
    )

    result = _gate(policy_document, baseline, candidate)

    row = next(item for item in result["metric_results"] if item["name"] == "p99_latency_ms")
    assert row["regression_pct"] > 5.0, "the point estimate alone would have failed this candidate"
    assert row["status"] == "inconclusive"
    assert row["confidence_interval_lower"] < 5.0 < row["confidence_interval_upper"]
    assert result["outcome"] == "inconclusive"
    assert "undecidable_metric:p99_latency_ms" in result["issues"]
    validate_physical_gate_result(result)


def test_gate_reports_inconclusive_when_timing_variability_exceeds_the_policy() -> None:
    policy_document = policy()
    bundle_id = hashlib.sha256(b"unstable").hexdigest()
    baseline = observation(
        bundle_id,
        "baseline",
        p99=[50.0, 150.0, 60.0, 140.0, 70.0, 130.0],
        throughput=[1000.0] * 6,
    )
    candidate = observation(bundle_id, "candidate", p99=[100.0] * 6, throughput=[1000.0] * 6)

    result = _gate(policy_document, baseline, candidate)

    row = next(item for item in result["metric_results"] if item["name"] == "p99_latency_ms")
    assert row["baseline_relative_iqr_pct"] > policy_document["noise"]["max_relative_iqr_pct"]
    assert row["status"] == "inconclusive"
    assert result["outcome"] == "inconclusive"
    assert "unstable_measurement:p99_latency_ms" in result["issues"]
    validate_physical_gate_result(result)


def test_gate_interval_is_deterministic_and_revalidation_rejects_a_forged_status() -> None:
    policy_document = policy()
    bundle_id = hashlib.sha256(b"deterministic").hexdigest()
    baseline = observation(bundle_id, "baseline", p99=[100.0] * 6, throughput=[1000.0] * 6)
    candidate = observation(bundle_id, "candidate", p99=[107.0] * 6, throughput=[1001.0] * 6)

    first = _gate(policy_document, baseline, candidate)
    second = _gate(policy_document, baseline, candidate)
    assert first == second
    assert first["outcome"] == "fail"

    forged = copy.deepcopy(first)
    row = next(item for item in forged["metric_results"] if item["name"] == "p99_latency_ms")
    row["status"] = "pass"
    with pytest.raises(SchemaError, match="status is inconsistent"):
        validate_physical_gate_result(forged)

    stripped = copy.deepcopy(first)
    row = next(item for item in stripped["metric_results"] if item["name"] == "p99_latency_ms")
    row["confidence_interval_lower"] = None
    row["confidence_interval_upper"] = None
    with pytest.raises(SchemaError, match="reports resamples without an interval"):
        validate_physical_gate_result(stripped)


def test_gate_no_longer_passes_a_regression_the_evidence_cannot_rule_out() -> None:
    """The severe-false-negative case the point-estimate gate produced.

    The candidate's median sits below the 5% boundary, so comparing medians
    alone returns ``pass``. The measurement is noisy enough that a much larger
    regression is entirely consistent with it, so the interval reaches well past
    the boundary and the honest answer is that this run did not settle it.
    """

    policy_document = policy()
    bundle_id = hashlib.sha256(b"false-negative").hexdigest()
    baseline = observation(bundle_id, "baseline", p99=[100.0] * 8, throughput=[1000.0] * 8)
    candidate = observation(
        bundle_id,
        "candidate",
        p99=[96.0, 112.0, 98.0, 110.0, 100.0, 108.0, 102.0, 106.0],
        throughput=[1000.0] * 8,
    )

    result = _gate(policy_document, baseline, candidate)
    row = next(item for item in result["metric_results"] if item["name"] == "p99_latency_ms")

    assert row["regression_pct"] <= 5.0, "median comparison alone would have passed this candidate"
    assert row["confidence_interval_upper"] > 5.0, "the evidence does not rule out a real regression"
    assert row["status"] == "inconclusive"
    assert result["outcome"] == "inconclusive"
    validate_physical_gate_result(result)
