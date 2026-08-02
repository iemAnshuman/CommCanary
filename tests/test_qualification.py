from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import commcanary.qualification_io as qualification_io_module
from commcanary.artifacts import (
    qualification_materialization_sha256,
    qualification_request_sha256,
    validate_qualification_request,
)
from commcanary.compiler import compile_trace, verify_canary_fidelity
from commcanary.errors import CommCanaryError, SchemaError
from commcanary.resources import ResourceLimits
from commcanary.services import (
    prepare_qualification_request,
    verify_qualification_request,
)
from commcanary.workflows import (
    materialize_qualification,
    verify_qualification_materialization,
)
from tests.builders import qualification_policy, qualification_trace, small_trace

ROOT = Path(__file__).resolve().parents[1]


def _prepare(tmp_path: Path) -> tuple[Path, dict]:
    trace = qualification_trace()
    canary = compile_trace(trace)
    bundle = tmp_path / "qualification"
    request = prepare_qualification_request(str(bundle), trace, canary, qualification_policy())
    return bundle, request


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _refresh_reference(request: dict, artifact_id: str, path: Path) -> None:
    raw = path.read_bytes()
    request["artifacts"][artifact_id]["sha256"] = hashlib.sha256(raw).hexdigest()
    request["artifacts"][artifact_id]["size_bytes"] = len(raw)
    request["request_id"] = qualification_request_sha256(request)


def test_prepare_and_verify_binds_exact_rank_local_work_without_calibration(
    tmp_path: Path,
) -> None:
    bundle, request = _prepare(tmp_path)

    assert {path.name for path in bundle.iterdir()} == {
        "qualification-request.json",
        "source.trace.json",
        "canary.json",
        "fidelity.json",
        "qualification-policy.json",
    }
    assert request["claims"] == {
        "source_correspondence": "source_verified",
        "physical_measurement": "not_included",
        "physical_fidelity": "unproven",
        "qualification_verdict": "policy_bound_not_issued",
    }
    assert request["decision_policy"] == {
        "policy_id": qualification_policy()["policy_id"],
        "policy_format": "commcanary.qualification_policy.v1",
        "application": "required_before_execution",
        "outcomes": ["fail", "incomparable", "inconclusive", "pass"],
    }
    target = request["target_execution"]
    assert target["materialization"] == "deterministic_from_verified_request"
    assert target["program_encoding"] == "commcanary.source-bound-compute-recipe.v2"
    assert target["executor_contract"] == "async-issue-rank-gemm-recipe-explicit-wait.v2"
    assert target["compute_work_source"] == "source-bound-per-rank-exact-recipe"
    assert target["compute_recipe_method"] == "explicit-wait-linked-contiguous-gemm.v1"
    assert len(target["compute_recipe_projection_sha256"]) == 64
    assert target["compute_recipe_event_count"] == 6
    assert target["compute_recipe_operation_count"] == 24
    assert target["target_compute_calibration"] == "not_used"
    assert target["rank_arrival_timing"] == "emerges-from-source-bound-rank-local-work"
    assert target["source_overlap_observation"] == "bound-not-duration-paced"
    assert target["overlap_structure"] == "async-issue-exact-rank-work-explicit-wait"
    assert target["privacy_disclosure"] == "gemm-shapes-and-dtypes-revealed"
    assert target["communication_inventory_source"] == "full-generated-program"
    assert target["communication_operations"] == ["all_reduce"]
    assert target["communication_dtypes"] == ["float32"]
    assert target["communication_reduction_ops"] == ["sum"]
    assert target["communication_message_shapes"] == [
        {
            "operation": "all_reduce",
            "dtype": "float32",
            "world_size": 4,
            "in_msg_size": 32768,
            "out_msg_size": 32768,
        }
    ]
    assert "compute_fill_dtype" not in target
    assert "compute_fill_gemm_dim" not in target
    assert verify_qualification_request(str(bundle)) == request


def test_partial_canary_cannot_qualify_unseen_suffix_semantics(tmp_path: Path) -> None:
    trace = qualification_trace()
    for event in trace["events"][3:]:
        event["dtype"] = "float16"
        event["reduction_op"] = "max"
        event["bytes"] = event["metadata"]["kineto_in_msg_nelems"] * 2
    prefix_canary = compile_trace(trace, max_events=3)

    fidelity = verify_canary_fidelity(trace, prefix_canary)
    assert fidelity["status"] == "partial_source_verified"
    assert fidelity["source_coverage"] == "partial"
    with pytest.raises(SchemaError, match="requires full source coverage"):
        prepare_qualification_request(
            str(tmp_path / "partial-request"),
            trace,
            prefix_canary,
            qualification_policy(),
        )
    assert not (tmp_path / "partial-request").exists()


def test_verifier_rejects_a_rehashed_partial_canary_bundle(tmp_path: Path) -> None:
    trace = qualification_trace()
    for event in trace["events"][3:]:
        event["dtype"] = "float16"
        event["reduction_op"] = "max"
        event["bytes"] = event["metadata"]["kineto_in_msg_nelems"] * 2
    bundle = tmp_path / "request"
    request = prepare_qualification_request(
        str(bundle),
        trace,
        compile_trace(trace),
        qualification_policy(),
    )
    prefix_canary = compile_trace(trace, max_events=3)
    canary_path = bundle / "canary.json"
    fidelity_path = bundle / "fidelity.json"
    _write_json(canary_path, prefix_canary)
    _write_json(fidelity_path, verify_canary_fidelity(trace, prefix_canary))
    _refresh_reference(request, "canary", canary_path)
    _refresh_reference(request, "fidelity_verification", fidelity_path)
    compiler = prefix_canary["compiler"]
    for field in request["bindings"]:
        request["bindings"][field] = compiler[field]
    request["request_id"] = qualification_request_sha256(request)
    _write_json(bundle / "qualification-request.json", request)

    with pytest.raises(SchemaError, match="requires full source coverage"):
        verify_qualification_request(str(bundle))


def test_historical_v1_physical_request_remains_verifiable() -> None:
    bundle = ROOT / "experiments" / "rostam" / "results" / "exact-work-artifacts" / "qualification-inputs" / "request"

    request = verify_qualification_request(str(bundle))

    assert request["format"] == "commcanary.qualification_request.v1"
    assert request["claims"]["qualification_verdict"] == "not_issued"


def test_elapsed_timing_mutations_cannot_change_executable_compute_work(
    tmp_path: Path,
) -> None:
    first = qualification_trace()
    second = copy.deepcopy(first)
    for index, event in enumerate(second["events"]):
        event["start_us"] = float(index * 10_000)
        event["rank_arrival_us"] = {str(rank): float((rank + 1) * 100 + index) for rank in event["ranks"]}
        event["compute_overlap_us"] = float(index)

    first_bundle = tmp_path / "first-request"
    second_bundle = tmp_path / "second-request"
    first_request = prepare_qualification_request(
        str(first_bundle),
        first,
        compile_trace(first),
        qualification_policy(),
    )
    second_request = prepare_qualification_request(
        str(second_bundle),
        second,
        compile_trace(second),
        qualification_policy(),
    )
    assert (
        first_request["target_execution"]["compute_recipe_projection_sha256"]
        == second_request["target_execution"]["compute_recipe_projection_sha256"]
    )

    first_materialization = materialize_qualification(
        str(first_bundle),
        str(tmp_path / "first-program"),
    )
    second_materialization = materialize_qualification(
        str(second_bundle),
        str(tmp_path / "second-program"),
    )
    assert first_materialization["program"]["sha256"] == second_materialization["program"]["sha256"]


def test_recipe_shape_mutation_changes_projection_and_materialized_program(
    tmp_path: Path,
) -> None:
    first = qualification_trace()
    second = copy.deepcopy(first)
    second["events"][0]["compute_recipe_by_rank"]["3"][0]["m"] += 1

    first_bundle = tmp_path / "first-request"
    second_bundle = tmp_path / "second-request"
    first_request = prepare_qualification_request(
        str(first_bundle),
        first,
        compile_trace(first),
        qualification_policy(),
    )
    second_request = prepare_qualification_request(
        str(second_bundle),
        second,
        compile_trace(second),
        qualification_policy(),
    )
    assert (
        first_request["target_execution"]["compute_recipe_projection_sha256"]
        != second_request["target_execution"]["compute_recipe_projection_sha256"]
    )
    first_materialization = materialize_qualification(
        str(first_bundle),
        str(tmp_path / "first-program"),
    )
    second_materialization = materialize_qualification(
        str(second_bundle),
        str(tmp_path / "second-program"),
    )
    assert first_materialization["program"]["sha256"] != second_materialization["program"]["sha256"]


def test_complete_rank_with_no_intervening_compute_remains_explicit(
    tmp_path: Path,
) -> None:
    trace = qualification_trace()
    for event in trace["events"]:
        event["compute_recipe_by_rank"]["3"] = []
    bundle = tmp_path / "request"
    request = prepare_qualification_request(
        str(bundle),
        trace,
        compile_trace(trace),
        qualification_policy(),
    )
    assert request["target_execution"]["compute_recipe_operation_count"] == 18
    materialization = materialize_qualification(
        str(bundle),
        str(tmp_path / "materialization"),
    )
    assert materialization["compute_work"]["rank_operation_counts"] == {
        "0": 6,
        "1": 6,
        "2": 6,
        "3": 0,
    }


def test_preparation_requires_derived_complete_per_rank_kineto_recipes(
    tmp_path: Path,
) -> None:
    not_kineto = small_trace()
    for event in not_kineto["events"]:
        event["dtype"] = "float32"
        event["reduction_op"] = "sum"
    with pytest.raises(SchemaError, match="requires a PyTorch Kineto source trace"):
        prepare_qualification_request(
            str(tmp_path / "not-kineto"),
            not_kineto,
            compile_trace(not_kineto),
            qualification_policy(),
        )

    missing = qualification_trace()
    missing["events"][0].pop("compute_recipe_by_rank")
    with pytest.raises(SchemaError, match="requires explicit compute_recipe_by_rank"):
        prepare_qualification_request(
            str(tmp_path / "missing"),
            missing,
            compile_trace(missing),
            qualification_policy(),
        )

    incomplete = qualification_trace()
    incomplete["events"][0]["compute_recipe_by_rank"].pop("3")
    with pytest.raises(SchemaError, match="keys must match ranks"):
        prepare_qualification_request(
            str(tmp_path / "incomplete"),
            incomplete,
            compile_trace(incomplete),
            qualification_policy(),
        )

    wrong_method = qualification_trace()
    wrong_method["events"][0]["metadata"]["kineto_compute_recipe_method"] = "duration-fit.v0"
    with pytest.raises(SchemaError, match="compute-recipe method is unsupported"):
        prepare_qualification_request(
            str(tmp_path / "wrong-method"),
            wrong_method,
            compile_trace(wrong_method),
            qualification_policy(),
        )

    unsupported = qualification_trace()
    unsupported["events"][0]["op"] = "send"
    unsupported["events"][0].pop("reduction_op")
    unsupported["events"][0]["sender_rank"] = 0
    unsupported["events"][0]["receiver_rank"] = 1
    with pytest.raises(SchemaError, match="outside the exact-work collective contract"):
        prepare_qualification_request(
            str(tmp_path / "unsupported"),
            unsupported,
            compile_trace(unsupported),
            qualification_policy(),
        )


def test_preparation_requires_exact_kineto_message_shapes(tmp_path: Path) -> None:
    malformed = qualification_trace()
    malformed["events"][0]["metadata"]["kineto_message_shape_status"] = "unavailable_in_out_mismatch"
    with pytest.raises(SchemaError, match="message shape is not exactly materializable"):
        prepare_qualification_request(
            str(tmp_path / "malformed"),
            malformed,
            compile_trace(malformed),
            qualification_policy(),
        )

    skipped = qualification_trace()
    skipped["workload"]["skipped_empty_events"] = 1
    with pytest.raises(SchemaError, match="skipped zero-sized or missing-size"):
        prepare_qualification_request(
            str(tmp_path / "skipped"),
            skipped,
            compile_trace(skipped),
            qualification_policy(),
        )


def test_prepare_refuses_existing_output_and_does_not_install_partial_manifest(
    tmp_path: Path,
) -> None:
    trace = qualification_trace()
    canary = compile_trace(trace)
    output = tmp_path / "owned"
    output.mkdir()
    marker = output / "owned-by-user"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(CommCanaryError, match="cannot create qualification bundle"):
        prepare_qualification_request(str(output), trace, canary, qualification_policy())
    assert marker.read_text(encoding="utf-8") == "preserve"

    over_budget = tmp_path / "over-budget"
    with pytest.raises(SchemaError, match="compute operation count"):
        prepare_qualification_request(
            str(over_budget),
            trace,
            canary,
            qualification_policy(),
            limits=ResourceLimits(max_param_compute_operations=23),
        )
    assert not over_budget.exists()


def test_request_verifier_detects_inventory_bytes_fidelity_and_binding_tampering(
    tmp_path: Path,
) -> None:
    missing_bundle, _ = _prepare(tmp_path / "missing")
    (missing_bundle / "fidelity.json").unlink()
    with pytest.raises(SchemaError, match="inventory mismatch"):
        verify_qualification_request(str(missing_bundle))

    tampered_bundle, _ = _prepare(tmp_path / "tampered")
    canary_path = tampered_bundle / "canary.json"
    canary_path.write_bytes(canary_path.read_bytes() + b" ")
    with pytest.raises(SchemaError, match="bytes do not match manifest"):
        verify_qualification_request(str(tampered_bundle))

    policy_bundle, _ = _prepare(tmp_path / "policy-tampered")
    policy_path = policy_bundle / "qualification-policy.json"
    policy_path.write_bytes(policy_path.read_bytes() + b" ")
    with pytest.raises(SchemaError, match="bytes do not match manifest"):
        verify_qualification_request(str(policy_bundle))

    rehashed_bundle, request = _prepare(tmp_path / "rehashed")
    trace_path = rehashed_bundle / "source.trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["events"][0]["bytes"] *= 2
    _write_json(trace_path, trace)
    _refresh_reference(request, "source_trace", trace_path)
    _write_json(rehashed_bundle / "qualification-request.json", request)
    with pytest.raises(SchemaError, match="source verified|does not recompute"):
        verify_qualification_request(str(rehashed_bundle))

    binding_bundle, request = _prepare(tmp_path / "binding")
    request["bindings"]["execution_semantic_sha256"] = "f" * 64
    request["request_id"] = qualification_request_sha256(request)
    _write_json(binding_bundle / "qualification-request.json", request)
    with pytest.raises(SchemaError, match="bindings do not match"):
        verify_qualification_request(str(binding_bundle))


def test_request_id_and_claims_are_semantically_validated(tmp_path: Path) -> None:
    _bundle, request = _prepare(tmp_path)
    bad_id = copy.deepcopy(request)
    bad_id["request_id"] = "0" * 64
    with pytest.raises(SchemaError, match="request_id does not match"):
        validate_qualification_request(bad_id)

    false_claim = copy.deepcopy(request)
    false_claim["claims"]["physical_fidelity"] = "verified"
    false_claim["request_id"] = qualification_request_sha256(false_claim)
    with pytest.raises(SchemaError, match="assurance boundary"):
        validate_qualification_request(false_claim)


def test_materialization_is_deterministic_exact_work_not_an_execution_claim(
    tmp_path: Path,
) -> None:
    bundle, request = _prepare(tmp_path / "request")
    output = tmp_path / "materialization"
    materialization = materialize_qualification(str(bundle), str(output))

    assert materialization["request"]["request_id"] == request["request_id"]
    assert "calibration" not in materialization
    assert "quantization" not in materialization
    assert materialization["compute_work"]["event_count"] == 6
    assert materialization["compute_work"]["operation_count"] == 24
    assert materialization["compute_work"]["rank_operation_counts"] == {
        "0": 6,
        "1": 6,
        "2": 6,
        "3": 6,
    }
    assert materialization["program"]["entry_count"] == 19
    assert materialization["program"]["compute_operation_count"] == 24
    assert materialization["executor"]["contract"] == ("async-issue-rank-gemm-recipe-explicit-wait.v2")
    assert materialization["claims"] == {
        "materialization": "request_bound",
        "compute_work_provenance": "source_trace_verified",
        "physical_execution": "not_included",
        "physical_measurement": "not_included",
        "qualification_verdict": "not_issued",
    }

    entries = json.loads((output / "replay-program.json").read_text(encoding="utf-8"))
    assert entries[0]["comms"] == "init"
    for event_index in range(6):
        issue, compute, wait = entries[1 + event_index * 3 : 4 + event_index * 3]
        assert issue["comms"] == "all_reduce"
        assert issue["source_event_index"] == event_index
        assert compute["compute"] == "gemm_recipe"
        assert compute["overlap_request"] == issue["req"]
        assert compute["source_event_index"] == event_index
        assert compute["recipe_by_rank"]["0"] == [{"op": "gemm", "dtype": "bfloat16", "m": 2, "n": 8, "k": 8}]
        assert "source_kernel_duration_us" not in compute["recipe_by_rank"]["0"][0]
        assert wait == {
            "comms": "wait",
            "req": issue["req"],
            "source_event_index": event_index,
            "markers": ["commcanary:qualification:complete:tp0:all_reduce"],
        }
    assert verify_qualification_materialization(str(bundle), str(output)) == materialization


def test_materialization_uses_the_verified_single_read_request_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, request = _prepare(tmp_path / "request")
    source_path = bundle / "source.trace.json"
    source_sha256 = request["artifacts"]["source_trace"]["sha256"]
    real_decode = qualification_io_module.decode_bounded_json_bytes
    mutated = False

    def decode_then_mutate(raw: bytes, **kwargs: object) -> object:
        nonlocal mutated
        if not mutated and hashlib.sha256(raw).hexdigest() == source_sha256:
            mutated = True
            source_path.write_bytes(b"{}")
        return real_decode(raw, **kwargs)

    monkeypatch.setattr(qualification_io_module, "decode_bounded_json_bytes", decode_then_mutate)
    output = tmp_path / "materialization"
    materialization = materialize_qualification(str(bundle), str(output))

    assert mutated
    assert materialization["compute_work"]["event_count"] == 6
    assert materialization["program"]["entry_count"] == 19
    with pytest.raises(SchemaError, match="bytes do not match manifest"):
        verify_qualification_request(str(bundle))


def test_materialization_refuses_existing_output_and_detects_program_tampering(
    tmp_path: Path,
) -> None:
    bundle, _ = _prepare(tmp_path / "request")
    output = tmp_path / "owned"
    output.mkdir()
    marker = output / "owned-by-user"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(CommCanaryError, match="cannot create qualification materialization"):
        materialize_qualification(str(bundle), str(output))
    assert marker.read_text(encoding="utf-8") == "preserve"

    materialized = tmp_path / "materialized"
    materialize_qualification(str(bundle), str(materialized))
    program = materialized / "replay-program.json"
    program.write_bytes(program.read_bytes() + b" ")
    with pytest.raises(SchemaError, match="program bytes do not match"):
        verify_qualification_materialization(str(bundle), str(materialized))


def test_materialization_verifier_recomputes_source_work_after_rehashed_tampering(
    tmp_path: Path,
) -> None:
    bundle, _ = _prepare(tmp_path / "request")
    output = tmp_path / "materialization"
    materialization = materialize_qualification(str(bundle), str(output))
    materialization["compute_work"]["source_kernel_duration_us"] += 1.0
    materialization["materialization_id"] = qualification_materialization_sha256(materialization)
    _write_json(output / "materialization.json", materialization)
    with pytest.raises(SchemaError, match="compute-work audit does not recompute"):
        verify_qualification_materialization(str(bundle), str(output))
