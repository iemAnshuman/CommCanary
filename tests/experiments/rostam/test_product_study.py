from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import pytest

from commcanary.adapters.chakra_capture import commcanary_trace_to_chakra
from commcanary.artifacts.json_codec import canonical_json_bytes
from experiments.rostam.product_canary import study as product_study
from experiments.rostam.product_canary.study_contract import (
    ProductStudyError,
    freeze_product_study,
    validate_perturbation_plan,
    verify_frozen_product_study,
    williams_schedule,
)
from tests.services.test_active_physical_synthesis import _captured_trace, _policy

ROOT = Path(__file__).resolve().parents[3]
PRODUCT = ROOT / "experiments" / "rostam" / "product_canary"


def _identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _runner(tmp_path: Path, *, runner_digest: str, wheel: Path) -> tuple[Path, Path]:
    build = tmp_path / "runner-build"
    build.mkdir()
    archive = build / "runner.oci.tar"
    archive.write_bytes(b"oci")
    sif = build / "runner.sif"
    sif.write_bytes(b"sif")
    log = build / "build.log"
    log.write_bytes(b"build log\n")
    container = tmp_path / "Containerfile"
    container.write_bytes(b"FROM scratch\n")
    lock = tmp_path / "runner-lock.json"
    lock.write_bytes(b"{}\n")
    descriptor: dict[str, object] = {
        "format": "commcanary.product_runner_descriptor.v1",
        "status": "complete",
        "host": "rostam",
        "architecture": "x86_64",
        "base_manifest_digest": f"sha256:{'0' * 64}",
        "runner_protocols": [
            "chakra-et-collective-graph.v1",
            "vllm-offline-tensor-parallel.v1",
        ],
        "oci_manifest": {
            "digest": runner_digest,
            "bytes": 3,
            "media_type": "application/vnd.oci.image.manifest.v1+json",
            "config_digest": f"sha256:{'3' * 64}",
            "layer_digests": [f"sha256:{'4' * 64}"],
        },
        "artifacts": {
            "runner.oci.tar": _identity(archive),
            "runner.sif": _identity(sif),
            "build.log": _identity(log),
        },
        "inputs": {
            "Containerfile": _identity(container),
            "commcanary.whl": _identity(wheel),
            "runner-lock.json": _identity(lock),
        },
        "versions": {
            "commcanary": "0.3.0",
            "torch": "2.11.0",
            "application_engine": "vllm",
            "application_engine_version": "0.26.0",
        },
    }
    descriptor["descriptor_id"] = hashlib.sha256(canonical_json_bytes(descriptor)).hexdigest()
    descriptor_path = build / "descriptor.json"
    _json(descriptor_path, descriptor)
    return descriptor_path, sif


def _freeze_fixture(tmp_path: Path) -> Path:
    runner_digest = f"sha256:{hashlib.sha256(b'runner').hexdigest()}"
    capture = commcanary_trace_to_chakra(_captured_trace(), opaque_attributes_reviewed=True)
    source = tmp_path / "source.et"
    source.write_bytes(capture.chakra_et)
    projection = tmp_path / "projection.json"
    _json(projection, capture.projection)
    policy = tmp_path / "policy.json"
    _json(policy, _policy(runner_digest))
    wheel = tmp_path / "commcanary.whl"
    wheel.write_bytes(b"wheel")
    descriptor, sif = _runner(tmp_path, runner_digest=runner_digest, wheel=wheel)
    model = tmp_path / "model"
    model.mkdir()
    _json(model / "config.json", {"model_type": "llama"})
    destination = tmp_path / "frozen-study"
    freeze_product_study(
        study_id="product-v1",
        output_directory=destination,
        source_et=source,
        projection_path=projection,
        policy_path=policy,
        perturbation_path=PRODUCT / "perturbations.json",
        model_directory=model,
        runner_descriptor_path=descriptor,
        runner_sif_path=sif,
        commcanary_wheel_path=wheel,
    )
    return destination


def test_williams_schedule_balances_positions_and_directed_carryover() -> None:
    identifiers = [f"c{index}" for index in range(8)]
    rows = williams_schedule(identifiers)

    assert len(rows) == 8
    assert all(set(row) == set(identifiers) for row in rows)
    for position in range(8):
        assert {row[position] for row in rows} == set(identifiers)
    transitions = [(left, right) for row in rows for left, right in zip(row, row[1:])]
    assert len(transitions) == 56
    assert len(set(transitions)) == 56


def test_perturbation_plan_is_decision_blind_and_has_one_historical_mechanism() -> None:
    plan = validate_perturbation_plan(
        json.loads((PRODUCT / "perturbations.json").read_text(encoding="utf-8")),
        engine="vllm",
    )

    assert len(plan["training"]) == 7
    assert len(plan["holdout"]) == 3
    assert sum(row["historical_reference"] is not None for row in plan["holdout"]) == 1
    assert "expected_decision" not in json.dumps(plan)


def test_sglang_has_an_independent_decision_blind_perturbation_plan() -> None:
    plan = validate_perturbation_plan(
        json.loads((PRODUCT / "sglang" / "perturbations.json").read_text(encoding="utf-8")),
        engine="sglang",
    )

    assert len(plan["training"]) == 7
    assert len(plan["holdout"]) == 3
    assert all(row["historical_reference"] is None for row in plan["holdout"])
    assert {row["id"] for row in plan["holdout"]} == {
        "nccl-tree-simple",
        "nccl-shm-disabled",
        "sglang-no-overlap-schedule",
    }


@pytest.mark.parametrize(
    ("engine", "driver_module", "configuration", "expected_flags", "absent_flags"),
    [
        (
            "vllm",
            "commcanary.product.vllm_application_driver",
            {
                "gpu_memory_utilization": 0.75,
                "kv_cache_memory_bytes": 536870912,
                "max_num_batched_tokens": 512,
                "max_num_seqs": 4,
                "disable_custom_all_reduce": True,
                "enforce_eager": False,
            },
            {"--disable-custom-all-reduce", "--kv-cache-memory-bytes"},
            {"--enforce-eager", "--disable-cuda-graph"},
        ),
        (
            "sglang",
            "commcanary.product.sglang_application_driver",
            {
                "mem_fraction_static": 0.75,
                "max_running_requests": 4,
                "max_total_tokens": 4096,
                "chunked_prefill_size": 512,
                "max_prefill_tokens": 1024,
                "disable_cuda_graph": True,
                "disable_custom_all_reduce": False,
                "disable_overlap_schedule": True,
            },
            {"--disable-cuda-graph", "--disable-overlap-schedule", "--max-total-tokens"},
            {"--disable-custom-all-reduce", "--enforce-eager"},
        ),
    ],
)
def test_application_child_argv_is_engine_specific_and_isolated(
    tmp_path: Path,
    engine: str,
    driver_module: str,
    configuration: Mapping[str, object],
    expected_flags: set[str],
    absent_flags: set[str],
) -> None:
    runner_digest = f"sha256:{hashlib.sha256(b'runner').hexdigest()}"
    manifest = {
        "application": {
            "driver_module": driver_module,
            "workload_arguments": {
                "tensor_parallel_size": 4,
                "batch_size": 4,
                "input_length": 128,
                "output_length": 8,
                "warmups": 1,
                "iterations": 4,
                "seed": 314159,
            },
        },
        "runner": {"oci_digest": runner_digest, "application_engine": engine},
    }
    subject = {
        "subject_sha256": hashlib.sha256(b"subject").hexdigest(),
        "perturbation_id": "fixture",
        "subject_configuration": {"engine_configuration": configuration},
    }

    argv = product_study._application_argv(
        manifest,
        subject,
        model_path=tmp_path / "model",
        output_path=tmp_path / "measurement.json",
        profile_path=tmp_path / "profile",
    )

    assert argv[1:3] == ["-I", "-m"]
    assert expected_flags.issubset(argv)
    assert absent_flags.isdisjoint(argv)
    assert argv[argv.index("--runner-oci-digest") + 1] == runner_digest


def test_frozen_product_study_recomputes_every_subject_and_file(tmp_path: Path) -> None:
    frozen = _freeze_fixture(tmp_path)

    manifest = verify_frozen_product_study(frozen)

    assert len(manifest["subjects"]) == 11
    assert len(manifest["schedule"]["training_configuration_order_by_repetition"]) == 8
    assert len(manifest["schedule"]["holdout_configuration_order_by_repetition"]) == 8
    assert manifest["execution"]["minimum_distinct_days"] == 2


def test_frozen_product_study_rejects_rehashed_subject_forgery(tmp_path: Path) -> None:
    frozen = _freeze_fixture(tmp_path)
    manifest_path = frozen / "manifest.json"
    os.chmod(manifest_path, 0o600)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["subjects"][0]["subject_sha256"] = "f" * 64
    manifest["manifest_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in manifest.items() if key != "manifest_id"})
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    checksum = frozen / "manifest.sha256"
    os.chmod(checksum, 0o600)
    checksum.write_text(f"{manifest['manifest_id']}  manifest.json\n", encoding="ascii")

    with pytest.raises(ProductStudyError, match="subject inventory"):
        verify_frozen_product_study(frozen)


def test_frozen_product_study_rejects_extra_executable_member(tmp_path: Path) -> None:
    frozen = _freeze_fixture(tmp_path)
    extra = frozen / "scripts" / "extra.py"
    extra.write_text("raise SystemExit(1)\n", encoding="utf-8")

    with pytest.raises(ProductStudyError, match="script member set"):
        verify_frozen_product_study(frozen)


@pytest.mark.parametrize("path_type", ["symlink", "directory"])
def test_holdout_guard_refuses_non_regular_selection_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_type: str,
) -> None:
    frozen = _freeze_fixture(tmp_path)
    selection_dir = frozen / "state" / "selection"
    selection_dir.mkdir(parents=True)
    selection_path = selection_dir / "selection.json"
    if path_type == "symlink":
        target = tmp_path / "selection-target.json"
        target.write_text("{}", encoding="utf-8")
        selection_path.symlink_to(target)
    else:
        selection_path.mkdir()

    monkeypatch.setattr(product_study, "_ensure_batch_jobs", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        product_study,
        "_application_oracles",
        lambda *args, **kwargs: {"baseline": {}},
    )
    monkeypatch.setattr(
        product_study,
        "synthesize_active_physical_canary",
        lambda *args, holdout_application_loader=None, **kwargs: holdout_application_loader(),
    )

    with pytest.raises(ProductStudyError, match="holdout access"):
        product_study._run_study(frozen, execute=False, no_wait=False, retry_failed=False)
