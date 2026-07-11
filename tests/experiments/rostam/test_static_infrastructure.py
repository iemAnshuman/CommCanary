from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from experiments.rostam.analysis.schemas import (
    PHYSICAL_MICRO_MEASUREMENT_SCHEMA,
    validate_scalar_measurement,
)
from experiments.rostam.harness import JSONResourceLimits, build_run_manifest, freeze_campaign
from experiments.rostam.lib import catalog as catalog_module
from experiments.rostam.lib import environment_contract as environment_contract_module
from experiments.rostam.lib import submission as submission_module
from experiments.rostam.lib.campaign import build_campaign
from experiments.rostam.lib.catalog import Catalog, CatalogValidationError, load_catalog
from experiments.rostam.lib.environment_contract import (
    EnvironmentContractError,
    audit_static_contracts,
    verify_ready_for_install,
)
from experiments.rostam.lib.physical_results import (
    FULL_MEASUREMENT_SCHEMA,
    FULL_PRODUCER_SCHEMA,
    FULL_STDOUT_SCHEMA,
    MICRO_MEASUREMENT_SCHEMA,
    MICRO_PRODUCER_SCHEMA,
    MICRO_STDOUT_SCHEMA,
    OVERLAP_MEASUREMENT_SCHEMA,
    OVERLAP_PRODUCER_SCHEMA,
    OVERLAP_STDOUT_SCHEMA,
    PARAM_MEASUREMENT_SCHEMA,
    PARAM_PRODUCER_SCHEMA,
    PhysicalResultError,
    adapt_physical_measurement,
    validate_param_trace,
)
from experiments.rostam.lib.submission import (
    SubmissionPlanError,
    build_submission_plan,
    load_submission_plan,
    submit_frozen_plan,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
EXPERIMENT_DIRECTORY = REPOSITORY_ROOT / "experiments" / "rostam"
CATALOG_PATH = EXPERIMENT_DIRECTORY / "configs.json"
PENDING_GEMM_CALIBRATION = "PENDING_ROSTAM_GEMM_CALIBRATION_US"


def test_catalog_plan_and_environment_loaders_apply_shared_byte_caps(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tiny_limits = JSONResourceLimits(max_document_bytes=8)
    monkeypatch.setattr(catalog_module, "DEFAULT_JSON_LIMITS", tiny_limits)
    with pytest.raises(CatalogValidationError, match="max_bytes=8"):
        load_catalog(CATALOG_PATH)

    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b'{"oversized":true}')
    monkeypatch.setattr(submission_module, "DEFAULT_JSON_LIMITS", tiny_limits)
    with pytest.raises(SubmissionPlanError, match="max_bytes=8"):
        load_submission_plan(plan_path)

    monkeypatch.setattr(environment_contract_module, "DEFAULT_JSON_LIMITS", tiny_limits)
    with pytest.raises(EnvironmentContractError, match="max_bytes=8"):
        audit_static_contracts(EXPERIMENT_DIRECTORY)


def _runtime() -> dict[str, object]:
    return {
        "hostname": "toranj0",
        "job_id": "12345",
        "python_version": "3.12.3",
        "torch_version": "2.4.1",
        "torch_cuda_version": "12.1",
        "runtime_nccl_version_code": 22005,
    }


def _micro_stdout() -> str:
    return json.dumps(
        {
            "schema": MICRO_STDOUT_SCHEMA,
            "rank": 0,
            "world_size": 4,
            "dtype": "bf16",
            "msg_sizes_bytes": [65536, 131072, 262144],
            "timings_us": [10.0, 20.0, 30.0],
            "metrics": {"median_us": 20.0, "iqr_us": 20.0, "count": 3},
        },
        sort_keys=True,
    )


def _full_stdout() -> str:
    return json.dumps(
        {
            "schema": FULL_STDOUT_SCHEMA,
            "rank": 0,
            "world_size": 4,
            "tokens": 3,
            "layers": 32,
            "hidden": 8192,
            "gemm_m_rank0": 256,
            "gemm_n": 8192,
            "dtype": "bf16",
            "msg_sizes_bytes": [65536, 131072, 262144],
            "inject_skew": 0.0,
            "timings_us": [10.0, 20.0, 30.0],
            "metrics": {"median_us": 20.0, "iqr_us": 20.0, "count": 3},
        },
        sort_keys=True,
    )


def _overlap_stdout() -> str:
    return json.dumps(
        {
            "schema": OVERLAP_STDOUT_SCHEMA,
            "rank": 0,
            "world_size": 4,
            "timings_us": [10.0, 20.0, 30.0],
            "metrics": {"median_us": 20.0, "iqr_us": 20.0, "count": 3},
        },
        sort_keys=True,
    )


def _micro_parameters() -> dict[str, object]:
    return {
        "adapter": "torch-json",
        "operation": "all_reduce",
        "world_size": 4,
        "global_ranks": [0, 1, 2, 3],
    }


def test_catalog_is_strict_declarative_and_manifest_ready() -> None:
    catalog = load_catalog(CATALOG_PATH)
    assert catalog.site.site_id == "rostam"
    assert catalog.site.scheduler == "slurm"
    assert catalog.site.partition == "cuda-A100"
    assert catalog.site.node_constraints == ("toranj0",)
    assert len(catalog.configurations) == 8
    assert len(catalog.workloads) == 8
    core = catalog.profile("core")
    assert core.workload_ids == ("micro", "full", "trace-build", "canary-param")
    assert "canary-overlap" not in core.workload_ids
    overlap = catalog.profile("overlap")
    assert overlap.workload_ids == ("overlap-trace-build", "canary-overlap")
    overlap_workloads = {workload.id: workload for workload in catalog.selected_workloads(overlap)}
    overlap_capture = overlap_workloads["overlap-trace-build"]
    overlap_capture_parameters = overlap_capture.parameters.to_value()
    assert overlap_capture_parameters["readiness"] == "requires-rostam-gemm-calibration"
    assert overlap_capture_parameters["outputs"]["param_trace"] == "{workspace}/param_trace_overlap.json"
    overlap_export = overlap_capture_parameters["transform_commands"][-1]
    assert "--overlap-structure" in overlap_export
    assert overlap_export[overlap_export.index("--compute-fill-us-per-gemm") + 1] == PENDING_GEMM_CALIBRATION
    overlap_replay = overlap_workloads["canary-overlap"]
    assert overlap_replay.depends_on == ("overlap-trace-build",)
    assert "{dependency:overlap-trace-build:param_trace}" in overlap_replay.parameters.to_value()["command"]

    shared_capture = catalog.profile("shared-capture")
    workload = catalog.selected_workloads(shared_capture)[0]
    shared_parameters = workload.parameters.to_value()
    assert shared_parameters["readiness"] == "requires-rostam-gemm-calibration"
    assert shared_parameters["profile_command"][-2:] == ["--profile", "{workspace}/profile.json"]
    assert shared_parameters["outputs"]["param_trace"] == "{workspace}/param_trace_overlap.json"
    shared_export = shared_parameters["transform_commands"][-1]
    assert "--overlap-structure" in shared_export
    assert shared_export[shared_export.index("--compute-fill-us-per-gemm") + 1] == PENDING_GEMM_CALIBRATION

    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    forged = copy.deepcopy(raw)
    forged["site"]["unknown"] = True
    with pytest.raises(CatalogValidationError, match="unknown fields"):
        Catalog.from_dict(forged)
    unsupported = copy.deepcopy(raw)
    unsupported["workloads"][0]["parameters"]["global_ranks"] = [0, 2, 1, 3]
    with pytest.raises(CatalogValidationError, match="dense world"):
        Catalog.from_dict(unsupported)


def test_physical_adapters_emit_distinct_strict_measurements() -> None:
    measurement = adapt_physical_measurement(
        measurement_schema=MICRO_MEASUREMENT_SCHEMA,
        producer_schema=MICRO_PRODUCER_SCHEMA,
        attempt_id="a-000001",
        parameters=_micro_parameters(),
        stdout=_micro_stdout(),
        stderr="",
        wall_time_s=0.25,
        runtime=_runtime(),
    )
    assert measurement["value_us"] == 20.0
    assert measurement["iqr_us"] == 20.0
    assert measurement["message_sizes_bytes"] == [65536, 131072, 262144]
    scalar = validate_scalar_measurement(
        PHYSICAL_MICRO_MEASUREMENT_SCHEMA,
        MICRO_PRODUCER_SCHEMA,
        "a-000001",
        measurement,
    )
    assert scalar.mode == "physical"
    assert scalar.samples_us == (10.0, 20.0, 30.0)

    full = adapt_physical_measurement(
        measurement_schema=FULL_MEASUREMENT_SCHEMA,
        producer_schema=FULL_PRODUCER_SCHEMA,
        attempt_id="a-000002",
        parameters=_micro_parameters(),
        stdout=_full_stdout(),
        stderr="",
        wall_time_s=0.5,
        runtime=_runtime(),
    )
    assert full["layers"] == 32
    assert full["gemm_m"] == 256

    overlap = adapt_physical_measurement(
        measurement_schema=OVERLAP_MEASUREMENT_SCHEMA,
        producer_schema=OVERLAP_PRODUCER_SCHEMA,
        attempt_id="a-000003",
        parameters={**_micro_parameters(), "replay_mode": "explicit-wait-overlap"},
        stdout=_overlap_stdout(),
        stderr="",
        wall_time_s=0.75,
        runtime=_runtime(),
        trace_sha256="b" * 64,
    )
    assert overlap["samples_us"] == [10.0, 20.0, 30.0]
    assert overlap["trace_sha256"] == "b" * 64

    overlap_with_placeholder = json.loads(_overlap_stdout())
    overlap_with_placeholder["dtype"] = "trace"
    with pytest.raises(PhysicalResultError, match="unknown fields: dtype"):
        adapt_physical_measurement(
            measurement_schema=OVERLAP_MEASUREMENT_SCHEMA,
            producer_schema=OVERLAP_PRODUCER_SCHEMA,
            attempt_id="a-000004",
            parameters={**_micro_parameters(), "replay_mode": "explicit-wait-overlap"},
            stdout=json.dumps(overlap_with_placeholder),
            stderr="",
            wall_time_s=0.75,
            runtime=_runtime(),
            trace_sha256="b" * 64,
        )

    with pytest.raises(PhysicalResultError, match="requires raw stdout schema"):
        adapt_physical_measurement(
            measurement_schema=OVERLAP_MEASUREMENT_SCHEMA,
            producer_schema=OVERLAP_PRODUCER_SCHEMA,
            attempt_id="a-000005",
            parameters={**_micro_parameters(), "replay_mode": "explicit-wait-overlap"},
            stdout=_micro_stdout(),
            stderr="",
            wall_time_s=0.75,
            runtime=_runtime(),
            trace_sha256="b" * 64,
        )

    param = adapt_physical_measurement(
        measurement_schema=PARAM_MEASUREMENT_SCHEMA,
        producer_schema=PARAM_PRODUCER_SCHEMA,
        attempt_id="a-000006",
        parameters={
            **_micro_parameters(),
            "adapter": "param-text",
            "replay_mode": "timestamp-paced-blocking",
        },
        stdout="Replayed all_reduce in block [x]... 12.5 us\n",
        stderr="[Warm-up] Replayed all_reduce in block [x]... 99.0 us\n",
        wall_time_s=1.0,
        runtime=_runtime(),
        trace_sha256="a" * 64,
    )
    assert param["samples_us"] == [12.5]
    assert param["trace_sha256"] == "a" * 64
    with pytest.raises(PhysicalResultError, match="requires producer"):
        adapt_physical_measurement(
            measurement_schema=MICRO_MEASUREMENT_SCHEMA,
            producer_schema=PARAM_PRODUCER_SCHEMA,
            attempt_id="a-000001",
            parameters=_micro_parameters(),
            stdout=_micro_stdout(),
            stderr="",
            wall_time_s=0.25,
            runtime=_runtime(),
        )


def test_param_trace_contract_rejects_aliasing_and_bad_request_lifetimes() -> None:
    blocking = [
        {"comms": "init", "pg_id": 0, "global_ranks": [0, 1, 2, 3]},
        {"comms": "all_reduce", "pg_id": 0, "req": 1, "in_msg_size": 16, "out_msg_size": 16, "dtype": "bfloat16"},
    ]
    assert validate_param_trace(blocking, world_size=4) == {
        "process_groups": 1,
        "collectives": 1,
        "waits": 0,
    }
    overlap = blocking + [{"comms": "wait", "req": 1}]
    assert validate_param_trace(overlap, world_size=4)["waits"] == 1
    non_world = copy.deepcopy(blocking)
    non_world[0]["global_ranks"] = [0, 1]
    with pytest.raises(PhysicalResultError, match="full world"):
        validate_param_trace(non_world, world_size=4)
    unknown_wait = blocking[:1] + [{"comms": "wait", "req": 99}]
    with pytest.raises(PhysicalResultError, match="unknown or already-completed"):
        validate_param_trace(unknown_wait, world_size=4)
    pending = blocking + [
        {"comms": "all_reduce", "pg_id": 0, "req": 2, "in_msg_size": 16, "out_msg_size": 16, "dtype": "bfloat16"},
        {"comms": "wait", "req": 1},
    ]
    with pytest.raises(PhysicalResultError, match="leaves 1 request"):
        validate_param_trace(pending, world_size=4)


def test_environment_is_pending_while_param_patch_contract_is_locally_reviewed(tmp_path: Path) -> None:
    audit = audit_static_contracts(EXPERIMENT_DIRECTORY)
    assert audit["environment_status"] == "pending-rostam-resolution"
    assert audit["patch_status"] == "reviewed"
    assert audit["param_patch_sha256"] == "59bf7dff99faf3d187a11424a641a9b2f0d190cf58794da2064d5542dc0141fc"
    assert audit["param_source_archive_sha256"] == ("d509a84fa3db007ab99be343b01f678d593628cda270af2ad571b15a2c06a7eb")
    assert audit["param_target_preimage_sha256"] == ("68dfa9362b66d47a1203f95cc0f1484397f7052def3e0e124f2e12e8fa912f8d")
    assert audit["param_target_postimage_sha256"] == (
        "219c95f65814d5db66762b96aa8ec5b34b7da4ca928b58abaaa48651880dd23a"
    )
    patch_contract = json.loads(
        (EXPERIMENT_DIRECTORY / "patches" / "param-patch-contract.json").read_text(encoding="utf-8")
    )
    assert patch_contract["upstream"]["commit"] == "a437fcebd3add1aee66fba880f28cec9fd744589"
    assert patch_contract["patch"]["apply_arguments"] == ["--check"]
    assert patch_contract["reviewed_at"] == "2026-07-11T00:03:06Z"
    assert patch_contract["collection_required"] == []
    patch_text = (EXPERIMENT_DIRECTORY / patch_contract["patch"]["path"]).read_text(encoding="utf-8")
    assert "@@ -742,7 +742,7 @@" in patch_text
    assert "--unidiff-zero" not in patch_text
    assert any(line.startswith(" ") for line in patch_text.splitlines())
    with pytest.raises(EnvironmentContractError, match="environment contract is not reviewed"):
        verify_ready_for_install(
            EXPERIMENT_DIRECTORY,
            wheel_path=tmp_path / "not-used.whl",
            wheel_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda contract: contract["upstream"].update({"source_archive_sha256": None}), "archive, preimage"),
        (
            lambda contract: contract["patch"].update({"apply_arguments": ["--check", "--unidiff-zero"]}),
            "ordinary-context",
        ),
        (lambda contract: contract["patch"].update({"sha256": "0" * 64}), "patch does not match"),
        (lambda contract: contract.update({"status": "complete"}), "unsupported PARAM patch contract status"),
        (lambda contract: contract.update({"reviewed_at": None}), "UTC reviewed_at"),
        (
            lambda contract: contract["target"].update({"postimage_sha256": contract["target"]["preimage_sha256"]}),
            "must differ",
        ),
        (lambda contract: contract.update({"collection_required": ["still pending"]}), "cannot retain"),
    ],
)
def test_reviewed_param_patch_contract_rejects_regressed_evidence(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    experiment = tmp_path / "rostam"
    shutil.copytree(EXPERIMENT_DIRECTORY, experiment)
    contract_path = experiment / "patches" / "param-patch-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    mutation(contract)
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(EnvironmentContractError, match=message):
        audit_static_contracts(experiment)


def test_reviewed_param_patch_rejects_zero_context_even_with_matching_patch_hash(tmp_path: Path) -> None:
    experiment = tmp_path / "rostam"
    shutil.copytree(EXPERIMENT_DIRECTORY, experiment)
    patch_path = experiment / "patches" / "param-use-triton-default.patch"
    zero_context = (
        "diff --git a/train/comms/pt/pytorch_dist_backend.py b/train/comms/pt/pytorch_dist_backend.py\n"
        "--- a/train/comms/pt/pytorch_dist_backend.py\n"
        "+++ b/train/comms/pt/pytorch_dist_backend.py\n"
        "@@ -1 +1 @@\n"
        "-        if collectiveArgs.use_triton:\n"
        '+        if getattr(collectiveArgs, "use_triton", False):\n'
    ).encode("utf-8")
    patch_path.write_bytes(zero_context)
    contract_path = experiment / "patches" / "param-patch-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["patch"]["sha256"] = hashlib.sha256(zero_context).hexdigest()
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(EnvironmentContractError, match="ordinary context lines"):
        audit_static_contracts(experiment)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _campaign_inputs(tmp_path: Path, *, reviewed: bool) -> dict[str, Path]:
    status = "reviewed" if reviewed else "pending-rostam-resolution"
    environment = _write_json(
        tmp_path / "environment.json",
        {"schema": "commcanary.rostam.environment-contract.v1", "status": status},
    )
    patch = _write_json(
        tmp_path / "param-patch.json",
        {
            "schema": "commcanary.rostam.param-patch-contract.v1",
            "status": "reviewed" if reviewed else "pending-upstream-preimage",
        },
    )
    wheel = tmp_path / "commcanary.whl"
    wheel.write_bytes(b"reviewed-wheel-fixture")
    return {
        "commcanary-wheel": wheel,
        "environment-lock": environment,
        "param-patch-contract": patch,
    }


def _frozen_core(tmp_path: Path, *, reviewed: bool):
    catalog = load_catalog(CATALOG_PATH)
    campaign = build_campaign(
        catalog=catalog,
        catalog_path=CATALOG_PATH,
        profile_id="core",
        run_id="rostam-static-fixture",
        repetitions=1,
        repository_commit="1" * 40,
        repository_dirty=False,
        repository_patch_sha256=None,
        source_archive_sha256="2" * 64,
        inputs=_campaign_inputs(tmp_path, reviewed=reviewed),
    )
    manifest = build_run_manifest(campaign)
    assert len(manifest.cells) == 8 * 4
    return freeze_campaign(campaign, tmp_path / "results")


@pytest.mark.parametrize(
    ("profile_id", "expected_cells"),
    (("overlap", 16), ("shared-capture", 1)),
)
def test_overlap_capture_profiles_are_structurally_complete_but_calibration_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
    expected_cells: int,
) -> None:
    catalog = load_catalog(CATALOG_PATH)
    campaign = build_campaign(
        catalog=catalog,
        catalog_path=CATALOG_PATH,
        profile_id=profile_id,
        run_id=f"rostam-{profile_id}-static-fixture",
        repetitions=1,
        repository_commit="1" * 40,
        repository_dirty=False,
        repository_patch_sha256=None,
        source_archive_sha256="2" * 64,
        inputs=_campaign_inputs(tmp_path, reviewed=True),
    )
    manifest = build_run_manifest(campaign)
    assert len(manifest.cells) == expected_cells
    frozen = freeze_campaign(campaign, tmp_path / f"results-{profile_id}")

    def forbidden(*args, **kwargs):  # pragma: no cover - proves no scheduler boundary is crossed
        raise AssertionError("subprocess must not run while calibration is unresolved")

    monkeypatch.setattr("experiments.rostam.lib.submission.subprocess.run", forbidden)
    with pytest.raises(SubmissionPlanError, match="requires-rostam-gemm-calibration"):
        build_submission_plan(frozen.directory, EXPERIMENT_DIRECTORY, dry_run=True)


def test_submission_planner_fails_before_scheduler_for_unreviewed_inputs(tmp_path: Path, monkeypatch) -> None:
    frozen = _frozen_core(tmp_path, reviewed=False)

    def forbidden(*args, **kwargs):  # pragma: no cover - proves no process boundary is crossed
        raise AssertionError("subprocess must not run while planning")

    monkeypatch.setattr("experiments.rostam.lib.submission.subprocess.run", forbidden)
    with pytest.raises(SubmissionPlanError, match="not a reviewed"):
        build_submission_plan(frozen.directory, EXPERIMENT_DIRECTORY, dry_run=True)


def test_submission_plan_precomputes_unique_owners_dependencies_and_exact_argv(tmp_path: Path, monkeypatch) -> None:
    frozen = _frozen_core(tmp_path, reviewed=True)

    def forbidden(*args, **kwargs):  # pragma: no cover - proves planner/guard stay static
        raise AssertionError("subprocess must not run while planning")

    monkeypatch.setattr("experiments.rostam.lib.submission.subprocess.run", forbidden)
    plan = build_submission_plan(frozen.directory, EXPERIMENT_DIRECTORY, dry_run=True)
    assert len(plan.cells) == 32
    assert len({cell.cell_id for cell in plan.cells}) == 32
    assert all(cell.action == "run" for cell in plan.cells)
    assert all(cell.sbatch_argv[0:2] == ("sbatch", "--parsable") for cell in plan.cells)
    assert all("--partition=cuda-A100" in cell.sbatch_argv for cell in plan.cells)
    assert all("--nodelist=toranj0" in cell.sbatch_argv for cell in plan.cells)
    assert all("--exclusive" in cell.sbatch_argv for cell in plan.cells)
    assert all(not any("*" in argument for argument in cell.sbatch_argv) for cell in plan.cells)
    canary_cells = [cell for cell in plan.cells if cell.workload_id == "canary-param"]
    assert len(canary_cells) == 8
    assert all(len(cell.dependency_attempts) == 1 for cell in canary_cells)
    assert all(len(cell.scheduler_dependency_cells) == 1 for cell in canary_cells)
    with pytest.raises(SubmissionPlanError, match="explicit --execute"):
        submit_frozen_plan(plan, execute=False)


def test_shell_layer_is_thin_and_contains_no_legacy_execution_scaffolding() -> None:
    wrappers = [
        "capture_shared_trace.sbatch",
        "run_canary.sbatch",
        "run_full.sbatch",
        "run_micro.sbatch",
        "run_shared.sbatch",
    ]
    forbidden = ("torchrun", "nvidia-smi", "eval ", "<<", "configs.json", "results/shared", "#SBATCH")
    for name in wrappers:
        path = EXPERIMENT_DIRECTORY / name
        text = path.read_text(encoding="utf-8")
        # Owner-execute only: git tracks a single executable bit, and checkouts
        # under a restrictive site umask (e.g. 077) drop group/other bits.
        assert path.stat().st_mode & 0o100 == 0o100
        assert len([line for line in text.splitlines() if line.strip()]) <= 5
        assert "lib/common.sh" in text
        assert all(token not in text for token in forbidden)
    for name in ("run_matrix.sh", "run_shared_matrix.sh"):
        path = EXPERIMENT_DIRECTORY / name
        text = path.read_text(encoding="utf-8")
        assert path.stat().st_mode & 0o100 == 0o100
        assert "experiments.rostam.lib.submission" in text
        assert "sbatch --parsable" not in text
        assert "results/shared" not in text


def test_setup_is_hash_locked_wheel_only_and_has_no_mutating_source_shortcuts() -> None:
    setup = EXPERIMENT_DIRECTORY / "setup.sh"
    common = EXPERIMENT_DIRECTORY / "lib" / "common.sh"
    text = setup.read_text(encoding="utf-8")
    assert setup.stat().st_mode & 0o100 == 0o100
    assert common.stat().st_mode & 0o100 == 0o100
    assert "--no-deps --require-hashes" in text
    assert "verify-param-preimage" in text
    assert "verify-param-postimage" in text
    assert "pip install --upgrade" not in text
    assert "numpy<2" not in text
    assert "git clone" not in text
    assert "sed -i" not in text
    assert "pip install -e" not in text
    assert "--unidiff-zero" not in text
    assert 'git -C "$PARAM_DIR" apply --check "$PATCH_PATH"' in text
    # Each venv must record the digest of the wheel it actually installed; the
    # cell entrypoint refuses venvs without this marker (or with a stale one).
    assert '>"$venv/commcanary-wheel.sha256"' in text
    assert "refusing to record a stale binding" in text


def test_vendored_third_party_checkouts_are_excluded_from_strict_typing() -> None:
    # The reviewed PARAM clone lives under third_party/ only on the cluster, so
    # a gate run there must not type-check vendored code. Ruff skips it via
    # .gitignore; mypy needs the explicit exclude.
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"^experiments/rostam/third_party/"' in pyproject
    gitignore = (EXPERIMENT_DIRECTORY / ".gitignore").read_text(encoding="utf-8")
    assert "third_party/" in gitignore


def test_design_marks_historical_evidence_and_the_exact_precluster_boundary() -> None:
    text = (EXPERIMENT_DIRECTORY / "DESIGN.md").read_text(encoding="utf-8")

    assert "complete raw attempt archive" in text
    assert "absent from the repository" in text
    assert "Pre-cluster handoff: deliberately unresolved evidence" in text
    assert "pending-rostam-resolution" in text
    assert "PARAM patch evidence is reviewed locally" in text
    assert "site.account` is `null" in text
    assert "Cluster mutation begins only at" in text
    assert "submission submit --plan PLAN" in text
    assert "No command in the completed local verification" in text


def test_historical_paper_does_not_claim_current_regeneration_or_test_counts() -> None:
    text = (REPOSITORY_ROOT / "paper" / "draft.md").read_text(encoding="utf-8")

    assert "historical campaign's complete raw attempts" in text
    assert "cannot regenerate the numeric tables above" in text
    assert "excluded from CommCanary release distributions" in text
    assert "Everything regenerates from a public repository" not in text
    assert "126 tests" not in text
