from __future__ import annotations

import builtins
import copy
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import commcanary.execution.qualification as execution_module
from commcanary.compiler import compile_trace
from commcanary.errors import CommCanaryError, SchemaError
from commcanary.execution import (
    distributed_execution_environment,
    execute_qualification_materialization,
    preflight_qualification_execution,
)
from commcanary.resources import ResourceLimits
from commcanary.services import prepare_qualification_request
from commcanary.workflows import materialize_qualification
from tests.builders import qualification_policy, qualification_trace


def _prepared_materialization(tmp_path: Path) -> tuple[Path, Path]:
    trace = qualification_trace()
    trace["events"][0]["op"] = "broadcast"
    trace["events"][0].pop("reduction_op")
    trace["events"][0]["root_rank"] = 2
    trace["events"][0]["dtype"] = "int64"
    exact_nelems = trace["events"][0]["bytes"] // 8
    trace["events"][0]["metadata"]["kineto_in_msg_nelems"] = exact_nelems
    trace["events"][0]["metadata"]["kineto_out_msg_nelems"] = exact_nelems
    canary = compile_trace(trace)
    request = tmp_path / "request"
    materialization = tmp_path / "materialization"
    prepare_qualification_request(str(request), trace, canary, qualification_policy())
    materialize_qualification(str(request), str(materialization))
    return request, materialization


def test_preflight_accepts_real_public_program_shape_and_bounds_all_work(
    tmp_path: Path,
) -> None:
    request, materialization = _prepared_materialization(tmp_path)
    plan = preflight_qualification_execution(
        str(request),
        str(materialization),
        world_size=4,
        iterations=2,
        warmup=1,
    )

    assert plan.world_size == 4
    assert plan.distributed_timeout_seconds == 300
    assert plan.groups == ((0, (0, 1, 2, 3)),)
    assert plan.communication_entries_per_pass == 6
    assert plan.compute_operations_per_pass == 24
    assert plan.rank_compute_operations_per_pass == (6, 6, 6, 6)
    assert plan.observation_samples == 48
    assert plan.rank_correctness_checks == (6, 6, 6, 6)
    assert len(plan.rank_tensor_bytes) == 4
    # One reusable exact rectangular GEMM recipe per rank, plus
    # request-isolated buffers for five float32 all-reduces and one int64
    # broadcast. The rank-local m dimension produces different bounded totals.
    assert plan.rank_tensor_bytes == (786_624, 786_656, 786_688, 786_720)
    operations = {
        entry.get("comms")
        for entry in plan.entries
        if entry.get("comms") is not None and entry.get("comms") not in {"init", "wait"}
    }
    assert operations == {"all_reduce", "broadcast"}
    assert {entry.get("dtype") for entry in plan.entries if entry.get("comms") in operations} == {"float32", "int64"}
    assert {
        entry["reduction_op"] for entry in plan.entries if entry.get("comms") in {"all_reduce", "reduce_scatter"}
    } == {"sum"}
    broadcast = next(entry for entry in plan.entries if entry.get("comms") == "broadcast")
    assert broadcast["root"] == 2


def test_preflight_rejects_rank_allocation_and_repeated_work_before_torch_import(
    tmp_path: Path,
) -> None:
    request, materialization = _prepared_materialization(tmp_path)

    with pytest.raises(SchemaError, match="outside world_size=3"):
        preflight_qualification_execution(
            str(request),
            str(materialization),
            world_size=3,
        )
    with pytest.raises(SchemaError, match="cover exactly the launched rank domain"):
        preflight_qualification_execution(
            str(request),
            str(materialization),
            world_size=5,
        )
    with pytest.raises(SchemaError, match="total tensor bytes"):
        preflight_qualification_execution(
            str(request),
            str(materialization),
            world_size=4,
            limits=ResourceLimits(max_execution_total_tensor_bytes=1024),
        )
    with pytest.raises(SchemaError, match="execution compute operations"):
        preflight_qualification_execution(
            str(request),
            str(materialization),
            world_size=4,
            iterations=10,
            warmup=1,
            limits=ResourceLimits(max_execution_compute_operations=200),
        )
    with pytest.raises(SchemaError, match="observation samples"):
        preflight_qualification_execution(
            str(request),
            str(materialization),
            world_size=4,
            iterations=2,
            warmup=0,
            limits=ResourceLimits(max_execution_observation_samples=47),
        )
    with pytest.raises(SchemaError, match="qualification communication operations"):
        preflight_qualification_execution(
            str(request),
            str(materialization),
            world_size=4,
            iterations=2,
            warmup=1,
            limits=ResourceLimits(max_replay_events=23),
        )
    with pytest.raises(SchemaError, match="distributed_timeout_seconds must be positive"):
        preflight_qualification_execution(
            str(request),
            str(materialization),
            world_size=4,
            distributed_timeout_seconds=0,
        )
    with pytest.raises(SchemaError, match="max_execution_timeout_seconds=60"):
        preflight_qualification_execution(
            str(request),
            str(materialization),
            world_size=4,
            distributed_timeout_seconds=61,
            limits=ResourceLimits(max_execution_timeout_seconds=60),
        )


def test_preflight_supports_every_materialized_communication_operation(
    tmp_path: Path,
) -> None:
    operations = [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "broadcast",
    ]
    events = []
    for index, operation in enumerate(operations):
        event = {
            "id": str(index),
            "op": operation,
            "bytes": 128,
            "ranks": [0, 1],
            "group": "g",
            "start_us": index * 10.0,
            "rank_arrival_us": {"0": 0.0, "1": 1.0},
            "compute_overlap_us": 1.0,
            "dtype": "float32",
            "metadata": {
                "kineto_in_msg_nelems": 32,
                "kineto_out_msg_nelems": 32,
                "kineto_in_split_sizes": [],
                "kineto_out_split_sizes": [],
                "kineto_message_shape_status": "derived",
                "kineto_message_shape_method": "record-param-comms-in-out-nelems.v1",
                "kineto_compute_recipe_status": "derived",
                "kineto_compute_recipe_method": "explicit-wait-linked-contiguous-gemm.v1",
            },
            "compute_recipe_by_rank": {
                "0": [
                    {
                        "op": "gemm",
                        "dtype": "float32",
                        "m": 2,
                        "n": 4,
                        "k": 4,
                        "source_kernel_count": 1,
                        "source_kernel_duration_us": 1.0,
                    }
                ],
                "1": [
                    {
                        "op": "gemm",
                        "dtype": "float32",
                        "m": 3,
                        "n": 4,
                        "k": 4,
                        "source_kernel_count": 1,
                        "source_kernel_duration_us": 1.0,
                    }
                ],
            },
        }
        if operation == "all_gather":
            event["metadata"]["kineto_in_msg_nelems"] = 16
        if operation == "reduce_scatter":
            event["metadata"]["kineto_out_msg_nelems"] = 16
        if operation == "broadcast":
            event["root_rank"] = 1
        if operation in {"all_reduce", "reduce_scatter"}:
            event["reduction_op"] = "sum"
        events.append(event)
    trace = {
        "format": "commcanary.trace.v1",
        "workload": {
            "name": "all-collectives",
            "import_source": "pytorch-kineto",
            "skipped_empty_events": 0,
        },
        "system": {"source_format": "pytorch-kineto"},
        "events": events,
    }
    canary = compile_trace(trace)
    request = tmp_path / "request"
    materialization = tmp_path / "materialization"
    prepare_qualification_request(str(request), trace, canary, qualification_policy())
    materialize_qualification(str(request), str(materialization))

    plan = preflight_qualification_execution(
        str(request),
        str(materialization),
        world_size=2,
        iterations=1,
        warmup=0,
    )
    assert {entry.get("comms") for entry in plan.entries if entry.get("comms") not in {None, "init", "wait"}} == {
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "broadcast",
    }
    assert plan.communication_entries_per_pass == 5
    assert plan.observation_samples == 10
    assert plan.rank_correctness_checks == (5, 5)
    broadcast = next(entry for entry in plan.entries if entry.get("comms") == "broadcast")
    assert broadcast["root"] == 1


def test_execution_imports_torch_only_after_complete_portable_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, materialization = _prepared_materialization(tmp_path)
    real_import = builtins.__import__
    torch_import_attempted = False

    def missing_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal torch_import_attempted
        if name == "torch" or name.startswith("torch."):
            torch_import_attempted = True
            raise ImportError("target torch absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_torch)
    with pytest.raises(CommCanaryError, match="target-compatible PyTorch"):
        execute_qualification_materialization(
            str(request),
            str(materialization),
            rank=0,
            world_size=4,
            local_rank=0,
            device="cpu",
            backend="gloo",
            iterations=1,
            warmup=0,
        )
    assert torch_import_attempted

    torch_import_attempted = False
    with pytest.raises(SchemaError, match="outside world_size=3"):
        execute_qualification_materialization(
            str(request),
            str(materialization),
            rank=0,
            world_size=3,
            local_rank=0,
            device="cpu",
            backend="gloo",
            iterations=1,
            warmup=0,
        )
    assert not torch_import_attempted


def test_preflight_binds_the_exact_parsed_program_after_directory_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, materialization = _prepared_materialization(tmp_path)
    real_load = execution_module.load_verified_qualification_materialization

    def load_then_mutate(
        request_directory: str,
        materialization_directory: str,
        **kwargs: Any,
    ) -> Any:
        result = real_load(
            request_directory,
            materialization_directory,
            **kwargs,
        )
        program_path = Path(materialization_directory) / "replay-program.json"
        program = json.loads(program_path.read_text(encoding="utf-8"))
        compute_entry = next(entry for entry in program if entry.get("compute") == "gemm_recipe")
        compute_entry["recipe_by_rank"]["0"][0]["m"] += 1
        program_path.write_text(
            json.dumps(program, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        execution_module,
        "load_verified_qualification_materialization",
        load_then_mutate,
    )
    plan = preflight_qualification_execution(
        str(request),
        str(materialization),
        world_size=4,
    )
    compute_entry = next(entry for entry in plan.entries if entry.get("compute") == "gemm_recipe")
    assert compute_entry["recipe_by_rank"]["0"][0]["m"] == 2


def test_injected_runtime_executes_bound_mixed_dtype_program_and_aggregates_all_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, materialization = _prepared_materialization(tmp_path)
    calls: dict[str, int] = {}

    class FakeTensor:
        def fill_(self, _value: int) -> FakeTensor:
            return self

        def zero_(self) -> FakeTensor:
            return self

        def narrow(self, _dimension: int, _start: int, _length: int) -> FakeTensor:
            return self

        def eq(self, _value: int) -> FakeTensor:
            return self

        def all(self) -> FakeTensor:
            return self

        def item(self) -> int:
            return 1

    class FakeWork:
        def wait(self) -> None:
            calls["wait"] = calls.get("wait", 0) + 1

    fake_torch = types.ModuleType("torch")
    fake_dist = types.ModuleType("torch.distributed")
    fake_torch.__path__ = []
    fake_torch.__version__ = "test-torch"  # type: ignore[attr-defined]
    fake_torch.version = types.SimpleNamespace(cuda=None)  # type: ignore[attr-defined]
    fake_dist.ReduceOp = types.SimpleNamespace(  # type: ignore[attr-defined]
        AVG="avg",
        MAX="max",
        MIN="min",
        PRODUCT="product",
        SUM="sum",
    )
    for dtype in (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
    ):
        setattr(fake_torch, dtype, dtype)

    def tensor_factory(*_shape: Any, **_kwargs: Any) -> FakeTensor:
        return FakeTensor()

    def mm(
        _left: FakeTensor,
        _right: FakeTensor,
        *,
        out: FakeTensor,
    ) -> FakeTensor:
        calls["mm"] = calls.get("mm", 0) + 1
        calls["mm_reused_output"] = calls.get("mm_reused_output", 0) + 1
        return out

    fake_torch.randn = tensor_factory  # type: ignore[attr-defined]
    fake_torch.ones = tensor_factory  # type: ignore[attr-defined]
    fake_torch.empty = tensor_factory  # type: ignore[attr-defined]
    fake_torch.mm = mm  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: False,
        nccl=types.SimpleNamespace(version=lambda: 0),
    )

    initialized = False

    def init_process_group(_backend: str, *, timeout: Any) -> None:
        nonlocal initialized
        initialized = True
        calls["init_timeout_seconds"] = int(timeout.total_seconds())

    def destroy_process_group() -> None:
        nonlocal initialized
        initialized = False

    def operation(name: str) -> FakeWork:
        calls[name] = calls.get(name, 0) + 1
        return FakeWork()

    fake_dist.group = types.SimpleNamespace(WORLD="world")  # type: ignore[attr-defined]
    fake_dist.init_process_group = init_process_group  # type: ignore[attr-defined]
    fake_dist.destroy_process_group = destroy_process_group  # type: ignore[attr-defined]
    fake_dist.is_initialized = lambda: initialized  # type: ignore[attr-defined]

    def new_group(ranks: list[int], *, timeout: Any) -> tuple[int, ...]:
        calls["new_group"] = calls.get("new_group", 0) + 1
        calls["new_group_timeout_seconds"] = int(timeout.total_seconds())
        return tuple(ranks)

    fake_dist.new_group = new_group  # type: ignore[attr-defined]
    fake_dist.barrier = lambda: None  # type: ignore[attr-defined]
    fake_dist.get_rank = lambda: 0  # type: ignore[attr-defined]
    fake_dist.get_world_size = lambda: 4  # type: ignore[attr-defined]
    fake_dist.all_reduce = lambda *_args, **_kwargs: operation("all_reduce")  # type: ignore[attr-defined]
    fake_dist.broadcast = lambda *_args, **_kwargs: operation("broadcast")  # type: ignore[attr-defined]
    fake_dist.all_gather_into_tensor = lambda *_args, **_kwargs: operation("all_gather")  # type: ignore[attr-defined]
    fake_dist.reduce_scatter_tensor = lambda *_args, **_kwargs: operation("reduce_scatter")  # type: ignore[attr-defined]
    fake_dist.all_to_all_single = lambda *_args, **_kwargs: operation("all_to_all")  # type: ignore[attr-defined]
    fake_dist.isend = lambda *_args, **_kwargs: operation("send")  # type: ignore[attr-defined]
    fake_dist.irecv = lambda *_args, **_kwargs: operation("recv")  # type: ignore[attr-defined]

    def all_gather_object(output: list[Any], local_value: Any) -> None:
        if isinstance(local_value, dict):
            for rank in range(len(output)):
                result = copy.deepcopy(local_value)
                result["rank"] = rank
                for sample in result.get("samples", []):
                    sample["rank"] = rank
                if "program_makespans_us" in result:
                    result["program_makespans_us"] = [float(rank + 1)]
                output[rank] = result
            return
        for rank in range(len(output)):
            samples = copy.deepcopy(local_value)
            for sample in samples:
                sample["rank"] = rank
            output[rank] = samples

    fake_dist.all_gather_object = all_gather_object  # type: ignore[attr-defined]
    fake_dist.get_backend = lambda: "gloo"  # type: ignore[attr-defined]
    fake_torch.distributed = fake_dist  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)

    result = execute_qualification_materialization(
        str(request),
        str(materialization),
        rank=0,
        world_size=4,
        local_rank=0,
        device="cpu",
        backend="gloo",
        iterations=1,
        warmup=1,
        distributed_timeout_seconds=45,
    )

    assert result is not None
    assert result["schema"] == "commcanary.reference-execution.stdout.v1"
    assert result["request_id"]
    assert result["materialization_id"]
    assert result["executor"]["claim"] == ("reference-implementation-not-yet-physically-conformance-validated")
    assert result["executor"]["distributed_timeout_seconds"] == 45
    assert result["correctness_validation"] == {
        "status": "passed",
        "semantics": "untimed-deterministic-communication-data-check",
        "checks_per_rank": [6, 6, 6, 6],
        "total_check_count": 24,
    }
    assert result["aggregate"]["count"] == 6
    assert result["program_makespan"]["count"] == 1
    assert result["program_makespan"]["semantics"] == ("maximum-rank-whole-program-wall-clock")
    assert result["program_makespan"]["timings_us"] == [4.0]
    assert result["program_makespan"]["median_us"] == 4.0
    assert set(result["program_makespan"]["rank_timings_us"]) == {
        "0",
        "1",
        "2",
        "3",
    }
    assert set(result["rank_samples"]) == {"0", "1", "2", "3"}
    assert result["claims"] == {
        "physical_execution": "self_reported_reference_executor",
        "physical_fidelity": "unproven",
        "qualification_verdict": "not_issued",
    }
    assert calls["mm"] == 12
    assert calls["mm_reused_output"] == 12
    assert calls["all_reduce"] == 15
    assert calls["broadcast"] == 3
    assert calls["new_group"] == 1
    assert calls["init_timeout_seconds"] == 45
    assert calls["new_group_timeout_seconds"] == 45
    assert initialized is False


def test_execution_aggregation_rejects_a_duplicate_rank_masking_a_missing_rank(
    tmp_path: Path,
) -> None:
    request, materialization = _prepared_materialization(tmp_path)
    plan = preflight_qualification_execution(
        str(request),
        str(materialization),
        world_size=4,
        iterations=1,
        warmup=0,
    )
    rank_samples: dict[str, list[dict[str, Any]]] = {}
    for rank in range(plan.world_size):
        rank_samples[str(rank)] = [
            {
                "rank": rank,
                "iteration": 0,
                "sequence": sequence,
                "request": entry["req"],
                "operation": entry["comms"],
                "duration_us": 1.0,
            }
            for sequence, entry in enumerate(plan.entries)
            if entry.get("comms") in execution_module._COLLECTIVES
        ]
    rank_samples["1"].pop(0)
    rank_samples["0"].append(copy.deepcopy(rank_samples["0"][0]))

    with pytest.raises(CommCanaryError, match="duplicates rank 0"):
        execution_module._execution_payload(
            plan,
            rank_samples=rank_samples,
            rank_program_makespans_us={str(rank): [1.0] for rank in range(plan.world_size)},
            device="cpu",
            backend="gloo",
            torch=object(),
            dist=object(),
            correctness_checks=plan.rank_correctness_checks,
        )


def test_correctness_validation_fails_collectively_on_wrong_rank_data(
    tmp_path: Path,
) -> None:
    request, materialization = _prepared_materialization(tmp_path)
    plan = preflight_qualification_execution(
        str(request),
        str(materialization),
        world_size=4,
        iterations=1,
        warmup=0,
    )
    gathered = [
        {
            "rank": rank,
            "check_count": plan.rank_correctness_checks[rank],
            "failure_count": 0,
            "failures": [],
        }
        for rank in range(plan.world_size)
    ]
    gathered[2]["failure_count"] = 1
    gathered[2]["failures"] = ["sequence 4 request 3 all_reduce"]

    with pytest.raises(
        CommCanaryError,
        match=r"rank 2: 1 failures .*all_reduce",
    ):
        execution_module._validated_correctness_checks(plan, gathered)


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ((2, 20, 5), 22005),
        ([2, 19, 3], 21903),
        (22005, 22005),
    ],
)
def test_runtime_nccl_version_code_preserves_supported_pytorch_representations(
    reported: object,
    expected: int,
) -> None:
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            nccl=types.SimpleNamespace(version=lambda: reported),
        ),
    )

    assert execution_module._runtime_nccl_version_code(torch) == expected


@pytest.mark.parametrize(
    "reported",
    [
        None,
        True,
        0,
        (2, 20),
        (2, "20", 5),
        (2, 100, 0),
    ],
)
def test_runtime_nccl_version_code_refuses_missing_or_ambiguous_identity(
    reported: object,
) -> None:
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            nccl=types.SimpleNamespace(version=lambda: reported),
        ),
    )

    with pytest.raises(CommCanaryError, match="NCCL execution reported"):
        execution_module._runtime_nccl_version_code(torch)


def test_runtime_dispatch_covers_every_materialized_operation() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeDist:
        ReduceOp = types.SimpleNamespace(
            AVG="avg",
            MAX="max",
            MIN="min",
            PRODUCT="product",
            SUM="sum",
        )

        def __getattr__(self, name: str) -> Any:
            def invoke(*_args: Any, **kwargs: Any) -> str:
                calls.append((name, kwargs))
                return name

            return invoke

    fake_dist = FakeDist()
    base = {
        "req": 7,
        "pg_id": 0,
        "global_ranks": [0, 1],
        "in_msg_size": 8,
        "out_msg_size": 8,
        "dtype": "float32",
        "src_rank": 0,
        "dst_rank": 1,
    }
    expected_methods = {
        "all_reduce": "all_reduce",
        "broadcast": "broadcast",
        "all_gather": "all_gather_into_tensor",
        "reduce_scatter": "reduce_scatter_tensor",
        "all_to_all": "all_to_all_single",
        "send": "isend",
        "recv": "irecv",
    }
    for operation, method in expected_methods.items():
        entry = {**base, "comms": operation}
        if operation == "broadcast":
            entry["root"] = 1
        if operation in {"all_reduce", "reduce_scatter"}:
            entry["reduction_op"] = "sum"
        key = execution_module._communication_buffer_key(entry)
        result = execution_module._issue_runtime_operation(
            entry,
            rank=0,
            group="group",
            buffers={"communication": {key: (object(), object())}},
            dist=fake_dist,
        )
        assert result == method

    assert [name for name, _kwargs in calls] == list(expected_methods.values())
    assert calls[0][1]["op"] == "sum"
    assert calls[1][1]["src"] == 1
    assert calls[3][1]["op"] == "sum"
    assert calls[-2][1]["dst"] == 1
    assert calls[-1][1]["src"] == 0


def test_communication_buffers_are_isolated_by_request_identity() -> None:
    base = {
        "comms": "all_reduce",
        "pg_id": 0,
        "in_msg_size": 8,
        "out_msg_size": 8,
        "dtype": "float32",
    }

    assert execution_module._communication_buffer_key({**base, "req": 1}) != (
        execution_module._communication_buffer_key({**base, "req": 2})
    )


def test_preflight_message_shape_rejects_uneven_all_to_all_split() -> None:
    with pytest.raises(SchemaError, match="invalid all_to_all"):
        execution_module._validate_message_shape(
            "all_to_all",
            in_elements=3,
            out_elements=3,
            group_size=2,
            index=1,
        )


def test_distributed_environment_is_explicit_and_fail_closed() -> None:
    assert distributed_execution_environment({"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "2"}) == (1, 4, 2)
    assert distributed_execution_environment({"RANK": "0", "WORLD_SIZE": "1"}) == (
        0,
        1,
        0,
    )
    with pytest.raises(CommCanaryError, match="must be valid integers"):
        distributed_execution_environment({})
    with pytest.raises(CommCanaryError, match="outside the declared world"):
        distributed_execution_environment({"RANK": "4", "WORLD_SIZE": "4"})


def test_reference_execution_output_is_not_a_qualification_observation_format(
    tmp_path: Path,
) -> None:
    request, materialization = _prepared_materialization(tmp_path)
    manifest = json.loads((materialization / "materialization.json").read_text(encoding="utf-8"))
    assert manifest["claims"]["physical_execution"] == "not_included"
    assert manifest["claims"]["qualification_verdict"] == "not_issued"
