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
    oracle_plan = preflight_qualification_execution(
        str(request),
        str(materialization),
        world_size=4,
        iterations=1,
        warmup=1,
        distributed_timeout_seconds=45,
    )
    validation_entries = [
        entry
        for entry in oracle_plan.entries
        if entry.get("comms") in execution_module._COLLECTIVES and 0 in dict(oracle_plan.groups)[int(entry["pg_id"])]
    ]
    validation_index = 0

    class FakeTensor(_ContentTensor):
        def __init__(self, length: int = 1) -> None:
            super().__init__([0] * max(1, length))

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

    def tensor_factory(*shape: Any, **_kwargs: Any) -> FakeTensor:
        length = int(shape[0]) if shape and isinstance(shape[0], int) else 1
        return FakeTensor(length)

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

    def operation(name: str, output: _ContentTensor) -> FakeWork:
        nonlocal validation_index
        calls[name] = calls.get(name, 0) + 1
        if validation_index < len(validation_entries):
            entry = validation_entries[validation_index]
            assert entry["comms"] == name
            expected = execution_module._validation_expected_output_values(
                entry,
                request_id=oracle_plan.request_id,
                rank=0,
                group_ranks=dict(oracle_plan.groups)[int(entry["pg_id"])],
            )
            assert len(output.values) == len(expected)
            output.values[:] = list(expected)
            validation_index += 1
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
    fake_dist.all_reduce = lambda tensor, **_kwargs: operation("all_reduce", tensor)  # type: ignore[attr-defined]
    fake_dist.broadcast = lambda tensor, **_kwargs: operation("broadcast", tensor)  # type: ignore[attr-defined]
    fake_dist.all_gather_into_tensor = lambda output, _input, **_kwargs: operation("all_gather", output)  # type: ignore[attr-defined]
    fake_dist.reduce_scatter_tensor = lambda output, _input, **_kwargs: operation("reduce_scatter", output)  # type: ignore[attr-defined]
    fake_dist.all_to_all_single = lambda output, _input, **_kwargs: operation("all_to_all", output)  # type: ignore[attr-defined]
    fake_dist.isend = lambda tensor, **_kwargs: operation("send", tensor)  # type: ignore[attr-defined]
    fake_dist.irecv = lambda tensor, **_kwargs: operation("recv", tensor)  # type: ignore[attr-defined]

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


class _ContentTensor:
    def __init__(
        self,
        values: list[float | int],
        *,
        indices: list[int] | None = None,
    ) -> None:
        self.values = values
        self.indices = list(range(len(values))) if indices is None else indices

    def narrow(self, _dimension: int, start: int, length: int) -> _ContentTensor:
        return _ContentTensor(self.values, indices=self.indices[start : start + length])

    def __getitem__(self, item: slice) -> _ContentTensor:
        return _ContentTensor(self.values, indices=self.indices[item])

    def fill_(self, value: float | int) -> _ContentTensor:
        for index in self.indices:
            self.values[index] = value
        return self

    def zero_(self) -> _ContentTensor:
        return self.fill_(0)

    def eq(self, expected: float | int) -> _ContentTensor:
        return _ContentTensor([int(self.values[index] == expected) for index in self.indices])

    def all(self) -> _ContentTensor:
        return _ContentTensor([int(all(self.values[index] for index in self.indices))])

    def item(self) -> int:
        assert len(self.indices) == 1
        return int(self.values[self.indices[0]])


def _oracle_entry(
    operation: str,
    *,
    reduction_op: str | None = None,
    dtype: str = "float32",
    group_ranks: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[str, Any]:
    group_size = len(group_ranks)
    out_elements = 16
    in_elements = out_elements
    if operation == "all_gather":
        in_elements = out_elements // group_size
    elif operation == "reduce_scatter":
        in_elements = out_elements * group_size
    entry: dict[str, Any] = {
        "comms": operation,
        "req": 17,
        "pg_id": 0,
        "global_ranks": list(group_ranks),
        "in_msg_size": in_elements,
        "out_msg_size": out_elements,
        "dtype": dtype,
    }
    if reduction_op is not None:
        entry["reduction_op"] = reduction_op
    return entry


def _oracle_matches(
    entry: dict[str, Any],
    values: list[float | int],
    *,
    rank: int,
    group_ranks: tuple[int, ...] = (0, 1, 2, 3),
    request_id: str = "request-oracle-test",
) -> bool:
    output = _ContentTensor(values)
    key = execution_module._communication_buffer_key(entry)
    return execution_module._validation_output_matches(
        entry,
        request_id=request_id,
        rank=rank,
        group_ranks=group_ranks,
        buffers={"communication": {key: (_ContentTensor([0] * int(entry["in_msg_size"])), output)}},
    )


def test_reduction_oracle_rejects_max_executed_for_requested_product() -> None:
    entry = _oracle_entry("all_reduce", reduction_op="product")
    expected = execution_module._validation_expected_output_values(
        entry,
        request_id="request-oracle-test",
        rank=0,
        group_ranks=(0, 1, 2, 3),
    )
    assert _oracle_matches(entry, list(expected), rank=0)
    wrong: list[float | int] = []
    for index in range(int(entry["out_msg_size"])):
        lane = index % execution_module._VALIDATION_LANE_PERIOD
        probe = execution_module._reduction_probe(
            request_id="request-oracle-test",
            request=int(entry["req"]),
            group_ranks=(0, 1, 2, 3),
            destination_rank=None,
            lane=lane,
            lane_count=execution_module._VALIDATION_LANE_PERIOD,
            dtype="float32",
            reduction_op="product",
        )
        wrong.append(probe.outcomes[execution_module._REDUCTION_OUTCOME_INDEX["max"]])
    assert not _oracle_matches(entry, wrong, rank=0)


def test_reduce_scatter_oracle_rejects_wrong_destination_shard() -> None:
    entry = _oracle_entry("reduce_scatter", reduction_op="sum")
    expected_rank_zero = execution_module._validation_expected_output_values(
        entry,
        request_id="request-oracle-test",
        rank=0,
        group_ranks=(0, 1, 2, 3),
    )
    wrong_rank_one = execution_module._validation_expected_output_values(
        entry,
        request_id="request-oracle-test",
        rank=1,
        group_ranks=(0, 1, 2, 3),
    )
    assert _oracle_matches(entry, list(expected_rank_zero), rank=0)
    assert not _oracle_matches(entry, list(wrong_rank_one), rank=0)


def test_all_gather_oracle_rejects_rank_block_permutation() -> None:
    entry = _oracle_entry("all_gather")
    expected = list(
        execution_module._validation_expected_output_values(
            entry,
            request_id="request-oracle-test",
            rank=0,
            group_ranks=(0, 1, 2, 3),
        )
    )
    block = int(entry["in_msg_size"])
    permuted = expected[block : 2 * block] + expected[:block] + expected[2 * block :]
    assert _oracle_matches(entry, expected, rank=0)
    assert not _oracle_matches(entry, permuted, rank=0)


def test_all_to_all_oracle_rejects_destination_chunk_permutation() -> None:
    entry = _oracle_entry("all_to_all")
    expected_rank_zero = execution_module._validation_expected_output_values(
        entry,
        request_id="request-oracle-test",
        rank=0,
        group_ranks=(0, 1, 2, 3),
    )
    wrong_rank_one = execution_module._validation_expected_output_values(
        entry,
        request_id="request-oracle-test",
        rank=1,
        group_ranks=(0, 1, 2, 3),
    )
    assert _oracle_matches(entry, list(expected_rank_zero), rank=0)
    assert not _oracle_matches(entry, list(wrong_rank_one), rank=0)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "int8"))
def test_low_precision_routing_signatures_are_injective_for_four_rank_all_to_all(dtype: str) -> None:
    signatures = {
        (source, destination): tuple(
            execution_module._routing_value(
                request_id="low-precision-routing",
                request=17,
                source_index=source,
                destination_index=destination,
                group_size=4,
                lane=lane,
                dtype=dtype,
            )
            for lane in range(execution_module._VALIDATION_LANE_PERIOD)
        )
        for source in range(4)
        for destination in range(4)
    }

    assert len(set(signatures.values())) == 16
    assert signatures[(0, 0)] != signatures[(2, 3)]


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "int8"))
def test_sparse_global_ranks_use_group_local_routing_indices(dtype: str) -> None:
    group_ranks = (0, 127)
    entry = _oracle_entry("all_gather", dtype=dtype, group_ranks=group_ranks)
    expected = list(
        execution_module._validation_expected_output_values(
            entry,
            request_id="sparse-rank-routing",
            rank=0,
            group_ranks=group_ranks,
        )
    )
    block = int(entry["in_msg_size"])
    permuted = expected[block:] + expected[:block]

    assert expected[:block] != expected[block:]
    assert _oracle_matches(
        entry,
        expected,
        rank=0,
        group_ranks=group_ranks,
        request_id="sparse-rank-routing",
    )
    assert not _oracle_matches(
        entry,
        permuted,
        rank=0,
        group_ranks=group_ranks,
        request_id="sparse-rank-routing",
    )


@pytest.mark.parametrize(
    ("dtype", "reduction_op"),
    tuple(
        (dtype, reduction_op)
        for dtype in ("float16", "bfloat16", "int8", "uint8", "int16", "int32", "int64")
        for reduction_op in ("sum", "avg", "min", "max", "product")
    ),
)
def test_maximum_rank_reduction_probes_are_bounded_and_operator_distinguishing(
    dtype: str,
    reduction_op: str,
) -> None:
    candidates = execution_module._reduction_probe_candidates(dtype, 65_536, reduction_op)

    assert 1 <= len(candidates) <= execution_module._REDUCTION_PROBE_TEMPLATE_COUNT
    assert len(
        {probe.outcomes[execution_module._REDUCTION_OUTCOME_INDEX[reduction_op]] for probe in candidates}
    ) == len(candidates)
    for probe in candidates:
        assert len(set(probe.outcomes)) == len(execution_module._REDUCTION_OUTCOME_INDEX)
        assert probe.value_at(65_535) in {probe.tail_even, probe.tail_odd}


def test_correctness_probe_work_is_resource_accounted() -> None:
    entry = _oracle_entry("all_reduce", reduction_op="sum")
    with pytest.raises(SchemaError, match="correctness probe work=32 exceeds limit=31"):
        execution_module._validate_correctness_probe_support(
            (entry,),
            {0: (0, 1, 2, 3)},
            limits=ResourceLimits(max_execution_correctness_probe_work=31),
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
