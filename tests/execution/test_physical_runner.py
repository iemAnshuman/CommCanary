from __future__ import annotations

import copy
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict

import pytest

from commcanary.adapters.chakra_capture import commcanary_trace_to_chakra
from commcanary.artifacts import physical_execution as physical_execution_module
from commcanary.artifacts.chakra import decode_chakra_execution_trace, encode_chakra_subgraph
from commcanary.artifacts.json_codec import canonical_json_bytes
from commcanary.artifacts.physical_canary import with_content_identity
from commcanary.errors import SchemaError
from commcanary.execution import physical_runner
from commcanary.execution.physical_runner import prepare_physical_program, validate_physical_execution_measurement
from commcanary.formats import TRACE_FORMAT


def _captured_trace() -> Dict[str, Any]:
    recipe = {
        "op": "gemm",
        "dtype": "bfloat16",
        "m": 2,
        "n": 4,
        "k": 3,
        "source_kernel_count": 1,
        "source_kernel_duration_us": 2.0,
    }
    events = []
    for index in range(2):
        events.append(
            {
                "id": f"collective-{index}",
                "op": "all_reduce",
                "dtype": "bfloat16",
                "bytes": 32,
                "ranks": [0, 1, 2, 3],
                "group": "default",
                "start_us": 10.0 + index * 10.0,
                "phase": "decode",
                "reduction_op": "sum",
                "compute_overlap_us": 2.0,
                "rank_arrival_us": {"0": 0.0, "1": 0.5, "2": 1.0, "3": 1.5},
                "compute_recipe_by_rank": {str(rank): [copy.deepcopy(recipe)] for rank in range(4)},
            }
        )
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "captured-vllm"},
        "system": {"source_format": "pytorch-kineto"},
        "events": events,
    }


def _program_inputs() -> tuple[bytes, bytes, Dict[str, Any]]:
    captured = commcanary_trace_to_chakra(_captured_trace(), opaque_attributes_reviewed=True)
    source = decode_chakra_execution_trace(captured.chakra_et)
    first_region = captured.projection["regions"][0]
    executable, _node_ids = encode_chakra_subgraph(source, first_region["node_ids"])
    return captured.chakra_et, executable, captured.projection


def test_prepare_physical_program_binds_exact_reduced_work() -> None:
    source, executable, projection = _program_inputs()

    program = prepare_physical_program(
        source,
        executable,
        projection,
        ["event-000000"],
        world_size=4,
        max_workspace_bytes=1024,
    )

    assert program.selected_region_ids == ("event-000000",)
    assert program.selected_node_ids == (1, 2)
    assert program.executed_collectives == 1
    assert program.executed_flops == 192
    assert program.workspace_bytes_max_rank == 180
    assert len(program.selected_node_ids) < len(decode_chakra_execution_trace(source).nodes)


def test_prepare_physical_program_supports_an_explicit_communication_only_baseline() -> None:
    source, _executable, projection = _program_inputs()
    decoded = decode_chakra_execution_trace(source)
    executable, _closure = encode_chakra_subgraph(decoded, [1])

    program = prepare_physical_program(
        source,
        executable,
        projection,
        ["event-000000"],
        world_size=4,
        max_workspace_bytes=1024,
        communication_only=True,
    )

    assert program.communication_only is True
    assert program.selected_node_ids == (1,)
    assert program.executed_collectives == 1
    assert program.executed_flops == 0
    assert program.workspace_bytes_max_rank == 128


def test_prepare_physical_program_rejects_wrong_executable_bytes() -> None:
    source, _executable, projection = _program_inputs()

    with pytest.raises(SchemaError, match="executable bytes"):
        prepare_physical_program(
            source,
            source,
            projection,
            ["event-000000"],
            world_size=4,
            max_workspace_bytes=1024,
        )


def test_prepare_physical_program_enforces_workspace_before_torch_import() -> None:
    source, executable, projection = _program_inputs()

    with pytest.raises(SchemaError, match="workspace bytes=180"):
        prepare_physical_program(
            source,
            executable,
            projection,
            ["event-000000"],
            world_size=4,
            max_workspace_bytes=179,
        )


def test_physical_runner_rejects_invalid_resource_and_identity_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, executable, projection = _program_inputs()
    with pytest.raises(SchemaError, match="world_size"):
        prepare_physical_program(source, executable, projection, ["event-000000"], world_size=1, max_workspace_bytes=1)
    with pytest.raises(SchemaError, match="max_workspace_bytes"):
        prepare_physical_program(source, executable, projection, ["event-000000"], world_size=4, max_workspace_bytes=0)
    with pytest.raises(SchemaError, match="non-empty strings"):
        physical_runner._sorted_unique_region_ids([])
    with pytest.raises(SchemaError, match="must be unique"):
        physical_runner._sorted_unique_region_ids(["same", "same"])
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(SchemaError, match="integer environment variable"):
        physical_runner._environment_int("LOCAL_RANK")
    with pytest.raises(SchemaError, match="lowercase SHA-256"):
        physical_runner._sha256("wrong", "identity")
    with pytest.raises(SchemaError, match="must use sha256"):
        physical_runner._oci_digest("wrong")


def test_prepare_physical_program_rejects_incomplete_rank_semantics() -> None:
    source, executable, projection = _program_inputs()
    changed = copy.deepcopy(projection)
    changed["nodes"][1]["execution"]["rank_recipes"] = changed["nodes"][1]["execution"]["rank_recipes"][:3]
    changed["nodes"][1]["executed_flops"] = 144
    changed["nodes"][1]["tensor_bytes"] = 156
    changed = with_content_identity(changed, "projection_id")

    with pytest.raises(SchemaError, match="recipes must cover ranks"):
        prepare_physical_program(
            source,
            executable,
            changed,
            ["event-000000"],
            world_size=4,
            max_workspace_bytes=1024,
        )


def _telemetry_snapshot(phase: str, timestamp: int) -> Dict[str, Any]:
    gpus = [
        {
            "index": rank,
            "uuid": f"GPU-{rank}",
            "pstate": "P0",
            "sm_clock_mhz": 1200.0,
            "memory_clock_mhz": 1215.0,
            "temperature_c": 40.0,
            "power_w": 100.0,
            "clock_event_reasons_active": "0x0",
            "corrected_volatile_ecc": 0,
            "uncorrected_volatile_ecc": 0,
        }
        for rank in range(4)
    ]
    xid_lines: list[str] = []
    return {
        "phase": phase,
        "monotonic_ns": timestamp,
        "gpu_observation": {
            "status": "complete",
            "command": ["nvidia-smi"],
            "returncode": 0,
            "gpus": gpus,
            "stderr": "",
        },
        "xid_observation": {
            "status": "complete",
            "command": ["journalctl"],
            "lines": xid_lines,
            "sha256": hashlib.sha256(canonical_json_bytes(xid_lines)).hexdigest(),
            "stderr": "",
        },
        "slurm_node_observation": {
            "status": "complete",
            "command": ["scontrol"],
            "node": "toranj0",
            "state": "ALLOCATED",
            "stderr": "",
        },
    }


def _measurement() -> Dict[str, Any]:
    telemetry = [
        _telemetry_snapshot("before_correctness", 1),
        _telemetry_snapshot("before_measured_cycle_1", 2),
        _telemetry_snapshot("after_measured_cycle_1", 3),
        _telemetry_snapshot("after_measured_cycle_2", 4),
        _telemetry_snapshot("final", 5),
    ]
    environment = {
        "hostname": "toranj0",
        "platform": "Linux",
        "python": "3.12.3",
        "torch": "2.11.0",
        "cuda": "13.0",
        "nccl": [2, 28, 9],
        "world_size": 4,
        "slurm_job_id": "1",
        "slurm_node": "toranj0",
        "nccl_configuration": {
            "NCCL_ALGO": None,
            "NCCL_PROTO": None,
            "NCCL_NET": None,
            "NCCL_P2P_DISABLE": None,
            "NCCL_SHM_DISABLE": None,
        },
    }
    document: Dict[str, Any] = {
        "format": "commcanary.physical_execution_measurement.v1",
        "role": "reduced_decision_canary",
        "runner": {
            "oci_digest": f"sha256:{hashlib.sha256(b'runner').hexdigest()}",
            "execution_protocol": "chakra-et-collective-graph.v1",
        },
        "subject_sha256": hashlib.sha256(b"subject").hexdigest(),
        "perturbation_id": "baseline",
        "source_et_sha256": hashlib.sha256(b"source").hexdigest(),
        "executable_sha256": hashlib.sha256(b"executable").hexdigest(),
        "projection_id": hashlib.sha256(b"projection").hexdigest(),
        "selected_region_ids": ["event-000000"],
        "selected_node_ids": [1, 2],
        "execution": {
            "world_size": 4,
            "warmups": 1,
            "iterations": 2,
            "disable_overlap": False,
            "communication_only": False,
            "rank_skew_us": 0.0,
            "rank_skew_mechanism": "host_issue_monotonic_spin",
            "collective_buffer_slots": 4,
            "workspace_bytes_max_rank": 180,
        },
        "correctness": [
            {
                "rank": rank,
                "passed": True,
                "validated_node_ids": [2, 1],
                "commitment_sha256": hashlib.sha256(f"rank:{rank}".encode()).hexdigest(),
            }
            for rank in range(4)
        ],
        "samples": [
            {
                "iteration": 0,
                "cuda_seconds_by_rank": [0.1, 0.1, 0.1, 0.1],
                "host_seconds_by_rank": [0.11, 0.11, 0.11, 0.11],
                "physical_runtime_seconds": 0.1,
                "peak_memory_bytes_by_rank": [100, 100, 100, 100],
            },
            {
                "iteration": 1,
                "cuda_seconds_by_rank": [0.2, 0.2, 0.2, 0.2],
                "host_seconds_by_rank": [0.21, 0.21, 0.21, 0.21],
                "physical_runtime_seconds": 0.2,
                "peak_memory_bytes_by_rank": [100, 100, 100, 100],
            },
        ],
        "physical_metrics": {
            "physical_runtime_seconds": 0.15000000000000002,
            "gpu_seconds": 0.6000000000000001,
            "gpu_count": 4,
            "executed_collectives": 1,
            "executed_flops": 192,
            "peak_memory_bytes": 100,
        },
        "environment": environment,
        "environment_sha256": hashlib.sha256(canonical_json_bytes(environment)).hexdigest(),
        "telemetry": telemetry,
        "telemetry_assessment": physical_execution_module.telemetry_assessment(telemetry, world_size=4),
    }
    document["measurement_id"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    return document


def test_physical_measurement_validator_recomputes_evidence_and_telemetry() -> None:
    validate_physical_execution_measurement(_measurement())


def test_physical_measurement_validator_rejects_forged_environment_identity() -> None:
    document = _measurement()
    document["environment_sha256"] = "0" * 64
    document["measurement_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in document.items() if key != "measurement_id"})
    ).hexdigest()

    with pytest.raises(SchemaError, match="environment_sha256"):
        validate_physical_execution_measurement(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(format="wrong"), "format is unsupported"),
        (lambda row: row.update(role="wrong"), "role is unsupported"),
        (lambda row: row["runner"].update(execution_protocol="wrong"), "protocol is unsupported"),
        (lambda row: row["correctness"][0].update(passed=False), "correctness must pass"),
        (lambda row: row["samples"][0].update(iteration=2), "iterations must be contiguous"),
        (lambda row: row["telemetry"][0].update(phase="wrong"), "phases are incomplete"),
    ],
)
def test_physical_measurement_validator_rejects_closed_semantic_mutations(
    mutation: Any,
    message: str,
) -> None:
    document = _measurement()
    mutation(document)
    document["measurement_id"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in document.items() if key != "measurement_id"})
    ).hexdigest()

    with pytest.raises(SchemaError, match=message):
        validate_physical_execution_measurement(document)


class _FakeScalar:
    def __init__(self, tensor: "_FakeTensor", index: int) -> None:
        self.tensor = tensor
        self.index = index

    def item(self) -> float:
        return self.tensor.values[self.index]


class _FakeTensor:
    def __init__(self, elements: int) -> None:
        self.values = [0.0 for _ in range(elements)]
        self.columns = elements

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            stop = len(self.values) if key.stop is None else int(key.stop)
            self.values = self.values[:stop]
            return self
        if isinstance(key, tuple):
            row, column = key
            return _FakeScalar(self, int(row) * self.columns + int(column))
        return _FakeScalar(self, int(key))

    def view(self, rows: int, columns: int) -> "_FakeTensor":
        assert rows * columns == len(self.values)
        self.columns = columns
        return self

    def fill_(self, value: float) -> "_FakeTensor":
        self.values[:] = [float(value) for _ in self.values]
        return self

    def zero_(self) -> "_FakeTensor":
        return self.fill_(0.0)


class _FakeWork:
    def __init__(self) -> None:
        self.wait_count = 0

    def wait(self) -> None:
        self.wait_count += 1


class _FakeEvent:
    def __init__(self, *, enable_timing: bool) -> None:
        assert enable_timing is True

    def record(self) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, other: Any) -> float:
        assert isinstance(other, _FakeEvent)
        return 100.0


def _fake_torch_and_dist(world_size: int) -> tuple[ModuleType, ModuleType]:
    torch = ModuleType("torch")
    torch.__path__ = []  # type: ignore[attr-defined]
    torch.__version__ = "2.11.0"  # type: ignore[attr-defined]
    torch.version = SimpleNamespace(cuda="13.0")  # type: ignore[attr-defined]
    for dtype in ("float16", "bfloat16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool"):
        setattr(torch, dtype, dtype)
    torch.device = lambda kind, rank: (kind, rank)  # type: ignore[attr-defined]
    torch.empty = lambda elements, **kwargs: _FakeTensor(elements)  # type: ignore[attr-defined]
    torch.inference_mode = nullcontext  # type: ignore[attr-defined]

    def mm(a: _FakeTensor, b: _FakeTensor, *, out: _FakeTensor) -> _FakeTensor:
        out.fill_(a.values[0] * b.values[0] * a.columns)
        return out

    torch.mm = mm  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        set_device=lambda rank: None,
        synchronize=lambda device: None,
        reset_peak_memory_stats=lambda device: None,
        Event=_FakeEvent,
        max_memory_allocated=lambda device: 2048,
        nccl=SimpleNamespace(version=lambda: (2, 28, 9)),
    )

    dist = ModuleType("torch.distributed")
    dist.ReduceOp = SimpleNamespace(SUM="sum")  # type: ignore[attr-defined]
    dist.init_process_group = lambda **kwargs: None  # type: ignore[attr-defined]
    dist.get_rank = lambda: 0  # type: ignore[attr-defined]
    dist.get_world_size = lambda: world_size  # type: ignore[attr-defined]
    dist.barrier = lambda: None  # type: ignore[attr-defined]
    dist.destroy_process_group = lambda: None  # type: ignore[attr-defined]

    def all_gather_object(destination: list[Any], value: Any) -> None:
        if isinstance(value, float):
            destination[:] = [float(rank + 1) for rank in range(world_size)]
        elif "rank" in value:
            destination[:] = [{**value, "rank": rank} for rank in range(world_size)]
        else:
            destination[:] = [
                {
                    **value,
                    "cuda_seconds": float(value["cuda_seconds"]) + rank / 100.0,
                    "host_seconds": float(value["host_seconds"]) + rank / 100.0,
                }
                for rank in range(world_size)
            ]

    def all_reduce(tensor: _FakeTensor, *, op: Any, async_op: bool) -> _FakeWork:
        assert op == "sum"
        assert async_op is True
        tensor.values[0] = sum(float(rank + 1) for rank in range(world_size))
        return _FakeWork()

    dist.all_gather_object = all_gather_object  # type: ignore[attr-defined]
    dist.all_reduce = all_reduce  # type: ignore[attr-defined]
    torch.distributed = dist  # type: ignore[attr-defined]
    return torch, dist


def test_physical_runner_executes_complete_program_and_writes_valid_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, executable, projection = _program_inputs()
    source_path = tmp_path / "source.et"
    executable_path = tmp_path / "canary.et"
    projection_path = tmp_path / "projection.json"
    output_path = tmp_path / "result" / "measurement.json"
    source_path.write_bytes(source)
    executable_path.write_bytes(executable)
    projection_path.write_bytes(canonical_json_bytes(projection) + b"\n")
    torch, dist = _fake_torch_and_dist(4)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "toranj0")
    runner_digest = f"sha256:{hashlib.sha256(b'runner').hexdigest()}"
    monkeypatch.setenv("COMMCANARY_RUNNER_OCI_DIGEST", runner_digest)
    timestamp = 0

    def telemetry(phase: str) -> Dict[str, Any]:
        nonlocal timestamp
        timestamp += 1
        return _telemetry_snapshot(phase, timestamp)

    monkeypatch.setattr(physical_runner, "capture_cycle_telemetry", telemetry)

    result = physical_runner.main(
        [
            "--source-et",
            str(source_path),
            "--executable-et",
            str(executable_path),
            "--projection",
            str(projection_path),
            "--selected-region",
            "event-000000",
            "--output",
            str(output_path),
            "--runner-oci-digest",
            runner_digest,
            "--subject-sha256",
            hashlib.sha256(b"subject").hexdigest(),
            "--perturbation-id",
            "baseline",
            "--role",
            "reduced_decision_canary",
            "--warmups",
            "1",
            "--iterations",
            "2",
            "--max-workspace-bytes",
            "1024",
            "--disable-overlap",
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert measurement["correctness"][0]["validated_node_ids"] == [2, 1]
    assert measurement["physical_metrics"]["executed_collectives"] == 1
    assert measurement["physical_metrics"]["executed_flops"] == 192
    assert measurement["telemetry_assessment"]["comparable"] is True
    validate_physical_execution_measurement(measurement)
