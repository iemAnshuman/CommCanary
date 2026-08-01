from __future__ import annotations

import argparse
import builtins
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rostam import (
    microbench_tp8,
    overlap_replay,
    qualification_physical,
    workload_overlap_capture,
    workload_tp8,
)
from experiments.rostam.lib import physical_results
from experiments.rostam.lib.physical_results import (
    FULL_STDOUT_SCHEMA,
    MICRO_STDOUT_SCHEMA,
    OVERLAP_STDOUT_SCHEMA,
    ParamTraceLimits,
    PhysicalResultError,
    load_validated_param_trace,
)


def _trace(*, ranks: list[int], explicit_wait: bool = True) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = [
        {
            "comms": "init",
            "pg_id": 0,
            "global_ranks": ranks,
            "world_size": len(ranks),
        },
        {
            "comms": "all_reduce",
            "pg_id": 0,
            "req": 1,
            "global_ranks": ranks,
            "world_size": len(ranks),
            "in_msg_size": 16,
            "out_msg_size": 16,
            "dtype": "bfloat16",
        },
    ]
    if explicit_wait:
        entries.append({"comms": "wait", "req": 1})
    return entries


def _write_trace(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def _qualification_sources(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = {}
    for source_id in (
        "request_manifest",
        "source_trace",
        "canary",
        "fidelity",
        "qualification_policy",
        "materialization_manifest",
        "replay_program",
    ):
        source = tmp_path / f"{source_id}.json"
        source.write_text(f'{{"source":"{source_id}"}}', encoding="utf-8")
        sources[source_id] = source
    return sources


def _source_capture_files(tmp_path: Path) -> tuple[Path, Path]:
    rank_timings = [
        [1.0, 5.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 3.0, 4.0],
        [4.0, 2.0, 5.0],
    ]
    timings = [4.0, 5.0, 6.0]
    stdout_payload = {
        "schema": qualification_physical.SOURCE_CAPTURE_STDOUT_SCHEMA,
        "rank": 0,
        "world_size": 4,
        "tokens": 2,
        "layers": 4,
        "hidden": 8192,
        "gemm_m_rank0": 12,
        "gemm_n": 8192,
        "dtype": "bf16",
        "msg_sizes_bytes": [65536, 131072],
        "inject_skew": 0.25,
        "distributed_timeout_seconds": 300,
        "execution_semantics": "async-all-reduce-then-gemm-then-explicit-wait",
        "warmup_programs": 5,
        "measurement_iterations": 3,
        "profile_warmup_programs": 1,
        "profiled_programs": 1,
        "rank_timings_us": rank_timings,
        "timings_us": timings,
        "timing_semantics": qualification_physical.SOURCE_TIMING_SEMANTICS,
        "metrics": {"count": 3, "median_us": 5.0, "iqr_us": 2.0},
        "profiles": [],
    }
    stdout_path = tmp_path / "capture.stdout.json"
    stdout_path.write_text(
        json.dumps(stdout_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    stdout_bytes = stdout_path.read_bytes()
    evidence_payload = {
        "schema": qualification_physical.SOURCE_CAPTURE_EVIDENCE_SCHEMA,
        "diagnostic_id": "source-diagnostic",
        "attempt_id": "a-000001",
        "scheduler": {"job_id": "177966", "node": "toranj1", "partition": "cuda-A100"},
        "input_bindings": {},
        "capture": {
            "execution_semantics": "async-all-reduce-then-gemm-then-explicit-wait",
            "measurement_iterations": 3,
            "metrics": {"count": 3, "median_us": 5.0, "iqr_us": 2.0},
            "timing_range_us": {"min": 4.0, "max": 6.0},
            "timing_semantics": qualification_physical.SOURCE_TIMING_SEMANTICS,
        },
        "import": {},
        "artifacts": {
            "capture.stdout.json": {
                "sha256": hashlib.sha256(stdout_bytes).hexdigest(),
                "size_bytes": len(stdout_bytes),
            }
        },
        "claims": {},
    }
    evidence_path = tmp_path / "capture-evidence.json"
    evidence_path.write_text(json.dumps(evidence_payload), encoding="utf-8")
    return evidence_path, stdout_path


def test_qualification_inputs_are_rank_local_canonical_and_non_overwriting(tmp_path: Path) -> None:
    sources = _qualification_sources(tmp_path)
    request, materialization = qualification_physical.stage_qualification_inputs(
        sources,
        rank=2,
        workspace=tmp_path,
    )

    assert request == tmp_path / "qualification-input-rank-00002" / "request"
    assert materialization == tmp_path / "qualification-input-rank-00002" / "materialization"
    assert sorted(path.name for path in request.iterdir()) == [
        "canary.json",
        "fidelity.json",
        "qualification-policy.json",
        "qualification-request.json",
        "source.trace.json",
    ]
    assert sorted(path.name for path in materialization.iterdir()) == [
        "materialization.json",
        "replay-program.json",
    ]
    assert all(path.stat().st_mode & 0o222 == 0 for path in (*request.iterdir(), *materialization.iterdir()))

    with pytest.raises(SystemExit, match="cannot create rank-local"):
        qualification_physical.stage_qualification_inputs(sources, rank=2, workspace=tmp_path)


def test_qualification_staging_preserves_legacy_v1_inventory(tmp_path: Path) -> None:
    sources = _qualification_sources(tmp_path)
    sources.pop("qualification_policy")

    request, _ = qualification_physical.stage_qualification_inputs(
        sources,
        rank=3,
        workspace=tmp_path,
    )

    assert sorted(path.name for path in request.iterdir()) == [
        "canary.json",
        "fidelity.json",
        "qualification-request.json",
        "source.trace.json",
    ]


def test_qualification_staging_refuses_symlinked_and_empty_sources(tmp_path: Path) -> None:
    sources = _qualification_sources(tmp_path)
    target = sources["canary"]
    symlink = tmp_path / "canary-link.json"
    symlink.symlink_to(target)
    sources["canary"] = symlink
    with pytest.raises(SystemExit, match="real regular file"):
        qualification_physical.stage_qualification_inputs(sources, rank=0, workspace=tmp_path)

    empty_sources = _qualification_sources(tmp_path / "empty-sources")
    empty_sources["fidelity"].write_bytes(b"")
    with pytest.raises(SystemExit, match="is empty"):
        qualification_physical.stage_qualification_inputs(
            empty_sources,
            rank=1,
            workspace=tmp_path,
        )


def test_qualification_source_observation_is_hash_bound_and_recomputed(tmp_path: Path) -> None:
    evidence_path, stdout_path = _source_capture_files(tmp_path)
    observed = qualification_physical.load_source_capture_observation(
        evidence_path,
        stdout_path,
        world_size=4,
        iterations=3,
    )

    assert observed["timings_us"] == [4.0, 5.0, 6.0]
    assert observed["metrics"] == {
        "count": 3,
        "median_us": 5.0,
        "iqr_us": 2.0,
        "min_us": 4.0,
        "max_us": 6.0,
    }
    stdout_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="bytes disagree"):
        qualification_physical.load_source_capture_observation(
            evidence_path,
            stdout_path,
            world_size=4,
            iterations=3,
        )


def test_producers_emit_distinct_honest_raw_contracts() -> None:
    micro = microbench_tp8._result_payload(
        rank=0,
        world_size=4,
        dtype="bf16",
        message_sizes=[65536],
        timings_us=[10.0, 20.0, 30.0],
    )
    assert microbench_tp8.MICRO_STDOUT_SCHEMA == MICRO_STDOUT_SCHEMA
    assert set(micro) == {
        "schema",
        "rank",
        "world_size",
        "dtype",
        "msg_sizes_bytes",
        "timings_us",
        "metrics",
    }

    workload = workload_tp8._result_payload(
        rank=0,
        world_size=4,
        tokens=3,
        layers=32,
        hidden=8192,
        gemm_m=256,
        gemm_n=8192,
        dtype="bf16",
        message_sizes=[65536],
        inject_skew=0.0,
        timings_us=[10.0, 20.0, 30.0],
    )
    assert workload_tp8.WORKLOAD_STDOUT_SCHEMA == FULL_STDOUT_SCHEMA
    assert set(workload) == {
        "schema",
        "rank",
        "world_size",
        "tokens",
        "layers",
        "hidden",
        "gemm_m_rank0",
        "gemm_n",
        "dtype",
        "msg_sizes_bytes",
        "inject_skew",
        "timings_us",
        "metrics",
    }

    overlap = overlap_replay._result_payload(
        rank=0,
        world_size=4,
        timings_us=[10.0, 20.0, 30.0],
    )
    assert overlap_replay.OVERLAP_STDOUT_SCHEMA == OVERLAP_STDOUT_SCHEMA
    assert set(overlap) == {"schema", "rank", "world_size", "timings_us", "metrics"}
    assert len({micro["schema"], workload["schema"], overlap["schema"]}) == 3


def test_overlap_capture_issues_async_communication_before_compute_and_wait() -> None:
    calls: list[object] = []

    class Work:
        def wait(self) -> None:
            calls.append("wait")

    class Dist:
        class ReduceOp:
            SUM = "sum"

        @staticmethod
        def all_reduce(buffer: object, *, op: object, async_op: bool) -> Work:
            calls.append(("all_reduce", buffer, op, async_op))
            return Work()

    class Torch:
        @staticmethod
        def matmul(activation: object, weight: object) -> None:
            calls.append(("matmul", activation, weight))

    workload_overlap_capture._run_overlap_layer(
        Torch(),
        Dist(),
        "activation",
        "weight",
        "buffer",
    )

    assert calls == [
        ("all_reduce", "buffer", "sum", True),
        ("matmul", "activation", "weight"),
        "wait",
    ]


def test_overlap_capture_profile_directory_is_new_and_non_overwriting(tmp_path: Path) -> None:
    directory = tmp_path / "profiles"
    output = workload_overlap_capture._profile_path(directory, rank=2)
    assert output == directory / "rank-00002.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="already exists"):
        workload_overlap_capture._profile_path(directory, rank=2)

    symlink = tmp_path / "profile-link"
    symlink.symlink_to(directory, target_is_directory=True)
    with pytest.raises(SystemExit, match="must not be a symlink"):
        workload_overlap_capture._profile_path(symlink, rank=0)


def test_overlap_capture_payload_binds_profiles_and_unprofiled_program_timing() -> None:
    profiles = [
        {
            "rank": rank,
            "filename": f"rank-{rank:05d}.json",
            "sha256": f"{rank + 1:064x}",
            "size_bytes": 100 + rank,
        }
        for rank in range(2)
    ]
    payload = workload_overlap_capture._result_payload(
        world_size=2,
        tokens=2,
        layers=3,
        hidden=8,
        gemm_m_rank0=4,
        gemm_n=8,
        dtype="bf16",
        message_sizes=[1024],
        inject_skew=0.0,
        warmup_programs=1,
        measurement_iterations=2,
        profile_warmup_programs=1,
        distributed_timeout_seconds=45,
        profiles=profiles,
        rank_timings=[[10.0, 30.0], [20.0, 25.0]],
    )

    assert payload["schema"] == workload_overlap_capture.OVERLAP_CAPTURE_STDOUT_SCHEMA
    assert payload["execution_semantics"] == "async-all-reduce-then-gemm-then-explicit-wait"
    assert payload["timing_semantics"] == "maximum-rank-unprofiled-whole-program-duration"
    assert payload["warmup_programs"] == 1
    assert payload["measurement_iterations"] == 2
    assert payload["profile_warmup_programs"] == 1
    assert payload["profiled_programs"] == 1
    assert payload["profiles"] == profiles
    assert payload["rank_timings_us"] == [[10.0, 30.0], [20.0, 25.0]]
    assert payload["timings_us"] == [20.0, 30.0]
    assert payload["metrics"] == {"median_us": 25.0, "iqr_us": 10.0, "count": 2}


def test_overlap_capture_requires_bounded_positive_measurement_iterations() -> None:
    args = workload_overlap_capture.build_parser().parse_args(["--profile-directory", "profiles"])
    args.measurement_iterations = 0
    with pytest.raises(SystemExit, match="measurement-iterations"):
        workload_overlap_capture._positive_arguments(args)

    args.measurement_iterations = 1001
    with pytest.raises(SystemExit, match="measurement-iterations"):
        workload_overlap_capture._positive_arguments(args)

    args.measurement_iterations = 1
    args.profile_warmup_programs = 0
    with pytest.raises(SystemExit, match="profile-warmup-programs"):
        workload_overlap_capture._positive_arguments(args)


def test_overlap_preflight_requires_dense_world_and_complete_waits(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.json"
    _write_trace(valid_path, _trace(ranks=[0, 1, 2, 3]))
    entries, audit = overlap_replay._prepare_replay(
        str(valid_path),
        world_size=4,
        iterations=1,
        warmup=0,
    )
    assert len(entries) == 3
    assert audit == {"process_groups": 1, "collectives": 1, "waits": 1}

    shifted_path = tmp_path / "shifted.json"
    _write_trace(shifted_path, _trace(ranks=[1, 2, 3, 4]))
    with pytest.raises(PhysicalResultError, match="full world ranks"):
        overlap_replay._prepare_replay(
            str(shifted_path),
            world_size=4,
            iterations=1,
            warmup=0,
        )

    permuted_path = tmp_path / "permuted.json"
    _write_trace(permuted_path, _trace(ranks=[0, 2, 1, 3]))
    with pytest.raises(PhysicalResultError, match="full world ranks"):
        overlap_replay._prepare_replay(
            str(permuted_path),
            world_size=4,
            iterations=1,
            warmup=0,
        )

    blocking_path = tmp_path / "blocking.json"
    _write_trace(blocking_path, _trace(ranks=[0, 1, 2, 3], explicit_wait=False))
    with pytest.raises(PhysicalResultError, match="exactly one explicit wait"):
        overlap_replay._prepare_replay(
            str(blocking_path),
            world_size=4,
            iterations=1,
            warmup=0,
        )


def test_invalid_trace_fails_before_any_torch_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "shifted.json"
    _write_trace(trace_path, _trace(ranks=[1, 2, 3, 4]))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch was imported before trace validation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = argparse.Namespace(
        trace_path=str(trace_path),
        iters=1,
        warmup=0,
        device="cpu",
        backend="gloo",
    )
    with pytest.raises(PhysicalResultError, match="full world ranks"):
        overlap_replay.run(args)


def test_trace_loader_is_strict_and_request_ids_are_single_use(tmp_path: Path) -> None:
    duplicate_key_path = tmp_path / "duplicate-key.json"
    duplicate_key_path.write_text(
        '[{"comms":"init","pg_id":0,"pg_id":1,"global_ranks":[0,1,2,3]}]',
        encoding="utf-8",
    )
    with pytest.raises(PhysicalResultError, match="duplicate JSON object key"):
        load_validated_param_trace(str(duplicate_key_path), world_size=4)

    reused = _trace(ranks=[0, 1, 2, 3])
    reused.extend(
        [
            {
                "comms": "all_reduce",
                "pg_id": 0,
                "req": 1,
                "in_msg_size": 16,
                "out_msg_size": 16,
                "dtype": "bfloat16",
            },
            {"comms": "wait", "req": 1},
        ]
    )
    reused_path = tmp_path / "reused.json"
    _write_trace(reused_path, reused)
    with pytest.raises(PhysicalResultError, match="duplicate request id"):
        load_validated_param_trace(str(reused_path), world_size=4)


def test_trace_loader_bounds_bytes_items_entries_and_depth(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.json"
    _write_trace(valid_path, _trace(ranks=[0, 1, 2, 3]))
    with pytest.raises(PhysicalResultError, match="max_input_bytes=8"):
        load_validated_param_trace(
            str(valid_path),
            world_size=4,
            limits=ParamTraceLimits(max_input_bytes=8),
        )
    with pytest.raises(PhysicalResultError, match="max_param_entries=2"):
        load_validated_param_trace(
            str(valid_path),
            world_size=4,
            limits=ParamTraceLimits(max_param_entries=2),
        )

    too_many_items = tmp_path / "items.json"
    too_many_items.write_text('[{"a": 1, "b": 2}]', encoding="utf-8")
    with pytest.raises(PhysicalResultError, match="max_json_items=2"):
        load_validated_param_trace(
            str(too_many_items),
            world_size=4,
            limits=ParamTraceLimits(max_json_items=2),
        )

    too_deep = tmp_path / "deep.json"
    too_deep.write_text("[[[]]]", encoding="utf-8")
    with pytest.raises(PhysicalResultError, match="max_json_depth=2"):
        load_validated_param_trace(
            str(too_deep),
            world_size=4,
            limits=ParamTraceLimits(max_json_depth=2),
        )


def test_trace_loader_normalizes_decoder_recursion_and_reuses_decoded_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_path = tmp_path / "valid.json"
    raw = _trace(ranks=[0, 1, 2, 3])
    _write_trace(valid_path, raw)

    def recursive_decoder(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError("decoder recursion")

    monkeypatch.setattr(physical_results.json, "loads", recursive_decoder)
    with pytest.raises(PhysicalResultError, match="cannot decode PARAM trace: decoder recursion"):
        load_validated_param_trace(str(valid_path), world_size=4)

    def decoded_without_copy(path: str, *, limits: ParamTraceLimits) -> Any:
        return raw

    monkeypatch.setattr(physical_results, "_load_bounded_param_json", decoded_without_copy)
    entries, _ = load_validated_param_trace(str(valid_path), world_size=4)
    assert entries is raw
