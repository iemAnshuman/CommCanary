from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rostam import microbench_tp8, overlap_replay, workload_tp8
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
