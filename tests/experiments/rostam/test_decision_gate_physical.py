from __future__ import annotations

from collections import Counter
from pathlib import Path

from commcanary.compiler import compile_trace
from commcanary.execution import preflight_qualification_execution
from commcanary.services import prepare_qualification_request
from commcanary.workflows import materialize_qualification
from experiments.rostam import decision_gate_physical
from tests.builders import qualification_policy, qualification_trace


class _TorchWithLocalVersion:
    __version__ = "2.4.1+cu121"


class _SelectedNcclLibrary:
    @staticmethod
    def ncclGetVersion(pointer) -> int:
        pointer._obj.value = 21903
        return 0


def _gate_inputs(tmp_path: Path):
    trace = qualification_trace()
    policy = qualification_policy()
    request_directory = tmp_path / "request"
    materialization_directory = tmp_path / "materialization"
    request = prepare_qualification_request(
        str(request_directory),
        trace,
        compile_trace(trace),
        policy,
    )
    materialization = materialize_qualification(
        str(request_directory),
        str(materialization_directory),
    )
    plan = preflight_qualification_execution(
        str(request_directory),
        str(materialization_directory),
        world_size=4,
        iterations=20,
        warmup=5,
    )
    return trace, policy, request, materialization, plan


def test_source_and_materialization_form_the_same_closed_gate_program(tmp_path: Path) -> None:
    trace, _policy, _request, _materialization, plan = _gate_inputs(tmp_path)

    source = decision_gate_physical.source_events(trace, world_size=4)
    materialized = decision_gate_physical.plan_events(plan)

    assert source == materialized
    assert len(source) == 6
    assert decision_gate_physical.stratified_indices(source) == (0,)


def test_stratified_gate_selects_one_event_per_collective_shape(tmp_path: Path) -> None:
    trace, _policy, _request, _materialization, _plan = _gate_inputs(tmp_path)
    for index, event in enumerate(trace["events"]):
        event["bytes"] = 65_536 if index % 2 == 0 else 131_072

    source = decision_gate_physical.source_events(trace, world_size=4)

    assert decision_gate_physical.stratified_indices(source) == (0, 1)


def test_representation_order_rotates_every_measured_iteration() -> None:
    orders = [decision_gate_physical.representation_order(index) for index in range(6)]

    assert len(set(orders)) == 6
    assert all(set(order) == set(decision_gate_physical.REPRESENTATION_IDS) for order in orders)
    assert [order[0] for order in orders] == list(decision_gate_physical.REPRESENTATION_IDS)


def test_replicated_schedule_balances_positions_carryover_and_repetition_start() -> None:
    positions = {representation: [0] * 6 for representation in decision_gate_physical.REPRESENTATION_IDS}
    predecessors = Counter()
    for iteration in range(24):
        order = decision_gate_physical.representation_order(
            iteration,
            configuration_repetition=3,
        )
        for position, representation in enumerate(order):
            positions[representation][position] += 1
        predecessors.update(zip(order, order[1:]))

    assert all(counts == [4, 4, 4, 4, 4, 4] for counts in positions.values())
    assert set(predecessors.values()) == {4}
    assert len(predecessors) == 30
    assert decision_gate_physical.representation_order(0, configuration_repetition=1)[0] == "exact_work"


def test_warmup_order_indices_rotate_before_measured_indices_restart() -> None:
    warmup = 5
    pass_indices = range(-warmup, 2)
    order_indices = [index if index >= 0 else index + warmup for index in pass_indices]

    assert order_indices == [0, 1, 2, 3, 4, 0, 1]


def test_runtime_torch_version_matches_the_normalized_cell_observation() -> None:
    assert decision_gate_physical._normalized_torch_version(_TorchWithLocalVersion()) == "2.4.1"


def test_runtime_nccl_version_queries_the_explicit_selected_library(
    tmp_path: Path,
    monkeypatch,
) -> None:
    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"test fixture")
    maps = tmp_path / "maps"
    maps.write_text(f"0000-1000 r-xp 0000 00:00 0 {library}\n", encoding="utf-8")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    monkeypatch.setattr(decision_gate_physical.ctypes, "CDLL", lambda path: _SelectedNcclLibrary())

    selected = decision_gate_physical._selected_nccl_library()

    assert selected == library.resolve()
    assert decision_gate_physical._runtime_nccl_version_code(selected, proc_maps_path=maps) == 21903


def test_result_payload_recomputes_max_rank_metrics_and_retains_raw_samples(tmp_path: Path) -> None:
    _trace, policy, request, materialization, plan = _gate_inputs(tmp_path)
    gathered = []
    for rank in range(4):
        gathered.append(
            {
                "rank": rank,
                "timings_us": {
                    representation: [float(10 + rank), float(20 + rank)]
                    for representation in decision_gate_physical.REPRESENTATION_IDS
                },
            }
        )

    payload = decision_gate_physical.result_payload(
        request=request,
        materialization_id=materialization["materialization_id"],
        program_sha256=plan.program_sha256,
        policy=policy,
        world_size=4,
        iterations=2,
        warmup=1,
        source_event_count=8,
        selected_indices=(0, 1),
        gathered=gathered,
        correctness_checks_per_rank=(2, 2, 2, 2),
        runtime={
            "torch_version": "2.4.1",
            "torch_cuda_version": "12.1",
            "runtime_nccl_version_code": 22005,
            "distributed_backend": "nccl",
        },
    )

    assert payload["representations"]["source"]["timings_us"] == [13.0, 23.0]
    assert payload["representations"]["source"]["metrics"] == {
        "count": 2,
        "median_us": 18.0,
        "iqr_us": 10.0,
        "min_us": 13.0,
        "max_us": 23.0,
    }
    assert payload["representations"]["stratified"]["executed_event_count"] == 2
    assert payload["representations"]["isolated"]["template_count"] == 2
    assert payload["correctness"]["total_check_count"] == 8
    assert payload["claims"]["physical_decision_fidelity"] == "not_analyzed"


def test_replicated_payload_declares_positive_control_and_block_schedule(tmp_path: Path) -> None:
    _trace, policy, request, materialization, plan = _gate_inputs(tmp_path)
    gathered = [
        {
            "rank": rank,
            "timings_us": {
                representation: [float(index + rank + 1) for index in range(24)]
                for representation in decision_gate_physical.REPRESENTATION_IDS
            },
        }
        for rank in range(4)
    ]

    payload = decision_gate_physical.result_payload(
        request=request,
        materialization_id=materialization["materialization_id"],
        program_sha256=plan.program_sha256,
        policy=policy,
        world_size=4,
        iterations=24,
        warmup=5,
        source_event_count=8,
        selected_indices=(0, 1),
        gathered=gathered,
        correctness_checks_per_rank=(2, 2, 2, 2),
        runtime={
            "torch_version": "2.4.1",
            "torch_cuda_version": "12.1",
            "runtime_nccl_version_code": 22005,
            "distributed_backend": "nccl",
        },
        configuration_repetition=2,
    )

    assert payload["schema"] == "commcanary.rostam.decision-gate.stdout.v2"
    assert payload["execution"]["configuration_repetition"] == 2
    assert payload["execution"]["representation_order_by_iteration"][0][0] == "stratified"
    assert payload["representations"]["exact_work"]["category"] == "positive_conformance_control"


def test_main_accepts_forwarded_bootstrap_arguments(monkeypatch) -> None:
    observed = {}

    def fake_run(args):
        observed["request"] = args.request_manifest
        observed["iterations"] = args.iterations
        return 7

    monkeypatch.setattr(decision_gate_physical, "run", fake_run)
    arguments = [
        "--request-manifest",
        "request.json",
        "--source-trace",
        "source.json",
        "--canary",
        "canary.json",
        "--fidelity",
        "fidelity.json",
        "--qualification-policy",
        "policy.json",
        "--materialization-manifest",
        "materialization.json",
        "--replay-program",
        "program.json",
        "--expected-request-id",
        "1" * 64,
        "--expected-materialization-id",
        "2" * 64,
        "--expected-program-sha256",
        "3" * 64,
        "--expected-policy-id",
        "4" * 64,
        "--iterations",
        "23",
    ]

    assert decision_gate_physical.main(arguments) == 7
    assert observed == {"request": Path("request.json"), "iterations": 23}
