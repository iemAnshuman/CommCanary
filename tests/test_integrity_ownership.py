from __future__ import annotations

import copy
import unittest

from commcanary.baselines import isolated_collective_baseline_trace
from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace, verify_canary_behavior, verify_canary_fidelity
from commcanary.interop import canary_to_param_comms_trace, kineto_trace_to_commcanary_trace
from commcanary.reduce import ddmin_ranking_reduction
from commcanary.replay import replay_canary, verify_report_against_canary
from commcanary.schema import (
    ARTIFACT_PROVENANCE_ALGORITHM,
    ASSURANCE_STATES,
    CANARY_INTEGRITY_PROFILE,
    TRACE_FORMAT,
    SchemaError,
    canary_artifact_provenance_sha256,
    canary_calibration_sha256,
    canary_execution_sha256,
    canary_scheduler_execution_sha256,
    validate_canary,
)


def nested_trace() -> dict:
    return {
        "format": TRACE_FORMAT,
        "workload": {
            "name": "ownership",
            "metadata": {"tags": ["decode", "tp"]},
        },
        "system": {"runtime": {"name": "test", "versions": [1]}},
        "events": [
            {
                "id": {"parts": ["event", index]},
                "phase": "decode",
                "op": "all_reduce",
                "bytes": 1024,
                "ranks": [0, 1],
                "group": "tp",
                "start_us": float(index * 10),
                "rank_arrival_us": {"0": 0.0, "1": float(index + 1)},
                "compute_overlap_us": 1.0,
            }
            for index in range(3)
        ],
    }


def motif_trace() -> dict:
    trace = nested_trace()
    trace["events"] = [
        {
            "id": {"parts": ["motif-event", index]},
            "phase": "decode",
            "op": "all_reduce" if index % 2 == 0 else "all_gather",
            "bytes": 1024 if index % 2 == 0 else 2048,
            "ranks": [0, 1],
            "group": "tp",
            "gap_us": 10.0,
            "rank_arrival_us": {"0": 0.0, "1": 1.0},
            "compute_overlap_us": 1.0,
        }
        for index in range(6)
    ]
    return trace


def refresh_producer_hashes(canary: dict) -> None:
    """Model a producer that can rewrite every stored artifact commitment."""

    compiler = canary["compiler"]
    compiler["execution_semantic_sha256"] = canary_execution_sha256(canary)
    compiler["scheduler_execution_sha256"] = canary_scheduler_execution_sha256(canary)
    compiler["calibration_evaluation_sha256"] = canary_calibration_sha256(canary)
    compiler["artifact_provenance_sha256"] = canary_artifact_provenance_sha256(canary)


class IntegrityProfileTests(unittest.TestCase):
    def test_assurance_ladder_is_machine_readable_without_changing_status(self) -> None:
        trace = nested_trace()
        canary = compile_trace(trace)
        fidelity = verify_canary_fidelity(trace, canary)
        behavior = verify_canary_behavior(trace, canary)
        report = replay_canary(canary)
        report_verification = verify_report_against_canary(report, canary)

        self.assertEqual(
            ASSURANCE_STATES,
            (
                "structurally_valid",
                "internally_consistent",
                "source_corresponding",
                "model_recomputed",
                "behaviorally_verified",
            ),
        )
        self.assertEqual(fidelity["status"], "source_verified")
        self.assertEqual(fidelity["assurance_state"], "source_corresponding")
        self.assertEqual(behavior["status"], "behaviorally_verified")
        self.assertEqual(behavior["assurance_state"], "behaviorally_verified")
        self.assertEqual(report_verification["status"], "model_recomputed")
        self.assertEqual(report_verification["assurance_state"], "model_recomputed")

        forged_report = copy.deepcopy(report)
        forged_report["workload"]["metadata"]["tags"].append("forged")
        failed_report_verification = verify_report_against_canary(forged_report, canary)
        self.assertEqual(failed_report_verification["status"], "failed")
        self.assertEqual(failed_report_verification["assurance_state"], "structurally_valid")

    def test_new_canaries_require_and_recompute_integrity_profile(self) -> None:
        canary = compile_trace(nested_trace())
        compiler = canary["compiler"]
        self.assertEqual(compiler["integrity_profile"], CANARY_INTEGRITY_PROFILE)
        self.assertEqual(
            compiler["artifact_provenance_algorithm"],
            ARTIFACT_PROVENANCE_ALGORITHM,
        )

        tampered = copy.deepcopy(canary)
        tampered["workload"]["metadata"]["tags"].append("tampered")
        with self.assertRaisesRegex(SchemaError, "artifact_provenance_sha256"):
            validate_canary(tampered)

    def test_each_profile_commitment_is_mandatory(self) -> None:
        canary = compile_trace(nested_trace())
        required = (
            "source_trace_sha256",
            "source_normalized_sha256",
            "execution_semantic_sha256",
            "scheduler_execution_sha256",
            "calibration_evaluation_sha256",
            "artifact_provenance_sha256",
        )
        for field in required:
            with self.subTest(field=field):
                missing = copy.deepcopy(canary)
                del missing["compiler"][field]
                with self.assertRaisesRegex(SchemaError, field):
                    validate_canary(missing)

        missing_source_format = copy.deepcopy(canary)
        del missing_source_format["source_format"]
        with self.assertRaisesRegex(SchemaError, "source_format"):
            validate_canary(missing_source_format)

    def test_every_recursive_source_block_requires_id_bounds_and_digest(self) -> None:
        canary = compile_trace(motif_trace())
        self.assertEqual(canary["events"][0]["program"], "sequence_motif")
        locations = (
            ("wrapper", lambda value: value["events"][0]["source"]),
            ("child", lambda value: value["events"][0]["events"][0]["source"]),
        )
        for location, select_source in locations:
            for field in ("first_id", "last_id", "digest"):
                with self.subTest(location=location, field=field):
                    missing = copy.deepcopy(canary)
                    del select_source(missing)[field]
                    with self.assertRaisesRegex(SchemaError, field):
                        validate_canary(missing)

    def test_tamper_matrix_rejects_each_protected_field_family(self) -> None:
        trace = nested_trace()
        for index, event in enumerate(trace["events"]):
            event["observed_exposed_us"] = 20.0 + index
        canary = compile_trace(trace)
        mutations = (
            ("source_format", lambda value: value.__setitem__("source_format", "forged.trace.v1")),
            ("workload", lambda value: value["workload"]["metadata"]["tags"].append("forged")),
            ("system", lambda value: value["system"]["runtime"]["versions"].append(2)),
            ("execution", lambda value: value["events"][0].__setitem__("bytes", 2048)),
            (
                "calibration",
                lambda value: value["events"][0]["timing_samples"][0].__setitem__("observed_exposed_us", 999.0),
            ),
            ("source", lambda value: value["events"][0]["source"].__setitem__("first_id", "forged")),
            ("compiler", lambda value: value["compiler"]["fidelity"].__setitem__("mode", "forged")),
            ("unknown_top_level", lambda value: value.__setitem__("forged", True)),
        )
        for family, mutate in mutations:
            with self.subTest(family=family):
                tampered = copy.deepcopy(canary)
                mutate(tampered)
                with self.assertRaises(SchemaError):
                    validate_canary(tampered)

        volatile = copy.deepcopy(canary)
        volatile["created_at"] = "different but deliberately unhashed"
        validate_canary(volatile)

    def test_source_verifier_defeats_producer_rehashing_source_metadata(self) -> None:
        traces_and_mutations = (
            (
                "source_format",
                nested_trace(),
                lambda value: value.__setitem__("source_format", "forged.trace.v1"),
                "source_format",
            ),
            (
                "flat_first_id",
                nested_trace(),
                lambda value: value["events"][0]["source"].__setitem__("first_id", "forged"),
                "source_commitments",
            ),
            (
                "flat_last_id",
                nested_trace(),
                lambda value: value["events"][0]["source"].__setitem__("last_id", "forged"),
                "source_commitments",
            ),
            (
                "flat_digest",
                nested_trace(),
                lambda value: value["events"][0]["source"].__setitem__("digest", "0" * 64),
                "source_commitments",
            ),
            (
                "motif_wrapper_first_id",
                motif_trace(),
                lambda value: value["events"][0]["source"].__setitem__("first_id", "forged"),
                "source_commitments",
            ),
            (
                "motif_wrapper_digest",
                motif_trace(),
                lambda value: value["events"][0]["source"].__setitem__("digest", "0" * 64),
                "source_commitments",
            ),
            (
                "motif_wrapper_last_id",
                motif_trace(),
                lambda value: value["events"][0]["source"].__setitem__("last_id", "forged"),
                "source_commitments",
            ),
            (
                "motif_child_first_id",
                motif_trace(),
                lambda value: value["events"][0]["events"][0]["source"].__setitem__("first_id", "forged"),
                "source_commitments",
            ),
            (
                "motif_child_last_id",
                motif_trace(),
                lambda value: value["events"][0]["events"][0]["source"].__setitem__("last_id", "forged"),
                "source_commitments",
            ),
            (
                "motif_child_digest",
                motif_trace(),
                lambda value: value["events"][0]["events"][0]["source"].__setitem__("digest", "0" * 64),
                "source_commitments",
            ),
            (
                "source_digest_aliases",
                nested_trace(),
                lambda value: value["compiler"].update(
                    {"source_trace_sha256": "0" * 64, "source_normalized_sha256": "0" * 64}
                ),
                "source_trace_sha256",
            ),
        )
        for name, trace, mutate, failed_check in traces_and_mutations:
            with self.subTest(name=name):
                canary = compile_trace(trace)
                mutate(canary)
                refresh_producer_hashes(canary)
                validate_canary(canary)

                verification = verify_canary_fidelity(trace, canary)
                self.assertEqual(verification["status"], "failed")
                self.assertEqual(verification["assurance_state"], "internally_consistent")
                checks = {check["name"]: check for check in verification["checks"]}
                self.assertEqual(checks[failed_check]["status"], "fail")

    def test_legacy_validation_is_explicit_and_never_the_default(self) -> None:
        legacy = compile_trace(nested_trace())
        legacy["compiler"].pop("integrity_profile")
        legacy["compiler"].pop("artifact_provenance_algorithm")
        legacy["compiler"].pop("artifact_provenance_sha256")

        with self.assertRaisesRegex(SchemaError, "integrity profile"):
            validate_canary(legacy)
        validate_canary(legacy, allow_legacy_unverified=True)

    def test_source_verification_checks_workload_and_system_correspondence(self) -> None:
        trace = nested_trace()
        canary = compile_trace(trace)
        changed_source = copy.deepcopy(trace)
        changed_source["workload"]["metadata"]["tags"].append("different")
        changed_source["system"]["runtime"]["versions"].append(2)

        verification = verify_canary_fidelity(changed_source, canary)
        self.assertEqual(verification["status"], "failed")
        failed = {check["name"] for check in verification["checks"] if check["status"] == "fail"}
        self.assertIn("workload", failed)
        self.assertIn("system", failed)


class DetachedOutputTests(unittest.TestCase):
    def test_compile_output_is_detached_from_trace(self) -> None:
        trace = nested_trace()
        canary = compile_trace(trace)

        trace["workload"]["metadata"]["tags"].append("input-mutation")
        trace["system"]["runtime"]["versions"].append(2)
        trace["events"][0]["id"]["parts"].append("changed")
        self.assertEqual(canary["workload"]["metadata"]["tags"], ["decode", "tp"])
        self.assertEqual(canary["system"]["runtime"]["versions"], [1])
        self.assertEqual(canary["events"][0]["source"]["first_id"]["parts"], ["event", 0])

        canary["workload"]["metadata"]["tags"].append("output-mutation")
        self.assertEqual(trace["workload"]["metadata"]["tags"], ["decode", "tp", "input-mutation"])

    def test_replay_and_compare_outputs_are_detached(self) -> None:
        canary = compile_trace(nested_trace())
        baseline = replay_canary(canary)
        candidate = replay_canary(canary, latency_floor_us=12.0)
        comparison = compare_reports(baseline, candidate)

        canary["workload"]["metadata"]["tags"].append("canary-mutation")
        canary["compiler"]["fidelity"]["mode"] = "changed"
        self.assertEqual(baseline["workload"]["metadata"]["tags"], ["decode", "tp"])
        self.assertNotEqual(baseline["canary_summary"]["fidelity"]["mode"], "changed")

        baseline["metrics"]["median_us"] = 999.0
        baseline["backend"]["label"] = "mutated"
        self.assertNotEqual(comparison["baseline"]["metrics"]["median_us"], 999.0)
        self.assertNotEqual(comparison["baseline"]["backend"]["label"], "mutated")

        comparison["candidate"]["metrics"]["median_us"] = -1.0
        self.assertNotEqual(candidate["metrics"]["median_us"], -1.0)

    def test_baseline_and_reduction_outputs_are_detached(self) -> None:
        trace = nested_trace()
        baseline = isolated_collective_baseline_trace(trace)
        reduced = ddmin_ranking_reduction(trace, max_oracle_calls=8)

        trace["workload"]["metadata"]["tags"].append("input-mutation")
        trace["system"]["runtime"]["versions"].append(2)
        self.assertNotIn("input-mutation", baseline["workload"]["metadata"]["tags"])
        self.assertNotIn("input-mutation", reduced["workload"]["metadata"]["tags"])
        self.assertEqual(baseline["system"]["runtime"]["versions"], [1])
        self.assertEqual(reduced["system"]["runtime"]["versions"], [1])

    def test_kineto_import_output_is_bidirectionally_detached(self) -> None:
        kineto = {
            "distributedInfo": {
                "backend": {"name": "nccl", "features": ["cuda"]},
                "rank": 0,
                "world_size": 2,
                "nccl_version": {"components": [2, 27, 5]},
            },
            "traceEvents": [
                {
                    "cat": "cpu_op",
                    "name": "record_param_comms",
                    "ts": 100.0,
                    "dur": 5.0,
                    "args": {
                        "Collective name": "allreduce",
                        "In msg nelems": 256,
                        "Out msg nelems": 256,
                        "dtype": "Float",
                        "Process Group Name": "tp",
                        "Process Group Ranks": "[0, 1]",
                    },
                }
            ],
        }
        imported = kineto_trace_to_commcanary_trace(kineto)

        kineto["distributedInfo"]["backend"]["features"].append("input-mutation")
        kineto["distributedInfo"]["nccl_version"]["components"].append(6)
        self.assertEqual(imported["system"]["kineto_backend"]["features"], ["cuda"])
        self.assertEqual(imported["system"]["kineto_nccl_version"]["components"], [2, 27, 5])

        imported["system"]["kineto_backend"]["features"].append("output-mutation")
        imported["system"]["kineto_nccl_version"]["components"].append(7)
        self.assertEqual(kineto["distributedInfo"]["backend"]["features"], ["cuda", "input-mutation"])
        self.assertEqual(kineto["distributedInfo"]["nccl_version"]["components"], [2, 27, 5, 6])

    def test_param_export_outputs_are_detached_from_input_and_siblings(self) -> None:
        canary = compile_trace(nested_trace())
        first = canary_to_param_comms_trace(canary)
        sibling = canary_to_param_comms_trace(canary)

        canary["events"][0]["ranks"].append(2)
        self.assertEqual(first[0]["global_ranks"], [0, 1])
        self.assertEqual(first[1]["global_ranks"], [0, 1])

        first[0]["global_ranks"].append(3)
        first[1]["markers"].append("output-mutation")
        self.assertEqual(canary["events"][0]["ranks"], [0, 1, 2])
        self.assertEqual(sibling[0]["global_ranks"], [0, 1])
        self.assertNotIn("output-mutation", sibling[1]["markers"])

    def test_fidelity_verification_output_is_bidirectionally_detached(self) -> None:
        trace = nested_trace()
        canary = compile_trace(trace)
        verification = verify_canary_fidelity(trace, canary)
        checks = {check["name"]: check for check in verification["checks"]}
        workload_check = checks["workload"]
        system_check = checks["system"]

        trace["workload"]["metadata"]["tags"].append("trace-mutation")
        canary["system"]["runtime"]["versions"].append(2)
        self.assertEqual(workload_check["expected"]["metadata"]["tags"], ["decode", "tp"])
        self.assertEqual(system_check["actual"]["runtime"]["versions"], [1])

        workload_check["expected"]["metadata"]["tags"].append("output-mutation")
        system_check["actual"]["runtime"]["versions"].append(3)
        self.assertEqual(trace["workload"]["metadata"]["tags"], ["decode", "tp", "trace-mutation"])
        self.assertEqual(canary["system"]["runtime"]["versions"], [1, 2])

    def test_report_verification_output_is_bidirectionally_detached(self) -> None:
        canary = compile_trace(nested_trace())
        report = replay_canary(canary)
        report["workload"]["metadata"]["tags"].append("forged")
        verification = verify_report_against_canary(report, canary)
        workload_check = next(check for check in verification["checks"] if check["name"] == "workload")

        canary["workload"]["metadata"]["tags"].append("canary-mutation")
        report["workload"]["metadata"]["tags"].append("report-mutation")
        self.assertEqual(workload_check["expected"]["metadata"]["tags"], ["decode", "tp"])
        self.assertEqual(workload_check["actual"]["metadata"]["tags"], ["decode", "tp", "forged"])

        workload_check["expected"]["metadata"]["tags"].append("expected-mutation")
        workload_check["actual"]["metadata"]["tags"].append("actual-mutation")
        self.assertEqual(canary["workload"]["metadata"]["tags"], ["decode", "tp", "canary-mutation"])
        self.assertEqual(
            report["workload"]["metadata"]["tags"],
            ["decode", "tp", "forged", "report-mutation"],
        )


if __name__ == "__main__":
    unittest.main()
