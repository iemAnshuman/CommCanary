from __future__ import annotations

import copy
import unittest

from commcanary.baselines import isolated_collective_baseline_trace
from commcanary.compare import compare_reports
from commcanary.compiler import compile_trace, verify_canary_fidelity
from commcanary.reduce import ddmin_ranking_reduction
from commcanary.replay import replay_canary
from commcanary.schema import (
    ARTIFACT_PROVENANCE_ALGORITHM,
    CANARY_INTEGRITY_PROFILE,
    SchemaError,
    TRACE_FORMAT,
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


class IntegrityProfileTests(unittest.TestCase):
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
        failed = {
            check["name"]
            for check in verification["checks"]
            if check["status"] == "fail"
        }
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


if __name__ == "__main__":
    unittest.main()
