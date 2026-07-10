import math
import unittest

from commcanary.behavior_config import (
    _behavioral_replay_args,
    _normalize_behavior_configurations,
)
from commcanary.compiler import (
    compile_trace,
    synthesize_behavioral_canary,
    verify_canary_behavior,
)
from commcanary.reduce import ddmin_ranking_reduction
from commcanary.replay import replay_canary
from commcanary.schema import TRACE_FORMAT, SchemaError


def _trace():
    return {
        "format": TRACE_FORMAT,
        "workload": {"name": "behavior-config-contract"},
        "events": [
            {
                "id": f"event-{index}",
                "op": "all_reduce",
                "bytes": 1024,
                "ranks": [0, 1],
                "start_us": float(index * 10),
                "rank_arrival_us": {"0": 0.0, "1": float(index)},
            }
            for index in range(2)
        ],
    }


def _two_configs(first=None, second=None):
    left = {"name": "left"}
    right = {"name": "right"}
    left.update(first or {})
    right.update(second or {})
    return [left, right]


class BehaviorConfigurationTests(unittest.TestCase):
    def test_none_uses_detached_explicit_defaults(self):
        first = _normalize_behavior_configurations(None)
        second = _normalize_behavior_configurations(None)

        self.assertEqual(
            [config["name"] for config in first],
            ["baseline", "low_latency", "high_bandwidth", "overlap_friendly", "congested"],
        )
        expected_keys = {
            "name",
            "bandwidth_gbps",
            "latency_floor_us",
            "compute_pressure",
            "overlap_efficiency",
            "iterations",
            "seed",
            "max_replay_events",
        }
        self.assertTrue(all(set(config) == expected_keys for config in first))
        with self.assertRaises(TypeError):
            first[0]["name"] = "mutated"
        with self.assertRaises(TypeError):
            first[0]["seed"] = 99
        self.assertEqual(second[0]["name"], "baseline")
        self.assertEqual(second[0]["seed"], 7)

    def test_configuration_collection_must_be_a_bounded_mapping_sequence(self):
        with self.assertRaisesRegex(SchemaError, "at least two"):
            _normalize_behavior_configurations([])
        with self.assertRaisesRegex(SchemaError, "at least two"):
            _normalize_behavior_configurations([{"name": "only"}])
        with self.assertRaisesRegex(SchemaError, "sequence of mappings"):
            _normalize_behavior_configurations({"name": "not-a-sequence"})
        with self.assertRaisesRegex(SchemaError, "sequence of mappings"):
            _normalize_behavior_configurations(iter(_two_configs()))
        with self.assertRaisesRegex(SchemaError, "must be a mapping"):
            _normalize_behavior_configurations([{"name": "left"}, "right"])

        maximum = _normalize_behavior_configurations(
            [{"name": f"config-{index}"} for index in range(32)]
        )
        self.assertEqual(len(maximum), 32)
        with self.assertRaisesRegex(SchemaError, "at most 32"):
            _normalize_behavior_configurations(
                [{"name": f"config-{index}"} for index in range(33)]
            )

    def test_names_are_required_nonempty_strings_and_unique(self):
        invalid = (
            [{}, {"name": "right"}],
            [{"name": ""}, {"name": "right"}],
            [{"name": "   "}, {"name": "right"}],
            [{"name": 1}, {"name": "right"}],
        )
        for configurations in invalid:
            with self.subTest(configurations=configurations):
                with self.assertRaisesRegex(SchemaError, "name must be a non-empty string"):
                    _normalize_behavior_configurations(configurations)

        with self.assertRaisesRegex(SchemaError, "unique configuration names"):
            _normalize_behavior_configurations(
                [{"name": "duplicate"}, {"name": "duplicate"}]
            )
        with self.assertRaisesRegex(SchemaError, "unique configuration names"):
            _normalize_behavior_configurations(
                [{"name": " duplicate "}, {"name": "duplicate"}]
            )

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(SchemaError, "unknown keys.*'mystery'"):
            _normalize_behavior_configurations(
                _two_configs({"mystery": 1})
            )

    def test_replay_fields_are_normalized_and_detached(self):
        raw = _two_configs(
            {
                "name": " left ",
                "bandwidth_gbps": "10.5",
                "latency_floor_us": "0",
                "compute_pressure": "1.25",
                "overlap_efficiency": "1",
                "iterations": "2",
                "seed": "3",
                "max_replay_events": "100",
            }
        )
        normalized = _normalize_behavior_configurations(tuple(raw))
        left = normalized[0]
        self.assertEqual(
            dict(left),
            {
                "name": "left",
                "bandwidth_gbps": 10.5,
                "latency_floor_us": 0.0,
                "compute_pressure": 1.25,
                "overlap_efficiency": 1.0,
                "iterations": 2,
                "seed": 3,
                "max_replay_events": 100,
            },
        )
        raw[0]["seed"] = 77
        self.assertEqual(left["seed"], 3)

        replay_args = _behavioral_replay_args(left)
        self.assertNotIn("name", replay_args)
        replay_args["seed"] = 88
        self.assertEqual(left["seed"], 3)

        report = replay_canary(
            compile_trace(_trace()),
            backend_label=left["name"],
            **_behavioral_replay_args(left),
        )
        self.assertEqual(report["backend"]["seed"], 3)
        self.assertEqual(report["replay_protocol"]["iterations"], 2)

    def test_invalid_replay_field_domains_fail_normalization(self):
        invalid_values = (
            ("bandwidth_gbps", 0),
            ("bandwidth_gbps", -1),
            ("bandwidth_gbps", math.nan),
            ("latency_floor_us", -1),
            ("compute_pressure", -1),
            ("overlap_efficiency", -0.01),
            ("overlap_efficiency", 1.01),
            ("iterations", 0),
            ("iterations", 1.5),
            ("seed", 1.5),
            ("seed", True),
            ("max_replay_events", 0),
            ("max_replay_events", 1.5),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(SchemaError, field):
                    _normalize_behavior_configurations(_two_configs({field: value}))

        normalized = _normalize_behavior_configurations(_two_configs({"seed": -1}))
        self.assertEqual(normalized[0]["seed"], -1)

    def test_all_behavioral_entry_points_share_empty_list_semantics(self):
        trace = _trace()
        canary = compile_trace(trace)

        with self.assertRaisesRegex(SchemaError, "at least two"):
            verify_canary_behavior(trace, canary, configurations=[])
        with self.assertRaisesRegex(SchemaError, "at least two"):
            synthesize_behavioral_canary(
                trace,
                min_timing_sample_limit=2,
                max_timing_sample_limit=2,
                behavior_configurations=[],
            )
        with self.assertRaisesRegex(SchemaError, "at least two"):
            ddmin_ranking_reduction(trace, configurations=[])


if __name__ == "__main__":
    unittest.main()
