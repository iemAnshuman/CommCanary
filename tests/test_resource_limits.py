from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from commcanary.capture import merge_trace_shards
from commcanary.compiler import compile_trace, synthesize_behavioral_canary
from commcanary.interop import canary_to_param_comms_trace, load_kineto_trace
from commcanary.reduce import ddmin_ranking_reduction
from commcanary.replay import replay_canary
from commcanary.resources import (
    MAX_CHECKED_COUNT,
    JsonResourceError,
    ResourceLimits,
    checked_add,
    checked_multiply,
    require_within,
    validate_json_mapping,
    validate_json_value,
)
from commcanary.schema import (
    SchemaError,
    iter_canary_logical_events,
    load_json,
    preflight_canary_expansion,
    validate_trace,
    write_json,
)


class BoundedJsonLoaderTests(unittest.TestCase):
    def _write(self, directory: str, payload: bytes, name: str = "input.json") -> str:
        path = Path(directory) / name
        path.write_bytes(payload)
        return str(path)

    def test_resource_limits_are_frozen_and_validate_configuration(self) -> None:
        limits = ResourceLimits()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            limits.max_input_bytes = 1  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "max_input_bytes"):
            ResourceLimits(max_input_bytes=0)
        with self.assertRaisesRegex(TypeError, "max_json_depth"):
            ResourceLimits(max_json_depth=True)
        with self.assertRaisesRegex(ValueError, "max_expanded_events"):
            ResourceLimits(max_expanded_events=0)
        with self.assertRaisesRegex(ValueError, "max_json_depth"):
            ResourceLimits(max_json_depth=0)
        with self.assertRaisesRegex(ValueError, "max_json_items"):
            ResourceLimits(max_json_items=-1)
        with self.assertRaisesRegex(ValueError, "max_json_string_bytes"):
            ResourceLimits(max_json_string_bytes=-1)
        with self.assertRaisesRegex(ValueError, "max_json_number_chars"):
            ResourceLimits(max_json_number_chars=0)
        with self.assertRaisesRegex(ValueError, "max_behavior_configurations must be at least 2"):
            ResourceLimits(max_behavior_configurations=1)

    def test_circular_references_are_rejected_without_recursion(self) -> None:
        circular_mapping: dict = {"payload": {}}
        circular_mapping["payload"]["loop"] = circular_mapping
        with self.assertRaisesRegex(JsonResourceError, "circular references"):
            validate_json_mapping(circular_mapping)
        circular_list: list = [[]]
        circular_list[0].append(circular_list)
        with self.assertRaisesRegex(JsonResourceError, "circular references"):
            validate_json_value(circular_list)

    def test_checked_work_arithmetic_rejects_overflow_before_expansion(self) -> None:
        self.assertEqual(checked_add(2, 3, label="events"), 5)
        self.assertEqual(checked_multiply(2, 3, label="events"), 6)
        self.assertEqual(require_within(6, 6, label="events"), 6)
        with self.assertRaisesRegex(JsonResourceError, "supported count range"):
            checked_add(MAX_CHECKED_COUNT, 1, label="events")
        with self.assertRaisesRegex(JsonResourceError, "supported count range"):
            checked_multiply(MAX_CHECKED_COUNT, 2, label="events")
        with self.assertRaisesRegex(JsonResourceError, "exceeds limit"):
            require_within(7, 6, label="events")

    def test_motif_and_timing_expansion_are_counted_before_iteration(self) -> None:
        child = {
            "repeat": 1,
            "source": {"count": 1},
            "timing_samples": [{"weight": 1}],
        }
        events = [
            {
                "program": "sequence_motif",
                "program_repeats": 3,
                "source": {"count": 6},
                "events": [dict(child), dict(child)],
            }
        ]
        exact = ResourceLimits(
            max_stored_events=3,
            max_stored_timing_records=2,
            max_expanded_events=6,
            max_expanded_timing_records=6,
        )
        counts = preflight_canary_expansion(events, limits=exact)
        self.assertEqual(counts.stored_events, 3)
        self.assertEqual(counts.stored_timing_records, 2)
        self.assertEqual(counts.logical_events, 6)
        self.assertEqual(counts.logical_timing_records, 6)

        too_small = dataclasses.replace(exact, max_expanded_events=5)
        with mock.patch(
            "commcanary.schema._expand_sequence_motif",
            side_effect=AssertionError("expansion must not begin"),
        ) as expand:
            with self.assertRaisesRegex(SchemaError, "logical canary events=6 exceeds limit=5"):
                list(iter_canary_logical_events(events, limits=too_small))
        expand.assert_not_called()

        timing_too_small = dataclasses.replace(
            exact,
            max_expanded_timing_records=5,
        )
        with self.assertRaisesRegex(SchemaError, "logical timing records=6 exceeds limit=5"):
            preflight_canary_expansion(events, limits=timing_too_small)

    def test_replay_iterations_obey_the_shared_preflight_budget(self) -> None:
        trace = {
            "format": "commcanary.trace.v1",
            "events": [
                {
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 1.0},
                },
                {
                    "op": "all_reduce",
                    "bytes": 32,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 1.0},
                },
            ],
        }
        canary = compile_trace(trace, enable_sequence_motifs=False)
        report = replay_canary(canary, iterations=2, max_replay_events=4)
        self.assertEqual(report["metrics"]["count"], 4)
        with self.assertRaisesRegex(
            SchemaError,
            "replay would execute 4 events, above max_replay_events=3",
        ):
            replay_canary(canary, iterations=2, max_replay_events=3)
        with self.assertRaisesRegex(SchemaError, "cannot exceed resource policy"):
            replay_canary(
                canary,
                max_replay_events=5,
                limits=ResourceLimits(max_replay_events=4),
            )

    def test_behavior_candidate_and_ledger_limits_precede_search(self) -> None:
        trace = {
            "format": "commcanary.trace.v1",
            "events": [
                {
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 1.0},
                }
            ],
        }
        with self.assertRaisesRegex(SchemaError, "evaluate 3 candidates"):
            synthesize_behavioral_canary(
                trace,
                min_timing_sample_limit=2,
                max_timing_sample_limit=4,
                limits=ResourceLimits(max_behavior_candidates=2),
            )
        with self.assertRaisesRegex(SchemaError, "retain 3 candidate rows"):
            synthesize_behavioral_canary(
                trace,
                min_timing_sample_limit=2,
                max_timing_sample_limit=4,
                limits=ResourceLimits(max_retained_ledger_rows=2),
            )
        with self.assertRaisesRegex(SchemaError, "max_oracle_calls cannot exceed"):
            ddmin_ranking_reduction(
                trace,
                max_oracle_calls=3,
                limits=ResourceLimits(max_reduction_oracle_calls=2),
            )

    def test_param_entry_limit_is_checked_before_export_iteration(self) -> None:
        trace = {
            "format": "commcanary.trace.v1",
            "events": [
                {
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 1.0},
                },
                {
                    "op": "all_reduce",
                    "bytes": 32,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 1.0},
                },
            ],
        }
        canary = compile_trace(trace, enable_sequence_motifs=False)
        self.assertEqual(
            len(
                canary_to_param_comms_trace(
                    canary,
                    limits=ResourceLimits(max_param_entries=3),
                )
            ),
            3,
        )
        with mock.patch(
            "commcanary.interop.iter_canary_logical_events",
            side_effect=AssertionError("export iteration must not begin"),
        ) as expand:
            with self.assertRaisesRegex(
                SchemaError,
                "PARAM trace entries=3 exceeds limit=2",
            ):
                canary_to_param_comms_trace(
                    canary,
                    limits=ResourceLimits(max_param_entries=2),
                )
        expand.assert_not_called()

    def test_capture_shard_and_aggregate_limits_precede_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "one.trace.json").write_text("{}", encoding="utf-8")
            Path(tmp, "two.trace.json").write_text("{}", encoding="utf-8")
            with mock.patch(
                "commcanary.capture.load_json",
                side_effect=AssertionError("shards must not be loaded"),
            ) as load:
                with self.assertRaisesRegex(SchemaError, "capture shards=2 exceeds limit=1"):
                    merge_trace_shards(
                        tmp,
                        workload_name="bounded",
                        limits=ResourceLimits(max_capture_shards=1),
                    )
            load.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp:
            trace = {
                "format": "commcanary.trace.v1",
                "workload": {"name": "bounded"},
                "events": [
                    {
                        "op": "all_reduce",
                        "bytes": size,
                        "ranks": [0],
                        "rank_arrival_us": {"0": 0.0},
                    }
                    for size in (16, 32)
                ],
            }
            write_json(str(Path(tmp, "one.trace.json")), trace)
            with self.assertRaisesRegex(
                SchemaError,
                "capture aggregate events=2 exceeds limit=1",
            ):
                merge_trace_shards(
                    tmp,
                    workload_name="bounded",
                    limits=ResourceLimits(max_capture_events=1),
                )

    def test_rank_count_obeys_the_shared_policy(self) -> None:
        trace = {
            "format": "commcanary.trace.v1",
            "events": [
                {
                    "op": "all_reduce",
                    "bytes": 16,
                    "ranks": [0, 1],
                    "rank_arrival_us": {"0": 0.0, "1": 1.0},
                }
            ],
        }
        with self.assertRaisesRegex(SchemaError, "rank count exceeds resource policy"):
            validate_trace(trace, limits=ResourceLimits(max_ranks=1))

    def test_standard_loader_requires_object_and_kineto_accepts_object_or_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            object_path = self._write(tmp, b'{"traceEvents": []}', "object.json")
            array_path = self._write(tmp, b'[{"name": "event"}]', "array.json")
            scalar_path = self._write(tmp, b"1", "scalar.json")

            self.assertEqual(load_json(object_path), {"traceEvents": []})
            with self.assertRaisesRegex(SchemaError, "must contain a JSON object"):
                load_json(array_path)
            self.assertEqual(
                load_kineto_trace(array_path),
                {"traceEvents": [{"name": "event"}]},
            )
            self.assertEqual(load_kineto_trace(object_path), {"traceEvents": []})
            with self.assertRaisesRegex(SchemaError, "object or event array"):
                load_kineto_trace(scalar_path)

    def test_byte_limit_accepts_exact_boundary_and_rejects_boundary_plus_one(self) -> None:
        payload = b'{"x":1}'
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, payload)
            self.assertEqual(
                load_json(path, limits=ResourceLimits(max_input_bytes=len(payload))),
                {"x": 1},
            )
            with self.assertRaisesRegex(SchemaError, "max_input_bytes"):
                load_json(
                    path,
                    limits=ResourceLimits(max_input_bytes=len(payload) - 1),
                )

    def test_reader_requests_only_limit_plus_one_bytes_in_binary_mode(self) -> None:
        mocked_open = mock.mock_open(read_data=b"{}")
        with mock.patch("builtins.open", mocked_open):
            self.assertEqual(
                load_json("virtual.json", limits=ResourceLimits(max_input_bytes=2)),
                {},
            )
        mocked_open.assert_called_once_with("virtual.json", "rb")
        mocked_open().read.assert_called_once_with(3)

    def test_invalid_utf8_and_invalid_json_are_stable_schema_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            invalid_utf8 = self._write(tmp, b'{"x":"\xff"}', "utf8.json")
            invalid_json = self._write(tmp, b'{"x":', "syntax.json")
            with self.assertRaisesRegex(SchemaError, "not UTF-8 JSON"):
                load_json(invalid_utf8)
            with self.assertRaisesRegex(SchemaError, "not valid JSON"):
                load_json(invalid_json)

    def test_duplicate_keys_are_rejected_at_root_and_nested_levels(self) -> None:
        payloads = (
            b'{"x":1,"x":2}',
            b'{"outer":{"x":1,"x":2}}',
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, payload in enumerate(payloads):
                path = self._write(tmp, payload, f"duplicate-{index}.json")
                with self.subTest(index=index):
                    with self.assertRaisesRegex(SchemaError, "duplicate JSON object key 'x'"):
                        load_json(path)

            kineto_path = self._write(
                tmp,
                b'{"traceEvents":[{"args":{"x":1,"x":2}}]}',
                "kineto-duplicate.json",
            )
            with self.assertRaisesRegex(SchemaError, "duplicate JSON object key 'x'"):
                load_kineto_trace(kineto_path)

    def test_nonstandard_and_out_of_range_floats_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for index, token in enumerate((b"NaN", b"Infinity", b"-Infinity", b"1e9999")):
                path = self._write(tmp, b'{"value":' + token + b"}", f"number-{index}.json")
                with self.subTest(token=token):
                    with self.assertRaisesRegex(SchemaError, "JSON (constant|number)"):
                        load_json(path)

    def test_numeric_token_limit_has_an_exact_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exact_path = self._write(tmp, b'{"value":9999}', "number-exact.json")
            oversized_path = self._write(
                tmp,
                b'{"value":99999}',
                "number-oversized.json",
            )
            limits = ResourceLimits(max_json_number_chars=4)
            self.assertEqual(load_json(exact_path, limits=limits)["value"], 9999)
            with self.assertRaisesRegex(SchemaError, "max_json_number_chars=4"):
                load_json(oversized_path, limits=limits)

    def test_depth_is_rejected_before_json_decoder_runs(self) -> None:
        depth_three = b'{"a":{"b":{"c":0}}}'
        depth_four = b'{"a":{"b":{"c":{"d":0}}}}'
        limits = ResourceLimits(max_json_depth=3)
        with tempfile.TemporaryDirectory() as tmp:
            valid_path = self._write(tmp, depth_three, "depth-three.json")
            invalid_path = self._write(tmp, depth_four, "depth-four.json")
            self.assertEqual(load_json(valid_path, limits=limits)["a"]["b"]["c"], 0)
            with mock.patch(
                "commcanary.resources.json.loads",
                side_effect=AssertionError("decoder must not run"),
            ):
                with self.assertRaisesRegex(SchemaError, "max_json_depth=3"):
                    load_json(invalid_path, limits=limits)

    def test_item_and_utf8_string_limits_have_exact_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item_path = self._write(tmp, b'{"items":[1,2]}', "items.json")
            self.assertEqual(
                load_json(item_path, limits=ResourceLimits(max_json_items=3))["items"],
                [1, 2],
            )
            with self.assertRaisesRegex(SchemaError, "max_json_items=2"):
                load_json(item_path, limits=ResourceLimits(max_json_items=2))

            string_path = self._write(
                tmp,
                json.dumps({"k": "éé"}, ensure_ascii=False).encode("utf-8"),
                "string.json",
            )
            self.assertEqual(
                load_json(
                    string_path,
                    limits=ResourceLimits(max_json_string_bytes=4),
                )["k"],
                "éé",
            )
            with self.assertRaisesRegex(SchemaError, "max_json_string_bytes=3"):
                load_json(
                    string_path,
                    limits=ResourceLimits(max_json_string_bytes=3),
                )

    def test_invalid_unicode_scalar_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, b'{"value":"\\ud800"}')
            with self.assertRaisesRegex(SchemaError, "valid Unicode scalar"):
                load_json(path)

    def test_decoder_recursion_overflow_and_value_errors_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, b"{}")
            with mock.patch(
                "commcanary.resources.json.loads",
                side_effect=RecursionError("decoder recursion"),
            ):
                with self.assertRaisesRegex(SchemaError, "parser nesting capacity"):
                    load_json(path)
            with mock.patch(
                "commcanary.resources.json.loads",
                side_effect=OverflowError("numeric overflow"),
            ):
                with self.assertRaisesRegex(SchemaError, "number that is too large"):
                    load_json(path)
            with mock.patch(
                "commcanary.resources.json.loads",
                side_effect=ValueError("decoder value error"),
            ):
                with self.assertRaisesRegex(SchemaError, "non-standard JSON"):
                    load_json(path)

    def test_iterative_validator_rejects_non_json_native_values(self) -> None:
        with self.assertRaisesRegex(JsonResourceError, "keys must be strings"):
            validate_json_value({1: "value"})
        with self.assertRaisesRegex(JsonResourceError, "got tuple"):
            validate_json_value({"value": (1, 2)})
        with self.assertRaisesRegex(JsonResourceError, "numbers must be finite"):
            validate_json_value({"value": math.nan})


if __name__ == "__main__":
    unittest.main()
