from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from commcanary.interop import load_kineto_trace
from commcanary.resources import JsonResourceError, ResourceLimits, validate_json_value
from commcanary.schema import SchemaError, load_json


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
