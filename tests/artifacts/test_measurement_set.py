from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from commcanary.artifacts.measurement_set import validate_measurement_set
from commcanary.errors import SchemaError

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "schemas" / "commcanary.measurement_set.v1.schema.json"
PACKAGED = ROOT / "src" / "commcanary" / "schemas" / "commcanary.measurement_set.v1.schema.json"


def test_measurement_set_schema_mirror_is_byte_identical_and_accepts_example() -> None:
    assert PUBLISHED.read_bytes() == PACKAGED.read_bytes()
    schema = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    example = json.loads((ROOT / "examples" / "fidelity" / "reference.json").read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(example)) == []
    validate_measurement_set(example)


def test_measurement_set_allows_signed_proxy_scores_but_rejects_nonpositive_reference_values() -> None:
    base = {
        "format": "commcanary.measurement_set.v1",
        "name": "proxy",
        "role": "proxy",
        "metric": "score",
        "direction": "higher_is_better",
        "measurements": {"a": [1.0], "b": [2.0]},
    }
    validate_measurement_set({**base, "measurements": {"a": [-1.0], "b": [0.0]}})
    with pytest.raises(SchemaError, match="reference measurement set.*must be positive"):
        validate_measurement_set({**base, "role": "reference", "measurements": {"a": [0.0], "b": [2.0]}})
    with pytest.raises(SchemaError, match="fields are not closed"):
        validate_measurement_set({**base, "tool": "commcanary"})


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"format": "commcanary.measurement_set.v2"}, "format is unsupported"),
        ({"name": " "}, "name must be a non-empty string"),
        ({"role": "candidate"}, "role is unsupported"),
        ({"direction": "smaller"}, "direction is unsupported"),
        ({"measurements": {"a": [1.0]}}, "at least two configurations"),
        ({"measurements": {"": [1.0], "b": [2.0]}}, "configuration names must be non-empty"),
        ({"measurements": {"a": [], "b": [2.0]}}, "must contain samples"),
        ({"measurements": {"a": [True], "b": [2.0]}}, "sample 0 must be numeric"),
        ({"measurements": {"a": [float("inf")], "b": [2.0]}}, "JSON numbers must be finite"),
    ),
)
def test_measurement_set_rejects_invalid_contract_fields(
    replacement: dict[str, object],
    message: str,
) -> None:
    measurement_set = {
        "format": "commcanary.measurement_set.v1",
        "name": "proxy",
        "role": "proxy",
        "metric": "score",
        "direction": "higher_is_better",
        "measurements": {"a": [1.0], "b": [2.0]},
        **replacement,
    }

    with pytest.raises(SchemaError, match=message):
        validate_measurement_set(measurement_set)
