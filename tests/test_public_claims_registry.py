from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "claims" / "public-claims.yaml"
SCHEMA_PATH = ROOT / "schemas" / "commcanary.public_claims.v1.schema.json"
PACKAGED_SCHEMA_PATH = ROOT / "src" / "commcanary" / "schemas" / "commcanary.public_claims.v1.schema.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_public_claims_registry_validates() -> None:
    schema = _json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_json(CLAIMS_PATH))


def test_public_claims_schema_mirror_is_exact() -> None:
    assert SCHEMA_PATH.read_bytes() == PACKAGED_SCHEMA_PATH.read_bytes()


def test_public_claim_ids_are_unique_across_groups() -> None:
    document = _json(CLAIMS_PATH)
    ids = [row["id"] for group in ("allowed", "qualified_only_after_evidence", "forbidden") for row in document[group]]
    assert len(ids) == len(set(ids))


def test_public_claim_registry_has_the_frozen_initial_claims() -> None:
    document = _json(CLAIMS_PATH)
    assert {row["id"] for row in document["allowed"]} == {
        "capture_build_gate_exists",
        "chakra_dependency_closed_selection_exists",
        "exact_work_point_estimates_published",
        "private_exchange_is_implemented",
    }
    assert {row["id"] for row in document["qualified_only_after_evidence"]} == {
        "predicts_stack_change_decisions",
        "preserves_held_out_regressions",
        "reduces_physical_runtime_by_10x",
        "safe_for_ci_gating",
    }
    assert {row["id"] for row in document["forbidden"]} == {
        "multi_node_qualified",
        "privacy_safe",
        "production_validated",
        "works_across_arbitrary_models",
    }
