from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from commcanary.compare import compare_reports
from commcanary.schema import validate_comparison

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "contracts"


def _load_json(name: str) -> Any:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _evaluation(comparison: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    return next(row for row in comparison["evaluations"] if row["metric"] == metric)


@pytest.mark.parametrize(
    "case_index",
    range(7),
    ids=[
        "relative-exact",
        "relative-above",
        "absolute-exact",
        "absolute-above",
        "hidden-exact",
        "hidden-above",
        "zero-baseline",
    ],
)
def test_literal_comparison_threshold_boundaries_and_structured_evaluation_codes(case_index: int) -> None:
    vectors = _load_json("comparison_boundary_vectors.v1.json")
    case = vectors["cases"][case_index]
    baseline = copy.deepcopy(_load_json(vectors["base_report_fixture"]))
    candidate = copy.deepcopy(baseline)
    baseline["metrics"].update(case["baseline_metrics"])
    candidate["metrics"].update(case["candidate_metrics"])

    comparison = compare_reports(baseline, candidate)
    expected = case["expected"]
    evaluation = _evaluation(comparison, expected["evaluation_code"])
    validate_comparison(comparison)

    assert vectors["structured_reason_code_path"] == "evaluations[].metric"
    assert comparison["verdict"] == expected["verdict"]
    assert comparison["derived_verdict"] == expected["verdict"]
    assert comparison["reasons"] == expected["reasons"]
    assert evaluation["metric"] == expected["evaluation_code"]
    assert evaluation["threshold_result"] == expected["threshold_result"]
    if expected["evaluation_code"] == "overall.communication_hidden_pct_drop":
        assert evaluation["drop_points"] == expected["absolute_delta"]
    else:
        assert evaluation["absolute_delta"] == expected["absolute_delta"]
        assert evaluation["relative_delta_pct"] == expected["relative_delta_pct"]
        assert evaluation["relative_status"] == expected["relative_status"]


def test_comparison_has_structured_evaluation_codes_but_no_dedicated_reason_code_array() -> None:
    vectors = _load_json("comparison_boundary_vectors.v1.json")
    case = vectors["cases"][1]
    baseline = copy.deepcopy(_load_json(vectors["base_report_fixture"]))
    candidate = copy.deepcopy(baseline)
    baseline["metrics"].update(case["baseline_metrics"])
    candidate["metrics"].update(case["candidate_metrics"])

    comparison = compare_reports(baseline, candidate)
    assert "reason_codes" not in comparison
    assert [row["metric"] for row in comparison["evaluations"][:4]] == [
        "overall.median",
        "overall.p95",
        "overall.p99",
        "overall.communication_hidden_pct_drop",
    ]
