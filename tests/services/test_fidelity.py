from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pytest

from commcanary.artifacts import load_json
from commcanary.errors import SchemaError
from commcanary.reporting.fidelity import render_fidelity_html
from commcanary.services.fidelity import score_fidelity

ROOT = Path(__file__).resolve().parents[2]


def _measurement_set(
    name: str,
    role: str,
    values: Mapping[str, Sequence[float]],
    *,
    direction: str = "lower_is_better",
) -> Dict[str, Any]:
    return {
        "format": "commcanary.measurement_set.v1",
        "name": name,
        "role": role,
        "metric": "score",
        "direction": direction,
        "measurements": {configuration: list(samples) for configuration, samples in values.items()},
    }


def _repeated(**values: float) -> Dict[str, Sequence[float]]:
    return {configuration: [value, value, value] for configuration, value in values.items()}


def _proxy(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(row for row in report["proxies"] if row["name"] == name)


def test_perfect_proxy_scores_zero_regret() -> None:
    reference = _measurement_set("reference", "reference", _repeated(a=10.0, b=20.0, c=30.0))
    proxy = _measurement_set("perfect", "proxy", _repeated(a=1.0, b=2.0, c=3.0))

    result = score_fidelity([reference, proxy], bootstrap_resamples=100)

    assert result["proxies"][0]["regret_at_1_pct"] == 0.0
    assert result["proxies"][0]["regret_at_1_confidence_interval"]["lower_pct"] == 0.0
    assert result["proxies"][0]["regret_at_1_confidence_interval"]["upper_pct"] == 0.0


def test_inverted_proxy_scores_maximum_possible_regret() -> None:
    reference = _measurement_set("reference", "reference", _repeated(a=10.0, b=20.0, c=30.0))
    proxy = _measurement_set("inverted", "proxy", _repeated(a=3.0, b=2.0, c=1.0))

    result = score_fidelity([reference, proxy], bootstrap_resamples=100)

    row = result["proxies"][0]
    assert row["recommended_configuration"] == "c"
    assert row["regret_at_1_pct"] == 200.0
    assert row["disaster_detection"]["configuration"] == "c"


def test_agreement_and_regret_rankings_can_diverge() -> None:
    reference = _measurement_set(
        "reference",
        "reference",
        _repeated(a=10.0, b=20.0, c=30.0, d=40.0),
    )
    high_agreement = _measurement_set(
        "high-agreement",
        "proxy",
        _repeated(a=2.0, b=1.0, c=3.0, d=4.0),
    )
    low_regret = _measurement_set(
        "low-regret",
        "proxy",
        _repeated(a=1.0, b=4.0, c=3.0, d=2.0),
    )

    result = score_fidelity([reference, high_agreement, low_regret], bootstrap_resamples=100)
    agreement_row = _proxy(result, "high-agreement")
    regret_row = _proxy(result, "low-regret")

    assert (
        agreement_row["ranking_quality"]["pairwise_decision_agreement"]
        > regret_row["ranking_quality"]["pairwise_decision_agreement"]
    )
    assert regret_row["regret_at_1_pct"] < agreement_row["regret_at_1_pct"]
    assert result["decision_ranking"] == ["low-regret", "high-agreement"]
    assert result["ranking_quality_ranking"] == ["high-agreement", "low-regret"]
    assert result["rankings_diverge"] is True


def test_proxy_configuration_mismatch_is_refused() -> None:
    reference = _measurement_set("reference", "reference", _repeated(a=10.0, b=20.0, c=30.0))
    proxy = _measurement_set("partial", "proxy", _repeated(a=1.0, b=2.0, d=3.0))

    with pytest.raises(SchemaError, match="configuration set does not match"):
        score_fidelity([reference, proxy], bootstrap_resamples=100)


def test_insufficient_replicates_are_refused_instead_of_reporting_an_interval() -> None:
    reference = _measurement_set(
        "reference",
        "reference",
        {"a": [10.0, 10.0], "b": [20.0, 20.0]},
    )
    proxy = _measurement_set("proxy", "proxy", _repeated(a=1.0, b=2.0))

    with pytest.raises(SchemaError, match="refusing bootstrap interval"):
        score_fidelity([reference, proxy], bootstrap_resamples=100)


def test_real_fidelity_examples_reproduce_regret_and_disaster_result() -> None:
    example = ROOT / "examples" / "fidelity"
    result = score_fidelity(
        [
            load_json(str(example / "reference.json")),
            load_json(str(example / "microbenchmark.json")),
            load_json(str(example / "overlap-canary.json")),
            load_json(str(example / "comm-only.json")),
        ],
        bootstrap_resamples=100,
    )
    overlap = _proxy(result, "overlap-canary")
    micro = _proxy(result, "microbenchmark")

    assert overlap["regret_at_1_pct"] == 0.0
    assert round(float(micro["regret_at_1_pct"]), 2) == 3.16
    assert micro["disaster_detection"]["in_bottom_quartile"] is False
    assert round(float(micro["disaster_detection"]["true_shortfall_pct"]), 1) == 38.3
    assert micro["ranking_quality"]["pairwise_decision_agreement_pct"] == pytest.approx(64.2857142857)
    assert overlap["ranking_quality"]["pairwise_decision_agreement_pct"] == pytest.approx(53.5714285714)


def test_bootstrap_and_html_report_are_reproducible_and_expose_divergence() -> None:
    reference = _measurement_set(
        "reference",
        "reference",
        {"a": [9.0, 10.0, 11.0], "b": [10.0, 11.0, 12.0], "c": [29.0, 30.0, 31.0]},
    )
    agreement_proxy = _measurement_set(
        "agreement-proxy",
        "proxy",
        _repeated(a=2.0, b=1.0, c=3.0),
    )
    decision_proxy = _measurement_set(
        "decision-proxy",
        "proxy",
        _repeated(a=1.0, b=3.0, c=2.0),
    )

    first = score_fidelity([reference, agreement_proxy, decision_proxy], bootstrap_resamples=100, seed=71)
    second = score_fidelity([reference, agreement_proxy, decision_proxy], bootstrap_resamples=100, seed=71)
    rendered = render_fidelity_html(first)

    assert first == second
    assert "Decision and ranking metrics disagree" in rendered
    assert "Decision metrics: ranked by regret at 1" in rendered
    assert "Ranking-quality metrics: ranked by pairwise agreement" in rendered
    assert rendered.index("decision-proxy") < rendered.index("agreement-proxy")
