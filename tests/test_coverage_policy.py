from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import coverage_policy


def _policy(
    *,
    statement_floor: object = "86.0",
    branch_floor: object = "90.0",
    required: bool = True,
    paths: object = None,
    globs: object = None,
) -> object:
    return {
        "version": 1,
        "statement_floor_percent": statement_floor,
        "responsibilities": [
            {
                "name": "critical path",
                "branch_floor_percent": branch_floor,
                "required": required,
                "paths": ["src/commcanary/critical.py"] if paths is None else paths,
                "globs": [] if globs is None else globs,
            }
        ],
    }


def _report(
    *,
    covered_statements: int = 86,
    num_statements: int = 100,
    covered_branches: int = 90,
    num_branches: int = 100,
    path: str = "src/commcanary/critical.py",
) -> object:
    summary = {
        "covered_lines": covered_statements,
        "num_statements": num_statements,
        "covered_branches": covered_branches,
        "num_branches": num_branches,
    }
    return {
        "meta": {"branch_coverage": True},
        "files": {path: {"summary": summary}},
        "totals": {
            "covered_lines": covered_statements,
            "num_statements": num_statements,
            "covered_branches": covered_branches,
            "num_branches": num_branches,
        },
    }


def test_exact_statement_and_branch_boundaries_pass() -> None:
    result = coverage_policy.check_coverage(_report(), _policy())

    assert (result.covered_statements, result.num_statements) == (86, 100)
    assert len(result.responsibilities) == 1
    assert (result.responsibilities[0].covered_branches, result.responsibilities[0].num_branches) == (
        90,
        100,
    )


@pytest.mark.parametrize(
    ("report", "message"),
    [
        (_report(covered_statements=85), "global statement coverage"),
        (_report(covered_branches=89), "branch coverage"),
    ],
)
def test_one_below_a_floor_fails(report: object, message: str) -> None:
    with pytest.raises(coverage_policy.CoverageFailure, match=message):
        coverage_policy.check_coverage(report, _policy())


def test_display_rounding_cannot_hide_a_statement_regression() -> None:
    report = _report(
        covered_statements=8_599,
        num_statements=10_000,
        covered_branches=9,
        num_branches=10,
    )

    with pytest.raises(coverage_policy.CoverageFailure, match=r"85\.9900%.*86\.0%"):
        coverage_policy.check_coverage(report, _policy())


def test_required_responsibility_must_match_a_measured_file() -> None:
    with pytest.raises(coverage_policy.CoverageFailure, match="matched no measured files"):
        coverage_policy.check_coverage(
            _report(path="src/commcanary/unrelated.py"),
            _policy(),
        )


def test_optional_future_responsibility_can_be_absent() -> None:
    result = coverage_policy.check_coverage(
        _report(path="src/commcanary/unrelated.py"),
        _policy(required=False),
    )

    assert result.responsibilities == ()


def test_matched_responsibility_with_zero_branches_fails_closed() -> None:
    with pytest.raises(coverage_policy.CoverageFailure, match="zero measurable branches"):
        coverage_policy.check_coverage(
            _report(covered_branches=0, num_branches=0),
            _policy(),
        )


def test_exact_path_aliases_normalize_windows_separators() -> None:
    result = coverage_policy.check_coverage(
        _report(path=r"src\commcanary\critical.py"),
        _policy(),
    )

    assert result.responsibilities[0].files == ("src/commcanary/critical.py",)


def test_explicit_glob_alias_supports_relocated_source_root() -> None:
    result = coverage_policy.check_coverage(
        _report(path="/workspace/checkout/src/commcanary/critical.py"),
        _policy(paths=[], globs=["*/src/commcanary/critical.py"]),
    )

    assert result.responsibilities[0].files == ("/workspace/checkout/src/commcanary/critical.py",)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.pop("totals"),
        lambda report: report["totals"].update({"covered_lines": True}),
        lambda report: report["files"]["src/commcanary/critical.py"].pop("summary"),
        lambda report: report["totals"].update({"num_statements": 101}),
        lambda report: report["files"]["src/commcanary/critical.py"]["summary"].update({"covered_branches": 101}),
    ],
)
def test_malformed_reports_fail_closed(mutate) -> None:
    report = _report()
    mutate(report)

    with pytest.raises(coverage_policy.CoveragePolicyError):
        coverage_policy.check_coverage(report, _policy())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda policy: policy.update({"version": 2}),
        lambda policy: policy.update({"statement_floor_percent": "NaN"}),
        lambda policy: policy["responsibilities"][0].update({"required": "yes"}),
        lambda policy: policy["responsibilities"][0].update({"paths": ["*.py"]}),
        lambda policy: policy["responsibilities"][0].update({"globs": ["exact.py"]}),
        lambda policy: policy["responsibilities"].append(copy.deepcopy(policy["responsibilities"][0])),
    ],
)
def test_malformed_policies_fail_closed(mutate) -> None:
    policy = _policy()
    mutate(policy)

    with pytest.raises(coverage_policy.CoveragePolicyError):
        coverage_policy.check_coverage(_report(), policy)


def test_cli_reads_json_and_prints_exact_counters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_path = tmp_path / "coverage.json"
    policy_path = tmp_path / "policy.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")

    assert coverage_policy.main(["--report", str(report_path), "--policy", str(policy_path)]) == 0
    output = capsys.readouterr()
    assert "statement coverage 86.0000% (86/100; floor=86.0%)" in output.out
    assert output.err == ""
