"""Enforce CommCanary's statement baseline and responsibility branch floors.

Coverage.py's combined branch-mode percentage is useful for display, but it is
not the release policy: CommCanary preserves statement coverage globally and
sets explicit branch floors for safety-critical responsibilities.  This module
reads ``coverage json`` output and evaluates those two dimensions separately.

The checker deliberately derives percentages from integer counters.  Displayed
or rounded percentages in the report are never trusted for pass/fail decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class CoveragePolicyError(RuntimeError):
    """A coverage report or policy is malformed."""


class CoverageFailure(RuntimeError):
    """Measured coverage does not satisfy a valid policy."""


@dataclass(frozen=True)
class ResponsibilityPolicy:
    """Branch floor and explicit path aliases for one responsibility."""

    name: str
    branch_floor_percent: Decimal
    required: bool
    paths: Tuple[str, ...]
    globs: Tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    """Validated coverage policy."""

    statement_floor_percent: Decimal
    responsibilities: Tuple[ResponsibilityPolicy, ...]


@dataclass(frozen=True)
class ResponsibilityResult:
    """Aggregate branch measurement for one matched responsibility."""

    name: str
    covered_branches: int
    num_branches: int
    branch_floor_percent: Decimal
    files: Tuple[str, ...]


@dataclass(frozen=True)
class CoverageResult:
    """Exact counters accepted by the policy."""

    covered_statements: int
    num_statements: int
    statement_floor_percent: Decimal
    responsibilities: Tuple[ResponsibilityResult, ...]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CoveragePolicyError(f"{label} must be a JSON object")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoveragePolicyError(f"{label} must be an integer")
    if value < 0:
        raise CoveragePolicyError(f"{label} must be non-negative")
    return value


def _percent(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CoveragePolicyError(f"{label} must be a decimal percentage")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CoveragePolicyError(f"{label} must be a decimal percentage") from exc
    if not result.is_finite() or result < 0 or result > 100:
        raise CoveragePolicyError(f"{label} must be between 0 and 100")
    return result


def _string_list(value: object, label: str, *, globs: bool) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise CoveragePolicyError(f"{label} must be a JSON array")
    result: List[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise CoveragePolicyError(f"{label}[{index}] must be a non-empty string")
        normalized = _normalize_path(item)
        has_magic = any(character in normalized for character in "*?[")
        if globs and not has_magic:
            raise CoveragePolicyError(f"{label}[{index}] must contain a glob wildcard")
        if not globs and has_magic:
            raise CoveragePolicyError(f"{label}[{index}] must be an exact path, not a glob")
        result.append(normalized)
    if len(set(result)) != len(result):
        raise CoveragePolicyError(f"{label} contains duplicate entries")
    return tuple(result)


def _normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def parse_policy(raw_policy: object) -> Policy:
    """Validate and parse a JSON policy object."""

    policy = _mapping(raw_policy, "coverage policy")
    if _integer(policy.get("version"), "coverage policy.version") != 1:
        raise CoveragePolicyError("coverage policy.version must be 1")
    statement_floor = _percent(
        policy.get("statement_floor_percent"),
        "coverage policy.statement_floor_percent",
    )
    raw_responsibilities = policy.get("responsibilities")
    if not isinstance(raw_responsibilities, list) or not raw_responsibilities:
        raise CoveragePolicyError("coverage policy.responsibilities must be a non-empty JSON array")

    responsibilities: List[ResponsibilityPolicy] = []
    names = set()
    for index, raw_responsibility in enumerate(raw_responsibilities):
        label = f"coverage policy.responsibilities[{index}]"
        responsibility = _mapping(raw_responsibility, label)
        name = responsibility.get("name")
        if not isinstance(name, str) or not name.strip():
            raise CoveragePolicyError(f"{label}.name must be a non-empty string")
        if name in names:
            raise CoveragePolicyError(f"duplicate coverage responsibility {name!r}")
        names.add(name)
        required = responsibility.get("required")
        if not isinstance(required, bool):
            raise CoveragePolicyError(f"{label}.required must be a boolean")
        paths = _string_list(responsibility.get("paths"), f"{label}.paths", globs=False)
        globs = _string_list(responsibility.get("globs"), f"{label}.globs", globs=True)
        if not paths and not globs:
            raise CoveragePolicyError(f"{label} must declare at least one path or glob")
        responsibilities.append(
            ResponsibilityPolicy(
                name=name,
                branch_floor_percent=_percent(
                    responsibility.get("branch_floor_percent"),
                    f"{label}.branch_floor_percent",
                ),
                required=required,
                paths=paths,
                globs=globs,
            )
        )
    return Policy(statement_floor_percent=statement_floor, responsibilities=tuple(responsibilities))


def _summary_counts(summary: Mapping[str, Any], label: str) -> Tuple[int, int, int, int]:
    covered_statements = _integer(summary.get("covered_lines"), f"{label}.covered_lines")
    num_statements = _integer(summary.get("num_statements"), f"{label}.num_statements")
    covered_branches = _integer(summary.get("covered_branches"), f"{label}.covered_branches")
    num_branches = _integer(summary.get("num_branches"), f"{label}.num_branches")
    if covered_statements > num_statements:
        raise CoveragePolicyError(f"{label}.covered_lines exceeds num_statements")
    if covered_branches > num_branches:
        raise CoveragePolicyError(f"{label}.covered_branches exceeds num_branches")
    return covered_statements, num_statements, covered_branches, num_branches


def _report_files(raw_report: object) -> Tuple[Mapping[str, Any], Dict[str, Tuple[int, int, int, int]]]:
    report = _mapping(raw_report, "coverage report")
    totals = _mapping(report.get("totals"), "coverage report.totals")
    raw_files = _mapping(report.get("files"), "coverage report.files")
    if not raw_files:
        raise CoveragePolicyError("coverage report.files must not be empty")

    files: Dict[str, Tuple[int, int, int, int]] = {}
    for raw_path, raw_file in raw_files.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise CoveragePolicyError("coverage report file paths must be non-empty strings")
        path = _normalize_path(raw_path)
        if path in files:
            raise CoveragePolicyError(f"coverage report contains duplicate normalized path {path!r}")
        file_data = _mapping(raw_file, f"coverage report.files[{raw_path!r}]")
        summary = _mapping(file_data.get("summary"), f"coverage report.files[{raw_path!r}].summary")
        files[path] = _summary_counts(summary, f"coverage report.files[{raw_path!r}].summary")
    return totals, files


def _meets_floor(covered: int, total: int, floor: Decimal) -> bool:
    """Compare exact counters with a decimal percentage without rounding."""

    return Decimal(covered) * Decimal(100) >= Decimal(total) * floor


def _format_percent(covered: int, total: int) -> str:
    if total == 0:
        return "undefined"
    return f"{(Decimal(covered) * Decimal(100) / Decimal(total)):.4f}%"


def check_coverage(raw_report: object, raw_policy: object) -> CoverageResult:
    """Return exact measurements or raise if the policy is invalid or unmet."""

    policy = parse_policy(raw_policy)
    totals, files = _report_files(raw_report)
    covered_statements = _integer(totals.get("covered_lines"), "coverage report.totals.covered_lines")
    num_statements = _integer(totals.get("num_statements"), "coverage report.totals.num_statements")
    if covered_statements > num_statements:
        raise CoveragePolicyError("coverage report.totals.covered_lines exceeds num_statements")
    if num_statements == 0:
        raise CoveragePolicyError("coverage report contains zero statements")

    summed_covered = sum(counts[0] for counts in files.values())
    summed_statements = sum(counts[1] for counts in files.values())
    if (covered_statements, num_statements) != (summed_covered, summed_statements):
        raise CoveragePolicyError(
            "coverage report totals do not equal the sum of file statement counters "
            f"(totals={covered_statements}/{num_statements}, files={summed_covered}/{summed_statements})"
        )

    failures: List[str] = []
    if not _meets_floor(covered_statements, num_statements, policy.statement_floor_percent):
        failures.append(
            "global statement coverage "
            f"{_format_percent(covered_statements, num_statements)} is below "
            f"{policy.statement_floor_percent}% ({covered_statements}/{num_statements})"
        )

    responsibility_results: List[ResponsibilityResult] = []
    for responsibility in policy.responsibilities:
        matched = tuple(
            sorted(
                path
                for path in files
                if path in responsibility.paths or any(fnmatchcase(path, pattern) for pattern in responsibility.globs)
            )
        )
        if not matched:
            if responsibility.required:
                failures.append(f"required responsibility {responsibility.name!r} matched no measured files")
            continue
        covered_branches = sum(files[path][2] for path in matched)
        num_branches = sum(files[path][3] for path in matched)
        result = ResponsibilityResult(
            name=responsibility.name,
            covered_branches=covered_branches,
            num_branches=num_branches,
            branch_floor_percent=responsibility.branch_floor_percent,
            files=matched,
        )
        responsibility_results.append(result)
        if num_branches == 0:
            failures.append(
                f"responsibility {responsibility.name!r} has zero measurable branches in {', '.join(matched)}"
            )
        elif not _meets_floor(covered_branches, num_branches, responsibility.branch_floor_percent):
            failures.append(
                f"responsibility {responsibility.name!r} branch coverage "
                f"{_format_percent(covered_branches, num_branches)} is below "
                f"{responsibility.branch_floor_percent}% ({covered_branches}/{num_branches}; "
                f"files={', '.join(matched)})"
            )

    if failures:
        raise CoverageFailure("coverage policy failed:\n" + "\n".join(f"- {failure}" for failure in failures))
    return CoverageResult(
        covered_statements=covered_statements,
        num_statements=num_statements,
        statement_floor_percent=policy.statement_floor_percent,
        responsibilities=tuple(responsibility_results),
    )


def _read_json(path: Path, label: str) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoveragePolicyError(f"cannot read {label} {path}: {exc}") from exc


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="coverage.py JSON report")
    parser.add_argument("--policy", required=True, type=Path, help="CommCanary coverage policy JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        result = check_coverage(
            _read_json(args.report, "coverage report"),
            _read_json(args.policy, "coverage policy"),
        )
    except (CoverageFailure, CoveragePolicyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        "statement coverage "
        f"{_format_percent(result.covered_statements, result.num_statements)} "
        f"({result.covered_statements}/{result.num_statements}; floor={result.statement_floor_percent}%)"
    )
    for responsibility in result.responsibilities:
        print(
            f"{responsibility.name} branch coverage "
            f"{_format_percent(responsibility.covered_branches, responsibility.num_branches)} "
            f"({responsibility.covered_branches}/{responsibility.num_branches}; "
            f"floor={responsibility.branch_floor_percent}%; files={','.join(responsibility.files)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
