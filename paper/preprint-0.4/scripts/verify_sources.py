#!/usr/bin/env python3
"""Verify paper-facing metrics against immutable repository evidence."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
METRICS_PATH = SOURCE_ROOT / "evidence" / "verified_metrics.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"{label}: expected {expected!r}, observed {actual!r}")


def _load(relative_path: str) -> dict[str, Any]:
    return json.loads((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    expected_head = metrics["paper"]["evidence_commit"]
    observed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_head != expected_head:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected_head, observed_head],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        if ancestry.returncode != 0:
            raise SystemExit(
                "repository HEAD is neither the evidence commit nor its descendant: "
                f"expected ancestor {expected_head}, observed {observed_head}"
            )

    for relative_path, expected_digest in metrics["source_files"].items():
        path = REPOSITORY_ROOT / relative_path
        observed_digest = _sha256(path)
        if observed_digest != expected_digest:
            raise SystemExit(
                f"source digest mismatch for {relative_path}: expected {expected_digest}, observed {observed_digest}"
            )

    trusted_path = "experiments/rostam/results/publications/trusted-join-core-shared-overlap-primary/aggregate.json"
    trusted = _load(trusted_path)
    snapshot = metrics["trusted_comparison"]
    if trusted["selected_cell_count"] != snapshot["selected_cells"]:
        raise SystemExit("trusted comparison selected-cell count mismatch")
    if not trusted["completeness"]["complete"]:
        raise SystemExit("trusted comparison is not complete")
    if trusted["provenance"]["trusted_join_sha256"] != snapshot["trusted_join_sha256"]:
        raise SystemExit("trusted join identity mismatch")

    gate_path = (
        "experiments/rostam/results/publications/"
        "decision-gate-20260801-r3-analysis-1300d12-primary/"
        "decision-fidelity-verdict.json"
    )
    gate = _load(gate_path)
    gate_snapshot = metrics["exact_work_gate"]
    if gate["outcome"] != gate_snapshot["outcome"]:
        raise SystemExit("exact-work outcome mismatch")
    observed = gate["representation_metrics"]["exact_work"]
    for field in (
        "false_negative_count",
        "false_positive_count",
        "kendall_tau_b",
        "median_absolute_relative_error_pct",
        "p95_absolute_relative_error_pct",
        "pairwise_agreement_count",
    ):
        expected = gate_snapshot[field]
        value = observed[field]
        if isinstance(expected, float):
            _require_close(float(value), expected, f"exact-work {field}")
        elif value != expected:
            raise SystemExit(f"exact-work {field} mismatch")

    diagnostic_path = (
        "experiments/rostam/results/exact-work-artifacts/publication/"
        "qualification-exact-20260730-r3-primary/aggregate.json"
    )
    diagnostic = _load(diagnostic_path)
    diagnostic_snapshot = metrics["same_node_diagnostic"]
    if diagnostic["selected_cell_count"] != diagnostic_snapshot["selected_cells"]:
        raise SystemExit("diagnostic selected-cell count mismatch")
    row = diagnostic["aggregates"][0]
    _require_close(
        float(row["median_us"]),
        float(diagnostic_snapshot["replay_median_us"]),
        "diagnostic replay median",
    )
    _require_close(
        float(row["iqr_us"]),
        float(diagnostic_snapshot["replay_iqr_us"]),
        "diagnostic replay IQR",
    )

    print(
        "verified preprint sources: evidence ancestry, 7 source digests, 280-cell trusted "
        "comparison, 8-cell exact-work gate, and 1-cell diagnostic"
    )


if __name__ == "__main__":
    main()
