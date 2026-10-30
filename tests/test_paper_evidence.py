"""Paper values must regenerate from committed evidence, like every other number."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_trusted_join_uncertainty_regenerates_exactly() -> None:
    script = runpy.run_path(str(ROOT / "paper" / "arxiv" / "scripts" / "trusted_join_uncertainty.py"))
    assert script["main"](["--check"]) == 0


def test_decision_regret_and_typeset_table_regenerate_exactly() -> None:
    script = runpy.run_path(str(ROOT / "paper" / "arxiv" / "scripts" / "decision_regret.py"))
    assert script["main"](["--check"]) == 0
