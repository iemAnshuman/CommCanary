"""docs/cli.md is the command-line contract; keep it complete."""

from __future__ import annotations

import argparse
from pathlib import Path

from commcanary.cli import _build_parser

ROOT = Path(__file__).resolve().parents[1]


def _public_subcommands() -> list[str]:
    parser = _build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    # Hidden subcommands keep their parser but suppress their help entry.
    listed = {choice.dest for choice in action._choices_actions if choice.help != argparse.SUPPRESS}
    return sorted(name for name in action.choices if name in listed)


def test_every_public_subcommand_is_in_the_cli_reference() -> None:
    """demo and fidelity shipped without a word in docs/cli.md."""

    reference = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    missing = [name for name in _public_subcommands() if f"commcanary {name}" not in reference]
    assert not missing, missing


def test_the_cli_reference_does_not_hardcode_the_format_count() -> None:
    """It said 25 while the package reported 28. The count lives in code."""

    reference = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    assert "exact artifact format IDs" not in reference
