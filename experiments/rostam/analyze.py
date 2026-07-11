#!/usr/bin/env python3
"""Completeness-gated Rostam analysis and deterministic publication CLI."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import List, Optional

if __package__ in {None, ""}:  # direct ``python experiments/rostam/analyze.py`` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rostam.analysis import CampaignEvidence, verify_regenerate_compare
from experiments.rostam.analysis import legacy as legacy_analysis
from experiments.rostam.harness import ContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a frozen manifest, explicit selection, and persisted completeness "
            "verdict before regenerating aggregate JSON/CSV and a paper fragment."
        )
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--verdict-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--regeneration-command")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--join-evidence",
        nargs=3,
        action="append",
        default=[],
        metavar=("RUN_DIRECTORY", "SELECTION_ID", "VERDICT_SHA256"),
        help="join another independently complete frozen campaign",
    )
    parser.add_argument("--archive-descriptor", type=Path)
    parser.add_argument("--raw-archive", type=Path)
    parser.add_argument("--baseline-config")
    parser.add_argument("--candidate-config")
    parser.add_argument("--median-threshold-pct", type=float, default=8.0)
    parser.add_argument("--median-absolute-threshold-us", type=float, default=1.0)
    parser.add_argument("--golden-directory", type=Path)
    return parser


def _derived_regeneration_command(args: argparse.Namespace) -> str:
    command = [
        "python",
        "-m",
        "experiments.rostam.analyze",
        "verify",
        "--run-directory",
        str(args.run_directory),
        "--selection-id",
        str(args.selection_id),
        "--verdict-sha256",
        str(args.verdict_sha256),
        "--output-directory",
        str(args.output_directory),
    ]
    if args.allow_incomplete:
        command.append("--allow-incomplete")
    for run_directory, selection_id, verdict_sha256 in args.join_evidence:
        command.extend(("--join-evidence", run_directory, selection_id, verdict_sha256))
    if args.archive_descriptor is not None:
        command.extend(("--archive-descriptor", str(args.archive_descriptor)))
    if args.raw_archive is not None:
        command.extend(("--raw-archive", str(args.raw_archive)))
    if args.baseline_config is not None:
        command.extend(("--baseline-config", args.baseline_config))
    if args.candidate_config is not None:
        command.extend(("--candidate-config", args.candidate_config))
    command.extend(("--median-threshold-pct", str(args.median_threshold_pct)))
    command.extend(("--median-absolute-threshold-us", str(args.median_absolute_threshold_us)))
    if args.golden_directory is not None:
        command.extend(("--golden-directory", str(args.golden_directory)))
    return shlex.join(command)


def _verified_main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    regeneration_command = args.regeneration_command or _derived_regeneration_command(args)
    try:
        publication = verify_regenerate_compare(
            args.run_directory,
            args.selection_id,
            args.verdict_sha256,
            args.output_directory,
            regeneration_command=regeneration_command,
            allow_incomplete=args.allow_incomplete,
            joined_evidence=tuple(
                CampaignEvidence(Path(run_directory), selection_id, verdict_sha256)
                for run_directory, selection_id, verdict_sha256 in args.join_evidence
            ),
            archive_descriptor=args.archive_descriptor,
            raw_archive=args.raw_archive,
            golden_directory=args.golden_directory,
            baseline_config=args.baseline_config,
            candidate_config=args.candidate_config,
            relative_threshold_pct=args.median_threshold_pct,
            absolute_threshold_us=args.median_absolute_threshold_us,
        )
    except (ContractError, OSError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    for filename, digest in sorted(publication.output_sha256.items()):
        print(f"wrote {publication.output_directory / filename} sha256={digest}")
    if publication.matched_golden:
        print("golden publication bytes match")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "legacy":
        return legacy_analysis.main(arguments[1:])
    if arguments and arguments[0] == "verify":
        arguments = arguments[1:]
    return _verified_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
