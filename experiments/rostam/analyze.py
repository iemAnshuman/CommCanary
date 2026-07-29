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

from experiments.rostam.analysis import (
    CampaignEvidence,
    prepare_cross_commit_compatibility,
    verify_regenerate_compare,
)
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
    parser.add_argument(
        "--cross-commit-contract",
        type=Path,
        help="reviewed exact-evidence bridge for a join spanning repository identities",
    )
    parser.add_argument(
        "--compatibility-golden-directory",
        type=Path,
        help="immutable publication bytes that the current analyzer must reproduce before a cross-commit join",
    )
    parser.add_argument("--compatibility-archive-descriptor", type=Path)
    parser.add_argument("--compatibility-raw-archive", type=Path)
    return parser


def build_compatibility_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate a same-repository ground-truth publication byte-for-byte, "
            "then prepare an exact manifest-bound cross-commit compatibility contract."
        )
    )
    parser.add_argument(
        "--ground-evidence",
        nargs=3,
        action="append",
        required=True,
        metavar=("RUN_DIRECTORY", "SELECTION_ID", "VERDICT_SHA256"),
    )
    parser.add_argument(
        "--extension-evidence",
        nargs=3,
        action="append",
        required=True,
        metavar=("RUN_DIRECTORY", "SELECTION_ID", "VERDICT_SHA256"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regeneration-command", required=True)
    parser.add_argument("--golden-directory", type=Path, required=True)
    parser.add_argument("--archive-descriptor", type=Path)
    parser.add_argument("--raw-archive", type=Path)
    parser.add_argument("--baseline-config")
    parser.add_argument("--candidate-config")
    parser.add_argument("--median-threshold-pct", type=float, default=8.0)
    parser.add_argument("--median-absolute-threshold-us", type=float, default=1.0)
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help=(
            "emit an executable reviewed contract after inspecting the exact candidate; "
            "without this acknowledgement the output cannot authorize a join"
        ),
    )
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
    if args.cross_commit_contract is not None:
        command.extend(("--cross-commit-contract", str(args.cross_commit_contract)))
    if args.compatibility_golden_directory is not None:
        command.extend(("--compatibility-golden-directory", str(args.compatibility_golden_directory)))
    if args.compatibility_archive_descriptor is not None:
        command.extend(("--compatibility-archive-descriptor", str(args.compatibility_archive_descriptor)))
    if args.compatibility_raw_archive is not None:
        command.extend(("--compatibility-raw-archive", str(args.compatibility_raw_archive)))
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
            cross_commit_contract=args.cross_commit_contract,
            compatibility_golden_directory=args.compatibility_golden_directory,
            compatibility_archive_descriptor=args.compatibility_archive_descriptor,
            compatibility_raw_archive=args.compatibility_raw_archive,
        )
    except (ContractError, OSError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2
    for filename, digest in sorted(publication.output_sha256.items()):
        print(f"wrote {publication.output_directory / filename} sha256={digest}")
    if publication.matched_golden:
        print("golden publication bytes match")
    return 0


def _compatibility_main(argv: List[str]) -> int:
    parser = build_compatibility_parser()
    args = parser.parse_args(argv)
    try:
        prepared = prepare_cross_commit_compatibility(
            tuple(
                CampaignEvidence(Path(run_directory), selection_id, verdict_sha256)
                for run_directory, selection_id, verdict_sha256 in args.ground_evidence
            ),
            tuple(
                CampaignEvidence(Path(run_directory), selection_id, verdict_sha256)
                for run_directory, selection_id, verdict_sha256 in args.extension_evidence
            ),
            args.output,
            regeneration_command=args.regeneration_command,
            golden_directory=args.golden_directory,
            archive_descriptor=args.archive_descriptor,
            raw_archive=args.raw_archive,
            baseline_config=args.baseline_config,
            candidate_config=args.candidate_config,
            relative_threshold_pct=args.median_threshold_pct,
            absolute_threshold_us=args.median_absolute_threshold_us,
            reviewed=args.reviewed,
        )
    except (ContractError, OSError) as exc:
        print(f"compatibility preparation failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {prepared.status} cross-commit contract {prepared.output_path} sha256={prepared.contract_sha256}")
    if prepared.status != "reviewed":
        print("candidate is non-executable; inspect it and repeat with --reviewed")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "legacy":
        return legacy_analysis.main(arguments[1:])
    if arguments and arguments[0] == "prepare-compatibility":
        return _compatibility_main(arguments[1:])
    if arguments and arguments[0] == "verify":
        arguments = arguments[1:]
    return _verified_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
