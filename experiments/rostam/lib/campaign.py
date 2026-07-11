"""Build a frozen Rostam campaign from explicit, hashed local inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..harness import (
    CAMPAIGN_SCHEMA,
    CampaignSpec,
    ContractError,
    canonical_json_bytes,
    file_sha256,
    freeze_campaign,
)
from .catalog import Catalog, CatalogValidationError, load_catalog

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_RE = re.compile(r"^([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)=(.+)$")
_EXECUTION_FILES = (
    "capture_shared_trace.sbatch",
    "lib/campaign.py",
    "lib/catalog.py",
    "lib/cell_entrypoint.py",
    "lib/common.sh",
    "lib/environment_contract.py",
    "lib/physical_results.py",
    "lib/submission.py",
    "microbench_tp8.py",
    "overlap_replay.py",
    "run_canary.sbatch",
    "run_full.sbatch",
    "run_micro.sbatch",
    "run_shared.sbatch",
    "setup.sh",
    "workload_tp8.py",
)


class CampaignPreparationError(ContractError):
    """Raised before freezing a campaign whose provenance is incomplete."""


def _input_pair(value: str) -> Tuple[str, Path]:
    match = _INPUT_RE.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("inputs must use ID=PATH with a safe lowercase ID")
    return match.group(1), Path(match.group(2)).expanduser()


def _artifact(input_id: str, path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CampaignPreparationError(f"input {input_id!r} must be a real regular file: {path}")
    return {"id": input_id, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def build_campaign(
    *,
    catalog: Catalog,
    catalog_path: Path,
    profile_id: str,
    run_id: str,
    repetitions: int,
    repository_commit: str,
    repository_dirty: bool,
    repository_patch_sha256: Optional[str],
    source_archive_sha256: Optional[str],
    inputs: Mapping[str, Path],
) -> CampaignSpec:
    if not _COMMIT_RE.fullmatch(repository_commit):
        raise CampaignPreparationError("repository_commit must be a full lowercase Git SHA")
    if repository_dirty != (repository_patch_sha256 is not None):
        raise CampaignPreparationError("repository patch SHA-256 must be present exactly for a dirty tree")
    if repository_patch_sha256 is not None and not _SHA256_RE.fullmatch(repository_patch_sha256):
        raise CampaignPreparationError("repository_patch_sha256 must be a lowercase SHA-256")
    if source_archive_sha256 is not None and not _SHA256_RE.fullmatch(source_archive_sha256):
        raise CampaignPreparationError("source_archive_sha256 must be a lowercase SHA-256")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or not 1 <= repetitions <= 1000:
        raise CampaignPreparationError("repetitions must be an integer in [1, 1000]")
    profile = catalog.profile(profile_id)
    bound_inputs = dict(inputs)
    if "rostam-catalog" in bound_inputs and bound_inputs["rostam-catalog"].resolve() != catalog_path.resolve():
        raise CampaignPreparationError("rostam-catalog input must name the catalog used to build the campaign")
    bound_inputs["rostam-catalog"] = catalog_path
    missing = sorted(set(profile.required_input_ids) - set(bound_inputs))
    unexpected = sorted(set(bound_inputs) - set(profile.required_input_ids))
    if missing or unexpected:
        raise CampaignPreparationError(
            f"profile input ownership mismatch: missing={missing!r}, unexpected={unexpected!r}"
        )
    artifacts = [_artifact(input_id, path) for input_id, path in sorted(bound_inputs.items())]
    configurations = []
    for configuration in catalog.selected_configurations(profile):
        configurations.append(
            {
                "id": configuration.id,
                "environment": configuration.environment.to_value(),
                "parameters": {"venv": configuration.venv},
                "expected_runtime": configuration.expected_runtime.to_value(),
            }
        )
    workloads = []
    for workload in catalog.selected_workloads(profile):
        parameters = workload.parameters.to_value()
        parameters.update(
            {
                "wrapper": workload.wrapper,
                "timeout_seconds": workload.timeout_seconds,
                "max_output_bytes": 67108864,
                "max_result_bytes": 4194304,
            }
        )
        workloads.append(
            {
                "id": workload.id,
                "producer_schema": workload.producer_schema,
                "measurement_schema": workload.measurement_schema,
                "parameters": parameters,
                "depends_on": list(workload.depends_on),
            }
        )
    input_paths = {input_id: str(path.resolve()) for input_id, path in sorted(bound_inputs.items())}
    experiment_directory = catalog_path.resolve().parent
    script_hashes: Dict[str, str] = {}
    for relative in _EXECUTION_FILES:
        path = experiment_directory / relative
        if path.is_symlink() or not path.is_file():
            raise CampaignPreparationError(f"execution script is missing or unsafe: {relative}")
        script_hashes[relative] = file_sha256(path)
    raw = {
        "schema": CAMPAIGN_SCHEMA,
        "run_id": run_id,
        "campaign_id": f"rostam-{profile.id}",
        "repository": {
            "commit": repository_commit,
            "dirty": repository_dirty,
            "patch_sha256": repository_patch_sha256,
            "source_archive_sha256": source_archive_sha256,
        },
        "inputs": artifacts,
        "axes": {
            "configurations": configurations,
            "workloads": workloads,
            "repetitions": repetitions,
        },
        "policy": {
            "aggregation": "median-of-cell-medians",
            "catalog_profile": profile.id,
            "cell_order": "repetition-workload-topology-configuration",
            "dependency_policy": "afterok-explicit-attempt-binding",
            "exclusion_policy": "explicit-terminal-record-only",
            "input_paths": input_paths,
            "interleave_configurations": True,
            "planner_schema": "commcanary.rostam.submission-plan.v1",
            "retry_policy": "append-only-explicit",
            "script_hashes": script_hashes,
            "tie_policy": "difference-below-either-config-iqr",
        },
        "expected_site": catalog.site.to_manifest_dict(),
    }
    return CampaignSpec.from_dict(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--repository-dirty", action="store_true")
    parser.add_argument("--repository-patch-sha256")
    parser.add_argument("--repository-patch-file", type=Path)
    parser.add_argument("--source-archive-sha256")
    parser.add_argument("--input", type=_input_pair, action="append", default=[])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--print-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.repository_dirty:
            if (
                args.repository_patch_file is None
                or args.repository_patch_file.is_symlink()
                or not args.repository_patch_file.is_file()
            ):
                raise CampaignPreparationError("a dirty repository requires --repository-patch-file")
            observed_patch_sha256 = file_sha256(args.repository_patch_file)
            if args.repository_patch_sha256 != observed_patch_sha256:
                raise CampaignPreparationError("repository patch file does not match --repository-patch-sha256")
        elif args.repository_patch_file is not None:
            raise CampaignPreparationError("a clean repository may not declare --repository-patch-file")
        input_pairs = dict(args.input)
        if len(input_pairs) != len(args.input):
            raise CampaignPreparationError("duplicate --input ownership")
        catalog = load_catalog(args.catalog)
        campaign = build_campaign(
            catalog=catalog,
            catalog_path=args.catalog,
            profile_id=args.profile,
            run_id=args.run_id,
            repetitions=args.repetitions,
            repository_commit=args.repository_commit,
            repository_dirty=args.repository_dirty,
            repository_patch_sha256=args.repository_patch_sha256,
            source_archive_sha256=args.source_archive_sha256,
            inputs=input_pairs,
        )
        if args.print_only:
            print(canonical_json_bytes(campaign.to_dict()).decode("utf-8"), end="")
            return 0
        frozen = freeze_campaign(campaign, args.results_root)
    except (CampaignPreparationError, CatalogValidationError, OSError, UnicodeError) as exc:
        raise SystemExit(f"campaign preparation error: {exc}") from exc
    print(
        json.dumps(
            {
                "run_directory": str(frozen.directory),
                "manifest_sha256": frozen.manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
