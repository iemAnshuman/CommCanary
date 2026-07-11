"""Fail-closed validation for Rostam locks, wheel provenance, and PARAM patch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..harness import (
    DEFAULT_JSON_LIMITS,
    CanonicalJSONError,
    ContractError,
    file_sha256,
    read_bounded_bytes,
    read_bounded_text,
    strict_json_loads,
)

ENVIRONMENT_SCHEMA = "commcanary.rostam.environment-contract.v1"
PARAM_PATCH_SCHEMA = "commcanary.rostam.param-patch-contract.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REVIEWED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PARAM_REPOSITORY = "https://github.com/facebookresearch/param.git"


class EnvironmentContractError(ContractError):
    """Raised before setup mutates any environment when evidence is incomplete."""


def _bounded_json(path: Path, field: str) -> Any:
    try:
        raw = read_bounded_bytes(
            path,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field=field,
        )
        return strict_json_loads(raw)
    except CanonicalJSONError as exc:
        raise EnvironmentContractError(f"cannot load {field}: {exc}") from exc


def _bounded_text(path: Path, field: str) -> str:
    try:
        return read_bounded_text(
            path,
            max_bytes=DEFAULT_JSON_LIMITS.max_document_bytes,
            field=field,
        )
    except CanonicalJSONError as exc:
        raise EnvironmentContractError(f"cannot load {field}: {exc}") from exc


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentContractError(f"{field} must be an object")
    return value


def _fields(value: Mapping[str, Any], field: str, required: Iterable[str]) -> None:
    required_set = set(required)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - required_set)
    if missing:
        raise EnvironmentContractError(f"{field} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise EnvironmentContractError(f"{field} has unknown fields: {', '.join(unknown)}")


def _sha256(value: Any, field: str, *, nullable: bool = False) -> Optional[str]:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EnvironmentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EnvironmentContractError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EnvironmentContractError(f"{field} must be a contained relative POSIX path")
    return value


def _contained(base: Path, relative: str) -> Path:
    root = base.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EnvironmentContractError(f"contract path escapes experiment directory: {relative}") from exc
    return path


def _load_environment_contract(path: Path) -> Mapping[str, Any]:
    data = _object(_bounded_json(path, "environment contract"), "environment contract")
    _fields(
        data,
        "environment contract",
        (
            "schema",
            "status",
            "reviewed_at",
            "target",
            "direct_requirement_files",
            "resolver",
            "environments",
            "commcanary_wheel",
            "collection_required",
        ),
    )
    if data["schema"] != ENVIRONMENT_SCHEMA:
        raise EnvironmentContractError(f"unsupported environment contract schema {data['schema']!r}")
    return data


def _load_patch_contract(path: Path) -> Mapping[str, Any]:
    data = _object(_bounded_json(path, "PARAM patch contract"), "PARAM patch contract")
    _fields(
        data,
        "PARAM patch contract",
        ("schema", "status", "upstream", "target", "patch", "reviewed_at", "collection_required"),
    )
    if data["schema"] != PARAM_PATCH_SCHEMA:
        raise EnvironmentContractError(f"unsupported PARAM patch contract schema {data['schema']!r}")
    return data


def audit_static_contracts(experiment_dir: Path) -> Dict[str, Any]:
    """Verify every hash knowable from the checkout without claiming readiness."""

    environment_path = experiment_dir / "constraints" / "environment-contract.json"
    patch_contract_path = experiment_dir / "patches" / "param-patch-contract.json"
    environment = _load_environment_contract(environment_path)
    patch_contract = _load_patch_contract(patch_contract_path)
    direct_rows = environment["direct_requirement_files"]
    if not isinstance(direct_rows, list) or not direct_rows:
        raise EnvironmentContractError("environment direct_requirement_files must be non-empty")
    direct_hashes = []
    for index, raw_row in enumerate(direct_rows):
        row = _object(raw_row, f"direct_requirement_files[{index}]")
        _fields(row, f"direct_requirement_files[{index}]", ("path", "sha256"))
        relative = _relative(row["path"], f"direct_requirement_files[{index}].path")
        expected = _sha256(row["sha256"], f"direct_requirement_files[{index}].sha256")
        path = _contained(experiment_dir, relative)
        if path.is_symlink() or not path.is_file():
            raise EnvironmentContractError(f"direct requirement file is missing or unsafe: {relative}")
        actual = file_sha256(path)
        if actual != expected:
            raise EnvironmentContractError(f"direct requirement hash mismatch for {relative}")
        direct_hashes.append({"path": relative, "sha256": actual})
    patch = _object(patch_contract["patch"], "PARAM patch contract.patch")
    _fields(patch, "PARAM patch contract.patch", ("path", "sha256", "apply_arguments"))
    patch_relative = _relative(patch["path"], "PARAM patch contract.patch.path")
    patch_path = _contained(experiment_dir, patch_relative)
    patch_sha256 = _sha256(patch["sha256"], "PARAM patch contract.patch.sha256")
    if patch_path.is_symlink() or not patch_path.is_file() or file_sha256(patch_path) != patch_sha256:
        raise EnvironmentContractError("committed PARAM patch does not match its declared SHA-256")
    apply_arguments = patch["apply_arguments"]
    if apply_arguments != ["--check"]:
        raise EnvironmentContractError("reviewed PARAM patch must use ordinary-context git apply --check")
    try:
        patch_text = _bounded_text(patch_path, "PARAM patch")
    except (OSError, UnicodeError) as exc:
        raise EnvironmentContractError(f"cannot read committed PARAM patch: {exc}") from exc
    if not any(line.startswith(" ") for line in patch_text.splitlines()):
        raise EnvironmentContractError("reviewed PARAM patch must contain ordinary context lines")
    upstream = _object(patch_contract["upstream"], "PARAM patch contract.upstream")
    _fields(upstream, "PARAM patch contract.upstream", ("repository", "commit", "source_archive_sha256"))
    if upstream["repository"] != _PARAM_REPOSITORY:
        raise EnvironmentContractError(f"PARAM upstream repository must be {_PARAM_REPOSITORY}")
    if not isinstance(upstream["commit"], str) or not _COMMIT_RE.fullmatch(upstream["commit"]):
        raise EnvironmentContractError("PARAM upstream commit must be a full lowercase Git SHA")
    source_archive_sha256 = _sha256(
        upstream["source_archive_sha256"],
        "PARAM patch contract.upstream.source_archive_sha256",
        nullable=True,
    )
    target = _object(patch_contract["target"], "PARAM patch contract.target")
    _fields(target, "PARAM patch contract.target", ("path", "preimage_sha256", "postimage_sha256"))
    _relative(target["path"], "PARAM patch contract.target.path")
    preimage_sha256 = _sha256(
        target["preimage_sha256"],
        "PARAM patch contract.target.preimage_sha256",
        nullable=True,
    )
    postimage_sha256 = _sha256(
        target["postimage_sha256"],
        "PARAM patch contract.target.postimage_sha256",
        nullable=True,
    )
    status = patch_contract["status"]
    reviewed_at = patch_contract["reviewed_at"]
    collection_required = patch_contract["collection_required"]
    if status not in {"pending-upstream-preimage", "reviewed"}:
        raise EnvironmentContractError(f"unsupported PARAM patch contract status {status!r}")
    if not isinstance(collection_required, list) or not all(
        isinstance(item, str) and item for item in collection_required
    ):
        raise EnvironmentContractError("PARAM patch contract.collection_required must contain non-empty strings")
    if status == "reviewed":
        if not isinstance(reviewed_at, str) or not _REVIEWED_AT_RE.fullmatch(reviewed_at):
            raise EnvironmentContractError("reviewed PARAM patch contract needs a UTC reviewed_at timestamp")
        if source_archive_sha256 is None or preimage_sha256 is None or postimage_sha256 is None:
            raise EnvironmentContractError(
                "reviewed PARAM patch contract needs archive, preimage, and postimage hashes"
            )
        if preimage_sha256 == postimage_sha256:
            raise EnvironmentContractError("reviewed PARAM patch preimage and postimage hashes must differ")
        if collection_required:
            raise EnvironmentContractError("reviewed PARAM patch contract cannot retain collection requirements")
    elif reviewed_at is not None:
        raise EnvironmentContractError("pending PARAM patch contract cannot have reviewed_at")
    return {
        "environment_status": environment["status"],
        "patch_status": status,
        "direct_requirement_files": direct_hashes,
        "param_patch_sha256": patch_sha256,
        "param_commit": upstream["commit"],
        "param_source_archive_sha256": source_archive_sha256,
        "param_target_preimage_sha256": preimage_sha256,
        "param_target_postimage_sha256": postimage_sha256,
    }


def _reviewed_lock(experiment_dir: Path, raw: Any, index: int) -> Tuple[str, Path]:
    row = _object(raw, f"environments[{index}]")
    _fields(
        row,
        f"environments[{index}]",
        ("id", "lock_path", "lock_sha256", "install_phases", "wheel_artifacts", "freeze_sha256"),
    )
    environment_id = row["id"]
    if environment_id not in {"nccl-2.19.3", "nccl-2.20.5"}:
        raise EnvironmentContractError(f"unknown environment id {environment_id!r}")
    lock_relative = _relative(row["lock_path"], f"environments[{index}].lock_path")
    expected = _sha256(row["lock_sha256"], f"environments[{index}].lock_sha256")
    path = _contained(experiment_dir, lock_relative)
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise EnvironmentContractError(f"reviewed lock is missing or stale: {lock_relative}")
    text = _bounded_text(path, f"requirement lock {lock_relative}")
    requirement_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not requirement_lines or any("--hash=sha256:" not in line for line in requirement_lines):
        raise EnvironmentContractError(f"every requirement in {lock_relative} must carry a SHA-256 hash")
    artifacts = row["wheel_artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise EnvironmentContractError(f"environment {environment_id} has no reviewed wheel inventory")
    for artifact_index, artifact_raw in enumerate(artifacts):
        artifact = _object(artifact_raw, f"environments[{index}].wheel_artifacts[{artifact_index}]")
        _fields(artifact, "wheel artifact", ("filename", "sha256", "size_bytes"))
        _sha256(artifact["sha256"], "wheel artifact.sha256")
        if not isinstance(artifact["filename"], str) or not artifact["filename"].endswith(".whl"):
            raise EnvironmentContractError("wheel artifact filename must end in .whl")
        if (
            isinstance(artifact["size_bytes"], bool)
            or not isinstance(artifact["size_bytes"], int)
            or artifact["size_bytes"] <= 0
        ):
            raise EnvironmentContractError("wheel artifact size_bytes must be positive")
    _sha256(row["freeze_sha256"], f"environments[{index}].freeze_sha256")
    return str(environment_id), path


def verify_ready_for_install(
    experiment_dir: Path,
    *,
    wheel_path: Path,
    wheel_sha256: str,
) -> Dict[str, Any]:
    audit = audit_static_contracts(experiment_dir)
    if audit["patch_status"] != "reviewed":
        raise EnvironmentContractError("PARAM patch contract is not reviewed")
    environment = _load_environment_contract(experiment_dir / "constraints" / "environment-contract.json")
    if environment["status"] != "reviewed" or not environment["reviewed_at"]:
        raise EnvironmentContractError(
            "environment contract is not reviewed; collect Rostam resolver output and artifact hashes first"
        )
    target = _object(environment["target"], "environment target")
    if not target.get("platform_tags") or not target.get("abi_tag"):
        raise EnvironmentContractError("reviewed environment contract lacks target platform/ABI evidence")
    resolver = _object(environment["resolver"], "environment resolver")
    if not resolver.get("commands") or _sha256(resolver.get("report_sha256"), "resolver.report_sha256") is None:
        raise EnvironmentContractError("reviewed environment contract lacks resolver command/report evidence")
    rows = environment["environments"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise EnvironmentContractError("environment contract must contain exactly two reviewed environments")
    locks = dict(_reviewed_lock(experiment_dir, row, index) for index, row in enumerate(rows))
    declared_wheel = _object(environment["commcanary_wheel"], "environment commcanary_wheel")
    _fields(declared_wheel, "environment commcanary_wheel", ("filename", "sha256", "repository_commit"))
    expected_wheel_sha = _sha256(declared_wheel["sha256"], "commcanary_wheel.sha256")
    if not isinstance(declared_wheel["repository_commit"], str) or not _COMMIT_RE.fullmatch(
        declared_wheel["repository_commit"]
    ):
        raise EnvironmentContractError("reviewed CommCanary wheel must declare a full repository commit")
    if _sha256(wheel_sha256, "wheel_sha256") != expected_wheel_sha:
        raise EnvironmentContractError("requested CommCanary wheel hash is not the reviewed hash")
    if wheel_path.is_symlink() or not wheel_path.is_file() or wheel_path.name != declared_wheel["filename"]:
        raise EnvironmentContractError("requested CommCanary wheel is missing, unsafe, or has the wrong filename")
    if file_sha256(wheel_path) != expected_wheel_sha:
        raise EnvironmentContractError("CommCanary wheel bytes do not match the reviewed SHA-256")
    return {
        **audit,
        "wheel_sha256": expected_wheel_sha,
        "locks": {key: str(value) for key, value in sorted(locks.items())},
    }


def verify_freeze(experiment_dir: Path, environment_id: str, freeze_path: Path) -> Dict[str, str]:
    environment = _load_environment_contract(experiment_dir / "constraints" / "environment-contract.json")
    if environment["status"] != "reviewed":
        raise EnvironmentContractError("environment contract is not reviewed")
    rows = environment["environments"]
    if not isinstance(rows, list):
        raise EnvironmentContractError("environment contract environments must be an array")
    try:
        row = next(
            _object(item, "environment") for item in rows if _object(item, "environment").get("id") == environment_id
        )
    except StopIteration as exc:
        raise EnvironmentContractError(f"unknown environment id {environment_id!r}") from exc
    expected = _sha256(row.get("freeze_sha256"), "environment.freeze_sha256")
    if freeze_path.is_symlink() or not freeze_path.is_file() or file_sha256(freeze_path) != expected:
        raise EnvironmentContractError(f"installed freeze for {environment_id} does not match reviewed evidence")
    return {"environment_id": environment_id, "freeze_sha256": expected or ""}


def _git_head(param_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(param_dir), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise EnvironmentContractError(f"cannot inspect PARAM checkout: {completed.stderr.strip()}")
    return completed.stdout.strip()


def verify_param_preimage(experiment_dir: Path, param_dir: Path) -> Dict[str, str]:
    audit_static_contracts(experiment_dir)
    contract = _load_patch_contract(experiment_dir / "patches" / "param-patch-contract.json")
    if contract["status"] != "reviewed" or not contract["reviewed_at"]:
        raise EnvironmentContractError(
            "PARAM patch contract is not reviewed; collect source archive/preimage/postimage hashes first"
        )
    upstream = _object(contract["upstream"], "PARAM patch contract.upstream")
    if _git_head(param_dir) != upstream["commit"]:
        raise EnvironmentContractError("PARAM checkout does not match the reviewed upstream commit")
    target = _object(contract["target"], "PARAM patch contract.target")
    _fields(target, "PARAM patch contract.target", ("path", "preimage_sha256", "postimage_sha256"))
    target_path = _contained(param_dir, _relative(target["path"], "PARAM patch target.path"))
    expected = _sha256(target["preimage_sha256"], "PARAM patch target.preimage_sha256")
    if target_path.is_symlink() or not target_path.is_file() or file_sha256(target_path) != expected:
        raise EnvironmentContractError("PARAM patch target does not match the reviewed preimage")
    return {"commit": upstream["commit"], "preimage_sha256": expected or "", "target": str(target_path)}


def verify_param_postimage(experiment_dir: Path, param_dir: Path) -> Dict[str, str]:
    audit_static_contracts(experiment_dir)
    contract = _load_patch_contract(experiment_dir / "patches" / "param-patch-contract.json")
    if contract["status"] != "reviewed" or not contract["reviewed_at"]:
        raise EnvironmentContractError("PARAM patch contract is not reviewed")
    target = _object(contract["target"], "PARAM patch contract.target")
    target_path = _contained(param_dir, _relative(target["path"], "PARAM patch target.path"))
    expected = _sha256(target["postimage_sha256"], "PARAM patch target.postimage_sha256")
    if target_path.is_symlink() or not target_path.is_file() or file_sha256(target_path) != expected:
        raise EnvironmentContractError("PARAM patch target does not match the reviewed postimage")
    return {"postimage_sha256": expected or "", "target": str(target_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    ready = subparsers.add_parser("verify-ready")
    ready.add_argument("--wheel", type=Path, required=True)
    ready.add_argument("--wheel-sha256", required=True)
    preimage = subparsers.add_parser("verify-param-preimage")
    preimage.add_argument("--param-dir", type=Path, required=True)
    postimage = subparsers.add_parser("verify-param-postimage")
    postimage.add_argument("--param-dir", type=Path, required=True)
    freeze = subparsers.add_parser("verify-freeze")
    freeze.add_argument("--environment-id", required=True)
    freeze.add_argument("--freeze", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            result = audit_static_contracts(args.experiment_dir)
        elif args.command == "verify-ready":
            result = verify_ready_for_install(
                args.experiment_dir,
                wheel_path=args.wheel,
                wheel_sha256=args.wheel_sha256,
            )
        elif args.command == "verify-param-preimage":
            result = verify_param_preimage(args.experiment_dir, args.param_dir)
        elif args.command == "verify-freeze":
            result = verify_freeze(args.experiment_dir, args.environment_id, args.freeze)
        else:
            result = verify_param_postimage(args.experiment_dir, args.param_dir)
    except (EnvironmentContractError, OSError, UnicodeError) as exc:
        raise SystemExit(f"environment contract error: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
