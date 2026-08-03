"""Execute exactly one manifest-owned physical cell inside a SLURM allocation."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import io
import json
import math
import os
import platform
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..harness import (
    ATTEMPT_SCHEMA,
    CELL_RESULT_SCHEMA,
    ArtifactReference,
    AttemptRecord,
    CellResult,
    ContractError,
    VerifiedArtifact,
    canonical_json_bytes,
    canonical_sha256,
    derive_attempt_id,
    file_sha256,
    load_attempt_record,
    load_cell_attempts,
    load_cell_result,
    load_frozen_run,
    read_bounded_text,
    strict_json_loads,
    utc_timestamp,
    verify_artifact_reference,
    verify_attempt_artifacts,
    write_attempt_record,
    write_cell_result,
)
from .executor_artifact import (
    EXECUTOR_ARTIFACT_INPUT_ID,
    EXECUTOR_POLICY_FORMAT,
)
from .param_artifact import (
    PARAM_RUNTIME_ARTIFACT_INPUT_ID,
    StagedParamRuntime,
    stage_param_artifact,
)
from .physical_results import (
    CAPTURE_MEASUREMENT_SCHEMA,
    PhysicalResultError,
    adapt_physical_measurement,
    load_and_validate_param_trace,
    validate_expected_runtime,
    validate_physical_layout,
)

_DEPENDENCY_RE = re.compile(r"^([^=]+)=(a-[0-9]{6})$")
_PLACEHOLDER_RE = re.compile(r"^\{([^{}]+)\}(.*)$")
_INHERITED_ENV = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "COMMCANARY_EXECUTOR_PATH",
    "COMMCANARY_EXECUTOR_SHA256",
    "SLURM_ACCOUNT",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_JOB_PARTITION",
    "SLURM_NODEID",
    "SLURM_NNODES",
    "SLURM_SUBMIT_DIR",
}
_STREAM_CHUNK_BYTES = 64 * 1024
_CAPTURE_JOIN_SECONDS = 2.0
_PROBE_TIMEOUT_SECONDS = 10
_PROBE_OUTPUT_BYTES = 64 * 1024
_MAX_GPU_COUNT = 64
_MAX_OBSERVED_TEXT_BYTES = 4096
_MAX_CPU_AFFINITY_ENTRIES = 4096
_OUTPUT_LIMIT_EXIT_CODE = 125
_WHEEL_INPUT_ID = "commcanary-wheel"
_WHEEL_MARKER_FILENAME = "commcanary-wheel.sha256"
_WHEEL_MARKER_MAX_BYTES = 256
_PARAM_CONTRACT_INPUT_ID = "param-patch-contract"
_PARAM_CONTRACT_MAX_BYTES = 1_048_576


class CellEntrypointError(ContractError):
    """Raised before or during one physical cell execution."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CellEntrypointError(f"{field} must be an object")
    return value


def _positive_integer(value: Any, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise CellEntrypointError(f"workload {field} is missing or invalid")
    return int(value)


def _dependency(value: str) -> Tuple[str, str]:
    match = _DEPENDENCY_RE.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("dependency attempts must use CELL_ID=a-NNNNNN")
    return match.group(1), match.group(2)


def _artifact_reference(run_directory: Path, path: Path) -> ArtifactReference:
    if path.is_symlink() or not path.is_file():
        raise CellEntrypointError(f"cannot reference missing or unsafe artifact: {path}")
    try:
        relative = path.resolve().relative_to(run_directory.resolve()).as_posix()
    except ValueError as exc:
        raise CellEntrypointError(f"artifact escapes frozen run: {path}") from exc
    return ArtifactReference(path=relative, sha256=file_sha256(path), size_bytes=path.stat().st_size)


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)


def _workspace(run_directory: Path, cell_id: str, attempt_id: str) -> Path:
    root = run_directory / "workspaces"
    cell_root = root / cell_id
    for parent in (root, cell_root):
        if parent.is_symlink():
            raise CellEntrypointError(f"workspace parent may not be a symlink: {parent}")
        parent.mkdir(exist_ok=True)
    workspace = cell_root / attempt_id
    try:
        workspace.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise CellEntrypointError(f"attempt workspace already exists: {workspace}") from exc
    os.chmod(workspace, 0o700)
    return workspace


def _validate_site(manifest: Any) -> Dict[str, str]:
    site = manifest.campaign.expected_site
    if site.site_id != "rostam" or site.scheduler != "slurm":
        raise CellEntrypointError("physical entrypoint requires the rostam/slurm manifest contract")
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if not job_id:
        raise CellEntrypointError("SLURM_JOB_ID is absent; refusing to run a physical cell outside its allocation")
    if partition is None or partition != site.partition:
        raise CellEntrypointError(f"SLURM partition mismatch: expected {site.partition!r}, observed {partition!r}")
    raw_nodes = os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES"))
    try:
        node_count = int(raw_nodes or "")
    except ValueError as exc:
        raise CellEntrypointError("SLURM node count is absent or invalid") from exc
    if node_count != site.nodes:
        raise CellEntrypointError(f"SLURM node-count mismatch: expected {site.nodes}, observed {node_count}")
    hostname = socket.gethostname() or "unknown-host"
    short_hostname = hostname.split(".", 1)[0]
    if site.node_constraints and short_hostname not in site.node_constraints:
        raise CellEntrypointError(
            f"allocated host {short_hostname!r} does not match manifest node constraints {site.node_constraints!r}"
        )
    account = os.environ.get("SLURM_JOB_ACCOUNT", os.environ.get("SLURM_ACCOUNT"))
    if site.account is not None and account != site.account:
        raise CellEntrypointError(f"SLURM account mismatch: expected {site.account!r}, observed {account!r}")
    return {
        "job_id": job_id,
        "partition": partition,
        "hostname": hostname,
        "account": account or "unrecorded",
    }


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while staging a verified artifact")
        view = view[written:]


def _stage_bound_input(
    source: Path,
    *,
    input_id: str,
    expected_size: int,
    expected_sha256: str,
    staging_directory: Path,
) -> Path:
    if source.is_symlink():
        raise CellEntrypointError(f"manifest input {input_id!r} is a symbolic link")
    try:
        resolved = source.resolve(strict=True)
        before_name = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise CellEntrypointError(f"manifest input {input_id!r} is missing or unsafe") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise CellEntrypointError(f"manifest input {input_id!r} cannot be opened safely") from exc
    suffix = resolved.suffix
    if not suffix or len(suffix) > 64 or not re.fullmatch(r"(?:\.[A-Za-z0-9_-]+)+", suffix):
        suffix = ".bin"
    destination = staging_directory / f"{input_id}{suffix}"
    destination_descriptor = -1
    completed = False
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or (before_name.st_dev, before_name.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise CellEntrypointError(f"manifest input {input_id!r} is not a stable regular file")
        if before.st_size != expected_size:
            raise CellEntrypointError(f"manifest input {input_id!r} has a stale size")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > expected_size:
                raise CellEntrypointError(f"manifest input {input_id!r} grew while it was staged")
            digest.update(chunk)
            _write_all(destination_descriptor, chunk)
        after = os.fstat(source_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or copied != expected_size:
            raise CellEntrypointError(f"manifest input {input_id!r} changed while it was staged")
        if digest.hexdigest() != expected_sha256:
            raise CellEntrypointError(f"manifest input {input_id!r} is stale")
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = -1
        os.chmod(destination, 0o400)
        if destination.stat().st_size != expected_size or file_sha256(destination) != expected_sha256:
            raise CellEntrypointError(f"staged manifest input {input_id!r} changed unexpectedly")
        completed = True
        return destination
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if not completed and (destination.exists() or destination.is_symlink()):
            destination.unlink()


def _verify_inputs(manifest: Any, workspace: Path) -> Dict[str, Path]:
    policy = _object(manifest.campaign.policy.to_value(), "campaign.policy")
    raw_paths = _object(policy.get("input_paths"), "campaign.policy.input_paths")
    artifacts = {artifact.id: artifact for artifact in manifest.campaign.inputs}
    if set(raw_paths) != set(artifacts):
        raise CellEntrypointError("manifest input_paths do not own exactly the declared inputs")
    staging_directory = workspace / "inputs"
    staging_directory.mkdir(mode=0o700)
    result: Dict[str, Path] = {}
    for input_id, artifact in artifacts.items():
        raw_path = raw_paths[input_id]
        if not isinstance(raw_path, str) or not raw_path:
            raise CellEntrypointError(f"manifest input path {input_id!r} is invalid")
        result[input_id] = _stage_bound_input(
            Path(raw_path),
            input_id=input_id,
            expected_size=artifact.size_bytes,
            expected_sha256=artifact.sha256,
            staging_directory=staging_directory,
        )
    return result


def _stage_verified_artifact(
    verified: VerifiedArtifact,
    *,
    staging_directory: Path,
    identity: str,
) -> Path:
    staging_directory.mkdir(mode=0o700, exist_ok=True)
    suffix = Path(verified.reference.path).suffix
    if not suffix or len(suffix) > 64 or not re.fullmatch(r"(?:\.[A-Za-z0-9_-]+)+", suffix):
        suffix = ".bin"
    name = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32] + suffix
    destination = staging_directory / name
    try:
        _write_exclusive(destination, verified.raw)
        os.chmod(destination, 0o400)
        if file_sha256(destination) != verified.reference.sha256:
            raise CellEntrypointError("private staged dependency artifact changed unexpectedly")
        return destination
    except BaseException:
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        raise


def _verify_venv_wheel_binding(venv_directory: Path, manifest: Any) -> None:
    artifacts = {artifact.id: artifact for artifact in manifest.campaign.inputs}
    bound = artifacts.get(_WHEEL_INPUT_ID)
    if bound is None:
        return
    marker = venv_directory / _WHEEL_MARKER_FILENAME
    if marker.is_symlink() or not marker.is_file():
        raise CellEntrypointError(
            "reviewed venv does not record its installed CommCanary wheel; archive it and rerun setup.sh"
        )
    recorded = read_bounded_text(marker, max_bytes=_WHEEL_MARKER_MAX_BYTES, field="venv wheel marker").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded):
        raise CellEntrypointError("venv wheel marker is not a sha256 digest")
    if recorded != bound.sha256:
        raise CellEntrypointError(
            f"venv holds CommCanary wheel {recorded}, but the manifest binds {bound.sha256}; "
            "archive the venvs and rerun setup.sh with the manifest-bound wheel"
        )


def _verify_param_postimage(param_directory: Path, input_paths: Mapping[str, Path]) -> None:
    """Refuse to execute a staged PARAM artifact without the reviewed postimage."""

    contract_path = input_paths.get(_PARAM_CONTRACT_INPUT_ID)
    if contract_path is None:
        return
    raw = read_bounded_text(contract_path, max_bytes=_PARAM_CONTRACT_MAX_BYTES, field="PARAM patch contract")
    contract = _object(strict_json_loads(raw), "PARAM patch contract")
    target = _object(contract.get("target"), "PARAM patch contract.target")
    relative_raw = target.get("path")
    postimage = target.get("postimage_sha256")
    if (
        not isinstance(relative_raw, str)
        or Path(relative_raw).is_absolute()
        or ".." in Path(relative_raw).parts
        or not isinstance(postimage, str)
        or not re.fullmatch(r"[0-9a-f]{64}", postimage)
    ):
        raise CellEntrypointError("PARAM patch contract target binding is malformed or unsafe")
    patched = param_directory / relative_raw
    if patched.is_symlink() or not patched.is_file():
        raise CellEntrypointError("reviewed PARAM checkout is missing its patched target; rerun setup.sh")
    if file_sha256(patched) != postimage:
        raise CellEntrypointError(
            "PARAM target does not match the reviewed postimage; the patch is not applied — rerun setup.sh"
        )


def _verify_execution_scripts(manifest: Any, experiment_directory: Path) -> None:
    policy = _object(manifest.campaign.policy.to_value(), "campaign.policy")
    script_hashes = _object(policy.get("script_hashes"), "campaign.policy.script_hashes")
    if not script_hashes:
        raise CellEntrypointError("manifest binds no physical execution scripts")
    for relative, expected in script_hashes.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise CellEntrypointError("manifest contains an unsafe execution-script binding")
        path = (experiment_directory / relative).resolve()
        try:
            path.relative_to(experiment_directory.resolve())
        except ValueError as exc:
            raise CellEntrypointError(f"execution script escapes experiment directory: {relative}") from exc
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise CellEntrypointError(f"manifest execution script is missing or stale: {relative}")


def _verify_running_executor(manifest: Any, experiment_directory: Path) -> Path:
    """Bind the imported entrypoint to the bootstrap's private executor copy."""

    policy = _object(manifest.campaign.policy.to_value(), "campaign.policy")
    raw_executor = policy.get("executor")
    if raw_executor is None:
        _verify_execution_scripts(manifest, experiment_directory)
        return experiment_directory.parent.parent
    executor = _object(raw_executor, "campaign.policy.executor")
    if (
        executor.get("format") != EXECUTOR_POLICY_FORMAT
        or executor.get("artifact_input_id") != EXECUTOR_ARTIFACT_INPUT_ID
    ):
        raise CellEntrypointError("manifest binds an unsupported Rostam executor")
    artifacts = {artifact.id: artifact for artifact in manifest.campaign.inputs}
    reference = artifacts.get(EXECUTOR_ARTIFACT_INPUT_ID)
    raw_path = os.environ.get("COMMCANARY_EXECUTOR_PATH")
    raw_sha256 = os.environ.get("COMMCANARY_EXECUTOR_SHA256")
    if reference is None or raw_sha256 != reference.sha256 or not raw_path:
        raise CellEntrypointError("running executor identity does not match the frozen campaign")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file() or file_sha256(path) != reference.sha256:
        raise CellEntrypointError("running executor artifact is missing or stale")
    module_path = str(Path(__file__))
    expected_prefix = f"{path}{os.sep}"
    if not module_path.startswith(expected_prefix):
        raise CellEntrypointError("cell entrypoint was imported outside the staged executor artifact")
    return path


def _dependency_artifacts(
    manifest: Any,
    run_directory: Path,
    cell: Any,
    bindings: Mapping[str, str],
    workspace: Path,
) -> Tuple[Dict[Tuple[str, str], Path], List[Dict[str, str]]]:
    if set(bindings) != set(cell.dependencies):
        raise CellEntrypointError("dependency-attempt bindings do not match the manifest cell")
    manifest_cells = {item.id: item for item in manifest.cells}
    workloads = {item.id: item for item in manifest.campaign.workloads}
    paths: Dict[Tuple[str, str], Path] = {}
    evidence: List[Dict[str, str]] = []
    for dependency_cell_id, attempt_id in sorted(bindings.items()):
        record, stored = load_attempt_record(run_directory, dependency_cell_id, attempt_id)
        if record.status != "success" or record.measurement is None:
            raise CellEntrypointError(f"dependency {dependency_cell_id}/{attempt_id} is not successful")
        verify_attempt_artifacts(run_directory, record)
        dependency_cell = manifest_cells[dependency_cell_id]
        dependency_workload = workloads[dependency_cell.workload_id]
        result = load_cell_result(
            verify_artifact_reference(run_directory, record.measurement).raw,
            cell_id=dependency_cell_id,
            cell_identity_sha256=dependency_cell.identity_sha256,
            producer_schema=dependency_workload.producer_schema,
            measurement_schema=dependency_workload.measurement_schema,
            max_bytes=max(1, record.measurement.size_bytes),
        )
        measurement = _object(result.measurement.to_value(), "dependency measurement")
        artifacts = _object(measurement.get("artifacts"), "dependency measurement.artifacts")
        for artifact_id, raw_reference in artifacts.items():
            reference = ArtifactReference.from_dict(raw_reference, f"dependency artifact {artifact_id}")
            verified = verify_artifact_reference(run_directory, reference)
            paths[(dependency_workload.id, str(artifact_id))] = _stage_verified_artifact(
                verified,
                staging_directory=workspace / "dependencies",
                identity=f"{dependency_cell_id}:{attempt_id}:{artifact_id}",
            )
        evidence.append(
            {
                "cell_id": dependency_cell_id,
                "attempt_id": attempt_id,
                "attempt_record_sha256": stored.record_sha256,
            }
        )
    return paths, evidence


def _resolve_argument(
    value: str,
    *,
    repetition: int,
    workspace: Path,
    experiment_directory: Path,
    venv_directory: Path,
    dependency_paths: Mapping[Tuple[str, str], Path],
    input_paths: Mapping[str, Path],
    safe_python: Optional[Path] = None,
    param_runtime: Optional[Path] = None,
) -> str:
    match = _PLACEHOLDER_RE.fullmatch(value)
    if match is None:
        if "{" in value or "}" in value:
            raise CellEntrypointError(f"unsupported command placeholder in {value!r}")
        return value
    token, suffix = match.groups()
    if token == "repetition":
        if suffix:
            raise CellEntrypointError("repetition placeholder cannot have a path suffix")
        return str(repetition)
    if token == "workspace":
        base = workspace
    elif token == "experiment_dir":
        base = experiment_directory
    elif token == "venv_python":
        base = venv_directory / "bin" / "python"
    elif token == "venv_bin":
        base = venv_directory / "bin"
    elif token == "safe_python":
        if safe_python is None:
            raise CellEntrypointError("safe Python runner is not available")
        base = safe_python
    elif token == "param_runtime":
        if param_runtime is None:
            raise CellEntrypointError("PARAM runtime artifact is not available")
        base = param_runtime
    elif token.startswith("dependency:"):
        parts = token.split(":")
        if len(parts) != 3 or (parts[1], parts[2]) not in dependency_paths:
            raise CellEntrypointError(f"unbound dependency artifact placeholder {token!r}")
        base = dependency_paths[(parts[1], parts[2])]
    elif token.startswith("input:"):
        input_id = token.split(":", 1)[1]
        if input_id not in input_paths:
            raise CellEntrypointError(f"unbound input placeholder {token!r}")
        base = input_paths[input_id]
    else:
        raise CellEntrypointError(f"unsupported command placeholder {token!r}")
    if suffix and not suffix.startswith("/"):
        raise CellEntrypointError("placeholder suffix must be a path component")
    return str(base) + suffix


def _commands(
    parameters: Mapping[str, Any],
    **resolution: Any,
) -> Tuple[Tuple[str, ...], ...]:
    raw_commands: List[Any] = []
    if parameters.get("adapter") == "capture":
        if "profile_command" not in parameters or "transform_commands" not in parameters:
            raise CellEntrypointError("capture workload lacks a fully manifest-bound pipeline")
        raw_commands.append(parameters["profile_command"])
        transforms = parameters["transform_commands"]
        if not isinstance(transforms, list):
            raise CellEntrypointError("capture transform_commands must be an array")
        raw_commands.extend(transforms)
    else:
        raw_commands.append(parameters.get("command"))
    result = []
    for command_index, raw in enumerate(raw_commands):
        if not isinstance(raw, list) or not raw:
            raise CellEntrypointError(f"physical command {command_index} must be a non-empty argv array")
        resolved = []
        for index, item in enumerate(raw):
            if not isinstance(item, str) or not item or "\x00" in item:
                raise CellEntrypointError(f"physical command {command_index} argument {index} is invalid")
            resolved.append(_resolve_argument(item, **resolution))
        result.append(tuple(resolved))
    return tuple(result)


def _find_trace_path(commands: Sequence[Sequence[str]]) -> Optional[Path]:
    trace: Optional[Path] = None
    for command in commands:
        for index, argument in enumerate(command):
            if argument == "--trace-path":
                if index + 1 >= len(command):
                    raise CellEntrypointError("--trace-path has no value")
                candidate = Path(command[index + 1])
                if trace is not None and candidate.resolve() != trace.resolve():
                    raise CellEntrypointError("cell resolves more than one replay trace")
                trace = candidate
    return trace


def _stage_bound_nccl_library(
    configuration: Any,
    input_paths: Mapping[str, Path],
    workspace: Path,
) -> Path:
    expected = _object(configuration.expected_runtime.to_value(), "configuration.expected_runtime")
    version = expected.get("nccl_version")
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version):
        raise CellEntrypointError("configuration has no valid NCCL library version binding")
    input_id = f"nccl-library-{version.replace('.', '-')}"
    source = input_paths.get(input_id)
    if source is None or source.is_symlink() or not source.is_file():
        raise CellEntrypointError(f"configuration lacks manifest-bound NCCL bytes {input_id!r}")
    destination_directory = workspace / "runtime-libraries"
    destination_directory.mkdir(mode=0o700)
    destination = destination_directory / "libnccl.so.2"
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as exc:
        raise CellEntrypointError("cannot privately stage the manifest-bound NCCL library") from exc
    os.chmod(destination, 0o400)
    if file_sha256(destination) != file_sha256(source):
        raise CellEntrypointError("private NCCL library bytes changed while they were staged")
    return destination


def _site_packages(venv_directory: Path) -> Path:
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    path = venv_directory / "lib" / python_version / "site-packages"
    if path.is_symlink() or not path.is_dir():
        raise CellEntrypointError(f"reviewed venv has no safe site-packages directory: {path}")
    return path.resolve()


def _prepare_safe_python(
    workspace: Path,
    configured_python: Path,
    executor_artifact: Path,
) -> Path:
    destination = workspace / "isolated-python"
    script = (
        "#!/bin/sh\n"
        f"exec {shlex.quote(str(configured_python))} -I -S {shlex.quote(str(executor_artifact))} "
        'run-python "$@"\n'
    ).encode("utf-8")
    _write_exclusive(destination, script)
    os.chmod(destination, 0o500)
    return destination


def _runtime_environment(
    configuration: Any,
    venv_directory: Path,
    nccl_library: Path,
    executor_artifact: Path,
) -> Dict[str, str]:
    result = {key: value for key, value in os.environ.items() if key in _INHERITED_ENV}
    configuration_environment = _object(configuration.environment.to_value(), "configuration.environment")
    for key, value in configuration_environment.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CellEntrypointError("configuration environment must contain strings")
        result[key] = value
    result["LD_LIBRARY_PATH"] = str(nccl_library.parent)
    result["COMMCANARY_EXECUTOR_PATH"] = str(executor_artifact)
    result["COMMCANARY_EXECUTOR_SHA256"] = file_sha256(executor_artifact)
    result["COMMCANARY_CHILD_IMPORT_PATHS"] = str(_site_packages(venv_directory))
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONSAFEPATH"] = "1"
    result.pop("PYTHONHOME", None)
    result.pop("PYTHONPATH", None)
    return result


def _command_environments(
    base: Mapping[str, str],
    commands: Sequence[Sequence[str]],
    param_runtime: Optional[StagedParamRuntime],
) -> Tuple[Dict[str, str], ...]:
    results = []
    param_root = None if param_runtime is None else param_runtime.root.resolve()
    for command in commands:
        environment = dict(base)
        uses_param = False
        if param_root is not None:
            for argument in command:
                candidate = Path(argument)
                if not candidate.is_absolute():
                    continue
                try:
                    candidate.resolve().relative_to(param_root)
                except (OSError, ValueError):
                    continue
                uses_param = True
                break
        if uses_param and param_runtime is not None:
            existing = environment["COMMCANARY_CHILD_IMPORT_PATHS"].split(os.pathsep)
            environment["COMMCANARY_CHILD_IMPORT_PATHS"] = os.pathsep.join(
                (*existing, *(str(path) for path in param_runtime.import_paths))
            )
            environment["COMMCANARY_ALLOWED_SCRIPT_ROOT"] = str(param_runtime.root)
        else:
            environment.pop("COMMCANARY_ALLOWED_SCRIPT_ROOT", None)
        results.append(environment)
    return tuple(results)


def _capture_bytes(
    stream: IO[bytes],
    destination: bytearray,
    limit: int,
    exceeded: threading.Event,
    failures: List[str],
) -> None:
    try:
        while True:
            chunk = stream.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            remaining = max(0, limit - len(destination))
            destination.extend(chunk[:remaining])
            if len(chunk) > remaining:
                exceeded.set()
    except Exception as exc:  # thread boundary; normalized by the caller
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _capture_file(
    stream: IO[bytes],
    destination: Path,
    limit: int,
    exceeded: threading.Event,
    failures: List[str],
) -> None:
    try:
        written = destination.stat().st_size
        if written > limit:
            exceeded.set()
            written = limit
        with destination.open("ab") as handle:
            while True:
                chunk = stream.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                remaining = max(0, limit - written)
                accepted = chunk[:remaining]
                if accepted:
                    handle.write(accepted)
                    written += len(accepted)
                if len(chunk) > remaining:
                    exceeded.set()
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:  # thread boundary; normalized by the caller
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _truncate_overflow(path: Path, limit: int) -> bool:
    """Enforce the persisted cap even if a child wrote the log path directly."""

    if path.stat().st_size <= limit:
        return False
    with path.open("r+b") as handle:
        handle.truncate(limit)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass


def _join_capture_threads(
    process: subprocess.Popen[Any],
    threads: Sequence[threading.Thread],
    failures: List[str],
) -> None:
    for thread in threads:
        thread.join(timeout=_CAPTURE_JOIN_SECONDS)
    if any(thread.is_alive() for thread in threads):
        _terminate(process)
        for thread in threads:
            thread.join(timeout=_CAPTURE_JOIN_SECONDS)
    if any(thread.is_alive() for thread in threads):
        failures.append("output capture thread did not terminate")


def _run_bounded_probe(
    command: Sequence[str],
    *,
    timeout_seconds: int = _PROBE_TIMEOUT_SECONDS,
    max_output_bytes: int = _PROBE_OUTPUT_BYTES,
) -> str:
    """Run one metadata probe with bounded time and per-stream memory."""

    if not command or any(not isinstance(part, str) or not part for part in command):
        raise CellEntrypointError("runtime probe command is invalid")
    timeout = _positive_integer(timeout_seconds, "probe timeout_seconds", maximum=60)
    output_limit = _positive_integer(max_output_bytes, "probe max_output_bytes", maximum=1024 * 1024)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise CellEntrypointError(f"runtime probe {command[0]!r} could not start: {type(exc).__name__}") from exc
    if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen contract
        _terminate(process)
        raise CellEntrypointError("runtime probe pipes were not created")
    stdout = bytearray()
    stderr = bytearray()
    exceeded = threading.Event()
    failures: List[str] = []
    threads = (
        threading.Thread(
            target=_capture_bytes,
            args=(process.stdout, stdout, output_limit, exceeded, failures),
            daemon=True,
        ),
        threading.Thread(
            target=_capture_bytes,
            args=(process.stderr, stderr, output_limit, exceeded, failures),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()
    started = time.monotonic()
    timed_out = False
    try:
        while process.poll() is None:
            if exceeded.is_set():
                _terminate(process)
                break
            if time.monotonic() - started >= timeout:
                timed_out = True
                _terminate(process)
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        _terminate(process)
        _join_capture_threads(process, threads, failures)
        raise CellEntrypointError(f"runtime probe {command[0]!r} was interrupted")
    return_code = process.wait()
    _join_capture_threads(process, threads, failures)
    if failures:
        raise CellEntrypointError(f"runtime probe output capture failed: {'; '.join(failures)}")
    if timed_out:
        raise CellEntrypointError(f"runtime probe {command[0]!r} exceeded {timeout} seconds")
    if exceeded.is_set():
        raise CellEntrypointError(f"runtime probe {command[0]!r} exceeded {output_limit} bytes per stream")
    if return_code != 0:
        raise CellEntrypointError(f"runtime probe {command[0]!r} exited {return_code}")
    try:
        return bytes(stdout).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CellEntrypointError(f"runtime probe {command[0]!r} returned non-UTF-8 output") from exc


def _observed_text(value: str, field: str, *, maximum: int = _MAX_OBSERVED_TEXT_BYTES) -> str:
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise CellEntrypointError(f"runtime observation {field} is not valid UTF-8") from exc
    if not value or "\x00" in value or size > maximum:
        raise CellEntrypointError(f"runtime observation {field} is empty, unsafe, or oversized")
    return value


def _gpu_inventory(raw: str) -> Tuple[Dict[str, Any], ...]:
    try:
        rows = list(csv.reader(io.StringIO(raw), strict=True))
    except csv.Error as exc:
        raise CellEntrypointError("nvidia-smi GPU inventory is not valid CSV") from exc
    if not rows or len(rows) > _MAX_GPU_COUNT:
        raise CellEntrypointError(f"nvidia-smi reported an invalid GPU count; maximum is {_MAX_GPU_COUNT}")
    result: List[Dict[str, Any]] = []
    observed_indices = set()
    for row_index, row in enumerate(rows):
        if len(row) != 12:
            raise CellEntrypointError(f"nvidia-smi GPU inventory row {row_index} has {len(row)} fields")
        fields = [item.strip() for item in row]
        if any(len(fields[index]) > 32 for index in (0, 7, 8, 9, 10, 11)):
            raise CellEntrypointError(f"nvidia-smi GPU inventory row {row_index} has oversized numbers")
        try:
            index = int(fields[0])
            temperature_c = int(fields[7])
            power_draw_w = float(fields[8])
            power_limit_w = float(fields[9])
            sm_clock_mhz = int(fields[10])
            memory_clock_mhz = int(fields[11])
        except ValueError as exc:
            raise CellEntrypointError(f"nvidia-smi GPU inventory row {row_index} has invalid numbers") from exc
        if not 0 <= index < _MAX_GPU_COUNT or index in observed_indices:
            raise CellEntrypointError("nvidia-smi GPU indices are duplicate or outside the supported range")
        if (
            not -50 <= temperature_c <= 200
            or not math.isfinite(power_draw_w)
            or power_draw_w < 0.0
            or not math.isfinite(power_limit_w)
            or power_limit_w <= 0.0
            or sm_clock_mhz < 0
            or memory_clock_mhz < 0
        ):
            raise CellEntrypointError("nvidia-smi GPU telemetry is outside the supported range")
        observed_indices.add(index)
        result.append(
            {
                "index": index,
                "uuid": _observed_text(fields[1], f"gpus[{row_index}].uuid", maximum=256),
                "name": _observed_text(fields[2], f"gpus[{row_index}].name", maximum=256),
                "driver_version": _observed_text(fields[3], f"gpus[{row_index}].driver_version", maximum=128),
                "pci_bus_id": _observed_text(fields[4], f"gpus[{row_index}].pci_bus_id", maximum=128),
                "persistence_mode": _observed_text(fields[5], f"gpus[{row_index}].persistence_mode", maximum=64),
                "performance_state": _observed_text(fields[6], f"gpus[{row_index}].performance_state", maximum=64),
                "temperature_c": temperature_c,
                "power_draw_w": power_draw_w,
                "power_limit_w": power_limit_w,
                "sm_clock_mhz": sm_clock_mhz,
                "memory_clock_mhz": memory_clock_mhz,
            }
        )
    return tuple(sorted(result, key=lambda row: int(row["index"])))


def _binding_observation() -> Dict[str, Any]:
    environment: Dict[str, Optional[str]] = {}
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_LOCALID",
        "SLURM_PROCID",
        "SLURM_NODEID",
        "SLURM_CPUS_PER_TASK",
        "OMP_NUM_THREADS",
    ):
        value = os.environ.get(key)
        environment[key] = None if value is None else _observed_text(value, f"binding.{key}")
    affinity_function = getattr(os, "sched_getaffinity", None)
    affinity: Optional[List[int]] = None
    if affinity_function is not None:
        try:
            affinity = sorted(int(cpu) for cpu in affinity_function(0))
        except (OSError, TypeError, ValueError) as exc:
            raise CellEntrypointError("cannot observe process CPU affinity") from exc
        if len(affinity) > _MAX_CPU_AFFINITY_ENTRIES or any(cpu < 0 for cpu in affinity):
            raise CellEntrypointError("observed process CPU affinity is invalid or oversized")
    return {
        "environment": environment,
        "cpu_affinity": affinity,
        "cpu_affinity_method": "sched_getaffinity" if affinity_function is not None else "unavailable",
    }


def _runtime_fingerprint(
    nccl_library: Path,
    site_observation: Mapping[str, str],
    *,
    site_packages: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if site_packages is not None:
        if site_packages.is_symlink() or not site_packages.is_dir():
            raise CellEntrypointError("reviewed site-packages directory is missing or unsafe")
        site_text = str(site_packages.resolve())
        if site_text not in sys.path:
            sys.path.append(site_text)
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise CellEntrypointError("reviewed venv cannot import torch") from exc
    try:
        library = ctypes.CDLL(str(nccl_library))
    except OSError as exc:
        raise CellEntrypointError("cannot load the manifest-selected NCCL library") from exc
    value = ctypes.c_int()
    try:
        nccl_status = library.ncclGetVersion(ctypes.byref(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise CellEntrypointError("manifest-selected NCCL library has no usable ncclGetVersion") from exc
    if nccl_status != 0:
        raise CellEntrypointError("ncclGetVersion failed for the manifest-selected NCCL library")
    torch_version_raw = getattr(torch, "__version__", None)
    torch_file_raw = getattr(torch, "__file__", None)
    if site_packages is not None:
        if not isinstance(torch_file_raw, str):
            raise CellEntrypointError("reviewed torch module has no source identity")
        try:
            Path(torch_file_raw).resolve().relative_to(site_packages.resolve())
        except ValueError as exc:
            raise CellEntrypointError("torch was imported outside the reviewed site-packages directory") from exc
    if torch_version_raw is None:
        raise CellEntrypointError("reviewed torch module has no version")
    torch_version = _observed_text(str(torch_version_raw).split("+", 1)[0], "torch_version", maximum=128)
    torch_runtime = getattr(torch, "version", None)
    cuda_raw = None if torch_runtime is None else getattr(torch_runtime, "cuda", None)
    torch_cuda_version = None if cuda_raw is None else _observed_text(str(cuda_raw), "torch_cuda_version", maximum=64)
    hostname = site_observation.get("hostname")
    job_id = site_observation.get("job_id")
    if not isinstance(hostname, str) or not isinstance(job_id, str):
        raise CellEntrypointError("site observation lacks hostname or job_id")
    runtime = {
        "hostname": _observed_text(hostname, "hostname", maximum=256),
        "job_id": _observed_text(job_id, "job_id", maximum=256),
        "python_version": _observed_text(platform.python_version(), "python_version", maximum=64),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "runtime_nccl_version_code": int(value.value),
    }
    inventory_command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,pci.bus_id,persistence_mode,pstate,temperature.gpu,"
        "power.draw,power.limit,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    )
    gpus = _gpu_inventory(_run_bounded_probe(inventory_command))
    driver_versions = sorted({str(gpu["driver_version"]) for gpu in gpus})
    if len(driver_versions) != 1:
        raise CellEntrypointError("nvidia-smi reported inconsistent driver versions across visible GPUs")
    topology = _observed_text(
        _run_bounded_probe(("nvidia-smi", "topo", "-m")).replace("\r\n", "\n").rstrip("\n"),
        "topology",
        maximum=_PROBE_OUTPUT_BYTES,
    )
    node_name = hostname.split(".", 1)[0]
    node_state = _observed_text(
        _run_bounded_probe(("scontrol", "show", "node", "--oneliner", node_name)).replace("\r\n", "\n").rstrip("\n"),
        "node_state",
        maximum=_PROBE_OUTPUT_BYTES,
    )
    evidence = {
        "schema": "commcanary.rostam.runtime-observation.v2",
        "runtime": dict(runtime),
        "nccl_library_sha256": file_sha256(nccl_library),
        "driver_version": driver_versions[0],
        "gpu_count": len(gpus),
        "gpus": list(gpus),
        "topology": {
            "method": "nvidia-smi topo -m",
            "text": topology,
        },
        "node_state": {
            "method": "scontrol show node --oneliner HOSTNAME",
            "text": node_state,
        },
        "binding": _binding_observation(),
        "probe_policy": {
            "timeout_seconds": _PROBE_TIMEOUT_SECONDS,
            "max_output_bytes_per_stream": _PROBE_OUTPUT_BYTES,
        },
    }
    return runtime, evidence


def _run_pipeline(
    commands: Sequence[Sequence[str]],
    *,
    workspace: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    max_output_bytes: int,
    command_environments: Optional[Sequence[Mapping[str, str]]] = None,
) -> Tuple[int, float, Optional[str], bool]:
    if command_environments is not None and len(command_environments) != len(commands):
        raise CellEntrypointError("physical command environment inventory is incomplete")
    started = time.monotonic()
    stdout_path.touch(exist_ok=False)
    stderr_path.touch(exist_ok=False)
    for command_index, command in enumerate(commands):
        marker = f"\n[commcanary physical step {command_index}] {json.dumps(list(command))}\n".encode("utf-8")
        current_stderr = stderr_path.stat().st_size
        remaining = max(0, max_output_bytes - current_stderr)
        with stderr_path.open("ab") as stderr_handle:
            stderr_handle.write(marker[:remaining])
            stderr_handle.flush()
            os.fsync(stderr_handle.fileno())
        if len(marker) > remaining:
            elapsed = time.monotonic() - started
            return (
                _OUTPUT_LIMIT_EXIT_CODE,
                elapsed,
                f"stdout or stderr exceeded {max_output_bytes} bytes",
                True,
            )
        try:
            process = subprocess.Popen(
                command,
                cwd=str(workspace),
                env=dict(environment if command_environments is None else command_environments[command_index]),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            return 127, time.monotonic() - started, f"cannot start step {command_index}: {exc}", False
        if process.stdout is None or process.stderr is None:  # pragma: no cover - Popen contract
            _terminate(process)
            return 126, time.monotonic() - started, "physical output pipes were not created", False
        exceeded = threading.Event()
        failures: List[str] = []
        threads = (
            threading.Thread(
                target=_capture_file,
                args=(process.stdout, stdout_path, max_output_bytes, exceeded, failures),
                daemon=True,
            ),
            threading.Thread(
                target=_capture_file,
                args=(process.stderr, stderr_path, max_output_bytes, exceeded, failures),
                daemon=True,
            ),
        )
        for thread in threads:
            thread.start()
        termination_reason: Optional[str] = None
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if exceeded.is_set():
                    termination_reason = f"stdout or stderr exceeded {max_output_bytes} bytes"
                    _terminate(process)
                    break
                if stdout_path.stat().st_size > max_output_bytes or stderr_path.stat().st_size > max_output_bytes:
                    exceeded.set()
                    termination_reason = f"stdout or stderr exceeded {max_output_bytes} bytes"
                    _terminate(process)
                    break
                if elapsed >= timeout_seconds:
                    termination_reason = f"pipeline exceeded {timeout_seconds} seconds"
                    _terminate(process)
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            _terminate(process)
            _join_capture_threads(process, threads, failures)
            return process.wait(), time.monotonic() - started, "execution interrupted", exceeded.is_set()
        return_code = process.wait()
        _join_capture_threads(process, threads, failures)
        if _truncate_overflow(stdout_path, max_output_bytes) or _truncate_overflow(stderr_path, max_output_bytes):
            exceeded.set()
        elapsed = time.monotonic() - started
        if failures:
            return 126, elapsed, f"output capture failed: {'; '.join(failures)}", exceeded.is_set()
        if exceeded.is_set():
            return (
                return_code if return_code != 0 else _OUTPUT_LIMIT_EXIT_CODE,
                elapsed,
                f"stdout or stderr exceeded {max_output_bytes} bytes",
                True,
            )
        if termination_reason is not None:
            return return_code, elapsed, termination_reason, False
        if return_code != 0:
            return return_code, elapsed, f"step {command_index} exited {return_code}", False
    return 0, time.monotonic() - started, None, False


def _capture_artifacts(
    parameters: Mapping[str, Any],
    *,
    resolution: Mapping[str, Any],
    run_directory: Path,
    workspace: Path,
) -> Tuple[Dict[str, Any], Tuple[ArtifactReference, ...]]:
    raw_outputs = _object(parameters.get("outputs"), "capture outputs")
    if not raw_outputs:
        raise CellEntrypointError("capture workload declares no named outputs")
    measurement: Dict[str, Any] = {}
    references: List[ArtifactReference] = []
    for output_id, raw_path in sorted(raw_outputs.items()):
        if not isinstance(output_id, str) or not isinstance(raw_path, str):
            raise CellEntrypointError("capture output map must contain string keys and paths")
        path = Path(_resolve_argument(raw_path, **resolution))
        try:
            path.resolve().relative_to(workspace.resolve())
        except ValueError as exc:
            raise CellEntrypointError(f"capture output {output_id!r} escapes its workspace") from exc
        reference = _artifact_reference(run_directory, path)
        references.append(reference)
        measurement[output_id] = reference.to_dict()
        os.chmod(path, 0o444)
    return measurement, tuple(sorted(references, key=lambda item: item.path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-wrapper",
        choices=tuple(sorted({"micro", "full", "canary", "qualification", "shared-capture", "shared"})),
        required=True,
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--dependency-attempt", type=_dependency, action="append", default=[])
    return parser


def _record_preflight_failure(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    failure: BaseException,
) -> None:
    """Preserve a scheduler-started cell whose fail-early check rejected it."""

    manifest, frozen = load_frozen_run(args.run_directory)
    if frozen.manifest_sha256 != args.manifest_sha256:
        raise CellEntrypointError("cannot bind preflight failure to a mismatched manifest")
    cells = {cell.id: cell for cell in manifest.cells}
    if args.cell_id not in cells:
        raise CellEntrypointError("cannot bind preflight failure to an unknown cell")
    cell = cells[args.cell_id]
    attempts = load_cell_attempts(frozen.directory, cell.id)
    if any(attempt.attempt_id == args.attempt_id for attempt in attempts):
        return
    expected_attempt_id = derive_attempt_id(len(attempts) + 1)
    if args.attempt_id != expected_attempt_id:
        raise CellEntrypointError("cannot record a preflight failure for stale attempt ownership")
    workspace = frozen.directory / "workspaces" / cell.id / args.attempt_id
    if workspace.is_symlink():
        raise CellEntrypointError("cannot preserve failure in a symlinked workspace")
    if not workspace.exists():
        workspace = _workspace(frozen.directory, cell.id, args.attempt_id)
    elif not workspace.is_dir():
        raise CellEntrypointError("attempt workspace is not a directory")
    stdout_path = workspace / "stdout.log"
    stderr_path = workspace / "stderr.log"
    if not stdout_path.exists():
        _write_exclusive(stdout_path, b"")
    if not stderr_path.exists():
        _write_exclusive(
            stderr_path,
            (f"physical cell preflight failed: {type(failure).__name__}: {failure}\n").encode("utf-8")[: 1024 * 1024],
        )
    reason = f"physical cell preflight failed: {type(failure).__name__}: {failure}"
    reason = reason.replace("\x00", "\\0")[:4096]
    partial_outputs: List[ArtifactReference] = []
    for path in sorted(workspace.iterdir(), key=lambda item: item.name):
        if path.name in {"stdout.log", "stderr.log", "execution_plan.json"}:
            continue
        if path.is_file() and not path.is_symlink():
            partial_outputs.append(_artifact_reference(frozen.directory, path))
    hostname = (socket.gethostname() or "unknown-host").split(".", 1)[0]
    job_id = os.environ.get("SLURM_JOB_ID")
    scheduler = "slurm" if job_id else None
    timestamp = utc_timestamp()
    command = [sys.executable, "-m", "experiments.rostam.lib.cell_entrypoint", *raw_argv]
    failure_identity = canonical_sha256(
        {
            "manifest_sha256": frozen.manifest_sha256,
            "cell_id": cell.id,
            "attempt_id": args.attempt_id,
            "command": command,
            "failure": reason,
        }
    )
    record = AttemptRecord.from_dict(
        {
            "schema": ATTEMPT_SCHEMA,
            "run_id": manifest.run_id,
            "manifest_sha256": frozen.manifest_sha256,
            "cell_id": cell.id,
            "cell_identity_sha256": cell.identity_sha256,
            "attempt_id": args.attempt_id,
            "attempt_number": len(attempts) + 1,
            "status": "parse-failed",
            "started_at": timestamp,
            "finished_at": timestamp,
            "command": command,
            "observed": {
                "executor": "slurm-preflight",
                "site_id": "rostam",
                "hostname": hostname,
                "scheduler": scheduler,
                "job_id": job_id,
                "nodes": [hostname],
                "account": os.environ.get("SLURM_JOB_ACCOUNT"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "metadata": {
                    "environment_sha256": canonical_sha256(
                        {key: value for key, value in os.environ.items() if key in _INHERITED_ENV}
                    ),
                    "execution_identity_sha256": failure_identity,
                    "execution_plan_sha256": failure_identity,
                    "failure_phase": "preflight",
                    "result_schema": CELL_RESULT_SCHEMA,
                },
            },
            "exit_code": None,
            "reason": reason,
            "stdout": _artifact_reference(frozen.directory, stdout_path).to_dict(),
            "stderr": _artifact_reference(frozen.directory, stderr_path).to_dict(),
            "measurement": None,
            "partial_outputs": [reference.to_dict() for reference in partial_outputs],
        }
    )
    write_attempt_record(frozen.directory, record)
    verify_attempt_artifacts(frozen.directory, record)


def run(args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    manifest, frozen = load_frozen_run(args.run_directory)
    if frozen.manifest_sha256 != args.manifest_sha256:
        raise CellEntrypointError("manifest checksum differs from the submitted ownership plan")
    site = _validate_site(manifest)
    cells = {cell.id: cell for cell in manifest.cells}
    if args.cell_id not in cells:
        raise CellEntrypointError(f"unknown manifest cell {args.cell_id!r}")
    cell = cells[args.cell_id]
    attempts = load_cell_attempts(frozen.directory, cell.id)
    expected_attempt_id = derive_attempt_id(len(attempts) + 1)
    if args.attempt_id != expected_attempt_id:
        raise CellEntrypointError(
            f"submitted attempt ownership is stale: expected {expected_attempt_id}, got {args.attempt_id}"
        )
    dependencies = dict(args.dependency_attempt)
    if len(dependencies) != len(args.dependency_attempt):
        raise CellEntrypointError("duplicate dependency-attempt ownership")
    configurations = {item.id: item for item in manifest.campaign.configurations}
    workloads = {item.id: item for item in manifest.campaign.workloads}
    configuration = configurations[cell.configuration_id]
    workload = workloads[cell.workload_id]
    parameters = _object(workload.parameters.to_value(), "workload.parameters")
    validate_physical_layout(parameters)
    if parameters.get("readiness", "ready") != "ready":
        raise CellEntrypointError(f"workload is not target-ready: {parameters.get('readiness')}")
    if parameters.get("wrapper") != args.site_wrapper:
        raise CellEntrypointError("spooled wrapper identity does not own this manifest workload")
    max_output_bytes = _positive_integer(
        parameters.get("max_output_bytes"),
        "max_output_bytes",
        maximum=1024**3,
    )
    max_result_bytes = _positive_integer(
        parameters.get("max_result_bytes"),
        "max_result_bytes",
        maximum=1024**3,
    )
    timeout_seconds = _positive_integer(
        parameters.get("timeout_seconds"),
        "timeout_seconds",
        maximum=86_400,
    )
    workspace = _workspace(frozen.directory, cell.id, args.attempt_id)
    stdout_path = workspace / "stdout.log"
    stderr_path = workspace / "stderr.log"
    result_path = workspace / "result.json"
    input_paths = _verify_inputs(manifest, workspace)
    dependency_paths, dependency_evidence = _dependency_artifacts(
        manifest,
        frozen.directory,
        cell,
        dependencies,
        workspace,
    )
    experiment_directory = Path(os.environ["COMMCANARY_EXPERIMENT_DIR"]).resolve()
    executor_artifact = _verify_running_executor(manifest, experiment_directory)
    repository_root = experiment_directory.parent.parent
    configuration_parameters = _object(configuration.parameters.to_value(), "configuration.parameters")
    venv_raw = configuration_parameters.get("venv")
    if not isinstance(venv_raw, str) or Path(venv_raw).is_absolute() or ".." in Path(venv_raw).parts:
        raise CellEntrypointError("configuration venv path is invalid")
    venv_directory = (repository_root / venv_raw).resolve()
    configured_python = venv_directory / "bin" / "python"
    if not configured_python.exists() or Path(sys.prefix).resolve() != venv_directory:
        raise CellEntrypointError("cell entrypoint is not running from the manifest-selected venv")
    _verify_venv_wheel_binding(venv_directory, manifest)
    safe_python = _prepare_safe_python(workspace, configured_python, executor_artifact)
    param_runtime: Optional[StagedParamRuntime] = None
    param_artifact = input_paths.get(PARAM_RUNTIME_ARTIFACT_INPUT_ID)
    if param_artifact is not None:
        try:
            param_runtime = stage_param_artifact(param_artifact, workspace / "param-pythonpath")
        except RuntimeError as exc:
            raise CellEntrypointError(f"cannot stage the manifest-bound PARAM runtime: {exc}") from exc
        _verify_param_postimage(param_runtime.root, input_paths)
    resolution = {
        "repetition": cell.repetition,
        "workspace": workspace,
        "experiment_directory": experiment_directory,
        "venv_directory": venv_directory,
        "dependency_paths": dependency_paths,
        "input_paths": input_paths,
        "safe_python": safe_python,
        "param_runtime": None if param_runtime is None else param_runtime.root,
    }
    commands = _commands(parameters, **resolution)
    trace_path = _find_trace_path(commands)
    if trace_path is not None:
        if trace_path.is_symlink() or not trace_path.is_file():
            raise CellEntrypointError("replay trace is missing or unsafe")
        world_size, _ranks = validate_physical_layout(parameters)
        load_and_validate_param_trace(str(trace_path), world_size=world_size)
    nccl_library = _stage_bound_nccl_library(configuration, input_paths, workspace)
    environment = _runtime_environment(
        configuration,
        venv_directory,
        nccl_library,
        executor_artifact,
    )
    command_environments = _command_environments(environment, commands, param_runtime)
    execution_plan = {
        "manifest_sha256": frozen.manifest_sha256,
        "cell_id": cell.id,
        "cell_identity_sha256": cell.identity_sha256,
        "attempt_id": args.attempt_id,
        "wrapper": args.site_wrapper,
        "commands": [list(command) for command in commands],
        "environment_sha256": canonical_sha256(command_environments),
        "dependency_attempts": dependency_evidence,
        "input_hashes": {artifact.id: artifact.sha256 for artifact in manifest.campaign.inputs},
        "timeout_seconds": timeout_seconds,
        "max_output_bytes": max_output_bytes,
        "max_result_bytes": max_result_bytes,
    }
    execution_plan_sha256 = canonical_sha256(execution_plan)
    _write_exclusive(workspace / "execution_plan.json", canonical_json_bytes(execution_plan))
    started_at = utc_timestamp()
    return_code, wall_time_s, execution_reason, output_exceeded = _run_pipeline(
        commands,
        workspace=workspace,
        environment=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        command_environments=command_environments,
    )
    finished_at = utc_timestamp()
    for path in (stdout_path, stderr_path):
        os.chmod(path, 0o444)
    result: Optional[CellResult] = None
    measurement_reference: Optional[ArtifactReference] = None
    partial_outputs: Tuple[ArtifactReference, ...] = ()
    runtime_observation: Optional[Dict[str, Any]] = None
    execution_succeeded = return_code == 0 and execution_reason is None and not output_exceeded
    status = "success" if execution_succeeded else "failed"
    reason = execution_reason
    if execution_succeeded:
        try:
            runtime, runtime_observation = _runtime_fingerprint(
                nccl_library,
                site,
                site_packages=_site_packages(venv_directory),
            )
            validate_expected_runtime(runtime, configuration.expected_runtime.to_value())
            capture_artifacts: Optional[Dict[str, Any]] = None
            if workload.measurement_schema == CAPTURE_MEASUREMENT_SCHEMA:
                capture_artifacts, partial_outputs = _capture_artifacts(
                    parameters,
                    resolution=resolution,
                    run_directory=frozen.directory,
                    workspace=workspace,
                )
            stdout = read_bounded_text(
                stdout_path,
                max_bytes=max_output_bytes,
                field="physical stdout",
                errors="replace",
            )
            stderr = read_bounded_text(
                stderr_path,
                max_bytes=max_output_bytes,
                field="physical stderr",
                errors="replace",
            )
            measurement = adapt_physical_measurement(
                measurement_schema=workload.measurement_schema,
                producer_schema=workload.producer_schema,
                attempt_id=args.attempt_id,
                parameters=parameters,
                stdout=stdout,
                stderr=stderr,
                wall_time_s=wall_time_s,
                runtime=runtime,
                repetition=cell.repetition,
                trace_sha256=None if trace_path is None else file_sha256(trace_path),
                artifacts=capture_artifacts,
            )
            result = CellResult.from_dict(
                {
                    "schema": CELL_RESULT_SCHEMA,
                    "cell_id": cell.id,
                    "cell_identity_sha256": cell.identity_sha256,
                    "producer_schema": workload.producer_schema,
                    "measurement_schema": workload.measurement_schema,
                    "measurement": measurement,
                }
            )
            write_cell_result(result_path, result)
            if result_path.stat().st_size > max_result_bytes:
                raise PhysicalResultError(f"cell result exceeds {max_result_bytes} bytes")
            measurement_reference = _artifact_reference(frozen.directory, result_path)
            reason = None
        except (ContractError, OSError, UnicodeError) as exc:
            status = "parse-failed"
            reason = f"cannot validate physical result: {exc}"
            if result_path.is_file() and not result_path.is_symlink():
                partial_outputs = tuple(
                    sorted(
                        partial_outputs + (_artifact_reference(frozen.directory, result_path),),
                        key=lambda item: item.path,
                    )
                )
    elif execution_reason == "execution interrupted" or output_exceeded or "exceeded" in (execution_reason or ""):
        status = "cancelled"
    reason = None if reason is None else reason.replace("\x00", "\\0")[:4096]
    stdout_reference = _artifact_reference(frozen.directory, stdout_path)
    stderr_reference = _artifact_reference(frozen.directory, stderr_path)
    command = [sys.executable, "-m", "experiments.rostam.lib.cell_entrypoint", *raw_argv]
    record_exit_code: Optional[int] = return_code
    if status in {"failed", "cancelled"} and record_exit_code == 0:
        record_exit_code = None
    record = AttemptRecord.from_dict(
        {
            "schema": ATTEMPT_SCHEMA,
            "run_id": manifest.run_id,
            "manifest_sha256": frozen.manifest_sha256,
            "cell_id": cell.id,
            "cell_identity_sha256": cell.identity_sha256,
            "attempt_id": args.attempt_id,
            "attempt_number": len(attempts) + 1,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "command": command,
            "observed": {
                "executor": "slurm-cell-entrypoint",
                "site_id": "rostam",
                "hostname": site["hostname"],
                "scheduler": "slurm",
                "job_id": site["job_id"],
                "nodes": [site["hostname"].split(".", 1)[0]],
                "account": None if site["account"] == "unrecorded" else site["account"],
                "partition": site["partition"],
                "metadata": {
                    "environment_sha256": canonical_sha256(command_environments),
                    "execution_identity_sha256": execution_plan_sha256,
                    "execution_plan_sha256": execution_plan_sha256,
                    "dependency_attempts": dependency_evidence,
                    "input_hashes": {artifact.id: artifact.sha256 for artifact in manifest.campaign.inputs},
                    "output_limit_exceeded": output_exceeded,
                    "physical_commands": [list(item) for item in commands],
                    "result_schema": CELL_RESULT_SCHEMA,
                    "runtime_observation": runtime_observation,
                },
            },
            "exit_code": record_exit_code,
            "reason": reason,
            "stdout": stdout_reference.to_dict(),
            "stderr": stderr_reference.to_dict(),
            "measurement": None if measurement_reference is None else measurement_reference.to_dict(),
            "partial_outputs": [item.to_dict() for item in partial_outputs],
        }
    )
    write_attempt_record(frozen.directory, record)
    verify_attempt_artifacts(frozen.directory, record)
    return 0 if status == "success" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    try:
        return run(args, raw_argv)
    except (CellEntrypointError, ContractError, OSError, UnicodeError) as exc:
        try:
            _record_preflight_failure(args, raw_argv, exc)
        except (CellEntrypointError, ContractError, OSError, UnicodeError) as record_error:
            raise SystemExit(
                f"physical cell error: {exc}; additionally could not preserve terminal evidence: {record_error}"
            ) from exc
        raise SystemExit(f"physical cell error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
