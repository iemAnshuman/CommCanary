"""Frozen contract for the Rostam physical decision-canary product study."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from commcanary.artifacts.application_measurement import (
    APPLICATION_PERTURBATION_ENVIRONMENT_NAMES,
    application_subject_configuration,
    application_subject_sha256,
)
from commcanary.artifacts.chakra import decode_chakra_execution_trace
from commcanary.artifacts.physical_canary import (
    validate_chakra_projection,
    validate_physical_canary_policy,
)

MANIFEST_FORMAT = "commcanary.rostam.product_study_manifest.v1"
PERTURBATION_FORMAT = "commcanary.product_perturbation_plan.v1"
RUNNER_DESCRIPTOR_FORMAT = "commcanary.product_runner_descriptor.v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_SCRIPT_BYTES = 4 * 1024 * 1024
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_MODEL_FILES = 1024
MAX_MODEL_BYTES = 256 * 1024 * 1024
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCRIPTS = ("study.py", "study_contract.py", "slurm_site.py")


class ProductStudyError(RuntimeError):
    """Raised before a frozen study or scheduler boundary can be crossed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductStudyError(f"{label} must be an object")
    return value


def _closed(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        missing = sorted(expected_set - set(value))
        unexpected = sorted(set(value) - expected_set)
        raise ProductStudyError(f"{label} fields are not closed: missing={missing}, unexpected={unexpected}")


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ProductStudyError(f"{label} must be a safe lowercase identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductStudyError(f"{label} must be a lowercase SHA-256")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProductStudyError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ProductStudyError(f"{label} exceeds {MAX_JSON_BYTES} bytes")

    def reject_duplicate(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductStudyError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductStudyError(f"{label} is not strict UTF-8 JSON") from exc
    return _object(value, label)


def _file_identity(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProductStudyError(f"study input is missing or unsafe: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _bounded_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ProductStudyError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ProductStudyError(f"{label} exceeds {maximum} bytes")
    return raw


def _relative_file_inventory(root: Path, *, label: str) -> List[Tuple[str, Path]]:
    if root.is_symlink() or not root.is_dir():
        raise ProductStudyError(f"{label} is missing or unsafe")
    rows: List[Tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ProductStudyError(f"{label} may not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ProductStudyError(f"{label} contains a non-regular entry")
        relative = path.relative_to(root).as_posix()
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ProductStudyError(f"{label} contains an unsafe path")
        rows.append((relative, path))
        if len(rows) > MAX_MODEL_FILES:
            raise ProductStudyError(f"{label} exceeds {MAX_MODEL_FILES} files")
    return rows


def _write_new(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _copy_new(source: Path, destination: Path) -> Dict[str, Any]:
    expected = _file_identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.chmod(destination, 0o444)
    observed = _file_identity(destination)
    if observed != expected:
        raise ProductStudyError(f"frozen copy changed while copying {source}")
    return observed


def validate_perturbation_plan(raw_plan: Mapping[str, Any], *, engine: str) -> Dict[str, Any]:
    """Validate a decision-blind perturbation inventory and return a copy."""

    _closed(raw_plan, {"format", "study_name", "baseline", "training", "holdout"}, "perturbation plan")
    if raw_plan.get("format") != PERTURBATION_FORMAT:
        raise ProductStudyError("perturbation plan format is unsupported")
    _safe_id(raw_plan.get("study_name"), "perturbation study_name")
    if engine not in {"vllm", "sglang"}:
        raise ProductStudyError("perturbation plan engine is unsupported")
    baseline = _validate_perturbation(raw_plan.get("baseline"), expected_split="baseline", engine=engine)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("training", "holdout"):
        rows = raw_plan.get(split)
        if not isinstance(rows, list) or len(rows) < 2:
            raise ProductStudyError(f"perturbation plan {split} must contain at least two rows")
        groups[split] = [_validate_perturbation(row, expected_split=split, engine=engine) for row in rows]
    all_rows = [baseline, *groups["training"], *groups["holdout"]]
    identifiers = [row["id"] for row in all_rows]
    if len(identifiers) != len(set(identifiers)):
        raise ProductStudyError("perturbation identifiers must be globally unique")
    if len(groups["training"]) + 1 != 8:
        raise ProductStudyError("the first product study requires baseline plus seven training configurations")
    historical = [row for row in groups["holdout"] if row["historical_reference"] is not None]
    if engine == "vllm" and len(historical) != 1:
        raise ProductStudyError("the holdout plan must contain exactly one historical regression mechanism")
    if engine == "sglang" and historical:
        raise ProductStudyError("the SGLang extension must not relabel a vLLM historical mechanism")
    return {
        "format": PERTURBATION_FORMAT,
        "study_name": str(raw_plan["study_name"]),
        "baseline": baseline,
        "training": groups["training"],
        "holdout": groups["holdout"],
    }


def _validate_perturbation(raw_value: Any, *, expected_split: str, engine: str) -> Dict[str, Any]:
    row = _object(raw_value, f"{expected_split} perturbation")
    _closed(
        row,
        {
            "id",
            "split",
            "description",
            "environment",
            "engine_configuration_overrides",
            "physical_runner_arguments",
            "historical_reference",
        },
        f"{expected_split} perturbation",
    )
    identifier = _safe_id(row.get("id"), f"{expected_split} perturbation id")
    if row.get("split") != expected_split:
        raise ProductStudyError(f"perturbation {identifier!r} has the wrong split")
    description = row.get("description")
    if not isinstance(description, str) or not description.strip() or len(description) > 512:
        raise ProductStudyError(f"perturbation {identifier!r} description is invalid")
    environment = _object(row.get("environment"), f"perturbation {identifier!r} environment")
    _closed(
        environment,
        APPLICATION_PERTURBATION_ENVIRONMENT_NAMES,
        f"perturbation {identifier!r} environment",
    )
    if any(value is not None and (not isinstance(value, str) or not value) for value in environment.values()):
        raise ProductStudyError(f"perturbation {identifier!r} environment values must be non-empty strings or null")
    overrides = _object(
        row.get("engine_configuration_overrides"),
        f"perturbation {identifier!r} engine overrides",
    )
    allowed_overrides = (
        {"enforce_eager", "disable_custom_all_reduce"}
        if engine == "vllm"
        else {"disable_cuda_graph", "disable_custom_all_reduce", "disable_overlap_schedule"}
    )
    if not set(overrides).issubset(allowed_overrides) or any(
        not isinstance(value, bool) for value in overrides.values()
    ):
        raise ProductStudyError(f"perturbation {identifier!r} has unsupported engine overrides")
    physical = _object(
        row.get("physical_runner_arguments"),
        f"perturbation {identifier!r} physical arguments",
    )
    _closed(physical, {"disable_overlap", "rank_skew_us"}, f"perturbation {identifier!r} physical arguments")
    if not isinstance(physical.get("disable_overlap"), bool):
        raise ProductStudyError(f"perturbation {identifier!r} disable_overlap must be boolean")
    rank_skew = physical.get("rank_skew_us")
    if not isinstance(rank_skew, (int, float)) or isinstance(rank_skew, bool) or not 0.0 <= float(rank_skew) <= 1e6:
        raise ProductStudyError(f"perturbation {identifier!r} rank_skew_us is invalid")
    historical = row.get("historical_reference")
    if historical is not None:
        historical = _object(historical, f"perturbation {identifier!r} historical reference")
        _closed(
            historical,
            {"kind", "url", "issue_id", "reported_symptom", "scope_note"},
            f"perturbation {identifier!r} historical reference",
        )
        if historical.get("kind") != "historical_mechanism_probe":
            raise ProductStudyError("historical reference kind is unsupported")
        url = historical.get("url")
        if not isinstance(url, str) or not url.startswith("https://github.com/"):
            raise ProductStudyError("historical reference must use an HTTPS GitHub URL")
        for field in ("issue_id", "reported_symptom", "scope_note"):
            if not isinstance(historical.get(field), str) or not historical[field]:
                raise ProductStudyError(f"historical reference {field} must be non-empty")
    return {
        "id": identifier,
        "split": expected_split,
        "description": description,
        "environment": {name: environment[name] for name in APPLICATION_PERTURBATION_ENVIRONMENT_NAMES},
        "engine_configuration_overrides": dict(overrides),
        "physical_runner_arguments": {
            "disable_overlap": bool(physical["disable_overlap"]),
            "rank_skew_us": float(rank_skew),
        },
        "historical_reference": copy.deepcopy(historical),
    }


def williams_schedule(configuration_ids: Sequence[str]) -> List[List[str]]:
    """Return an even-order Williams square with exact position/carryover balance."""

    identifiers = list(configuration_ids)
    if len(identifiers) < 2 or len(identifiers) % 2:
        raise ProductStudyError("Williams scheduling requires a positive even configuration count")
    if len(identifiers) != len(set(identifiers)):
        raise ProductStudyError("Williams scheduling requires unique configuration identifiers")
    size = len(identifiers)
    base = [0]
    for position in range(1, size):
        base.append((position + 1) // 2 if position % 2 else size - position // 2)
    return [[identifiers[(value + row) % size] for value in base] for row in range(size)]


def _default_application(engine: str) -> Dict[str, Any]:
    common = {
        "tensor_parallel_size": 4,
        "batch_size": 8,
        "input_length": 256,
        "output_length": 32,
        "warmups": 3,
        "iterations": 12,
        "seed": 314159,
    }
    if engine == "vllm":
        return {
            "driver_module": "commcanary.product.application_driver",
            "execution_protocol": "vllm-offline-tensor-parallel.v1",
            "workload_arguments": common,
            "engine_configuration": {
                "kind": "vllm.v1",
                "enforce_eager": False,
                "disable_custom_all_reduce": False,
                "gpu_memory_utilization": 0.75,
                "kv_cache_memory_bytes": 536870912,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 8,
            },
        }
    if engine == "sglang":
        return {
            "driver_module": "commcanary.product.sglang_application_driver",
            "execution_protocol": "sglang-offline-tensor-parallel.v1",
            "workload_arguments": common,
            "engine_configuration": {
                "kind": "sglang.v1",
                "disable_cuda_graph": False,
                "disable_custom_all_reduce": False,
                "disable_overlap_schedule": False,
                "mem_fraction_static": 0.75,
                "max_running_requests": 8,
                "max_total_tokens": 4096,
                "chunked_prefill_size": 2048,
                "max_prefill_tokens": 2048,
            },
        }
    raise ProductStudyError("application engine is unsupported")


def _subject_rows(
    *,
    plan: Mapping[str, Any],
    application: Mapping[str, Any],
    runner_digest: str,
    engine: str,
    engine_version: str,
) -> List[Dict[str, Any]]:
    default_engine_configuration = _object(
        application.get("engine_configuration"),
        "application engine configuration",
    )
    raw_training = plan.get("training")
    raw_holdout = plan.get("holdout")
    if not isinstance(raw_training, list) or not isinstance(raw_holdout, list):
        raise ProductStudyError("validated perturbation plan lost its arrays")
    perturbations = [plan["baseline"], *raw_training, *raw_holdout]
    result: List[Dict[str, Any]] = []
    for raw_row in perturbations:
        row = _object(raw_row, "validated perturbation")
        engine_configuration = copy.deepcopy(dict(default_engine_configuration))
        engine_configuration.update(_object(row["engine_configuration_overrides"], "engine overrides"))
        subject_configuration = application_subject_configuration(
            runner_oci_digest=runner_digest,
            application_engine=engine,
            application_engine_version=engine_version,
            engine_configuration=engine_configuration,
            perturbation_environment=_object(row["environment"], "perturbation environment"),
        )
        result.append(
            {
                "perturbation_id": row["id"],
                "split": row["split"],
                "subject_sha256": application_subject_sha256(subject_configuration),
                "subject_configuration": subject_configuration,
                "physical_runner_arguments": copy.deepcopy(row["physical_runner_arguments"]),
                "historical_reference": copy.deepcopy(row["historical_reference"]),
            }
        )
    subjects = [row["subject_sha256"] for row in result]
    if len(subjects) != len(set(subjects)):
        raise ProductStudyError("perturbation plan contains physically duplicate stack subjects")
    return result


def _validate_runner_descriptor(descriptor: Mapping[str, Any]) -> None:
    _closed(
        descriptor,
        {
            "format",
            "status",
            "host",
            "architecture",
            "base_manifest_digest",
            "runner_protocols",
            "oci_manifest",
            "artifacts",
            "inputs",
            "versions",
            "descriptor_id",
        },
        "runner descriptor",
    )
    if descriptor.get("format") != RUNNER_DESCRIPTOR_FORMAT or descriptor.get("status") != "complete":
        raise ProductStudyError("runner descriptor is not a completed product runner build")
    expected_id = hashlib.sha256(
        _canonical_bytes({key: value for key, value in descriptor.items() if key != "descriptor_id"})
    ).hexdigest()
    if descriptor.get("descriptor_id") != expected_id:
        raise ProductStudyError("runner descriptor_id does not match canonical content")
    protocols = descriptor.get("runner_protocols")
    if (
        not isinstance(protocols, list)
        or protocols != sorted(protocols)
        or len(protocols) != len(set(protocols))
        or any(not isinstance(value, str) or not value for value in protocols)
    ):
        raise ProductStudyError("runner descriptor protocols are not canonical")
    versions = _object(descriptor.get("versions"), "runner descriptor versions")
    engine = versions.get("application_engine")
    if not isinstance(engine, str):
        raise ProductStudyError("runner descriptor application engine is missing")
    expected_protocol = {
        "vllm": "vllm-offline-tensor-parallel.v1",
        "sglang": "sglang-offline-tensor-parallel.v1",
    }.get(engine)
    if expected_protocol is None or set(protocols) != {
        "chakra-et-collective-graph.v1",
        expected_protocol,
    }:
        raise ProductStudyError("runner descriptor protocols do not match its application engine")
    manifest = _object(descriptor.get("oci_manifest"), "runner descriptor OCI manifest")
    digest = manifest.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ProductStudyError("runner descriptor OCI digest is malformed")
    _sha256(digest.partition(":")[2], "runner descriptor OCI digest")


def _verify_runner_descriptor(descriptor: Mapping[str, Any], *, descriptor_dir: Path, sif_path: Path) -> None:
    _validate_runner_descriptor(descriptor)
    artifacts = _object(descriptor.get("artifacts"), "runner descriptor artifacts")
    _closed(artifacts, {"runner.oci.tar", "runner.sif", "build.log"}, "runner descriptor artifacts")
    sif_identity = _object(artifacts.get("runner.sif"), "runner descriptor SIF")
    if _file_identity(sif_path) != dict(sif_identity):
        raise ProductStudyError("runner SIF does not match its completed build descriptor")
    for name in ("runner.oci.tar", "build.log"):
        identity = _object(artifacts.get(name), f"runner descriptor {name}")
        artifact = descriptor_dir / name
        if _file_identity(artifact) != dict(identity):
            raise ProductStudyError(f"runner build artifact {name} does not match its descriptor")


def freeze_product_study(
    *,
    study_id: str,
    output_directory: Path,
    source_et: Path,
    projection_path: Path,
    policy_path: Path,
    perturbation_path: Path,
    model_directory: Path,
    runner_descriptor_path: Path,
    runner_sif_path: Path,
    commcanary_wheel_path: Path,
) -> Mapping[str, Any]:
    """Freeze all executable and scientific inputs before any study job exists."""

    study_id = _safe_id(study_id, "product study id")
    destination = output_directory.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite product study directory: {destination}")
    destination.mkdir(parents=True, mode=0o700)

    trace_bytes = _bounded_bytes(source_et, maximum=MAX_TRACE_BYTES, label="source Chakra ET")
    trace = decode_chakra_execution_trace(trace_bytes)
    projection = _read_json(projection_path, "Chakra projection")
    policy = _read_json(policy_path, "physical canary policy")
    validate_chakra_projection(projection, trace)
    validate_physical_canary_policy(policy)
    descriptor = _read_json(runner_descriptor_path, "runner descriptor")
    _verify_runner_descriptor(
        descriptor,
        descriptor_dir=runner_descriptor_path.resolve().parent,
        sif_path=runner_sif_path.resolve(),
    )
    runner_digest = _object(descriptor.get("oci_manifest"), "runner OCI manifest").get("digest")
    if policy["runner"]["oci_digest"] != runner_digest:
        raise ProductStudyError("physical policy runner digest does not match the built runner")
    if int(policy["runner"]["world_size"]) != 4:
        raise ProductStudyError("first product study requires exactly four physical ranks")
    versions = _object(descriptor.get("versions"), "runner descriptor versions")
    engine = str(versions.get("application_engine"))
    engine_version = str(versions.get("application_engine_version"))
    application = _default_application(engine)
    if application["execution_protocol"] not in descriptor["runner_protocols"]:
        raise ProductStudyError("runner descriptor does not bind the application protocol")
    wheel_identity = _file_identity(commcanary_wheel_path)
    descriptor_inputs = _object(descriptor.get("inputs"), "runner descriptor inputs")
    if dict(_object(descriptor_inputs.get("commcanary.whl"), "runner descriptor wheel")) != wheel_identity:
        raise ProductStudyError("host orchestration wheel does not match the wheel installed in the runner")
    plan = validate_perturbation_plan(_read_json(perturbation_path, "perturbation plan"), engine=engine)

    model_root = model_directory.resolve()
    model_files = _relative_file_inventory(model_root, label="application model directory")
    if not model_files:
        raise ProductStudyError("application model directory contains no files")
    model_bytes = sum(_file_identity(path)["bytes"] for _relative, path in model_files)
    if model_bytes > MAX_MODEL_BYTES:
        raise ProductStudyError(f"application model directory exceeds {MAX_MODEL_BYTES} bytes")

    copied_inputs: Dict[str, Any] = {}
    for name, path in (
        ("source.et", source_et),
        ("projection.json", projection_path),
        ("policy.json", policy_path),
        ("perturbations.json", perturbation_path),
        ("runner-descriptor.json", runner_descriptor_path),
        ("commcanary.whl", commcanary_wheel_path),
    ):
        copied_inputs[name] = _copy_new(path.resolve(), destination / "inputs" / name)
    copied_model = []
    for relative, path in model_files:
        identity = _copy_new(path, destination / "inputs" / "model" / relative)
        copied_model.append({"path": relative, **identity})

    script_root = Path(__file__).resolve().parent
    copied_scripts: Dict[str, Any] = {}
    for name in _SCRIPTS:
        path = script_root / name
        identity = _file_identity(path)
        if identity["bytes"] > MAX_SCRIPT_BYTES:
            raise ProductStudyError(f"study script {name} exceeds its byte budget")
        copied_scripts[name] = _copy_new(path, destination / "scripts" / name)

    subject_rows = _subject_rows(
        plan=plan,
        application=application,
        runner_digest=str(runner_digest),
        engine=engine,
        engine_version=engine_version,
    )
    training_ids = [plan["baseline"]["id"], *(row["id"] for row in plan["training"])]
    holdout_ids = [plan["baseline"]["id"], *(row["id"] for row in plan["holdout"])]
    holdout_square = williams_schedule(holdout_ids)

    raw_manifest: Dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "study_id": study_id,
        "frozen_at": _timestamp(),
        "site": {
            "site_id": "rostam",
            "scheduler": "slurm",
            "partition": "cuda-A100",
            "nodes": 1,
            "exclusive": True,
            "gpu_count": 4,
            "maximum_repetition_span_seconds": 7200,
        },
        "runner": {
            "descriptor_id": descriptor["descriptor_id"],
            "oci_digest": runner_digest,
            "sif_path": str(runner_sif_path.resolve()),
            "sif_identity": _file_identity(runner_sif_path.resolve()),
            "application_engine": engine,
            "application_engine_version": engine_version,
        },
        "inputs": copied_inputs,
        "model_files": copied_model,
        "scripts": copied_scripts,
        "application": application,
        "subjects": subject_rows,
        "schedule": {
            "method": "williams-even-order-position-and-first-order-carryover.v1",
            "training_configuration_order_by_repetition": williams_schedule(training_ids),
            "holdout_configuration_order_by_repetition": [*holdout_square, *holdout_square],
        },
        "execution": {
            "application_repetitions": 8,
            "minimum_distinct_days": 2,
            "repetition_day_offset_by_repetition": [0, 0, 0, 0, 1, 1, 1, 1],
            "application_timeout_seconds": 7200,
            "physical_timeout_seconds": 1800,
            "candidate_warmups": 3,
            "candidate_iterations": 12,
            "max_candidate_evaluations": 64,
            "poll_interval_seconds": 10,
        },
        "claims": {
            "scientific_status": "blocked_until_measured",
            "historical_scope": (
                "current_version_probe_not_historical_reproduction"
                if engine == "vllm"
                else "not_part_of_sglang_extension"
            ),
            "heldout_access": "forbidden_before_persisted_selection_sha256",
        },
    }
    raw_manifest["manifest_id"] = hashlib.sha256(_canonical_bytes(raw_manifest)).hexdigest()
    manifest_bytes = _canonical_bytes(raw_manifest) + b"\n"
    _write_new(destination / "manifest.json", manifest_bytes)
    _write_new(
        destination / "manifest.sha256",
        f"{raw_manifest['manifest_id']}  manifest.json\n".encode("ascii"),
    )
    directory_fd = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return raw_manifest


def verify_frozen_product_study(directory: Path) -> Mapping[str, Any]:
    """Recompute a frozen study and every external runtime-image byte."""

    root = directory.resolve()
    manifest = _read_json(root / "manifest.json", "product study manifest")
    _closed(
        manifest,
        {
            "format",
            "study_id",
            "frozen_at",
            "site",
            "runner",
            "inputs",
            "model_files",
            "scripts",
            "application",
            "subjects",
            "schedule",
            "execution",
            "claims",
            "manifest_id",
        },
        "product study manifest",
    )
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ProductStudyError("product study manifest format is unsupported")
    _safe_id(manifest.get("study_id"), "product study id")
    expected_id = hashlib.sha256(
        _canonical_bytes({key: value for key, value in manifest.items() if key != "manifest_id"})
    ).hexdigest()
    if manifest.get("manifest_id") != expected_id:
        raise ProductStudyError("product study manifest_id does not match canonical content")
    checksum = _bounded_bytes(
        root / "manifest.sha256",
        maximum=256,
        label="product study manifest checksum",
    ).decode("ascii")
    if checksum != f"{expected_id}  manifest.json\n":
        raise ProductStudyError("product study manifest checksum marker is invalid")
    inputs = _object(manifest.get("inputs"), "product study inputs")
    expected_inputs = {
        "source.et",
        "projection.json",
        "policy.json",
        "perturbations.json",
        "runner-descriptor.json",
        "commcanary.whl",
    }
    if set(inputs) != expected_inputs:
        raise ProductStudyError("product study input inventory is incomplete")
    for name, identity in inputs.items():
        if _file_identity(root / "inputs" / str(name)) != dict(_object(identity, f"input {name}")):
            raise ProductStudyError(f"frozen product study input changed: {name}")
    model_rows = manifest.get("model_files")
    if not isinstance(model_rows, list) or not model_rows:
        raise ProductStudyError("product study model inventory is missing")
    for raw_row in model_rows:
        row = _object(raw_row, "product study model row")
        _closed(row, {"path", "sha256", "bytes"}, "product study model row")
        path = row.get("path")
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ProductStudyError("product study model path is unsafe")
        if _file_identity(root / "inputs" / "model" / path) != {
            "sha256": row.get("sha256"),
            "bytes": row.get("bytes"),
        }:
            raise ProductStudyError(f"frozen product study model changed: {path}")
    expected_model_paths = {str(_object(row, "product study model row")["path"]) for row in model_rows}
    observed_model_paths = {
        relative for relative, _path in _relative_file_inventory(root / "inputs" / "model", label="frozen model")
    }
    if observed_model_paths != expected_model_paths:
        raise ProductStudyError("frozen product study model member set changed")
    observed_input_paths = {
        relative for relative, _path in _relative_file_inventory(root / "inputs", label="frozen inputs")
    }
    expected_input_paths = expected_inputs | {f"model/{path}" for path in expected_model_paths}
    if observed_input_paths != expected_input_paths:
        raise ProductStudyError("frozen product study input member set changed")
    scripts = _object(manifest.get("scripts"), "product study scripts")
    if set(scripts) != set(_SCRIPTS):
        raise ProductStudyError("product study script inventory is incomplete")
    for name, identity in scripts.items():
        if _file_identity(root / "scripts" / str(name)) != dict(_object(identity, f"script {name}")):
            raise ProductStudyError(f"frozen product study script changed: {name}")
    if {relative for relative, _path in _relative_file_inventory(root / "scripts", label="frozen scripts")} != set(
        _SCRIPTS
    ):
        raise ProductStudyError("frozen product study script member set changed")
    runner = _object(manifest.get("runner"), "product study runner")
    _closed(
        runner,
        {
            "descriptor_id",
            "oci_digest",
            "sif_path",
            "sif_identity",
            "application_engine",
            "application_engine_version",
        },
        "product study runner",
    )
    sif_path = Path(str(runner.get("sif_path")))
    if _file_identity(sif_path) != dict(_object(runner.get("sif_identity"), "product study SIF identity")):
        raise ProductStudyError("external content-addressed runner SIF changed or disappeared")

    descriptor = _read_json(root / "inputs" / "runner-descriptor.json", "frozen runner descriptor")
    _validate_runner_descriptor(descriptor)
    if descriptor.get("descriptor_id") != runner.get("descriptor_id"):
        raise ProductStudyError("frozen runner descriptor identity changed")
    descriptor_manifest = _object(descriptor.get("oci_manifest"), "frozen runner OCI manifest")
    if descriptor_manifest.get("digest") != runner.get("oci_digest"):
        raise ProductStudyError("frozen runner OCI digest changed")
    descriptor_artifacts = _object(descriptor.get("artifacts"), "frozen runner artifacts")
    if dict(_object(descriptor_artifacts.get("runner.sif"), "frozen runner SIF")) != dict(
        _object(runner.get("sif_identity"), "product study SIF identity")
    ):
        raise ProductStudyError("frozen runner descriptor no longer binds the external SIF")
    descriptor_inputs = _object(descriptor.get("inputs"), "frozen runner inputs")
    if dict(_object(descriptor_inputs.get("commcanary.whl"), "frozen runner wheel")) != dict(
        _object(inputs.get("commcanary.whl"), "frozen host wheel")
    ):
        raise ProductStudyError("frozen orchestration wheel differs from the runner wheel")

    trace = decode_chakra_execution_trace(
        _bounded_bytes(root / "inputs" / "source.et", maximum=MAX_TRACE_BYTES, label="frozen source Chakra ET")
    )
    projection = _read_json(root / "inputs" / "projection.json", "frozen projection")
    policy = _read_json(root / "inputs" / "policy.json", "frozen policy")
    validate_chakra_projection(projection, trace)
    validate_physical_canary_policy(policy)
    if policy["runner"]["oci_digest"] != runner.get("oci_digest"):
        raise ProductStudyError("frozen policy no longer matches the runner")
    if int(policy["runner"]["world_size"]) != 4:
        raise ProductStudyError("frozen product study no longer uses four physical ranks")
    application = _object(manifest.get("application"), "product study application")
    engine = str(runner.get("application_engine"))
    if dict(application) != _default_application(engine):
        raise ProductStudyError("frozen application contract does not recompute")
    versions = _object(descriptor.get("versions"), "frozen runner versions")
    if versions.get("application_engine") != engine or versions.get("application_engine_version") != runner.get(
        "application_engine_version"
    ):
        raise ProductStudyError("frozen runner versions do not match the study runner")
    if application["execution_protocol"] not in descriptor["runner_protocols"]:
        raise ProductStudyError("frozen runner does not support the application protocol")
    plan = validate_perturbation_plan(
        _read_json(root / "inputs" / "perturbations.json", "frozen perturbations"),
        engine=engine,
    )
    expected_subjects = _subject_rows(
        plan=plan,
        application=application,
        runner_digest=str(runner["oci_digest"]),
        engine=engine,
        engine_version=str(runner["application_engine_version"]),
    )
    if manifest.get("subjects") != expected_subjects:
        raise ProductStudyError("frozen subject inventory does not recompute")
    expected_training = [plan["baseline"]["id"], *(row["id"] for row in plan["training"])]
    expected_schedule = williams_schedule(expected_training)
    schedule = _object(manifest.get("schedule"), "product study schedule")
    _closed(
        schedule,
        {
            "method",
            "training_configuration_order_by_repetition",
            "holdout_configuration_order_by_repetition",
        },
        "product study schedule",
    )
    if schedule.get("method") != "williams-even-order-position-and-first-order-carryover.v1":
        raise ProductStudyError("frozen product study schedule method is unsupported")
    if schedule.get("training_configuration_order_by_repetition") != expected_schedule:
        raise ProductStudyError("frozen Williams schedule does not recompute")
    expected_holdout_ids = [plan["baseline"]["id"], *(row["id"] for row in plan["holdout"])]
    holdout_square = williams_schedule(expected_holdout_ids)
    if schedule.get("holdout_configuration_order_by_repetition") != [*holdout_square, *holdout_square]:
        raise ProductStudyError("frozen holdout schedule does not recompute")

    site = _object(manifest.get("site"), "product study site")
    if site != {
        "site_id": "rostam",
        "scheduler": "slurm",
        "partition": "cuda-A100",
        "nodes": 1,
        "exclusive": True,
        "gpu_count": 4,
        "maximum_repetition_span_seconds": 7200,
    }:
        raise ProductStudyError("frozen Rostam site contract is unsupported")
    execution = _object(manifest.get("execution"), "product study execution")
    _closed(
        execution,
        {
            "application_timeout_seconds",
            "application_repetitions",
            "minimum_distinct_days",
            "repetition_day_offset_by_repetition",
            "physical_timeout_seconds",
            "candidate_warmups",
            "candidate_iterations",
            "max_candidate_evaluations",
            "poll_interval_seconds",
        },
        "product study execution",
    )
    if execution != {
        "application_repetitions": 8,
        "minimum_distinct_days": 2,
        "repetition_day_offset_by_repetition": [0, 0, 0, 0, 1, 1, 1, 1],
        "application_timeout_seconds": 7200,
        "physical_timeout_seconds": 1800,
        "candidate_warmups": 3,
        "candidate_iterations": 12,
        "max_candidate_evaluations": 64,
        "poll_interval_seconds": 10,
    }:
        raise ProductStudyError("frozen product study execution contract is unsupported")
    claims = _object(manifest.get("claims"), "product study claims")
    if claims != {
        "scientific_status": "blocked_until_measured",
        "historical_scope": (
            "current_version_probe_not_historical_reproduction" if engine == "vllm" else "not_part_of_sglang_extension"
        ),
        "heldout_access": "forbidden_before_persisted_selection_sha256",
    }:
        raise ProductStudyError("frozen product study claim boundary is unsupported")
    return manifest


def subject_map(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = manifest.get("subjects")
    if not isinstance(rows, list):
        raise ProductStudyError("product study subject inventory is missing")
    result = {}
    for raw_row in rows:
        row = _object(raw_row, "product study subject")
        identifier = _safe_id(row.get("perturbation_id"), "product study perturbation id")
        result[identifier] = row
    if len(result) != len(rows):
        raise ProductStudyError("product study subject inventory contains duplicate perturbations")
    return result


__all__ = [
    "MANIFEST_FORMAT",
    "PERTURBATION_FORMAT",
    "ProductStudyError",
    "freeze_product_study",
    "subject_map",
    "validate_perturbation_plan",
    "verify_frozen_product_study",
    "williams_schedule",
]
