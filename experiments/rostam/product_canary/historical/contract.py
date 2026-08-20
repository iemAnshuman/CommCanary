"""Frozen boundary for the vLLM issue 2971 version-sensitivity study."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__:
    _images = importlib.import_module(f"{__package__}.build_images")
    _model = importlib.import_module(f"{__package__}.stage_model")
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _images = importlib.import_module("build_images")
    _model = importlib.import_module("stage_model")

validate_image_descriptor = _images.validate_image_descriptor
validate_image_lock = _images.validate_image_lock
validate_model_descriptor = _model.validate_model_descriptor
validate_model_lock = _model.validate_model_lock

MANIFEST_FORMAT = "commcanary.historical_vllm_study_manifest.v1"
STUDY_LOCK_FORMAT = "commcanary.historical_vllm_study_lock.v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_SCRIPT_BYTES = 2 * 1024 * 1024
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCRIPTS = ("build_images.py", "contract.py", "driver.py", "stage_model.py", "study.py")


class HistoricalStudyError(RuntimeError):
    """Raised before an unverifiable historical study can run."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalStudyError(f"{label} must be an object")
    return value


def _closed(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    if set(value) != set(expected):
        raise HistoricalStudyError(f"{label} fields are not closed")


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise HistoricalStudyError(f"{label} must be a safe lowercase identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HistoricalStudyError(f"{label} must be a lowercase SHA-256")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalStudyError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise HistoricalStudyError(f"{label} exceeds {MAX_JSON_BYTES} bytes")

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HistoricalStudyError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalStudyError(f"{label} is not strict UTF-8 JSON") from exc
    return _object(value, label)


def _read_small_ascii(path: Path, label: str, *, maximum: int) -> str:
    if path.is_symlink() or not path.is_file():
        raise HistoricalStudyError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise HistoricalStudyError(f"{label} exceeds {maximum} bytes")
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise HistoricalStudyError(f"{label} must be ASCII") from exc


def _identity(path: Path, *, maximum: Optional[int] = None) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalStudyError(f"historical study artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            if maximum is not None and size > maximum:
                raise HistoricalStudyError(f"historical study artifact exceeds {maximum} bytes: {path}")
    return {"sha256": digest.hexdigest(), "bytes": size}


def _write_new(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _copy_new(source: Path, target: Path) -> Dict[str, Any]:
    expected = _identity(source, maximum=MAX_SCRIPT_BYTES)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.chmod(target, 0o444)
    observed = _identity(target, maximum=MAX_SCRIPT_BYTES)
    if observed != expected:
        raise HistoricalStudyError(f"historical study input changed while copied: {source}")
    return observed


def williams_schedule(configuration_ids: Sequence[str]) -> List[List[str]]:
    identifiers = list(configuration_ids)
    if len(identifiers) < 2 or len(identifiers) % 2 or len(identifiers) != len(set(identifiers)):
        raise HistoricalStudyError("Williams scheduling requires a unique positive even configuration set")
    size = len(identifiers)
    base = [0]
    for position in range(1, size):
        base.append((position + 1) // 2 if position % 2 else size - position // 2)
    return [[identifiers[(value + row) % size] for value in base] for row in range(size)]


def validate_study_lock(raw_lock: Mapping[str, Any]) -> Dict[str, Any]:
    _closed(
        raw_lock,
        {"format", "scope", "workload", "execution", "decision", "claim_boundary"},
        "historical study lock",
    )
    if raw_lock.get("format") != STUDY_LOCK_FORMAT:
        raise HistoricalStudyError("historical study lock format is unsupported")
    if raw_lock.get("scope") != "issue-2971-version-sensitivity-on-rostam-a100-tp4":
        raise HistoricalStudyError("historical study scope is unsupported")
    workload = _object(raw_lock.get("workload"), "historical study workload")
    expected_workload = {
        "tensor_parallel_size": 4,
        "batch_size": 8,
        "input_length": 256,
        "output_length": 32,
        "max_model_len": 8192,
        "warmups": 3,
        "iterations": 12,
        "seed": 2971,
        "gpu_memory_utilization": 0.85,
        "swap_space_gib": 4,
    }
    if workload != expected_workload:
        raise HistoricalStudyError("historical study workload is not the reviewed fixed workload")
    execution = _object(raw_lock.get("execution"), "historical study execution")
    expected_execution = {
        "repetitions": 8,
        "minimum_distinct_days": 2,
        "repetition_day_offset_by_repetition": [0, 0, 0, 0, 1, 1, 1, 1],
        "allocation_timeout_seconds": 7200,
        "maximum_repetition_span_seconds": 7200,
        "poll_interval_seconds": 10,
    }
    if execution != expected_execution:
        raise HistoricalStudyError("historical study execution is not the reviewed replicated design")
    decision = _object(raw_lock.get("decision"), "historical study decision")
    expected_decision = {
        "regression_severity_threshold_pct": 5.0,
        "reported_version_comparison": ["vllm-0.2.7-default", "vllm-0.3.2-default"],
        "eager_mitigation_comparison": ["vllm-0.3.3-default", "vllm-0.3.3-enforce-eager"],
    }
    if decision != expected_decision:
        raise HistoricalStudyError("historical study decision rule is unsupported")
    claim = _object(raw_lock.get("claim_boundary"), "historical study claim boundary")
    expected_claim = {
        "application": "actual-openhermes-2.5-mistral-7b-at-issue-date-revision",
        "hardware": "four-a100-gpus-not-the-reported-single-rtx-4090",
        "execution": "fixed-offline-batches-not-the-unreported-original-server-request-stream",
        "interpretation": "historical-version-sensitivity-probe-not-an-exact-reproduction-of-the-user-report",
    }
    if claim != expected_claim:
        raise HistoricalStudyError("historical study claim boundary is unsupported")
    return {
        "format": STUDY_LOCK_FORMAT,
        "scope": str(raw_lock["scope"]),
        "workload": dict(workload),
        "execution": dict(execution),
        "decision": dict(decision),
        "claim_boundary": dict(claim),
    }


def freeze_historical_study(
    *,
    study_id: str,
    output_directory: Path,
    image_lock_path: Path,
    model_lock_path: Path,
    study_lock_path: Path,
    image_descriptor_path: Path,
    model_descriptor_path: Path,
) -> Dict[str, Any]:
    """Freeze all executable bytes and external content-addressed artifacts."""

    study_id = _safe_id(study_id, "historical study id")
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical study: {output}")
    image_lock = validate_image_lock(_read_json(image_lock_path.resolve(strict=True), "historical image lock"))
    model_lock = validate_model_lock(_read_json(model_lock_path.resolve(strict=True), "historical model lock"))
    study_lock = validate_study_lock(_read_json(study_lock_path.resolve(strict=True), "historical study lock"))
    image_descriptor = validate_image_descriptor(image_descriptor_path)
    model_descriptor = validate_model_descriptor(model_descriptor_path)
    built_lock = validate_image_lock(
        _read_json(image_descriptor_path.resolve(strict=True).parent / "image-lock.json", "built historical image lock")
    )
    staged_lock = validate_model_lock(
        _read_json(
            model_descriptor_path.resolve(strict=True).parent / "model-lock.json", "staged historical model lock"
        )
    )
    if built_lock != image_lock:
        raise HistoricalStudyError("historical image build does not match the selected image lock")
    if staged_lock != model_lock:
        raise HistoricalStudyError("historical model staging does not match the selected model lock")
    script_root = Path(__file__).resolve().parent
    for name in _SCRIPTS:
        _identity(script_root / name, maximum=MAX_SCRIPT_BYTES)

    output.mkdir(mode=0o700)
    try:
        inputs = output / "inputs"
        scripts = output / "scripts"
        inputs.mkdir(mode=0o700)
        scripts.mkdir(mode=0o700)
        lock_sources = {
            "image-lock.json": image_lock_path.resolve(strict=True),
            "model-lock.json": model_lock_path.resolve(strict=True),
            "study-lock.json": study_lock_path.resolve(strict=True),
            "image-descriptor.json": image_descriptor_path.resolve(strict=True),
            "model-descriptor.json": model_descriptor_path.resolve(strict=True),
        }
        input_identities = {name: _copy_new(source, inputs / name) for name, source in lock_sources.items()}
        script_identities = {name: _copy_new(script_root / name, scripts / name) for name in _SCRIPTS}
        conditions = [dict(row) for row in image_lock["conditions"]]
        square = williams_schedule([str(row["id"]) for row in conditions])
        raw_manifest: Dict[str, Any] = {
            "format": MANIFEST_FORMAT,
            "study_id": study_id,
            "created_at": _timestamp(),
            "inputs": input_identities,
            "scripts": script_identities,
            "external_artifacts": {
                "image_descriptor_path": str(image_descriptor_path.resolve(strict=True)),
                "image_descriptor_id": image_descriptor["descriptor_id"],
                "model_descriptor_path": str(model_descriptor_path.resolve(strict=True)),
                "model_descriptor_id": model_descriptor["descriptor_id"],
            },
            "runtime_images": [dict(row) for row in image_descriptor["images"]],
            "model": dict(model_descriptor),
            "conditions": conditions,
            "schedule": {
                "method": "two-replicated-williams-squares-position-and-first-order-carryover.v1",
                "configuration_order_by_repetition": [*square, *square],
            },
            "site": {
                "site_id": "rostam",
                "scheduler": "slurm",
                "partition": "cuda-A100",
                "nodes": 1,
                "exclusive": True,
                "gpu_count": 4,
            },
            "scope": study_lock["scope"],
            "workload": dict(study_lock["workload"]),
            "execution": dict(study_lock["execution"]),
            "decision": dict(study_lock["decision"]),
            "claim_boundary": dict(study_lock["claim_boundary"]),
            "scientific_status": "blocked_until_complete_measured_allocations",
        }
        raw_manifest["manifest_id"] = hashlib.sha256(_canonical_bytes(raw_manifest)).hexdigest()
        _write_new(output / "manifest.json", _canonical_bytes(raw_manifest) + b"\n")
        _write_new(output / "manifest.sha256", f"{raw_manifest['manifest_id']}  manifest.json\n".encode("ascii"))
        (output / "state").mkdir(mode=0o700)
        (output / "publication").mkdir(mode=0o700)
        os.chmod(inputs, 0o555)
        os.chmod(scripts, 0o555)
        os.chmod(output, 0o555)
        verify_frozen_historical_study(output)
        return raw_manifest
    except BaseException:
        # The missing manifest or failed verification keeps a partial freeze
        # visibly unusable. No external model or image bytes are removed.
        raise


def verify_frozen_historical_study(study_directory: Path) -> Dict[str, Any]:
    """Rehash the frozen files, external SIFs, and complete model snapshot."""

    root = study_directory.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise HistoricalStudyError("historical study directory is unsafe")
    manifest = _read_json(root / "manifest.json", "historical study manifest")
    _closed(
        manifest,
        {
            "format",
            "manifest_id",
            "study_id",
            "created_at",
            "inputs",
            "scripts",
            "external_artifacts",
            "runtime_images",
            "model",
            "conditions",
            "schedule",
            "site",
            "scope",
            "workload",
            "execution",
            "decision",
            "claim_boundary",
            "scientific_status",
        },
        "historical study manifest",
    )
    if manifest.get("format") != MANIFEST_FORMAT:
        raise HistoricalStudyError("historical study manifest format is unsupported")
    expected_manifest_id = hashlib.sha256(
        _canonical_bytes({key: value for key, value in manifest.items() if key != "manifest_id"})
    ).hexdigest()
    if manifest.get("manifest_id") != expected_manifest_id:
        raise HistoricalStudyError("historical study manifest identity does not recompute")
    checksum = _read_small_ascii(
        root / "manifest.sha256",
        "historical study manifest checksum",
        maximum=96,
    )
    if checksum != f"{expected_manifest_id}  manifest.json\n":
        raise HistoricalStudyError("historical study manifest checksum disagrees")
    _safe_id(manifest.get("study_id"), "historical study id")

    input_names = {
        "image-lock.json",
        "model-lock.json",
        "study-lock.json",
        "image-descriptor.json",
        "model-descriptor.json",
    }
    inputs = _object(manifest.get("inputs"), "historical study inputs")
    if set(inputs) != input_names or {path.name for path in (root / "inputs").iterdir()} != input_names:
        raise HistoricalStudyError("historical study input member set is not exact")
    for name in input_names:
        if inputs[name] != _identity(root / "inputs" / name, maximum=MAX_SCRIPT_BYTES):
            raise HistoricalStudyError(f"historical study input identity changed: {name}")
    scripts = _object(manifest.get("scripts"), "historical study scripts")
    if set(scripts) != set(_SCRIPTS) or {path.name for path in (root / "scripts").iterdir()} != set(_SCRIPTS):
        raise HistoricalStudyError("historical study script member set is not exact")
    for name in _SCRIPTS:
        if scripts[name] != _identity(root / "scripts" / name, maximum=MAX_SCRIPT_BYTES):
            raise HistoricalStudyError(f"historical study script identity changed: {name}")

    image_lock = validate_image_lock(_read_json(root / "inputs" / "image-lock.json", "frozen image lock"))
    model_lock = validate_model_lock(_read_json(root / "inputs" / "model-lock.json", "frozen model lock"))
    study_lock = validate_study_lock(_read_json(root / "inputs" / "study-lock.json", "frozen study lock"))
    external = _object(manifest.get("external_artifacts"), "historical external artifacts")
    _closed(
        external,
        {
            "image_descriptor_path",
            "image_descriptor_id",
            "model_descriptor_path",
            "model_descriptor_id",
        },
        "historical external artifacts",
    )
    image_descriptor_path = Path(str(external["image_descriptor_path"]))
    model_descriptor_path = Path(str(external["model_descriptor_path"]))
    image_descriptor = validate_image_descriptor(image_descriptor_path)
    model_descriptor = validate_model_descriptor(model_descriptor_path)
    if image_descriptor.get("descriptor_id") != external.get("image_descriptor_id"):
        raise HistoricalStudyError("historical image descriptor identity changed")
    if model_descriptor.get("descriptor_id") != external.get("model_descriptor_id"):
        raise HistoricalStudyError("historical model descriptor identity changed")
    if _read_json(root / "inputs" / "image-descriptor.json", "frozen image descriptor") != image_descriptor:
        raise HistoricalStudyError("frozen image descriptor differs from the external artifact")
    if _read_json(root / "inputs" / "model-descriptor.json", "frozen model descriptor") != model_descriptor:
        raise HistoricalStudyError("frozen model descriptor differs from the external artifact")
    built_lock = validate_image_lock(
        _read_json(image_descriptor_path.parent / "image-lock.json", "external image lock")
    )
    staged_lock = validate_model_lock(
        _read_json(model_descriptor_path.parent / "model-lock.json", "external model lock")
    )
    if image_lock != built_lock or model_lock != staged_lock:
        raise HistoricalStudyError("frozen locks differ from their external artifacts")
    if manifest.get("runtime_images") != image_descriptor["images"] or manifest.get("model") != model_descriptor:
        raise HistoricalStudyError("historical runtime or model inventory no longer recomputes")
    if manifest.get("conditions") != image_lock["conditions"]:
        raise HistoricalStudyError("historical conditions no longer recompute")
    square = williams_schedule([str(row["id"]) for row in image_lock["conditions"]])
    if manifest.get("schedule") != {
        "method": "two-replicated-williams-squares-position-and-first-order-carryover.v1",
        "configuration_order_by_repetition": [*square, *square],
    }:
        raise HistoricalStudyError("historical Williams schedule no longer recomputes")
    if manifest.get("site") != {
        "site_id": "rostam",
        "scheduler": "slurm",
        "partition": "cuda-A100",
        "nodes": 1,
        "exclusive": True,
        "gpu_count": 4,
    }:
        raise HistoricalStudyError("historical site contract is unsupported")
    for field in ("scope", "workload", "execution", "decision", "claim_boundary"):
        if manifest.get(field) != study_lock[field]:
            raise HistoricalStudyError(f"historical study {field} differs from its lock")
    if manifest.get("scientific_status") != "blocked_until_complete_measured_allocations":
        raise HistoricalStudyError("historical study frozen status is unsupported")
    return dict(manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--study-id", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--image-lock", required=True)
    freeze.add_argument("--model-lock", required=True)
    freeze.add_argument("--study-lock", required=True)
    freeze.add_argument("--image-descriptor", required=True)
    freeze.add_argument("--model-descriptor", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--study", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        manifest = freeze_historical_study(
            study_id=args.study_id,
            output_directory=Path(args.output),
            image_lock_path=Path(args.image_lock),
            model_lock_path=Path(args.model_lock),
            study_lock_path=Path(args.study_lock),
            image_descriptor_path=Path(args.image_descriptor),
            model_descriptor_path=Path(args.model_descriptor),
        )
    else:
        manifest = verify_frozen_historical_study(Path(args.study))
    print(json.dumps({"manifest_id": manifest["manifest_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HistoricalStudyError",
    "freeze_historical_study",
    "validate_study_lock",
    "verify_frozen_historical_study",
    "williams_schedule",
]
