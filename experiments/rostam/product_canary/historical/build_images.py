"""Build and verify content-addressed SIFs for the vLLM issue 2971 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOCK_FORMAT = "commcanary.historical_vllm_image_lock.v1"
DESCRIPTOR_FORMAT = "commcanary.historical_vllm_image_descriptor.v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_LOG_BYTES = 16 * 1024 * 1024
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class HistoricalImageError(RuntimeError):
    """Raised when a historical runtime image is not exactly bound."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _closed(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    if set(value) != set(expected):
        raise HistoricalImageError(f"{label} fields are not closed")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalImageError(f"{label} must be an object")
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise HistoricalImageError(f"{label} must be a safe lowercase identifier")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalImageError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise HistoricalImageError(f"{label} exceeds {MAX_JSON_BYTES} bytes")

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HistoricalImageError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalImageError(f"{label} is not strict UTF-8 JSON") from exc
    return _object(value, label)


def _identity(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalImageError(f"artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _write_new(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new(path, _canonical_bytes(value) + b"\n")


def validate_image_lock(raw_lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the closed, canonical runtime-image lock."""

    _closed(raw_lock, {"format", "source_issue", "images", "conditions"}, "historical image lock")
    if raw_lock.get("format") != LOCK_FORMAT:
        raise HistoricalImageError("historical image lock format is unsupported")
    issue = _object(raw_lock.get("source_issue"), "historical source issue")
    _closed(issue, {"issue_id", "url"}, "historical source issue")
    if issue.get("issue_id") != "vllm-2971" or issue.get("url") != "https://github.com/vllm-project/vllm/issues/2971":
        raise HistoricalImageError("historical image lock names the wrong source issue")
    raw_images = raw_lock.get("images")
    if not isinstance(raw_images, list) or len(raw_images) != 3:
        raise HistoricalImageError("historical image lock must contain three runtime images")
    images: List[Dict[str, Any]] = []
    for index, raw_image in enumerate(raw_images):
        image = _object(raw_image, f"historical image {index}")
        _closed(
            image,
            {
                "id",
                "repository",
                "tag_observed",
                "manifest_digest",
                "expected_vllm_version",
                "digest_source",
            },
            f"historical image {index}",
        )
        identifier = _safe_id(image.get("id"), "historical image id")
        version = image.get("expected_vllm_version")
        if identifier != f"vllm-{version}" or version not in {"0.2.7", "0.3.2", "0.3.3"}:
            raise HistoricalImageError("historical image identity and vLLM version disagree")
        if image.get("repository") != "docker.io/vllm/vllm-openai":
            raise HistoricalImageError("historical image repository is unsupported")
        if image.get("tag_observed") != f"v{version}":
            raise HistoricalImageError("historical image tag and version disagree")
        digest = image.get("manifest_digest")
        if not isinstance(digest, str) or _OCI_DIGEST.fullmatch(digest) is None:
            raise HistoricalImageError("historical image manifest digest is malformed")
        digest_source = image.get("digest_source")
        if not isinstance(digest_source, str) or not digest_source.startswith(
            f"https://hub.docker.com/layers/vllm/vllm-openai/v{version}/images/sha256-"
        ):
            raise HistoricalImageError("historical image digest source is unsupported")
        images.append(dict(image))
    if [row["id"] for row in images] != ["vllm-0.2.7", "vllm-0.3.2", "vllm-0.3.3"]:
        raise HistoricalImageError("historical images are not in canonical version order")

    raw_conditions = raw_lock.get("conditions")
    if not isinstance(raw_conditions, list) or len(raw_conditions) != 4:
        raise HistoricalImageError("historical image lock must contain four conditions")
    conditions: List[Dict[str, Any]] = []
    image_ids = {str(row["id"]) for row in images}
    for index, raw_condition in enumerate(raw_conditions):
        condition = _object(raw_condition, f"historical condition {index}")
        _closed(condition, {"id", "image_id", "enforce_eager"}, f"historical condition {index}")
        identifier = _safe_id(condition.get("id"), "historical condition id")
        if condition.get("image_id") not in image_ids or not isinstance(condition.get("enforce_eager"), bool):
            raise HistoricalImageError("historical condition is malformed")
        conditions.append(
            {
                "id": identifier,
                "image_id": str(condition["image_id"]),
                "enforce_eager": bool(condition["enforce_eager"]),
            }
        )
    expected_conditions = [
        "vllm-0.2.7-default",
        "vllm-0.3.2-default",
        "vllm-0.3.3-default",
        "vllm-0.3.3-enforce-eager",
    ]
    if [row["id"] for row in conditions] != expected_conditions:
        raise HistoricalImageError("historical conditions are not canonical")
    return {
        "format": LOCK_FORMAT,
        "source_issue": dict(issue),
        "images": images,
        "conditions": conditions,
    }


def _run_logged(argv: Sequence[str], *, log: Any, environment: Mapping[str, str]) -> None:
    log.write(f"command={json.dumps(list(argv), separators=(',', ':'))}\n")
    log.flush()
    completed = subprocess.run(
        list(argv),
        check=False,
        env=dict(environment),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.flush()
    if completed.returncode != 0:
        raise HistoricalImageError(f"command failed with exit {completed.returncode}: {argv[0]}")


def _probe(sif_path: Path, *, log: Any, environment: Mapping[str, str]) -> Mapping[str, Any]:
    script = (
        "import json,platform,torch,vllm;"
        "print('COMMCANARY_VERSION='+json.dumps({"
        "'python':platform.python_version(),'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'vllm':vllm.__version__},sort_keys=True))"
    )
    argv = ["apptainer", "exec", "--cleanenv", str(sif_path), "python3", "-c", script]
    log.write(f"command={json.dumps(argv, separators=(',', ':'))}\n")
    log.flush()
    completed = subprocess.run(argv, check=False, env=dict(environment), capture_output=True, text=True)
    log.write(completed.stdout)
    log.write(completed.stderr)
    log.flush()
    if completed.returncode != 0:
        raise HistoricalImageError(f"runtime version probe failed with exit {completed.returncode}")
    markers = [
        line.removeprefix("COMMCANARY_VERSION=")
        for line in completed.stdout.splitlines()
        if line.startswith("COMMCANARY_VERSION=")
    ]
    if len(markers) != 1:
        raise HistoricalImageError("runtime version probe did not emit one result marker")
    try:
        value = json.loads(markers[0])
    except json.JSONDecodeError as exc:
        raise HistoricalImageError("runtime version probe emitted invalid JSON") from exc
    return _object(value, "runtime version probe")


def build_historical_images(lock_path: Path, output_directory: Path) -> Mapping[str, Any]:
    """Build each locked Docker manifest into one measured SIF artifact."""

    lock_path = lock_path.resolve(strict=True)
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical image build: {output}")
    lock = validate_image_lock(_read_json(lock_path, "historical image lock"))
    output.mkdir(mode=0o700)
    _write_new(output / "image-lock.json", _canonical_bytes(lock) + b"\n")
    built_rows: List[Dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="commcanary-historical-images-") as scratch_text:
            scratch = Path(scratch_text)
            environment = dict(os.environ)
            environment["APPTAINER_CACHEDIR"] = str(scratch / "cache")
            environment["APPTAINER_TMPDIR"] = str(scratch / "tmp")
            for image in lock["images"]:
                identifier = str(image["id"])
                sif_path = output / f"{identifier}.sif"
                log_path = output / f"{identifier}.build.log"
                with log_path.open("x", encoding="utf-8") as log:
                    _run_logged(
                        [
                            "apptainer",
                            "build",
                            "--disable-cache",
                            str(sif_path),
                            f"docker://{image['repository']}@{image['manifest_digest']}",
                        ],
                        log=log,
                        environment=environment,
                    )
                    versions = _probe(sif_path, log=log, environment=environment)
                if versions.get("vllm") != image["expected_vllm_version"]:
                    raise HistoricalImageError(f"built {identifier} reports vLLM {versions.get('vllm')!r}")
                os.chmod(sif_path, 0o444)
                os.chmod(log_path, 0o444)
                built_rows.append(
                    {
                        "id": identifier,
                        "repository": image["repository"],
                        "tag_observed": image["tag_observed"],
                        "manifest_digest": image["manifest_digest"],
                        "sif_path": str(sif_path),
                        "sif_identity": _identity(sif_path),
                        "build_log_identity": _identity(log_path),
                        "versions": dict(versions),
                    }
                )
        raw_descriptor: Dict[str, Any] = {
            "format": DESCRIPTOR_FORMAT,
            "status": "complete",
            "host": platform.node(),
            "architecture": platform.machine(),
            "image_lock_identity": _identity(output / "image-lock.json"),
            "images": built_rows,
        }
        raw_descriptor["descriptor_id"] = hashlib.sha256(_canonical_bytes(raw_descriptor)).hexdigest()
        _write_json(output / "descriptor.json", raw_descriptor)
        return raw_descriptor
    except BaseException as exc:
        failure: Dict[str, Any] = {
            "format": DESCRIPTOR_FORMAT,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_image_ids": [row["id"] for row in built_rows],
        }
        failure["failure_id"] = hashlib.sha256(_canonical_bytes(failure)).hexdigest()
        _write_json(output / "failure.json", failure)
        raise


def validate_image_descriptor(descriptor_path: Path) -> Dict[str, Any]:
    """Rehash every SIF and log named by a completed build descriptor."""

    descriptor_path = descriptor_path.resolve(strict=True)
    descriptor = _read_json(descriptor_path, "historical image descriptor")
    _closed(
        descriptor,
        {"format", "descriptor_id", "status", "host", "architecture", "image_lock_identity", "images"},
        "historical image descriptor",
    )
    if descriptor.get("format") != DESCRIPTOR_FORMAT or descriptor.get("status") != "complete":
        raise HistoricalImageError("historical image descriptor is not complete")
    expected_id = hashlib.sha256(
        _canonical_bytes({key: value for key, value in descriptor.items() if key != "descriptor_id"})
    ).hexdigest()
    if descriptor.get("descriptor_id") != expected_id:
        raise HistoricalImageError("historical image descriptor identity does not recompute")
    root = descriptor_path.parent
    lock_path = root / "image-lock.json"
    lock = validate_image_lock(_read_json(lock_path, "built historical image lock"))
    if descriptor.get("image_lock_identity") != _identity(lock_path):
        raise HistoricalImageError("historical image lock identity changed after build")
    images = descriptor.get("images")
    if not isinstance(images, list) or len(images) != 3:
        raise HistoricalImageError("historical image descriptor must contain three images")
    locked = {str(row["id"]): row for row in lock["images"]}
    expected_names = {"descriptor.json", "image-lock.json"}
    for raw_row in images:
        row = _object(raw_row, "historical image descriptor row")
        _closed(
            row,
            {
                "id",
                "repository",
                "tag_observed",
                "manifest_digest",
                "sif_path",
                "sif_identity",
                "build_log_identity",
                "versions",
            },
            "historical image descriptor row",
        )
        identifier = _safe_id(row.get("id"), "historical descriptor image id")
        if identifier not in locked:
            raise HistoricalImageError("historical descriptor contains an unlocked image")
        locked_row = locked[identifier]
        for field in ("repository", "tag_observed", "manifest_digest"):
            if row.get(field) != locked_row[field]:
                raise HistoricalImageError(f"historical descriptor image {identifier} changed {field}")
        sif_path = Path(str(row.get("sif_path"))).resolve(strict=True)
        if sif_path != root / f"{identifier}.sif" or row.get("sif_identity") != _identity(sif_path):
            raise HistoricalImageError(f"historical image {identifier} SIF identity changed")
        log_path = root / f"{identifier}.build.log"
        if row.get("build_log_identity") != _identity(log_path):
            raise HistoricalImageError(f"historical image {identifier} build log identity changed")
        versions = _object(row.get("versions"), "historical image versions")
        if versions.get("vllm") != locked_row["expected_vllm_version"]:
            raise HistoricalImageError(f"historical image {identifier} version no longer matches its lock")
        expected_names.update({sif_path.name, log_path.name})
    names = {path.name for path in root.iterdir()}
    if names != expected_names:
        raise HistoricalImageError("historical image build inventory is not exact")
    return dict(descriptor)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    descriptor = build_historical_images(Path(args.lock), Path(args.output))
    print(json.dumps({"descriptor_id": descriptor["descriptor_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HistoricalImageError",
    "build_historical_images",
    "validate_image_descriptor",
    "validate_image_lock",
]
