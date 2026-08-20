"""Stage and inventory the issue-date OpenHermes model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOCK_FORMAT = "commcanary.historical_vllm_model_lock.v1"
DESCRIPTOR_FORMAT = "commcanary.historical_vllm_model_descriptor.v1"
MAX_JSON_BYTES = 4 * 1024 * 1024
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class HistoricalModelError(RuntimeError):
    """Raised when the historical model snapshot is incomplete or changed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _closed(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    if set(value) != set(expected):
        raise HistoricalModelError(f"{label} fields are not closed")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalModelError(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalModelError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as handle:
        raw = handle.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise HistoricalModelError(f"{label} exceeds {MAX_JSON_BYTES} bytes")

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HistoricalModelError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalModelError(f"{label} is not strict UTF-8 JSON") from exc
    return _object(value, label)


def _identity(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HistoricalModelError(f"model artifact is missing or unsafe: {path}")
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


def _copy_new(source: Path, target: Path) -> Dict[str, Any]:
    expected = _identity(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.chmod(target, 0o444)
    observed = _identity(target)
    if observed != expected:
        raise HistoricalModelError(f"model file changed while staging: {target.name}")
    return observed


def validate_model_lock(raw_lock: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the exact issue-date model source lock."""

    _closed(
        raw_lock,
        {"format", "repository", "revision", "revision_source", "required_files"},
        "historical model lock",
    )
    if raw_lock.get("format") != LOCK_FORMAT:
        raise HistoricalModelError("historical model lock format is unsupported")
    if raw_lock.get("repository") != "teknium/OpenHermes-2.5-Mistral-7B":
        raise HistoricalModelError("historical model repository is unsupported")
    revision = raw_lock.get("revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise HistoricalModelError("historical model revision must be a full commit hash")
    if raw_lock.get("revision_source") != (
        f"https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/commit/{revision}"
    ):
        raise HistoricalModelError("historical model revision source is unsupported")
    files = raw_lock.get("required_files")
    if (
        not isinstance(files, list)
        or not files
        or files != sorted(set(files))
        or any(not isinstance(name, str) or not name or Path(name).name != name or "\x00" in name for name in files)
    ):
        raise HistoricalModelError("historical model required_files must be canonical basenames")
    expected = {
        "added_tokens.json",
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
    if set(files) != expected:
        raise HistoricalModelError("historical model file inventory is not the reviewed safetensors subset")
    return {
        "format": LOCK_FORMAT,
        "repository": str(raw_lock["repository"]),
        "revision": revision,
        "revision_source": str(raw_lock["revision_source"]),
        "required_files": list(files),
    }


def stage_historical_model(lock_path: Path, cache_directory: Path, output_directory: Path) -> Mapping[str, Any]:
    """Download only the locked files, then copy and hash each regular file."""

    lock_path = lock_path.resolve(strict=True)
    cache = cache_directory.resolve()
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical model staging: {output}")
    lock = validate_model_lock(_read_json(lock_path, "historical model lock"))
    if cache.is_symlink():
        raise HistoricalModelError("historical model cache may not be a symlink")
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    model = output / "model"
    model.mkdir(mode=0o700)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        rows: List[Dict[str, Any]] = []
        for name in lock["required_files"]:
            downloaded = Path(
                hf_hub_download(
                    repo_id=str(lock["repository"]),
                    filename=str(name),
                    revision=str(lock["revision"]),
                    cache_dir=str(cache),
                )
            ).resolve(strict=True)
            identity = _copy_new(downloaded, model / str(name))
            rows.append({"path": str(name), **identity})
        raw_descriptor: Dict[str, Any] = {
            "format": DESCRIPTOR_FORMAT,
            "status": "complete",
            "source": {
                "repository": lock["repository"],
                "revision": lock["revision"],
                "revision_source": lock["revision_source"],
            },
            "model_path": str(model),
            "files": rows,
            "model_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
        }
        raw_descriptor["descriptor_id"] = hashlib.sha256(_canonical_bytes(raw_descriptor)).hexdigest()
        _write_new(output / "model-lock.json", _canonical_bytes(lock) + b"\n")
        _write_new(output / "descriptor.json", _canonical_bytes(raw_descriptor) + b"\n")
        os.chmod(model, 0o555)
        return raw_descriptor
    except BaseException as exc:
        failure: Dict[str, Any] = {
            "format": DESCRIPTOR_FORMAT,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failure["failure_id"] = hashlib.sha256(_canonical_bytes(failure)).hexdigest()
        _write_new(output / "failure.json", _canonical_bytes(failure) + b"\n")
        raise


def validate_model_descriptor(descriptor_path: Path) -> Dict[str, Any]:
    """Recompute the complete staged model inventory and descriptor identity."""

    descriptor_path = descriptor_path.resolve(strict=True)
    descriptor = _read_json(descriptor_path, "historical model descriptor")
    _closed(
        descriptor,
        {"format", "descriptor_id", "status", "source", "model_path", "files", "model_sha256"},
        "historical model descriptor",
    )
    if descriptor.get("format") != DESCRIPTOR_FORMAT or descriptor.get("status") != "complete":
        raise HistoricalModelError("historical model descriptor is not complete")
    expected_id = hashlib.sha256(
        _canonical_bytes({key: value for key, value in descriptor.items() if key != "descriptor_id"})
    ).hexdigest()
    if descriptor.get("descriptor_id") != expected_id:
        raise HistoricalModelError("historical model descriptor identity does not recompute")
    lock_path = descriptor_path.parent / "model-lock.json"
    lock = validate_model_lock(_read_json(lock_path, "staged historical model lock"))
    source = _object(descriptor.get("source"), "historical model source")
    if source != {
        "repository": lock["repository"],
        "revision": lock["revision"],
        "revision_source": lock["revision_source"],
    }:
        raise HistoricalModelError("historical model source differs from its lock")
    model = Path(str(descriptor.get("model_path"))).resolve(strict=True)
    if model != descriptor_path.parent / "model" or model.is_symlink() or not model.is_dir():
        raise HistoricalModelError("historical model path is unsafe")
    names = sorted(path.name for path in model.iterdir())
    if names != lock["required_files"]:
        raise HistoricalModelError("historical model directory inventory is not exact")
    if any(path.is_symlink() or not path.is_file() for path in model.iterdir()):
        raise HistoricalModelError("historical model directory contains an unsafe member")
    rows = [{"path": name, **_identity(model / name)} for name in names]
    if descriptor.get("files") != rows:
        raise HistoricalModelError("historical model file inventory changed after staging")
    expected_model_sha256 = hashlib.sha256(_canonical_bytes(rows)).hexdigest()
    if descriptor.get("model_sha256") != expected_model_sha256:
        raise HistoricalModelError("historical model identity does not recompute")
    root_names = {path.name for path in descriptor_path.parent.iterdir()}
    if root_names != {"descriptor.json", "model-lock.json", "model"}:
        raise HistoricalModelError("historical model staging inventory is not exact")
    return dict(descriptor)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    descriptor = stage_historical_model(Path(args.lock), Path(args.cache), Path(args.output))
    print(json.dumps({"descriptor_id": descriptor["descriptor_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HistoricalModelError",
    "stage_historical_model",
    "validate_model_descriptor",
    "validate_model_lock",
]
