"""Build one content-addressed OCI archive and Apptainer SIF on Rostam."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

FORMAT = "commcanary.product_runner_descriptor.v1"
MAX_OCI_JSON_BYTES = 4 * 1024 * 1024
MAX_RUNNER_LOCK_BYTES = 256 * 1024
MAX_CONTAINERFILE_BYTES = 1024 * 1024
_ENGINE_PROTOCOLS = {
    "vllm": "vllm-offline-tensor-parallel.v1",
    "sglang": "sglang-offline-tensor-parallel.v1",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _identity(path: Path) -> Dict[str, Any]:
    digest, size = _sha256_file(path)
    return {"sha256": digest, "bytes": size}


def _run(command: Sequence[str], *, log: Any, capture: bool = False) -> str:
    log.write(f"command={json.dumps(list(command), separators=(',', ':'))}\n")
    log.flush()
    if capture:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        log.write(completed.stdout)
        log.write(completed.stderr)
        log.flush()
        if completed.returncode != 0:
            raise RuntimeError(f"command failed with exit {completed.returncode}: {command[0]}")
        return completed.stdout
    completed = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT, text=True)
    log.flush()
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {command[0]}")
    return ""


def _read_oci_json(archive: tarfile.TarFile, member_name: str) -> Mapping[str, Any]:
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise RuntimeError(f"OCI archive lacks {member_name}") from exc
    if not member.isfile() or member.size > MAX_OCI_JSON_BYTES:
        raise RuntimeError(f"OCI archive member {member_name} is not a bounded regular file")
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError(f"cannot read OCI archive member {member_name}")
    raw = stream.read(MAX_OCI_JSON_BYTES + 1)
    if len(raw) != member.size:
        raise RuntimeError(f"OCI archive member {member_name} changed while read")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"OCI archive member {member_name} must contain an object")
    return value


def _read_oci_config(
    archive: tarfile.TarFile,
    manifest: Mapping[str, Any],
    *,
    expected_architecture: str,
) -> Dict[str, str]:
    descriptor = manifest.get("config")
    if not isinstance(descriptor, Mapping):
        raise RuntimeError("OCI manifest config descriptor is missing")
    if descriptor.get("mediaType") != "application/vnd.oci.image.config.v1+json":
        raise RuntimeError("OCI manifest config media type is unsupported")
    digest = descriptor.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError("OCI manifest config must use a SHA-256 digest")
    digest_hex = digest.partition(":")[2]
    if len(digest_hex) != 64 or any(character not in "0123456789abcdef" for character in digest_hex):
        raise RuntimeError("OCI manifest config digest is malformed")
    declared_size = descriptor.get("size")
    if (
        isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or declared_size < 1
        or declared_size > MAX_OCI_JSON_BYTES
    ):
        raise RuntimeError("OCI manifest config size is invalid")

    member_name = f"blobs/sha256/{digest_hex}"
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise RuntimeError("OCI archive lacks its content-addressed config blob") from exc
    if not member.isfile() or member.size != declared_size:
        raise RuntimeError("OCI archive config blob size does not match its descriptor")
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError("cannot read OCI config blob")
    raw = stream.read(MAX_OCI_JSON_BYTES + 1)
    if len(raw) != declared_size or hashlib.sha256(raw).hexdigest() != digest_hex:
        raise RuntimeError("OCI archive config blob does not match its descriptor")
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("OCI config blob must be strict UTF-8 JSON") from exc
    if not isinstance(config, Mapping):
        raise RuntimeError("OCI config blob must contain an object")
    if config.get("os") != "linux":
        raise RuntimeError("OCI config operating system must be linux")
    if config.get("architecture") != expected_architecture:
        raise RuntimeError(f"OCI config architecture must be {expected_architecture!r}")
    return {"os": "linux", "architecture": expected_architecture}


def _oci_manifest(archive_path: Path, *, expected_architecture: str) -> Dict[str, Any]:
    with tarfile.open(archive_path, mode="r") as archive:
        names = archive.getnames()
        if len(names) != len(set(names)):
            raise RuntimeError("OCI archive contains duplicate member names")
        index = _read_oci_json(archive, "index.json")
        manifests = index.get("manifests")
        if not isinstance(manifests, list) or len(manifests) != 1 or not isinstance(manifests[0], Mapping):
            raise RuntimeError("OCI archive index must contain exactly one manifest")
        descriptor = manifests[0]
        digest = descriptor.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise RuntimeError("OCI archive manifest must use a SHA-256 digest")
        digest_hex = digest.partition(":")[2]
        if len(digest_hex) != 64 or any(character not in "0123456789abcdef" for character in digest_hex):
            raise RuntimeError("OCI archive manifest digest is malformed")
        blob_name = f"blobs/sha256/{digest_hex}"
        try:
            blob = archive.getmember(blob_name)
        except KeyError as exc:
            raise RuntimeError("OCI archive lacks its content-addressed manifest blob") from exc
        if not blob.isfile() or blob.size != descriptor.get("size"):
            raise RuntimeError("OCI archive manifest blob size does not match its descriptor")
        stream = archive.extractfile(blob)
        if stream is None:
            raise RuntimeError("cannot read OCI manifest blob")
        raw = stream.read(MAX_OCI_JSON_BYTES + 1)
        if len(raw) != blob.size or hashlib.sha256(raw).hexdigest() != digest_hex:
            raise RuntimeError("OCI archive manifest blob does not match its digest")
        manifest = json.loads(raw)
        if not isinstance(manifest, Mapping):
            raise RuntimeError("OCI manifest must contain an object")
        platform_identity = _read_oci_config(
            archive,
            manifest,
            expected_architecture=expected_architecture,
        )
        return {
            "digest": digest,
            "bytes": blob.size,
            "media_type": descriptor.get("mediaType"),
            "config_digest": manifest.get("config", {}).get("digest"),
            "layer_digests": [layer.get("digest") for layer in manifest.get("layers", [])],
            "platform": platform_identity,
        }


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_bounded_file(path: Path, *, limit: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return raw


def _read_lock(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"runner lock contains duplicate key {key!r}")
            result[key] = value
        return result

    raw = _read_bounded_file(path, limit=MAX_RUNNER_LOCK_BYTES, label="runner lock")
    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runner lock must be strict UTF-8 JSON") from exc
    return _validate_lock(value)


def _validate_lock(raw_lock: Any) -> Mapping[str, Any]:
    if not isinstance(raw_lock, Mapping):
        raise ValueError("runner lock must contain an object")
    expected = {
        "format",
        "architecture",
        "base_image",
        "base_manifest_digest",
        "base_tag_observed",
        "expected_application_engine",
        "expected_application_engine_version",
        "runner_protocols",
    }
    if set(raw_lock) != expected:
        raise ValueError("runner lock fields are not closed")
    if raw_lock.get("format") != "commcanary.product_runner_lock.v1":
        raise ValueError("runner lock format is unsupported")
    if raw_lock.get("architecture") != "amd64":
        raise ValueError("runner lock architecture must be amd64")
    digest = raw_lock.get("base_manifest_digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError("runner lock base manifest digest is malformed")
    digest_hex = digest.partition(":")[2]
    if len(digest_hex) != 64 or any(character not in "0123456789abcdef" for character in digest_hex):
        raise ValueError("runner lock base manifest digest is malformed")
    engine = raw_lock.get("expected_application_engine")
    if engine not in _ENGINE_PROTOCOLS:
        raise ValueError("runner lock application engine is unsupported")
    version = raw_lock.get("expected_application_engine_version")
    if not isinstance(version, str) or not version:
        raise ValueError("runner lock application engine version is missing")
    protocols = raw_lock.get("runner_protocols")
    if not isinstance(protocols, list) or protocols != sorted(set(protocols)):
        raise ValueError("runner lock protocols must be a canonical unique array")
    required_protocols = {"chakra-et-collective-graph.v1", _ENGINE_PROTOCOLS[str(engine)]}
    if set(protocols) != required_protocols:
        raise ValueError("runner lock protocols do not match the selected engine")
    return raw_lock


def _version_probe(engine: str) -> str:
    if engine == "vllm":
        engine_import = "import vllm;engine_version=vllm.__version__"
    elif engine == "sglang":
        engine_import = "from sglang.version import __version__ as engine_version"
    else:
        raise ValueError("runner version probe engine is unsupported")
    return (
        "import json,commcanary,torch;" + engine_import + ";print(json.dumps({'commcanary':commcanary.__version__,"
        "'torch':torch.__version__,'application_engine':"
        + repr(engine)
        + ",'application_engine_version':engine_version},sort_keys=True))"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    context = Path(args.context).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runner build attempt: {output}")
    expected_names = {"Containerfile", "commcanary.whl", "runner-lock.json"}
    actual_names = {path.name for path in context.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            f"runner context inventory mismatch: missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for name in expected_names:
        path = context / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"runner context member is missing or unsafe: {name}")
    lock = _read_lock(context / "runner-lock.json")
    container_bytes = _read_bounded_file(
        context / "Containerfile",
        limit=MAX_CONTAINERFILE_BYTES,
        label="runner Containerfile",
    )
    try:
        container_text = container_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("runner Containerfile must be UTF-8") from exc
    if str(lock["base_manifest_digest"]) not in container_text:
        raise ValueError("runner Containerfile does not bind the locked base manifest digest")

    output.mkdir(mode=0o700)
    log_path = output / "build.log"
    podman_root = output / "podman-root"
    podman_runroot = output / "podman-runroot"
    archive_path = output / "runner.oci.tar"
    sif_path = output / "runner.sif"
    apptainer_cache = output / "apptainer-cache"
    apptainer_tmp = output / "apptainer-tmp"
    os.environ["APPTAINER_CACHEDIR"] = str(apptainer_cache)
    os.environ["APPTAINER_TMPDIR"] = str(apptainer_tmp)
    wheel_digest = _identity(context / "commcanary.whl")["sha256"]
    image_name = f"localhost/commcanary-runner:{str(wheel_digest)[:16]}"
    podman = [
        "podman",
        "--storage-driver=vfs",
        "--root",
        str(podman_root),
        "--runroot",
        str(podman_runroot),
    ]
    try:
        with log_path.open("x", encoding="utf-8") as log:
            _run(
                [
                    *podman,
                    "build",
                    "--platform",
                    "linux/amd64",
                    "--pull=always",
                    "--no-cache",
                    "--tag",
                    image_name,
                    "--file",
                    str(context / "Containerfile"),
                    str(context),
                ],
                log=log,
            )
            _run(
                [
                    *podman,
                    "push",
                    "--format=oci",
                    "--remove-signatures",
                    image_name,
                    f"oci-archive:{archive_path}",
                ],
                log=log,
            )
            manifest = _oci_manifest(
                archive_path,
                expected_architecture=str(lock["architecture"]),
            )
            _run(
                ["apptainer", "build", str(sif_path), f"oci-archive://{archive_path}"],
                log=log,
            )
            version_output = (
                _run(
                    [
                        "apptainer",
                        "exec",
                        "--cleanenv",
                        str(sif_path),
                        "python3",
                        "-c",
                        _version_probe(str(lock["expected_application_engine"])),
                    ],
                    log=log,
                    capture=True,
                )
                .strip()
                .splitlines()[-1]
            )
            for scratch in (podman_root, podman_runroot, apptainer_cache, apptainer_tmp):
                if scratch.parent != output or scratch.is_symlink():
                    raise RuntimeError("runner scratch path escaped its build attempt")
                if scratch.is_dir():
                    shutil.rmtree(scratch)
                    log.write(f"removed_scratch={scratch.name}\n")
        versions = json.loads(version_output)
        if versions.get("application_engine") != lock["expected_application_engine"]:
            raise RuntimeError("built runner application engine disagrees with its lock")
        if versions.get("application_engine_version") != lock["expected_application_engine_version"]:
            raise RuntimeError("built runner application engine version disagrees with its lock")
        raw_descriptor: Dict[str, Any] = {
            "format": FORMAT,
            "status": "complete",
            "host": platform.node(),
            "architecture": platform.machine(),
            "base_manifest_digest": lock["base_manifest_digest"],
            "runner_protocols": list(lock["runner_protocols"]),
            "oci_manifest": manifest,
            "artifacts": {
                "runner.oci.tar": _identity(archive_path),
                "runner.sif": _identity(sif_path),
                "build.log": _identity(log_path),
            },
            "inputs": {
                name: _identity(context / name) for name in ("Containerfile", "commcanary.whl", "runner-lock.json")
            },
            "versions": versions,
        }
        raw_descriptor["descriptor_id"] = hashlib.sha256(_canonical_bytes(raw_descriptor)).hexdigest()
        _write_new_json(output / "descriptor.json", raw_descriptor)
        print(json.dumps({"descriptor_id": raw_descriptor["descriptor_id"], "oci_digest": manifest["digest"]}))
        return 0
    except BaseException as exc:
        failure = {
            "format": FORMAT,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "inputs": {
                name: _identity(context / name) for name in ("Containerfile", "commcanary.whl", "runner-lock.json")
            },
        }
        failure["descriptor_id"] = hashlib.sha256(_canonical_bytes(failure)).hexdigest()
        _write_new_json(output / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
