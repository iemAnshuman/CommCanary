from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from experiments.rostam.product_canary.build_runner import (
    MAX_RUNNER_LOCK_BYTES,
    _oci_manifest,
    _read_lock,
    _validate_lock,
    _version_probe,
)

ROOT = Path(__file__).resolve().parents[3]
RUNNERS = ROOT / "experiments" / "rostam" / "product_canary"


@pytest.mark.parametrize(
    ("lock_path", "container_path", "engine"),
    [
        (RUNNERS / "runner-lock.json", RUNNERS / "Containerfile", "vllm"),
        (RUNNERS / "sglang" / "runner-lock.json", RUNNERS / "sglang" / "Containerfile", "sglang"),
    ],
)
def test_product_runner_locks_bind_their_base_and_engine(
    lock_path: Path,
    container_path: Path,
    engine: str,
) -> None:
    lock = _validate_lock(json.loads(lock_path.read_text(encoding="utf-8")))

    assert lock["expected_application_engine"] == engine
    assert lock["base_manifest_digest"] in container_path.read_text(encoding="utf-8")
    compile(_version_probe(engine), f"<{engine}-version-probe>", "exec")


def test_product_runner_lock_rejects_cross_engine_protocol() -> None:
    lock = json.loads((RUNNERS / "sglang" / "runner-lock.json").read_text(encoding="utf-8"))
    lock["runner_protocols"][-1] = "vllm-offline-tensor-parallel.v1"

    with pytest.raises(ValueError, match="protocols do not match"):
        _validate_lock(lock)


def test_product_runner_lock_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "runner-lock.json"
    path.write_text('{"format":"first","format":"second"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key 'format'"):
        _read_lock(path)


def test_product_runner_lock_reader_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "runner-lock.json"
    path.write_bytes(b" " * (MAX_RUNNER_LOCK_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds"):
        _read_lock(path)


def _oci_archive(
    path: Path,
    config: Dict[str, Any],
    *,
    digest_override: Optional[str] = None,
    size_override: Optional[int] = None,
    include_config: bool = True,
) -> None:
    config_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
    config_digest = digest_override or hashlib.sha256(config_bytes).hexdigest()
    config_size = len(config_bytes) if size_override is None else size_override
    manifest = {
        "schemaVersion": 2,
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": f"sha256:{config_digest}",
            "size": config_size,
        },
        "layers": [],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    index = {
        "schemaVersion": 2,
        "manifests": [
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": f"sha256:{manifest_digest}",
                "size": len(manifest_bytes),
            }
        ],
    }
    members = {
        "index.json": json.dumps(index, sort_keys=True).encode("utf-8"),
        f"blobs/sha256/{manifest_digest}": manifest_bytes,
    }
    if include_config:
        members[f"blobs/sha256/{config_digest}"] = config_bytes
    with tarfile.open(path, mode="w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_oci_manifest_binds_linux_amd64_config(tmp_path: Path) -> None:
    path = tmp_path / "runner.oci.tar"
    _oci_archive(path, {"os": "linux", "architecture": "amd64"})

    manifest = _oci_manifest(path, expected_architecture="amd64")

    assert manifest["platform"] == {"os": "linux", "architecture": "amd64"}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"os": "windows", "architecture": "amd64"}, "operating system"),
        ({"os": "linux", "architecture": "arm64"}, "architecture"),
    ],
)
def test_oci_manifest_rejects_wrong_platform(
    tmp_path: Path,
    config: Dict[str, Any],
    message: str,
) -> None:
    path = tmp_path / "runner.oci.tar"
    _oci_archive(path, config)

    with pytest.raises(RuntimeError, match=message):
        _oci_manifest(path, expected_architecture="amd64")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"digest_override": "0" * 64}, "does not match"),
        ({"size_override": 1}, "size does not match"),
        ({"include_config": False}, "lacks its content-addressed config"),
    ],
)
def test_oci_manifest_rejects_unbound_config_blob(
    tmp_path: Path,
    overrides: Dict[str, Any],
    message: str,
) -> None:
    path = tmp_path / "runner.oci.tar"
    _oci_archive(path, {"os": "linux", "architecture": "amd64"}, **overrides)

    with pytest.raises(RuntimeError, match=message):
        _oci_manifest(path, expected_architecture="amd64")
