from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from tools import release_metadata

PROJECT = "commcanary"
VERSION = "0.3.0"
EPOCH = "1704067200"


def _metadata(*, version: str = VERSION, runtime_requirement: Optional[str] = None) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {PROJECT}",
        f"Version: {version}",
        "License: MIT",
        "Requires-Python: >=3.9",
        "Provides-Extra: test",
        'Requires-Dist: pytest>=8; extra == "test"',
    ]
    if runtime_requirement is not None:
        lines.append(f"Requires-Dist: {runtime_requirement}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def _zip_info(name: str, *, mode: Optional[int] = None) -> zipfile.ZipInfo:
    timestamp = datetime.fromtimestamp(int(EPOCH), tz=timezone.utc).timetuple()[:6]
    info = zipfile.ZipInfo(name, date_time=timestamp)
    info.compress_type = zipfile.ZIP_DEFLATED
    if mode is not None:
        info.create_system = 3
        info.external_attr = mode << 16
    return info


def _write_wheel(
    path: Path,
    *,
    metadata: Optional[bytes] = None,
    extra: Optional[List[Tuple[zipfile.ZipInfo, bytes]]] = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(_zip_info("commcanary/__init__.py"), b'__version__ = "0.3.0"\n')
        archive.writestr(
            _zip_info("commcanary-0.3.0.dist-info/METADATA"),
            _metadata() if metadata is None else metadata,
        )
        for info, payload in extra or []:
            archive.writestr(info, payload)


def _tar_info(name: str, payload: Optional[bytes] = None) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mtime = int(EPOCH)
    if payload is not None:
        info.size = len(payload)
    return info


def _write_sdist(
    path: Path,
    *,
    metadata: Optional[bytes] = None,
    special: Optional[tarfile.TarInfo] = None,
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        package_metadata = _metadata() if metadata is None else metadata
        for name, payload in (
            ("commcanary-0.3.0/PKG-INFO", package_metadata),
            ("commcanary-0.3.0/src/commcanary/__init__.py", b'__version__ = "0.3.0"\n'),
        ):
            archive.addfile(_tar_info(name, payload), io.BytesIO(payload))
        if special is not None:
            archive.addfile(special)


def _distributions(tmp_path: Path) -> release_metadata.DistributionSet:
    directory = tmp_path / "dist"
    directory.mkdir()
    wheel = directory / "commcanary-0.3.0-py3-none-any.whl"
    sdist = directory / "commcanary-0.3.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)
    return release_metadata.DistributionSet(wheel=wheel, sdist=sdist)


def _write(
    artifacts: release_metadata.DistributionSet,
    output: Path,
) -> release_metadata.ReleaseMetadataFiles:
    return release_metadata.write_release_metadata(
        artifacts,
        output,
        project=PROJECT,
        version=VERSION,
        source_date_epoch=EPOCH,
    )


def test_release_metadata_is_deterministic_canonical_and_spdx_shaped(tmp_path: Path) -> None:
    artifacts = _distributions(tmp_path)

    first = _write(artifacts, tmp_path / "metadata-one")
    second = _write(artifacts, tmp_path / "metadata-two")

    first_payloads = {path.name: path.read_bytes() for path in first.__dict__.values()}
    second_payloads = {path.name: path.read_bytes() for path in second.__dict__.values()}
    assert first_payloads == second_payloads
    assert first.inventory.read_bytes().endswith(b"\n")
    assert b"\n " not in first.inventory.read_bytes()

    inventory = json.loads(first.inventory.read_text(encoding="utf-8"))
    assert inventory["schema"] == "commcanary.release-inventory.v1"
    assert inventory["project"] == PROJECT
    assert inventory["version"] == VERSION
    assert inventory["source_date_epoch"] == int(EPOCH)
    assert inventory["package_metadata"]["runtime_requirements"] == []
    assert len(inventory["artifacts"]) == 2
    for artifact in inventory["artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["size"] > 0
        assert artifact["members"]
        for member in artifact["members"]:
            assert member["size"] >= 0
            if member["type"] == "file":
                assert len(member["sha256"]) == 64

    expected_checksum_lines = [f"{artifact['sha256']}  {artifact['filename']}" for artifact in inventory["artifacts"]]
    assert first.checksums.read_text(encoding="ascii").splitlines() == expected_checksum_lines

    sbom = json.loads(first.sbom.read_text(encoding="utf-8"))
    assert sbom["spdxVersion"] == "SPDX-2.3"
    assert sbom["SPDXID"] == "SPDXRef-DOCUMENT"
    assert sbom["dataLicense"] == "CC0-1.0"
    assert sbom["creationInfo"]["created"] == "2024-01-01T00:00:00Z"
    assert sbom["packages"][0]["versionInfo"] == VERSION
    assert sbom["packages"][0]["filesAnalyzed"] is True
    assert {file["fileName"] for file in sbom["files"]} == {
        f"./{artifacts.wheel.name}",
        f"./{artifacts.sdist.name}",
    }
    assert all(relationship["relationshipType"] != "DEPENDS_ON" for relationship in sbom["relationships"])


def test_release_metadata_rejects_extra_missing_and_colliding_distributions(tmp_path: Path) -> None:
    artifacts = _distributions(tmp_path)
    extra = artifacts.wheel.parent / "unrelated-1.0-py3-none-any.whl"
    extra.write_bytes(b"not a wheel")
    with pytest.raises(release_metadata.ReleaseMetadataError, match="extra"):
        _write(artifacts, tmp_path / "extra-output")

    extra.unlink()
    artifacts.sdist.unlink()
    with pytest.raises(release_metadata.ReleaseMetadataError, match="missing sdist"):
        _write(artifacts, tmp_path / "missing-output")

    artifacts.sdist.mkdir()
    with pytest.raises(release_metadata.ReleaseMetadataError, match="missing sdist"):
        _write(artifacts, tmp_path / "directory-output")

    artifacts.sdist.rmdir()
    os.link(artifacts.wheel, artifacts.sdist)
    with pytest.raises(release_metadata.ReleaseMetadataError, match="paths collide"):
        _write(artifacts, tmp_path / "colliding-output")


def test_release_metadata_rejects_stale_and_runtime_dependency_metadata(tmp_path: Path) -> None:
    artifacts = _distributions(tmp_path)
    _write_wheel(artifacts.wheel, metadata=_metadata(version="0.2.0"))
    with pytest.raises(release_metadata.ReleaseMetadataError, match="metadata version.*stale"):
        _write(artifacts, tmp_path / "stale-output")

    disguised_runtime = 'requests>=2; python_version >= "3.9" or extra == "test"'
    _write_wheel(artifacts.wheel, metadata=_metadata(runtime_requirement=disguised_runtime))
    _write_sdist(artifacts.sdist, metadata=_metadata(runtime_requirement=disguised_runtime))
    with pytest.raises(release_metadata.ReleaseMetadataError, match="runtime dependencies"):
        _write(artifacts, tmp_path / "dependency-output")


@pytest.mark.parametrize("member_name", ["../escape.py", "/absolute.py", "a/../../escape.py", "C:/escape.py"])
def test_release_metadata_rejects_unsafe_wheel_members(tmp_path: Path, member_name: str) -> None:
    artifacts = _distributions(tmp_path)
    _write_wheel(artifacts.wheel, extra=[(_zip_info(member_name), b"escape")])

    with pytest.raises(release_metadata.ReleaseMetadataError, match="unsafe archive member"):
        _write(artifacts, tmp_path / "metadata")


def test_release_metadata_rejects_duplicate_and_case_colliding_members(tmp_path: Path) -> None:
    artifacts = _distributions(tmp_path)
    _write_wheel(
        artifacts.wheel,
        extra=[
            (_zip_info("commcanary/DUPLICATE.py"), b"first"),
            (_zip_info("commcanary/duplicate.py"), b"second"),
        ],
    )

    with pytest.raises(release_metadata.ReleaseMetadataError, match="duplicate or colliding"):
        _write(artifacts, tmp_path / "metadata")


def test_release_metadata_rejects_symlinks_and_special_archive_members(tmp_path: Path) -> None:
    artifacts = _distributions(tmp_path)
    symlink = _zip_info("commcanary/link", mode=stat.S_IFLNK | 0o777)
    _write_wheel(artifacts.wheel, extra=[(symlink, b"target")])
    with pytest.raises(release_metadata.ReleaseMetadataError, match="symlink or special"):
        _write(artifacts, tmp_path / "wheel-output")

    _write_wheel(artifacts.wheel)
    fifo = _tar_info("commcanary-0.3.0/fifo")
    fifo.type = tarfile.FIFOTYPE
    _write_sdist(artifacts.sdist, special=fifo)
    with pytest.raises(release_metadata.ReleaseMetadataError, match="symlink or special"):
        _write(artifacts, tmp_path / "sdist-output")


def test_release_metadata_rejects_nonempty_output_and_active_writer(tmp_path: Path) -> None:
    artifacts = _distributions(tmp_path)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(release_metadata.ReleaseMetadataError, match="not empty"):
        _write(artifacts, nonempty)
    assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "keep"

    destination = tmp_path / "locked"
    lock = tmp_path / ".locked.commcanary-release-metadata.lock"
    lock.write_text("another writer", encoding="utf-8")
    with pytest.raises(release_metadata.ReleaseMetadataError, match="already being created"):
        _write(artifacts, destination)
    assert not destination.exists()


def test_release_metadata_fails_closed_on_nondeterministic_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _distributions(tmp_path)
    original = release_metadata._render_release_metadata
    calls = 0

    def unstable(*args: object, **kwargs: object) -> Dict[str, bytes]:
        nonlocal calls
        calls += 1
        rendered = original(*args, **kwargs)  # type: ignore[arg-type]
        if calls == 2:
            rendered = dict(rendered)
            rendered[release_metadata.INVENTORY_NAME] += b" "
        return rendered

    monkeypatch.setattr(release_metadata, "_render_release_metadata", unstable)
    output = tmp_path / "metadata"
    with pytest.raises(release_metadata.ReleaseMetadataError, match="nondeterministic"):
        _write(artifacts, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".metadata.staging-*"))


def test_release_metadata_binds_output_to_previously_tested_hashes(tmp_path: Path) -> None:
    artifacts = _distributions(tmp_path)
    tested_hashes = {
        artifacts.wheel.name: "0" * 64,
        artifacts.sdist.name: hashlib.sha256(artifacts.sdist.read_bytes()).hexdigest(),
    }
    output = tmp_path / "metadata"

    with pytest.raises(release_metadata.ReleaseMetadataError, match="changed after artifact testing"):
        release_metadata.write_release_metadata(
            artifacts,
            output,
            project=PROJECT,
            version=VERSION,
            source_date_epoch=EPOCH,
            expected_sha256=tested_hashes,
        )
    assert not output.exists()


def test_release_metadata_output_directory_is_read_only_with_respect_to_collisions(tmp_path: Path) -> None:
    artifacts = _distributions(tmp_path)
    output = tmp_path / "metadata"
    files = _write(artifacts, output)
    original = files.checksums.read_bytes()

    with pytest.raises(release_metadata.ReleaseMetadataError, match="not empty"):
        _write(artifacts, output)
    assert files.checksums.read_bytes() == original
    assert os.path.samefile(files.checksums.parent, output)
