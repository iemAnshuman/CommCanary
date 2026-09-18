#!/usr/bin/env python3
"""Create a deterministic source tarball with a member hash manifest."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "commcanary-arxiv-source"
EXCLUDED_PARTS = {"__pycache__", ".DS_Store"}


def _source_files() -> list[Path]:
    files = []
    for path in SOURCE_ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(SOURCE_ROOT).as_posix())


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tar_info(name: str, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: package_source.py OUTPUT.tar.gz")
    output = Path(sys.argv[1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    members: list[tuple[str, bytes, int]] = []
    manifest: dict[str, str] = {}
    for path in _source_files():
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        data = path.read_bytes()
        mode = 0o755 if relative.startswith("scripts/") else 0o644
        members.append((relative, data, mode))
        manifest[relative] = _digest(data)

    manifest_data = (
        json.dumps(
            {
                "archive_root": ARCHIVE_ROOT,
                "members": manifest,
                "schema": "commcanary.paper.source-archive-manifest.v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    members.append(("MANIFEST.json", manifest_data, 0o644))

    with output.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for relative, data, mode in sorted(members):
                    name = f"{ARCHIVE_ROOT}/{relative}"
                    tar.addfile(_tar_info(name, len(data), mode), io.BytesIO(data))

    print(f"wrote {output} ({output.stat().st_size} bytes, sha256={_digest(output.read_bytes())})")


if __name__ == "__main__":
    main()
