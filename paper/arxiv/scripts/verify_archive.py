#!/usr/bin/env python3
"""Verify a source archive's safety, normalization, and member hashes."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

ARCHIVE_ROOT = "commcanary-arxiv-source"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_archive.py ARCHIVE.tar.gz")
    archive = Path(sys.argv[1]).resolve()
    observed: dict[str, str] = {}
    manifest_data: bytes | None = None

    with tarfile.open(archive, mode="r:gz") as tar:
        names: set[str] = set()
        for member in tar.getmembers():
            if member.name in names:
                raise SystemExit(f"duplicate archive member: {member.name}")
            names.add(member.name)
            if member.issym() or member.islnk() or not member.isfile():
                raise SystemExit(f"unsupported archive member type: {member.name}")
            if member.mtime != 0 or member.uid != 0 or member.gid != 0:
                raise SystemExit(f"non-normalized archive metadata: {member.name}")
            prefix = f"{ARCHIVE_ROOT}/"
            if not member.name.startswith(prefix):
                raise SystemExit(f"member escapes archive root: {member.name}")
            relative = member.name[len(prefix) :]
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise SystemExit(f"unsafe archive path: {member.name}")
            stream = tar.extractfile(member)
            if stream is None:
                raise SystemExit(f"unreadable archive member: {member.name}")
            data = stream.read()
            if relative == "MANIFEST.json":
                manifest_data = data
            else:
                observed[relative] = _digest(data)

    if manifest_data is None:
        raise SystemExit("archive has no MANIFEST.json")
    manifest = json.loads(manifest_data)
    if manifest.get("archive_root") != ARCHIVE_ROOT:
        raise SystemExit("archive root does not match manifest")
    if manifest.get("schema") != "commcanary.paper.source-archive-manifest.v1":
        raise SystemExit("unexpected archive manifest schema")
    if observed != manifest.get("members"):
        raise SystemExit("archive member hashes do not match MANIFEST.json")
    print(f"verified source archive: {len(observed)} members, sha256={_digest(archive.read_bytes())}")


if __name__ == "__main__":
    main()
