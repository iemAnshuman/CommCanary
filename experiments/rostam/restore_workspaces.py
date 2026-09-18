"""Restore a campaign's selected staging workspaces from its raw archive.

A campaign directory under ``experiments/rostam/results/`` carries its
manifest, attempts, selection and verdicts, but the per-attempt files those
attempts reference (``workspaces/<cell>/<attempt>/result.json``, the logs, and
for trace builds the profile) live only inside the campaign's normalized raw
archive. The analyzer verifies every referenced file before it regenerates a
publication, so a fresh clone cannot regenerate anything until they are back.

This restores exactly the files the selected attempts reference, into the run
directory's ``workspaces/`` tree, which Git ignores. It checks the archive
against its descriptor first, refuses members that are links or would land
outside the workspace tree, refuses to start when the free disk space cannot
hold the files, and checks every restored file against the SHA-256 its
attempt recorded. Files already present with the right hash are left alone.

Usage, from the repository root after ``git lfs pull``::

    python -m experiments.rostam.restore_workspaces \\
        --run-directory experiments/rostam/results/shared-replay-20260720-r2 \\
        --descriptor experiments/rostam/results/archives/<run>-primary-<verdict>.raw-archive-descriptor.json \\
        --archive experiments/rostam/results/archives/<run>-primary-<verdict>.raw.tar.gz

Then run the regeneration command recorded in the publication's
``aggregate.json`` with ``--output-directory`` pointed at a scratch directory,
and compare the three files with the committed ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, List, Optional, Sequence

DESCRIPTOR_SCHEMA = "commcanary.rostam.raw-archive-descriptor.v1"
WORKSPACES = "workspaces"
#: Room to leave for everything else on the disk after restoring.
FREE_SPACE_MARGIN_BYTES = 512 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


class RestoreError(Exception):
    """The workspaces cannot be restored safely or exactly."""


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RestoreError(f"cannot read {path}: {exc}") from exc


def _selected_references(run_directory: Path, selection_id: str) -> Dict[str, str]:
    """Workspace path -> SHA-256 for every file the selected attempts reference."""

    selection = _load_json(run_directory / "selections" / selection_id / "selection.json")
    entries = selection.get("entries") if isinstance(selection, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RestoreError(f"selection {selection_id!r} in {run_directory} lists no attempts")
    references: Dict[str, str] = {}

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            path, digest = value.get("path"), value.get("sha256")
            if isinstance(path, str) and isinstance(digest, str) and path.startswith(f"{WORKSPACES}/"):
                previous = references.setdefault(path, digest)
                if previous != digest:
                    raise RestoreError(f"attempts disagree about the SHA-256 of {path}")
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for entry in entries:
        if not isinstance(entry, dict):
            raise RestoreError("selection entry is not an object")
        attempt = run_directory / "attempts" / str(entry.get("cell_id")) / str(entry.get("attempt_id")) / "attempt.json"
        collect(_load_json(attempt))
    if not references:
        raise RestoreError(f"the selected attempts in {run_directory} reference no workspace files")
    return references


def _check_descriptor(descriptor: Any, archive: Path, run_id: str) -> None:
    if not isinstance(descriptor, dict) or descriptor.get("schema") != DESCRIPTOR_SCHEMA:
        raise RestoreError(f"descriptor is not a {DESCRIPTOR_SCHEMA} document")
    campaigns = descriptor.get("campaigns")
    if not isinstance(campaigns, list) or run_id not in {c.get("run_id") for c in campaigns if isinstance(c, dict)}:
        raise RestoreError(f"descriptor does not describe run {run_id!r}")
    size = archive.stat().st_size
    if size != descriptor.get("size_bytes"):
        raise RestoreError(
            f"{archive} is {size} bytes, the descriptor says {descriptor.get('size_bytes')}; "
            "if it is a few hundred bytes, run `git lfs pull` first"
        )
    observed = _sha256_file(archive)
    if observed != descriptor.get("sha256"):
        raise RestoreError(f"{archive} SHA-256 is {observed}, the descriptor says {descriptor.get('sha256')}")


def _safe_relative(name: str, run_id: str) -> Optional[str]:
    """The workspace-relative path of an archive member, or None if not ours."""

    prefix = f"{run_id}/"
    if not name.startswith(prefix):
        return None
    relative = name[len(prefix) :]
    parts = PurePosixPath(relative).parts
    if not parts or parts[0] != WORKSPACES or any(part in ("", ".", "..") for part in parts):
        return None
    return relative


def restore(
    run_directory: Path,
    archive: Path,
    descriptor_path: Path,
    *,
    selection_id: str = "primary",
    dry_run: bool = False,
) -> List[str]:
    run_directory = run_directory.resolve()
    run_id = run_directory.name
    references = _selected_references(run_directory, selection_id)
    _check_descriptor(_load_json(descriptor_path), archive, run_id)
    workspace_root = (run_directory / WORKSPACES).resolve()

    pending = {
        path: digest
        for path, digest in references.items()
        if not ((run_directory / path).is_file() and _sha256_file(run_directory / path) == digest)
    }
    with tarfile.open(archive, "r:gz") as bundle:
        members = {}
        for member in bundle.getmembers():
            relative = _safe_relative(member.name, run_id)
            if relative is not None and relative in pending:
                if not member.isfile():
                    raise RestoreError(f"archive member {member.name} is not a regular file")
                members[relative] = member
        missing = sorted(set(pending) - set(members))
        if missing:
            raise RestoreError(f"archive lacks {len(missing)} referenced files, first {missing[0]}")
        needed = sum(member.size for member in members.values())
        free = shutil.disk_usage(run_directory).free
        if needed + FREE_SPACE_MARGIN_BYTES > free:
            raise RestoreError(
                f"restoring needs {needed / 2**30:.2f} GiB plus a {FREE_SPACE_MARGIN_BYTES / 2**30:.1f} GiB margin, "
                f"and {free / 2**30:.2f} GiB is free on the disk holding {run_directory}"
            )
        if dry_run:
            return sorted(members)
        for relative, member in sorted(members.items(), key=lambda item: item[1].offset_data):
            destination = run_directory / relative
            if workspace_root not in destination.resolve().parents:
                raise RestoreError(f"{relative} would be written outside {workspace_root}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RestoreError(f"cannot read archive member {member.name}")
            temporary = destination.with_name(destination.name + ".restoring")
            digest = hashlib.sha256()
            with temporary.open("wb") as output:
                for block in iter(lambda: source.read(CHUNK_BYTES), b""):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != pending[relative]:
                temporary.unlink()
                raise RestoreError(
                    f"{relative} restored with SHA-256 {digest.hexdigest()}, its attempt recorded {pending[relative]}"
                )
            os.replace(temporary, destination)
    return sorted(members)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--selection-id", default="primary")
    parser.add_argument("--dry-run", action="store_true", help="check everything and report, write nothing")
    args = parser.parse_args(argv)
    try:
        restored = restore(
            args.run_directory,
            args.archive,
            args.descriptor,
            selection_id=args.selection_id,
            dry_run=args.dry_run,
        )
    except (RestoreError, OSError, tarfile.TarError) as exc:
        print(f"restore failed: {exc}", file=sys.stderr)
        return 1
    verb = "would restore" if args.dry_run else "restored"
    print(f"{verb} {len(restored)} workspace files into {args.run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
