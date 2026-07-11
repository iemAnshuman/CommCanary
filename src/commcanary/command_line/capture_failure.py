"""Bounded, immutable failed-capture evidence bundles."""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

from ..artifacts import write_json
from ..errors import CommCanaryError


def preserve_capture_failure(
    trace_dir: str,
    destination: str,
    *,
    workload_name: str,
    session_id: str,
    child_returncode: int,
) -> None:
    """Create an immutable, bounded bundle without exposing command/environment data."""

    destination_path = Path(destination).expanduser()
    try:
        destination_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise CommCanaryError(f"capture failure destination already exists: {destination_path}") from exc
    except OSError as exc:
        raise CommCanaryError(f"cannot create capture failure destination {destination_path}: {exc}") from exc

    shard_destination = destination_path / "shards"
    shard_destination.mkdir(mode=0o700)
    source_root = Path(trace_dir).resolve()
    candidates: Set[Path] = set()
    for pattern in ("*.trace.json", "*.trace.*.json", "*.rank-*-pid-*.json"):
        candidates.update(source_root.glob(pattern))

    maximum_file_bytes = 64 * 1024 * 1024
    maximum_total_bytes = 256 * 1024 * 1024
    total_bytes = 0
    shard_records = []
    for source in sorted(candidates, key=lambda path: path.name):
        if source.is_symlink() or not source.is_file() or source.parent.resolve() != source_root:
            continue
        encoded_name = source.name.encode("utf-8", errors="strict")
        if len(encoded_name) > 255:
            raise CommCanaryError("capture shard filename exceeds the failure-bundle limit")
        target = shard_destination / source.name
        digest = hashlib.sha256()
        copied = 0
        try:
            with source.open("rb") as reader, target.open("xb") as writer:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    total_bytes += len(chunk)
                    if copied > maximum_file_bytes or total_bytes > maximum_total_bytes:
                        raise CommCanaryError("partial capture shards exceed the failure-bundle byte limit")
                    digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        except CommCanaryError:
            raise
        except (OSError, UnicodeError) as exc:
            raise CommCanaryError(f"cannot preserve partial capture shard {source.name}: {exc}") from exc
        os.chmod(target, 0o400)
        shard_records.append(
            {
                "name": source.name,
                "size_bytes": copied,
                "sha256": digest.hexdigest(),
            }
        )

    manifest = {
        "format": "commcanary.capture_failure.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": {"name": str(workload_name)},
        "capture_session_id": str(session_id),
        "child_returncode": int(child_returncode),
        "partial_shards": shard_records,
    }
    manifest_path = destination_path / "capture_failure.json"
    write_json(str(manifest_path), manifest)
    os.chmod(manifest_path, 0o400)
    print(f"preserved failed capture: {destination_path}", file=sys.stderr)


__all__ = ["preserve_capture_failure"]
