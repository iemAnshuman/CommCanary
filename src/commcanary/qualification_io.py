"""Descriptor-bound snapshots for untrusted qualification directories."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Set

from .errors import CommCanaryIOError, SchemaError
from .resources import JsonResourceError, ResourceLimits, decode_bounded_json_bytes


@dataclass(frozen=True)
class FileSnapshot:
    """Exact bytes and identity read from one opened regular file."""

    raw: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class JsonFileSnapshot(FileSnapshot):
    """A parsed JSON value bound to the exact bytes that were hashed."""

    value: Any


class VerifiedDirectory:
    """One opened real directory used for descriptor-relative artifact reads."""

    def __init__(self, path: Path, *, label: str) -> None:
        self.path = path
        self.label = label
        self._descriptor = self._open_directory(path)

    def __enter__(self) -> VerifiedDirectory:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor < 0:
            return
        self._descriptor = -1
        os.close(descriptor)

    def names(self) -> Set[str]:
        try:
            names = os.listdir(self._descriptor)
        except OSError as exc:
            raise CommCanaryIOError(
                f"cannot list {self.label} {self.path}: {exc}",
                path=str(self.path),
                operation=f"list {self.label}",
            ) from exc
        if any(not isinstance(name, str) for name in names):
            raise SchemaError(f"{self.label} inventory contains a non-text name")
        return set(names)

    def read_bytes(self, name: str, *, limits: ResourceLimits) -> FileSnapshot:
        """Open one entry once, then fstat and bounded-read that descriptor."""

        if not name or Path(name).name != name:
            raise SchemaError(f"{self.label} artifact name is not canonical: {name!r}")
        descriptor = -1
        try:
            before_name = os.stat(name, dir_fd=self._descriptor, follow_symlinks=False)
            if stat.S_ISLNK(before_name.st_mode) or not stat.S_ISREG(before_name.st_mode):
                raise SchemaError(f"{self.label} artifact must be a regular file: {name}")
            flags = os.O_RDONLY
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(name, flags, dir_fd=self._descriptor)
            before_read = os.fstat(descriptor)
            if not stat.S_ISREG(before_read.st_mode) or (
                before_name.st_dev,
                before_name.st_ino,
            ) != (
                before_read.st_dev,
                before_read.st_ino,
            ):
                raise SchemaError(f"{self.label} artifact changed before it was opened: {name}")
            if before_read.st_size > limits.max_input_bytes:
                raise SchemaError(f"{self.label} artifact {name} exceeds max_input_bytes={limits.max_input_bytes}")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                raw = handle.read(limits.max_input_bytes + 1)
                after_read = os.fstat(handle.fileno())
            if len(raw) > limits.max_input_bytes:
                raise SchemaError(f"{self.label} artifact {name} exceeds max_input_bytes={limits.max_input_bytes}")
            before_identity = (
                before_read.st_dev,
                before_read.st_ino,
                before_read.st_size,
                before_read.st_mtime_ns,
                before_read.st_ctime_ns,
            )
            after_identity = (
                after_read.st_dev,
                after_read.st_ino,
                after_read.st_size,
                after_read.st_mtime_ns,
                after_read.st_ctime_ns,
            )
            if before_identity != after_identity or len(raw) != before_read.st_size:
                raise SchemaError(f"{self.label} artifact changed while it was read: {name}")
        except SchemaError:
            raise
        except FileNotFoundError as exc:
            raise SchemaError(f"{self.label} artifact disappeared before it was opened: {name}") from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SchemaError(f"{self.label} artifact must not be a symlink: {name}") from exc
            raise CommCanaryIOError(
                f"cannot read {self.label} artifact {self.path / name}: {exc}",
                path=str(self.path / name),
                operation=f"read {self.label} artifact",
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return FileSnapshot(
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
        )

    def read_json(
        self,
        name: str,
        *,
        limits: ResourceLimits,
        require_object: bool,
    ) -> JsonFileSnapshot:
        """Hash and parse the same single-read byte snapshot."""

        snapshot = self.read_bytes(name, limits=limits)
        try:
            value = decode_bounded_json_bytes(snapshot.raw, limits=limits)
        except UnicodeDecodeError as exc:
            raise SchemaError(f"{self.label} artifact {name} is not UTF-8 JSON: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{self.label} artifact {name} is not valid JSON: {exc.msg}") from exc
        except JsonResourceError as exc:
            raise SchemaError(f"{self.label} artifact {name} violates JSON resource constraints: {exc}") from exc
        except RecursionError as exc:
            raise SchemaError(f"{self.label} artifact {name} exceeds the JSON parser nesting capacity") from exc
        except OverflowError as exc:
            raise SchemaError(f"{self.label} artifact {name} contains a number that is too large") from exc
        except ValueError as exc:
            raise SchemaError(f"{self.label} artifact {name} contains non-standard JSON: {exc}") from exc
        if require_object and not isinstance(value, dict):
            raise SchemaError(f"{self.label} artifact {name} must contain a JSON object")
        return JsonFileSnapshot(
            raw=snapshot.raw,
            sha256=snapshot.sha256,
            size_bytes=snapshot.size_bytes,
            value=value,
        )

    def _open_directory(self, path: Path) -> int:
        try:
            before = os.lstat(str(path))
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise SchemaError(f"{self.label} path must be a real directory, not a symlink")
            flags = os.O_RDONLY
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(str(path), flags)
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode) or (before.st_dev, before.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                os.close(descriptor)
                raise SchemaError(f"{self.label} path changed before it was opened")
            return descriptor
        except SchemaError:
            raise
        except FileNotFoundError as exc:
            raise SchemaError(f"{path} does not exist") from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SchemaError(f"{self.label} path must be a real directory, not a symlink") from exc
            raise CommCanaryIOError(
                f"cannot open {self.label} {path}: {exc}",
                path=str(path),
                operation=f"open {self.label}",
            ) from exc


__all__ = ["FileSnapshot", "JsonFileSnapshot", "VerifiedDirectory"]
