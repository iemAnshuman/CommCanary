"""Atomic artifact writes with explicit policy at every call site.

Policy matrix
-------------
``SENSITIVE_JSON_POLICY`` and ``PARAM_TRACE_POLICY`` create parents, replace
the final path atomically, reject an existing final-component symlink, install
mode ``0o600``, flush and fsync the file, and fsync the parent directory on
POSIX. ``SHAREABLE_HTML_POLICY`` has the same durability and symlink policy but
installs mode ``0o644``. All temporary files live beside the target so the
install cannot cross filesystems.

Directory fsync has no portable Windows equivalent and is therefore skipped on
Windows; file flush/fsync and atomic replacement still apply. Cleanup runs for
every ``BaseException``. Expected CommCanary exceptions and control-flow
exceptions are never replaced; operating-system failures become
``CommCanaryIOError`` with their original cause attached.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from ..errors import CommCanaryError, CommCanaryIOError
from .json_codec import formatted_json_bytes

Pathish = Union[str, os.PathLike[str]]


class TempPlacement(str, Enum):
    """Supported placement for a same-filesystem atomic temporary file."""

    TARGET_PARENT = "target_parent"


class SymlinkPolicy(str, Enum):
    """Policy for a symlink occupying the final path."""

    REJECT = "reject"
    REPLACE_LINK = "replace_link"


@dataclass(frozen=True)
class AtomicWritePolicy:
    """Complete mechanics and durability policy for one artifact class."""

    artifact_label: str
    create_parents: bool
    overwrite: bool
    temp_placement: TempPlacement
    mode: int
    flush: bool
    fsync_file: bool
    fsync_parent: bool
    symlink: SymlinkPolicy

    def __post_init__(self) -> None:
        if not self.artifact_label:
            raise ValueError("artifact_label must not be empty")
        if self.temp_placement is not TempPlacement.TARGET_PARENT:
            raise ValueError("atomic writes require target-parent temporary placement")
        if not isinstance(self.mode, int) or isinstance(self.mode, bool) or not 0 <= self.mode <= 0o777:
            raise ValueError("mode must contain only POSIX permission bits")
        if self.fsync_file and not self.flush:
            raise ValueError("fsync_file requires flush")


SENSITIVE_JSON_POLICY = AtomicWritePolicy(
    artifact_label="JSON artifact",
    create_parents=True,
    overwrite=True,
    temp_placement=TempPlacement.TARGET_PARENT,
    mode=0o600,
    flush=True,
    fsync_file=True,
    fsync_parent=True,
    symlink=SymlinkPolicy.REJECT,
)

PARAM_TRACE_POLICY = AtomicWritePolicy(
    artifact_label="PARAM trace",
    create_parents=True,
    overwrite=True,
    temp_placement=TempPlacement.TARGET_PARENT,
    mode=0o600,
    flush=True,
    fsync_file=True,
    fsync_parent=True,
    symlink=SymlinkPolicy.REJECT,
)

SHAREABLE_HTML_POLICY = AtomicWritePolicy(
    artifact_label="HTML report",
    create_parents=True,
    overwrite=True,
    temp_placement=TempPlacement.TARGET_PARENT,
    mode=0o644,
    flush=True,
    fsync_file=True,
    fsync_parent=True,
    symlink=SymlinkPolicy.REJECT,
)


def atomic_write_text(
    path: Pathish,
    content: str,
    *,
    policy: AtomicWritePolicy,
    encoding: str = "utf-8",
) -> None:
    """Encode and atomically install text under an explicit policy."""

    atomic_write_bytes(path, content.encode(encoding), policy=policy)


def atomic_write_json(
    path: Pathish,
    data: Any,
    *,
    indent: int,
    policy: AtomicWritePolicy,
    trailing_newline: bool = True,
) -> None:
    """Encode deterministic JSON and atomically install it under ``policy``."""

    atomic_write_bytes(
        path,
        formatted_json_bytes(data, indent=indent, trailing_newline=trailing_newline),
        policy=policy,
    )


def atomic_write_bytes(path: Pathish, content: bytes, *, policy: AtomicWritePolicy) -> None:
    """Atomically install bytes while applying ``policy`` exactly.

    With ``overwrite=False`` publication uses a same-directory hard link,
    providing race-safe create-if-absent semantics on supported filesystems.
    An existing symlink is never followed: it is either rejected or atomically
    replaced as a directory entry according to ``policy.symlink``.
    """

    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    target = Path(path)
    parent = target.parent
    operation = "prepare parent"
    fd: Optional[int] = None
    temp_path: Optional[Path] = None
    try:
        if policy.create_parents:
            parent.mkdir(parents=True, exist_ok=True)
        elif not parent.is_dir():
            raise FileNotFoundError(f"parent directory does not exist: {parent}")

        operation = "inspect target"
        _enforce_target_policy(target, policy)

        operation = "create temporary file"
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
        temp_path = Path(temp_name)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, policy.mode)

        operation = "write temporary file"
        with os.fdopen(fd, "wb") as stream:
            fd = None
            written = stream.write(content)
            if written != len(content):
                raise OSError(f"short write: wrote {written} of {len(content)} bytes")
            if policy.flush:
                stream.flush()
            if policy.fsync_file:
                operation = "fsync temporary file"
                os.fsync(stream.fileno())

        operation = "install target"
        _enforce_target_policy(target, policy)
        if policy.overwrite:
            os.replace(str(temp_path), str(target))
            temp_path = None
        else:
            os.link(str(temp_path), str(target), follow_symlinks=False)
            temp_path.unlink()
            temp_path = None

        if policy.fsync_parent:
            operation = "fsync parent directory"
            _fsync_parent_directory(parent)
    except BaseException as exc:
        _close_quietly(fd)
        _unlink_quietly(temp_path)
        if isinstance(exc, CommCanaryError):
            raise
        if isinstance(exc, OSError):
            raise CommCanaryIOError(
                f"cannot write {policy.artifact_label} {target}: {operation}: {exc}",
                path=str(target),
                operation=operation,
            ) from exc
        raise


def _enforce_target_policy(target: Path, policy: AtomicWritePolicy) -> None:
    try:
        target_mode = os.lstat(str(target)).st_mode
    except FileNotFoundError:
        return
    if not policy.overwrite:
        raise FileExistsError(f"target already exists: {target}")
    if policy.symlink is SymlinkPolicy.REJECT and stat.S_ISLNK(target_mode):
        raise CommCanaryIOError(
            f"cannot write {policy.artifact_label} {target}: final path is a symlink",
            path=str(target),
            operation="inspect target",
        )


def _fsync_parent_directory(parent: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    directory_fd = os.open(str(parent), flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _close_quietly(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _unlink_quietly(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


__all__ = [
    "AtomicWritePolicy",
    "PARAM_TRACE_POLICY",
    "SENSITIVE_JSON_POLICY",
    "SHAREABLE_HTML_POLICY",
    "SymlinkPolicy",
    "TempPlacement",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
]
