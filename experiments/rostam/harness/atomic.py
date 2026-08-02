"""Race-safe installation primitives for immutable evidence directories."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, "Path"]

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_UNSUPPORTED_RENAME_ERRORS = {
    errno.EINVAL,
    errno.ENOSYS,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def atomic_rename_noreplace(
    source: PathLike,
    destination: PathLike,
    *,
    commit_name: Optional[str] = None,
) -> None:
    """Install a directory without ever replacing an existing destination.

    Linux and macOS use their native exclusive rename operations. Windows
    already gives ``os.rename`` no-replace behavior. The remaining fallback
    reserves the destination with exclusive ``mkdir``, installs regular files
    with exclusive hard links, and publishes the caller's checksum file last.
    Existing loaders already reject a directory that lacks that checksum.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    if sys.platform.startswith("linux") and _linux_rename_noreplace(source_path, destination_path):
        return
    if sys.platform == "darwin" and _darwin_rename_noreplace(source_path, destination_path):
        return
    if os.name == "nt":
        os.rename(source_path, destination_path)
        return
    _reserved_directory_install(
        source_path,
        destination_path,
        commit_name=commit_name,
    )


def _linux_rename_noreplace(source: Path, destination: Path) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError:
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in _UNSUPPORTED_RENAME_ERRORS:
        return False
    raise OSError(error, os.strerror(error), str(destination))


def _darwin_rename_noreplace(source: Path, destination: Path) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renamex_np = libc.renamex_np
    except AttributeError:
        return False
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    result = renamex_np(
        os.fsencode(source),
        os.fsencode(destination),
        _RENAME_EXCL,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in _UNSUPPORTED_RENAME_ERRORS:
        return False
    raise OSError(error, os.strerror(error), str(destination))


def _reserved_directory_install(
    source: Path,
    destination: Path,
    *,
    commit_name: Optional[str],
) -> None:
    if commit_name is None or Path(commit_name).name != commit_name:
        raise OSError(errno.ENOTSUP, "portable exclusive directory installation requires a commit filename")
    source_status = os.lstat(source)
    if stat.S_ISLNK(source_status.st_mode) or not stat.S_ISDIR(source_status.st_mode):
        raise OSError(errno.ENOTDIR, "exclusive install source must be a real directory", str(source))
    children = {entry.name: entry for entry in source.iterdir()}
    if commit_name not in children:
        raise OSError(errno.ENOENT, "exclusive install commit file is missing", str(source / commit_name))
    if any(entry.is_symlink() or not entry.is_file() for entry in children.values()):
        raise OSError(errno.ENOTSUP, "portable exclusive directory installation supports regular files only")

    os.mkdir(destination, mode=stat.S_IMODE(source_status.st_mode))
    content_names = [name for name in sorted(children) if name != commit_name]
    for name in content_names:
        os.link(children[name], destination / name, follow_symlinks=False)
    _fsync_directory(destination)
    os.link(children[commit_name], destination / commit_name, follow_symlinks=False)
    _fsync_directory(destination)
    _fsync_directory(destination.parent)
    shutil.rmtree(source)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


__all__ = ["atomic_rename_noreplace"]
