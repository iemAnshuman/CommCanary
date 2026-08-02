"""Race-safe installation primitives for immutable evidence directories."""

from __future__ import annotations

import ctypes
import errno
import os
import secrets
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

    token_path = destination.parent / f".{destination.name}.reservation-{secrets.token_hex(16)}"
    token_descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    token_status = os.fstat(token_descriptor)
    reservation_status: Optional[os.stat_result] = None
    token_payload = b""
    try:
        os.mkdir(destination, mode=stat.S_IMODE(source_status.st_mode))
        reservation_status = os.lstat(destination)
        token_payload = (
            f"commcanary-reservation-v1 {reservation_status.st_dev} {reservation_status.st_ino}\n".encode("ascii")
        )
        _write_all(token_descriptor, token_payload)
        os.fsync(token_descriptor)
        content_names = [name for name in sorted(children) if name != commit_name]
        for name in content_names:
            os.link(children[name], destination / name, follow_symlinks=False)
        _fsync_directory(destination)
        os.link(children[commit_name], destination / commit_name, follow_symlinks=False)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        shutil.rmtree(source)
    except BaseException:
        if reservation_status is not None:
            _cleanup_owned_reservation(
                destination,
                children=children,
                commit_name=commit_name,
                reservation_status=reservation_status,
                token_path=token_path,
                token_status=token_status,
                token_payload=token_payload,
            )
        raise
    finally:
        os.close(token_descriptor)
        _unlink_owned_token(token_path, token_status)


def _cleanup_owned_reservation(
    destination: Path,
    *,
    children: dict[str, Path],
    commit_name: str,
    reservation_status: os.stat_result,
    token_path: Path,
    token_status: os.stat_result,
    token_payload: bytes,
) -> None:
    """Remove only an uncommitted fallback directory still owned by this call."""

    try:
        observed_token_status = os.lstat(token_path)
        destination_status = os.lstat(destination)
        if (
            stat.S_ISLNK(observed_token_status.st_mode)
            or not stat.S_ISREG(observed_token_status.st_mode)
            or (observed_token_status.st_dev, observed_token_status.st_ino)
            != (token_status.st_dev, token_status.st_ino)
            or token_path.read_bytes() != token_payload
            or stat.S_ISLNK(destination_status.st_mode)
            or not stat.S_ISDIR(destination_status.st_mode)
            or (destination_status.st_dev, destination_status.st_ino)
            != (reservation_status.st_dev, reservation_status.st_ino)
            or (destination / commit_name).exists()
            or (destination / commit_name).is_symlink()
        ):
            return
        installed = {entry.name: entry for entry in destination.iterdir()}
        expected_names = set(children) - {commit_name}
        if not set(installed) <= expected_names:
            return
        for name, installed_path in installed.items():
            source_status = os.lstat(children[name])
            installed_status = os.lstat(installed_path)
            if (
                stat.S_ISLNK(installed_status.st_mode)
                or not stat.S_ISREG(installed_status.st_mode)
                or (source_status.st_dev, source_status.st_ino)
                != (installed_status.st_dev, installed_status.st_ino)
            ):
                return
        for installed_path in installed.values():
            installed_path.unlink()
        destination.rmdir()
        _fsync_directory(destination.parent)
    except OSError:
        return


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "short write while recording installation reservation")
        offset += written


def _unlink_owned_token(token_path: Path, expected: os.stat_result) -> None:
    try:
        observed = os.lstat(token_path)
        if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
            return
        token_path.unlink()
    except FileNotFoundError:
        pass


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
