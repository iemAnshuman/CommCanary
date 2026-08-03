#!/usr/bin/env bash
# This file is spooled by SLURM. Its inline standard-library loader stages the
# exact manifest-bound bootstrap before any repository Python module can run.
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: run_cell.sbatch <venv-python> <bootstrap> <bootstrap-sha256> <wrapper-kind> [cell arguments...]" >&2
  exit 2
fi

PYTHON_EXECUTABLE="$1"
BOOTSTRAP_PATH="$2"
BOOTSTRAP_SHA256="$3"
WRAPPER_KIND="$4"
shift 4

if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "unsafe or missing reviewed venv interpreter: $PYTHON_EXECUTABLE" >&2
  exit 2
fi

exec "$PYTHON_EXECUTABLE" -I -S -c '
import hashlib
import os
import runpy
import stat
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
expected = sys.argv[2]
wrapper = sys.argv[3]
runner_args = sys.argv[4:]
if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
    raise SystemExit("executor bootstrap SHA-256 is malformed")
if source.is_symlink():
    raise SystemExit("executor bootstrap must not be a symbolic link")
resolved = source.resolve(strict=True)
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(resolved, flags)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1048576:
        raise SystemExit("executor bootstrap is not a bounded regular file")
    chunks = []
    size = 0
    while True:
        chunk = os.read(descriptor, min(65536, 1048577 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > 1048576:
            raise SystemExit("executor bootstrap exceeds its byte limit")
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
raw = b"".join(chunks)
if identity_before != identity_after or len(raw) != before.st_size:
    raise SystemExit("executor bootstrap changed while it was read")
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit("executor bootstrap does not match the frozen campaign")
with tempfile.TemporaryDirectory(prefix="commcanary-rostam-bootstrap-") as temporary:
    root = Path(temporary)
    os.chmod(root, 0o700)
    staged = root / "executor_bootstrap.py"
    with staged.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(staged, 0o400)
    if hashlib.sha256(staged.read_bytes()).hexdigest() != expected:
        raise SystemExit("staged executor bootstrap changed before execution")
    sys.argv = [str(staged), "--site-wrapper", wrapper, *runner_args]
    runpy.run_path(str(staged), run_name="__main__")
' "$BOOTSTRAP_PATH" "$BOOTSTRAP_SHA256" "$WRAPPER_KIND" "$@"
