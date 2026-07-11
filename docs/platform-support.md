# Platform support

## Python versions

The supported interpreter range is CPython 3.9 through 3.13. Linux is tested on
every supported Python version. macOS is tested on CPython 3.12 as a portability
representative. Build metadata and the documented compatibility matrix are the
source of truth; support is not implied for an interpreter outside that range.

## Operating systems

| Platform | Support level | Scope |
|---|---|---|
| Linux | Supported | Library, CLI, capture, package gate, local experiment harness, and benchmark runner |
| macOS | Supported | Library, CLI, capture, package/install gate, and benchmark smoke; physical GPU/SLURM experiments are excluded |
| Windows | Unsupported in 0.3 | The core library may work, but it is not in CI and filesystem permission, subprocess, capture ownership, and shell contracts are not claimed |

Windows is explicitly unsupported rather than silently treated as portable.
A future support change requires a CI matrix entry plus path, atomic-write,
permissions, subprocess-exit, installed-wheel, and command-line tests on Windows.

Shell validation and the `experiments/rostam/` scripts are POSIX-only. Those
scripts are site-specific experiment infrastructure, not a portable package
feature. No supported local verification command connects to Rostam or another
cluster.

## Filesystems and processes

Atomic replacement and single-owner capture semantics assume a local filesystem
with normal same-directory rename and exclusive-create behavior. Network and
distributed filesystems can have weaker cache, lock, durability, or permission
semantics; they are unsupported for concurrent writers unless independently
qualified.

Fork handling is exercised where `os.register_at_fork` exists. Spawn-only
process models still receive unique capture ownership and must not reuse one
direct output path concurrently.

## Performance portability

Benchmark timing and peak RSS are evidence only for the recorded environment.
Semantic hashes can be compared across supported machines; wall time and memory
thresholds require a stable named runner class. The physical PARAM/NCCL path has
its own narrower hardware/software evidence contract in `RESEARCH_SPEC.md` and
must not be generalized to unmeasured systems.

