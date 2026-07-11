# ADR 0007: Platform support claims

- Status: Accepted
- Date: 2026-07-11

## Context

Core Python logic may happen to execute on many systems, but atomic filesystem
semantics, process behavior, shell scripts, package installation, and peak-RSS
measurement differ by platform. An untested platform should not be described as
supported merely because imports succeed.

## Decision

CommCanary 0.3 supports CPython 3.9 through 3.13 on Linux. macOS is a supported
library, CLI, capture, package, and local-benchmark platform on the CI-tested
representative Python version. Windows is explicitly unsupported until its own
CI jobs exercise path containment, atomic writes and permissions, subprocess
exit mapping, capture ownership, installation, and CLI contracts.

Rostam shell, SLURM, PARAM, GPU, and NCCL workflows are site-specific
experiment infrastructure, not portable package support. A physical campaign
supports only the exact hardware/software environment recorded in its evidence.
Performance thresholds do not transfer between unnamed runner classes.

The detailed current matrix is maintained in `docs/platform-support.md` and
must agree with package classifiers and CI. Adding or removing support is a
release-note and matrix change, not a prose-only edit.

## Consequences

- Best-effort behavior outside the matrix is not a compatibility promise.
- POSIX shell validation does not establish Windows support.
- Physical replay results are not generalized to unmeasured systems.
- Filesystem durability claims are limited to the documented local-filesystem
  assumptions.

## Alternatives considered

- Claiming every platform accepted by `requires-python` was rejected because
  interpreter and operating-system support are distinct.
- Calling Windows supported without CI was rejected because the highest-risk
  filesystem and process contracts are platform-specific.
- Treating site scripts as part of the portable library contract was rejected
  because they encode scheduler and environment policy.
