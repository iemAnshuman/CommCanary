"""Dependency-free public exception hierarchy for CommCanary.

The hierarchy lives below artifact contracts and engines so every layer can
report typed failures without importing :mod:`commcanary.schema` and creating
cycles.  ``CommCanaryIOError`` remains a ``SchemaError`` for compatibility with
callers that historically caught ``SchemaError`` around artifact reads/writes.
"""

from __future__ import annotations

from typing import Optional


class CommCanaryError(Exception):
    """Base exception for expected CommCanary failures."""


class SchemaError(CommCanaryError):
    """Raised when an artifact does not satisfy a CommCanary contract."""


class CommCanaryIOError(SchemaError):
    """Raised when artifact I/O fails while preserving the original cause.

    ``path`` and ``operation`` are stable structured context for applications;
    the underlying exception remains available through ``__cause__``.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.operation = operation


__all__ = ["CommCanaryError", "CommCanaryIOError", "SchemaError"]
