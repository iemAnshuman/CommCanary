"""Package-version access with build metadata as the single source of truth."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

SOURCE_TREE_VERSION = "0+unknown"


def package_version() -> str:
    """Return build metadata for an installed package or source checkout."""

    source_version = _source_tree_version()
    if source_version is not None:
        return source_version

    try:
        return version("commcanary")
    except PackageNotFoundError:
        return SOURCE_TREE_VERSION


def _source_tree_version() -> Optional[str]:
    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or Path(__file__).resolve().parent.parent != root / "src":
        return None
    try:
        text = pyproject.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    project = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
    if project is None:
        return None
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project.group(1), flags=re.MULTILINE)
    return match.group(1) if match is not None else None


__version__ = package_version()

__all__ = ["SOURCE_TREE_VERSION", "__version__", "package_version"]
