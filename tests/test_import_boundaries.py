from __future__ import annotations

from pathlib import Path

from tools.import_boundaries import check_boundaries


def _write(root: Path, relative: str, text: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _base_tree(tmp_path: Path) -> Path:
    source = tmp_path / "src"
    _write(source, "commcanary/__init__.py")
    _write(source, "commcanary/errors.py")
    _write(source, "commcanary/artifacts/__init__.py")
    _write(source, "commcanary/artifacts/wire.py", "from ..errors import CommCanaryError\n")
    _write(source, "commcanary/compilation/__init__.py")
    _write(source, "commcanary/verification/__init__.py")
    _write(source, "commcanary/services/__init__.py")
    return source


def test_valid_downward_dependency_graph_passes(tmp_path: Path) -> None:
    source = _base_tree(tmp_path)
    _write(source, "commcanary/version.py", "__version__ = '0+test'\n")
    _write(source, "commcanary/cli.py", "from .version import __version__\n")
    _write(source, "commcanary/compilation/core.py", "from ..artifacts.wire import JsonDict\n")
    _write(source, "commcanary/verification/check.py", "from ..compilation.core import compile_trace_core\n")
    _write(source, "commcanary/replay/__init__.py", "from ..verification.check import verify\n")
    _write(source, "commcanary/services/compile.py", "from ..verification.check import verify\n")

    assert check_boundaries(source) == []


def test_upward_and_cross_boundary_private_imports_fail(tmp_path: Path) -> None:
    source = _base_tree(tmp_path)
    _write(source, "commcanary/services/workflow.py", "def _private():\n    return None\n")
    _write(
        source,
        "commcanary/compilation/core.py",
        "from ..services.workflow import _private\n",
    )

    messages = [violation.message for violation in check_boundaries(source)]
    assert any("imports upward from services" in message for message in messages)
    assert any("cross-boundary private import" in message for message in messages)


def test_module_cycles_fail_even_within_one_boundary(tmp_path: Path) -> None:
    source = _base_tree(tmp_path)
    _write(source, "commcanary/compilation/left.py", "from .right import value\n")
    _write(source, "commcanary/compilation/right.py", "from .left import value\n")

    messages = [violation.message for violation in check_boundaries(source)]
    assert any(message.startswith("dependency cycle:") for message in messages)


def test_unclassified_module_fails_closed(tmp_path: Path) -> None:
    source = _base_tree(tmp_path)
    _write(source, "commcanary/mystery.py")

    messages = [violation.message for violation in check_boundaries(source)]
    assert messages == ["module 'commcanary.mystery' has no declared dependency boundary"]
