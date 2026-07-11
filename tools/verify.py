"""Run CommCanary's canonical local, CI, and release verification gate.

``python -m tools.verify --fast`` runs repository, static, test, data, shell,
workflow, and documentation checks. The default mode also builds and tests a
wheel and sdist outside the checkout. ``--reproducible`` adds a second clean
build under a fixed ``SOURCE_DATE_EPOCH`` and requires byte-identical artifacts;
``--release`` also requires finalized release identity.
``--metadata-dir`` emits checksums, a content inventory, and an SPDX SBOM for
the same distribution pair that passed the install tests.

Install the project dev extra before invoking the gate. CI calls this module
instead of maintaining a second command list in workflow YAML.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import venv
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from email.parser import BytesParser
from email.policy import default as email_policy
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

from tools import release_metadata

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SOURCE_DIRS = ("src", "tests", "examples", "experiments", "benchmarks", "tools")
MYPY_TARGETS = ("src", "tools", "experiments/rostam", "benchmarks")
SHELL_SUFFIXES = {".sh", ".sbatch"}
SOURCE_DATE_EPOCH = "1704067200"
INSTALLED_TEST_SUBDIRECTORIES = (
    "adapters",
    "comparison",
    "compilation",
    "contracts",
    "replay",
    "verification",
)
INSTALLED_TEST_EXCLUSIONS = {
    "test_cli_decomposition.py",
    "test_coverage_policy.py",
    "test_foundation_contracts.py",
    "test_import_boundaries.py",
    "test_phase5_boundaries.py",
    "test_phase5_edge_facades.py",
    "test_release_metadata.py",
    "test_verify_tool.py",
}
RELEASE_ROOT_FILES = (
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "RESEARCH_SPEC.md",
    "SECURITY.md",
    "pyproject.toml",
)
RELEASE_SOURCE_DIRS = (
    "benchmarks",
    "docs",
    "examples",
    "experiments",
    "schemas",
    "src",
    "tests",
    "tools",
)
FORBIDDEN_TRACKED_PARTS = {
    ".benchmark-data",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "release-metadata",
}
FORBIDDEN_TRACKED_NAMES = {".DS_Store", "context.md"}
MARKDOWN_LINK_RE = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
REPOSITORY_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])((?:docs|examples|experiments)/[A-Za-z0-9_./-]+)")
WORKFLOW_USES_PREFIX_RE = re.compile(r"^\s*(?:-\s*)?uses\s*:")
PINNED_ACTION_RE = re.compile(r"^[^/@\s]+/[^@\s]+@[0-9a-f]{40}$")
VERSION_COMMENT_RE = re.compile(r"^v[0-9]+(?:\.[0-9]+){0,2}(?:[-+][A-Za-z0-9.-]+)?$")


class VerificationError(RuntimeError):
    """A verification step failed or a required tool is unavailable."""


@dataclass(frozen=True)
class Artifacts:
    """One wheel/sdist pair produced by a clean build."""

    wheel: Path
    sdist: Path


def _base_env(
    extra: Optional[Mapping[str, str]] = None,
    *,
    unset: Iterable[str] = (),
) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in unset:
        env.pop(name, None)
    if extra:
        env.update(extra)
    return env


def _run(
    label: str,
    argv: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Optional[Mapping[str, str]] = None,
    unset_env: Iterable[str] = (),
) -> None:
    print(f"\n== {label} ==", flush=True)
    print("+ " + " ".join(argv), flush=True)
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=_base_env(env, unset=unset_env),
        check=False,
    )
    if completed.returncode != 0:
        raise VerificationError(f"{label} failed with exit code {completed.returncode}")


def _capture(argv: Sequence[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=_base_env(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise VerificationError(f"{' '.join(argv)} failed: {detail}")
    return completed.stdout


def _missing_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, AttributeError, ValueError):
        return True


def _preflight(*, include_artifacts: bool) -> None:
    tools = ["git", "mypy", "ruff"]
    modules = ["coverage", "jsonschema", "pytest"]
    if os.name != "nt":
        tools.extend(("bash", "shellcheck"))
    if (ROOT / ".github" / "workflows").exists():
        tools.append("actionlint")
    if include_artifacts:
        modules.extend(("build.__main__", "pip", "twine"))

    missing_tools = sorted(name for name in tools if shutil.which(name) is None)
    missing_modules = sorted(name for name in modules if _missing_module(name))
    if not missing_tools and not missing_modules:
        return

    details: List[str] = []
    if missing_tools:
        details.append("executables: " + ", ".join(missing_tools))
    if missing_modules:
        details.append("Python modules: " + ", ".join(missing_modules))
    raise VerificationError(
        "missing required dev tooling ("
        + "; ".join(details)
        + "); install this checkout with `python -m pip install -e '.[dev]'`"
    )


def _runtime_package_snapshot(root: Path) -> Dict[str, bytes]:
    snapshot: Dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            path.suffix == ".py"
            or path.name == "py.typed"
            or (relative.parts[:1] == ("schemas",) and path.suffix == ".json")
        ):
            snapshot[relative.as_posix()] = path.read_bytes()
    return snapshot


def _compare_runtime_package_trees(source: Path, installed: Path) -> None:
    source_snapshot = _runtime_package_snapshot(source)
    installed_snapshot = _runtime_package_snapshot(installed)
    missing = sorted(set(source_snapshot) - set(installed_snapshot))
    extra = sorted(set(installed_snapshot) - set(source_snapshot))
    mismatched = sorted(
        path
        for path in set(source_snapshot) & set(installed_snapshot)
        if source_snapshot[path] != installed_snapshot[path]
    )
    if missing or extra or mismatched:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        if mismatched:
            details.append("mismatched=" + ",".join(mismatched))
        raise VerificationError(
            "installed CommCanary does not match the checkout; reinstall from a clean build: " + "; ".join(details)
        )


def _validate_installed_source_mirror() -> None:
    spec = importlib.util.find_spec("commcanary")
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if len(locations) != 1:
        raise VerificationError("canonical tests require exactly one installed CommCanary package location")
    installed = Path(locations[0]).resolve()
    source = (ROOT / "src" / "commcanary").resolve()
    _compare_runtime_package_trees(source, installed)
    print(f"installed source mirror matches {installed}")


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file():
            yield root
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                yield path


def _validate_repository_hygiene() -> None:
    print("\n== repository hygiene ==", flush=True)
    git = shutil.which("git")
    if git is None:
        raise VerificationError("git disappeared after tool preflight")
    # core.pager=cat: on a dumb interactive terminal git may otherwise page and
    # stall the gate waiting for a keypress.
    _run("whitespace and conflict-marker check", (git, "-c", "core.pager=cat", "diff", "--check", "HEAD", "--", "."))

    tracked = _capture((git, "ls-files", "-z")).split("\0")
    forbidden: List[str] = []
    for raw_path in tracked:
        if not raw_path:
            continue
        path = Path(raw_path)
        if _forbidden_tracked_path(path):
            forbidden.append(raw_path)
    if forbidden:
        raise VerificationError("generated files are tracked: " + ", ".join(sorted(forbidden)))
    print(f"checked {len(tracked) - 1} tracked paths")


def _forbidden_tracked_path(path: Path) -> bool:
    if FORBIDDEN_TRACKED_PARTS.intersection(path.parts):
        return True
    if path.name in FORBIDDEN_TRACKED_NAMES or path.name.startswith("repomix-output."):
        return True
    if path.suffix in {".pyc", ".pyo"} or any(part.endswith(".egg-info") for part in path.parts):
        return True
    return path.parts[:3] == ("experiments", "rostam", "results")


def _validate_json_files() -> None:
    print("\n== JSON and JSON-Schema ==", flush=True)
    roots = (
        ROOT / "schemas",
        ROOT / "docs",
        ROOT / "examples",
        ROOT / "experiments",
        ROOT / "tests",
    )
    checked = 0
    schema_checked = 0
    for path in _iter_files(roots):
        if path.suffix != ".json":
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise VerificationError(f"invalid JSON file {path.relative_to(ROOT)}: {exc}") from exc
        checked += 1
        if not path.name.endswith(".schema.json"):
            continue
        try:
            jsonschema = importlib.import_module("jsonschema")
            validator = jsonschema.validators.validator_for(payload)
            validator.check_schema(payload)
        except Exception as exc:
            raise VerificationError(f"invalid JSON Schema {path.relative_to(ROOT)}: {exc}") from exc
        schema_checked += 1
    _validate_packaged_schema_mirror()
    print(f"validated {checked} JSON files ({schema_checked} JSON Schemas)")


def _validate_packaged_schema_mirror() -> None:
    """Keep normal ``pip install .`` builds complete, not only release staging."""

    published = {path.name: path.read_bytes() for path in _published_schema_files()}
    package_directory = ROOT / "src" / "commcanary" / "schemas"
    packaged_paths = {path.name: path for path in package_directory.glob("*.json") if path.is_file()}
    missing = sorted(set(published) - set(packaged_paths))
    extra = sorted(set(packaged_paths) - set(published))
    mismatched = sorted(
        name for name in set(published) & set(packaged_paths) if packaged_paths[name].read_bytes() != published[name]
    )
    if missing or extra or mismatched:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        if mismatched:
            details.append("mismatched=" + ",".join(mismatched))
        raise VerificationError("packaged schema mirror is stale: " + "; ".join(details))


def _shell_files() -> List[Path]:
    return [path for path in _iter_files((ROOT / "experiments",)) if path.suffix in SHELL_SUFFIXES]


def _validate_shell_files() -> None:
    shell_files = _shell_files()
    if not shell_files:
        print("\n== shell checks ==\nno shell files found", flush=True)
        return
    if os.name == "nt":
        print("\n== shell checks ==\nskipped on Windows (unsupported experiment host)", flush=True)
        return

    bash = shutil.which("bash")
    shellcheck = shutil.which("shellcheck")
    if bash is None or shellcheck is None:
        raise VerificationError("bash or shellcheck disappeared after tool preflight")
    for path in shell_files:
        _run(f"bash -n {path.relative_to(ROOT)}", (bash, "-n", str(path)))
    _run(
        "shellcheck",
        (shellcheck, "--shell=bash", *(str(path) for path in shell_files)),
    )


def _validate_workflows() -> None:
    workflow_dir = ROOT / ".github" / "workflows"
    workflows = sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")))
    if not workflows:
        print("\n== workflow checks ==\nno workflows found", flush=True)
        return
    _validate_workflow_action_pins(workflows)
    actionlint = shutil.which("actionlint")
    if actionlint is None:
        raise VerificationError("actionlint disappeared after tool preflight")
    _run("actionlint", (actionlint, *(str(path) for path in workflows)))


def _validate_workflow_action_pins(workflows: Sequence[Path]) -> None:
    """Require immutable, human-auditable references for every remote action."""

    problems: List[str] = []
    checked = 0
    for path in workflows:
        absolute_path = path.absolute()
        try:
            display_path = absolute_path.relative_to(ROOT.absolute())
        except ValueError:
            display_path = path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise VerificationError(f"cannot read workflow {display_path}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            prefix = WORKFLOW_USES_PREFIX_RE.match(line)
            if prefix is None:
                continue
            value, separator, raw_comment = line[prefix.end() :].partition("#")
            reference = value.strip()
            if len(reference) >= 2 and reference[0] == reference[-1] and reference[0] in {'"', "'"}:
                reference = reference[1:-1]
            comment = raw_comment.strip() if separator else ""
            location = f"{display_path}:{line_number}"
            if not reference or any(character.isspace() for character in reference):
                problems.append(f"{location}: malformed uses reference")
                continue
            if reference.startswith("./"):
                checked += 1
                continue
            if PINNED_ACTION_RE.fullmatch(reference) is None:
                problems.append(f"{location}: remote action must use a full lowercase 40-character commit SHA")
                continue
            if VERSION_COMMENT_RE.fullmatch(comment) is None:
                problems.append(f"{location}: pinned action must have a version comment such as '# v4.3.1'")
                continue
            checked += 1
    if problems:
        raise VerificationError("workflow action pin policy failed:\n" + "\n".join(problems))
    print(f"validated {checked} immutable workflow action references")


def _local_markdown_target(raw_target: str) -> Optional[str]:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    return unquote(target.split("#", 1)[0].split("?", 1)[0])


def _validate_readme() -> None:
    print("\n== README links, commands, and examples ==", flush=True)
    readme = ROOT / "README.md"
    try:
        text = readme.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise VerificationError(f"cannot read README.md: {exc}") from exc

    referenced: Set[str] = set()
    for match in MARKDOWN_LINK_RE.finditer(text):
        target = _local_markdown_target(match.group(1))
        if target is not None:
            referenced.add(target)
    for match in REPOSITORY_PATH_RE.finditer(text):
        referenced.add(match.group(1).rstrip(".,:;)"))

    missing = sorted(target for target in referenced if not (ROOT / target).exists())
    if missing:
        raise VerificationError("README.md references missing repository paths: " + ", ".join(missing))
    if "commcanary " not in text:
        raise VerificationError("README.md contains no commcanary command examples")
    documentation_candidates = [
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        ROOT / "RESEARCH_SPEC.md",
        ROOT / "paper" / "draft.md",
        ROOT / "experiments" / "rostam" / "DESIGN.md",
    ]
    docs_root = ROOT / "docs"
    if docs_root.is_dir():
        documentation_candidates.extend(sorted(docs_root.rglob("*.md")))
    documentation = {path for path in documentation_candidates if path.is_file()}
    checked_links = 0
    link_problems: List[str] = []
    root = ROOT.resolve()
    for document in sorted(documentation):
        try:
            document_text = document.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise VerificationError(f"cannot read documentation file {document}: {exc}") from exc
        for match in MARKDOWN_LINK_RE.finditer(document_text):
            target = _local_markdown_target(match.group(1))
            if target is None:
                continue
            checked_links += 1
            resolved = (document.parent / target).resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                link_problems.append(f"{document.relative_to(ROOT)}: local link escapes repository: {target}")
                continue
            if not resolved.exists():
                link_problems.append(f"{document.relative_to(ROOT)}: missing local link: {relative}")
    if link_problems:
        raise VerificationError("documentation link policy failed:\n" + "\n".join(link_problems))
    print(
        f"validated {len(referenced)} local README references, {checked_links} documentation links, "
        "and command examples"
    )


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
    if match is None:
        raise VerificationError("cannot determine project.version from pyproject.toml")
    return match.group(1)


def _verify_release_changelog(version: str, *, changelog: Optional[Path] = None) -> None:
    """Require one dated changelog heading for the version being released."""

    path = changelog if changelog is not None else ROOT / "CHANGELOG.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise VerificationError(f"cannot read release changelog {path}: {exc}") from exc
    pattern = re.compile(
        rf"^## {re.escape(version)} - (\d{{4}}-\d{{2}}-\d{{2}})$",
        flags=re.MULTILINE,
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise VerificationError(
            f"CHANGELOG.md must contain exactly one dated '## {version} - YYYY-MM-DD' release heading; "
            f"found {len(matches)}"
        )
    try:
        parsed = datetime.strptime(matches[0], "%Y-%m-%d")
    except ValueError as exc:
        raise VerificationError(f"CHANGELOG.md release date for {version} is invalid: {matches[0]!r}") from exc
    if parsed.strftime("%Y-%m-%d") != matches[0]:
        raise VerificationError(f"CHANGELOG.md release date for {version} is not canonical: {matches[0]!r}")


def _artifact_name(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise VerificationError(f"expected exactly one {suffix} artifact in {directory}, found {len(matches)}")
    return matches[0]


def _artifact_set(directory: Path) -> Artifacts:
    return Artifacts(
        wheel=_artifact_name(directory, ".whl"),
        sdist=_artifact_name(directory, ".tar.gz"),
    )


def _published_schema_files() -> List[Path]:
    schema_dir = ROOT / "schemas"
    schemas = sorted(schema_dir.glob("*.json"))
    if not schemas:
        raise VerificationError(f"no published JSON Schemas found in {schema_dir}")
    seen: Dict[str, str] = {}
    for path in schemas:
        if path.is_symlink() or not path.is_file():
            raise VerificationError(f"published schema must be a regular file: {path}")
        key = path.name.casefold()
        previous = seen.get(key)
        if previous is not None:
            raise VerificationError(f"published schema filenames collide: {previous!r} and {path.name!r}")
        seen[key] = path.name
    return schemas


def _release_path_is_ignored(relative: Path) -> bool:
    if _forbidden_tracked_path(relative):
        return True
    if relative.name == ".gitignore":
        return True
    return any(part.startswith(".") and part not in {".github"} for part in relative.parts)


def _release_repository_files() -> List[Path]:
    """Return the exact reviewed checkout files staged into the source release."""

    relative_files: List[Path] = []
    for name in RELEASE_ROOT_FILES:
        path = ROOT / name
        if path.is_symlink():
            raise VerificationError(f"release source file must not be a symlink: {name}")
        if path.is_file():
            relative_files.append(Path(name))
    for directory_name in RELEASE_SOURCE_DIRS:
        directory = ROOT / directory_name
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise VerificationError(f"release source directory must be a regular directory: {directory_name}")
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(ROOT)
            if _release_path_is_ignored(relative):
                continue
            if path.is_symlink():
                raise VerificationError(f"release source member must not be a symlink: {relative}")
            if path.is_file():
                relative_files.append(relative)
    return sorted(set(relative_files), key=lambda path: path.as_posix())


def _validate_release_source_state() -> None:
    """Prove release staging reads only clean files represented by ``HEAD``."""

    git = shutil.which("git")
    if git is None:
        raise VerificationError("git disappeared after tool preflight")
    status = _capture((git, "status", "--porcelain", "--untracked-files=all"))
    if status:
        raise VerificationError("release mode requires a clean Git worktree and index")
    tracked = {Path(raw_path) for raw_path in _capture((git, "ls-files", "-z", "--")).split("\0") if raw_path}
    untracked_release_files = sorted(set(_release_repository_files()) - tracked)
    if untracked_release_files:
        raise VerificationError(
            "release staging found files not represented by HEAD: "
            + ", ".join(path.as_posix() for path in untracked_release_files)
        )


def _stage_release_source(destination: Path) -> None:
    destination.mkdir(parents=True)
    for relative in _release_repository_files():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    package_schemas = destination / "src" / "commcanary" / "schemas"
    if package_schemas.is_symlink():
        raise VerificationError(f"staged package schema directory must not be a symlink: {package_schemas}")
    package_schemas.mkdir(parents=True, exist_ok=True)
    for schema in _published_schema_files():
        target = package_schemas / schema.name
        if target.exists() and target.read_bytes() != schema.read_bytes():
            raise VerificationError(f"staged package schema conflicts with published schema: {schema.name}")
        shutil.copy2(schema, target)


def _build_artifacts(workspace: Path, *, source_date_epoch: str) -> Artifacts:
    source = workspace / "source"
    output = workspace / "dist"
    _stage_release_source(source)
    output.mkdir()
    _run(
        "build wheel and sdist",
        (PYTHON, "-m", "build", "--no-isolation", "--outdir", str(output)),
        cwd=source,
        env={"SOURCE_DATE_EPOCH": source_date_epoch},
        unset_env=("PYTHONHOME", "PYTHONPATH"),
    )
    artifacts = _artifact_set(output)
    _normalize_wheel(artifacts.wheel, source_date_epoch=source_date_epoch)
    _normalize_sdist(artifacts.sdist, source_date_epoch=source_date_epoch)
    return artifacts


def _normalize_wheel(path: Path, *, source_date_epoch: str) -> None:
    """Repack a wheel with stable ordering, timestamps, and ZIP metadata."""

    epoch = int(source_date_epoch)
    date_time = datetime.fromtimestamp(epoch, tz=timezone.utc).timetuple()[:6]
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(path) as source:
        members = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(temporary, "w") as destination:
        for original, data in sorted(members, key=lambda item: item[0].filename):
            normalized = zipfile.ZipInfo(original.filename, date_time=date_time)
            normalized.comment = original.comment
            normalized.compress_type = original.compress_type
            normalized.create_system = original.create_system
            normalized.external_attr = original.external_attr
            normalized.internal_attr = original.internal_attr
            destination.writestr(normalized, data, compress_type=original.compress_type, compresslevel=9)
    os.replace(temporary, path)


def _normalize_sdist(path: Path, *, source_date_epoch: str) -> None:
    """Repack a gzip-compressed sdist tarball with stable archive metadata."""

    epoch = int(source_date_epoch)
    members: List[Tuple[tarfile.TarInfo, Optional[bytes]]] = []
    with tarfile.open(path, "r:gz") as source:
        for member in source.getmembers():
            extracted = source.extractfile(member) if member.isfile() else None
            data = extracted.read() if extracted is not None else None
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = epoch
            member.pax_headers = {}
            members.append((member, data))

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as destination:
                for member, data in sorted(members, key=lambda item: item[0].name):
                    if data is None:
                        destination.addfile(member)
                    else:
                        destination.addfile(member, BytesIO(data))
    os.replace(temporary, path)


def _wheel_metadata(archive: zipfile.ZipFile) -> Message:
    metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise VerificationError(f"wheel has {len(metadata_names)} METADATA files")
    return BytesParser(policy=email_policy).parsebytes(archive.read(metadata_names[0]))


def _forbidden_sdist_member(name: str) -> bool:
    return (
        "__pycache__/" in name
        or name.endswith((".pyc", ".pyo", "/.DS_Store", "/context.md"))
        or "/repomix-output." in name
        or "/paper/" in name
        or "/experiments/rostam/results/" in name
    )


def _inspect_artifacts(artifacts: Artifacts, *, expected_version: Optional[str] = None) -> None:
    project_version = _project_version()
    if expected_version is not None and project_version != expected_version:
        raise VerificationError(
            f"requested version {expected_version!r} does not match pyproject version {project_version!r}"
        )

    source_package = ROOT / "src" / "commcanary"
    required_wheel = {
        path.relative_to(ROOT / "src").as_posix()
        for path in source_package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".typed"}
    }
    required_wheel.add("commcanary/py.typed")
    required_wheel.update(f"commcanary/schemas/{path.name}" for path in _published_schema_files())
    with zipfile.ZipFile(artifacts.wheel) as archive:
        names = set(archive.namelist())
        metadata = _wheel_metadata(archive)
    missing_wheel = sorted(required_wheel - names)
    if missing_wheel:
        raise VerificationError(f"wheel is missing package files: {missing_wheel}")
    forbidden_wheel = sorted(
        name
        for name in names
        if name.endswith((".pyc", ".pyo"))
        or "__pycache__/" in name
        or name.startswith(("docs/", "examples/", "experiments/", "tests/", "tools/"))
    )
    if forbidden_wheel:
        raise VerificationError(f"wheel contains forbidden files: {forbidden_wheel}")
    if metadata.get("Name") != "commcanary":
        raise VerificationError(f"wheel project name is {metadata.get('Name')!r}, expected 'commcanary'")
    if metadata.get("Version") != project_version:
        raise VerificationError(f"wheel version {metadata.get('Version')!r} does not match {project_version!r}")
    if metadata.get("Requires-Python") != ">=3.9":
        raise VerificationError("wheel Requires-Python must be exactly '>=3.9'")

    with tarfile.open(artifacts.sdist, "r:gz") as archive:
        sdist_names = set(archive.getnames())
    roots = {name.split("/", 1)[0] for name in sdist_names if name}
    if len(roots) != 1:
        raise VerificationError(f"sdist must have one top-level directory, found {sorted(roots)}")
    root = next(iter(roots))
    required_sdist = {
        *(f"{root}/{path.as_posix()}" for path in _release_repository_files()),
        *(f"{root}/src/{path}" for path in required_wheel),
    }
    missing_sdist = sorted(required_sdist - sdist_names)
    if missing_sdist:
        raise VerificationError(f"sdist is missing required files: {missing_sdist}")
    forbidden_sdist = sorted(name for name in sdist_names if _forbidden_sdist_member(name))
    if forbidden_sdist:
        raise VerificationError(f"sdist contains generated files: {forbidden_sdist}")


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _venv_command(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "commcanary.exe"
    return environment / "bin" / "commcanary"


def _run_captured(
    label: str,
    argv: Sequence[str],
    *,
    cwd: Path,
    unset_env: Iterable[str],
) -> subprocess.CompletedProcess[str]:
    print(f"\n== {label} ==", flush=True)
    print("+ " + " ".join(argv), flush=True)
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        env=_base_env(unset=unset_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        raise VerificationError(f"{label} failed with exit code {completed.returncode}: {detail}")
    return completed


def _test_installed_wheel_typing(python: Path, workspace: Path) -> None:
    """Type-check a downstream fixture and prove mypy used the installed wheel."""

    python = python.resolve()
    fixture = ROOT / "tests" / "fixtures" / "contracts" / "public_api_example.py"
    if not fixture.is_file():
        raise VerificationError(f"missing downstream typing fixture {fixture.relative_to(ROOT)}")
    downstream = workspace / "downstream-typing"
    downstream.mkdir()
    example = downstream / fixture.name
    shutil.copy2(fixture, example)

    clean_env = ("PYTHONHOME", "PYTHONPATH", "MYPYPATH")
    probe_code = (
        "from pathlib import Path; import commcanary; "
        "assert commcanary.__file__ is not None; "
        "print(Path(commcanary.__file__).resolve().parent)"
    )
    probe = _run_captured(
        "locate installed typed package",
        (str(python), "-c", probe_code),
        cwd=workspace,
        unset_env=clean_env,
    )
    probe_lines = probe.stdout.strip().splitlines()
    if len(probe_lines) != 1:
        raise VerificationError(f"installed package probe returned unexpected output: {probe.stdout!r}")
    installed_package = Path(probe_lines[0]).resolve()
    environment = python.parent.parent.resolve()
    checkout_package = (ROOT / "src" / "commcanary").resolve()
    if environment not in installed_package.parents:
        raise VerificationError(f"typing probe resolved outside the wheel environment: {installed_package}")
    if checkout_package == installed_package or checkout_package in installed_package.parents:
        raise VerificationError(
            f"typing probe resolved checkout source instead of the installed wheel: {installed_package}"
        )
    marker = installed_package / "py.typed"
    if not marker.is_file():
        raise VerificationError(f"installed wheel is missing its PEP 561 marker: {marker}")

    mypy_executable = shutil.which("mypy")
    if mypy_executable is None:
        raise VerificationError("mypy disappeared after tool preflight")
    cache = workspace / "mypy-installed-wheel-cache"
    mypy = _run_captured(
        "strict downstream typing against installed wheel",
        (
            mypy_executable,
            "--strict",
            "--python-version",
            "3.9",
            "--python-executable",
            str(python),
            "--no-incremental",
            "--cache-dir",
            str(cache),
            "--show-error-codes",
            "--verbose",
            str(example),
        ),
        cwd=workspace,
        unset_env=clean_env,
    )
    diagnostics = mypy.stdout + "\n" + mypy.stderr
    normalized_diagnostics = os.path.normcase(diagnostics)
    if os.path.normcase(str(installed_package)) not in normalized_diagnostics:
        raise VerificationError("mypy succeeded but did not report resolving commcanary from the installed wheel")
    if os.path.normcase(str(checkout_package)) in normalized_diagnostics:
        raise VerificationError("mypy resolved commcanary typing information from checkout source")
    print(f"mypy resolved typed public API from {installed_package}")


def _test_installed_wheel(wheel: Path, workspace: Path) -> None:
    environment = workspace / "venv"
    # No system site-packages: a base interpreter that happens to carry test
    # tools (or another commcanary) must not be able to mask a broken wheel.
    venv.EnvBuilder(with_pip=True).create(environment)
    python = _venv_python(environment)
    clean_env = ("PYTHONHOME", "PYTHONPATH", "MYPYPATH")
    _run(
        "install exact wheel",
        (str(python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)),
        cwd=workspace,
        unset_env=clean_env,
    )
    # The test toolchain (pytest, jsonschema, ...) is the gate environment's
    # own reviewed toolchain, linked in via a .pth rather than resolved from
    # the network or inherited from the base interpreter. The .pth path sorts
    # after the scratch site-packages, so the wheel-installed commcanary wins;
    # the import-source assertion below proves it.
    gate_purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    scratch_purelib = Path(
        _capture((str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])")).strip()
    )
    (scratch_purelib / "commcanary-gate-toolchain.pth").write_text(f"{gate_purelib}\n", encoding="utf-8")
    assertion = (
        "from pathlib import Path; import commcanary; "
        f"checkout=Path({str(ROOT)!r}).resolve(); "
        f"gate=Path({str(gate_purelib)!r}).resolve(); "
        "loaded=Path(commcanary.__file__).resolve(); "
        "assert checkout not in loaded.parents, (checkout, loaded); "
        "assert gate not in loaded.parents, (gate, loaded); print(loaded)"
    )
    _run(
        "prove imports come from installed wheel",
        (str(python), "-c", assertion),
        cwd=workspace,
        unset_env=clean_env,
    )
    _run(
        "installed module smoke",
        (str(python), "-m", "commcanary", "--help"),
        cwd=workspace,
        unset_env=clean_env,
    )
    _run(
        "installed console-script smoke",
        (str(_venv_command(environment)), "--help"),
        cwd=workspace,
        unset_env=clean_env,
    )
    _test_installed_wheel_typing(python, workspace)
    installed_tests = [str(path) for path in _installed_wheel_test_paths()]
    if not installed_tests:
        raise VerificationError("no installed-package tests were found")
    _run(
        "installed wheel tests",
        (
            str(python),
            "-m",
            "pytest",
            "-q",
            "--import-mode=importlib",
            "-p",
            "no:cacheprovider",
            *installed_tests,
        ),
        cwd=workspace,
        unset_env=clean_env,
    )


def _installed_wheel_test_paths() -> List[Path]:
    """Return behavioral tests that must run against the installed artifact.

    Repository/tooling and source-architecture tests belong to the source gate.
    Runtime capability suites remain here even when they move below ``tests/``;
    selecting only top-level files would silently weaken this gate after a
    test-architecture refactor.
    """

    test_root = ROOT / "tests"
    selected = [path for path in sorted(test_root.glob("test_*.py")) if path.name not in INSTALLED_TEST_EXCLUSIONS]
    for directory in INSTALLED_TEST_SUBDIRECTORIES:
        selected.extend(sorted((test_root / directory).rglob("test_*.py")))
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(artifacts: Artifacts) -> Dict[str, str]:
    return {
        artifacts.wheel.name: _sha256(artifacts.wheel),
        artifacts.sdist.name: _sha256(artifacts.sdist),
    }


def _copy_artifacts(artifacts: Artifacts, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    existing = sorted((*destination.glob("*.whl"), *destination.glob("*.tar.gz")))
    if existing:
        raise VerificationError(f"artifact output directory {destination} already contains release artifacts")
    shutil.copy2(artifacts.wheel, destination / artifacts.wheel.name)
    shutil.copy2(artifacts.sdist, destination / artifacts.sdist.name)


def _package_gate(
    *,
    release: bool,
    reproducible: bool = False,
    artifact_dir: Optional[Path],
    metadata_dir: Optional[Path],
    expected_version: Optional[str],
) -> None:
    if release:
        _verify_release_changelog(expected_version or _project_version())
        _validate_release_source_state()
    with tempfile.TemporaryDirectory(prefix="commcanary-package-") as raw_tmp:
        tmp = Path(raw_tmp)
        first = _build_artifacts(tmp / "first", source_date_epoch=SOURCE_DATE_EPOCH)
        _inspect_artifacts(first, expected_version=expected_version)
        _run(
            "twine metadata and README check",
            (PYTHON, "-m", "twine", "check", str(first.wheel), str(first.sdist)),
            cwd=tmp,
            unset_env=("PYTHONHOME", "PYTHONPATH"),
        )
        _test_installed_wheel(first.wheel, tmp / "installed-test")

        hashes = _artifact_hashes(first)
        if release or reproducible:
            second = _build_artifacts(tmp / "second", source_date_epoch=SOURCE_DATE_EPOCH)
            second_hashes = _artifact_hashes(second)
            if hashes != second_hashes:
                raise VerificationError(f"artifacts are not reproducible: first={hashes!r}, second={second_hashes!r}")
            print("\n== reproducible artifacts ==", flush=True)
            for name, digest in sorted(hashes.items()):
                print(f"{digest}  {name}")

        if metadata_dir is not None:
            if artifact_dir is not None and metadata_dir == artifact_dir:
                raise VerificationError("--metadata-dir and --artifact-dir must be different directories")
            try:
                files = release_metadata.write_release_metadata(
                    release_metadata.DistributionSet(wheel=first.wheel, sdist=first.sdist),
                    metadata_dir,
                    project="commcanary",
                    version=_project_version(),
                    source_date_epoch=SOURCE_DATE_EPOCH,
                    expected_sha256=hashes,
                )
            except release_metadata.ReleaseMetadataError as exc:
                raise VerificationError(f"release metadata generation failed: {exc}") from exc
            print(f"wrote release checksums to {files.checksums}")
            print(f"wrote release inventory to {files.inventory}")
            print(f"wrote release SPDX SBOM to {files.sbom}")

        if artifact_dir is not None:
            _copy_artifacts(first, artifact_dir)
            print(f"copied tested artifacts to {artifact_dir}")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fast",
        action="store_true",
        help="skip artifact build/install tests",
    )
    mode.add_argument(
        "--reproducible",
        action="store_true",
        help="run two clean fixed-epoch builds and require byte-identical artifacts",
    )
    mode.add_argument(
        "--release",
        action="store_true",
        help="run the reproducible gate and require finalized release identity",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="copy the exact tested wheel and sdist here (not valid with --fast)",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="write checksums, content inventory, and SPDX SBOM here (not valid with --fast)",
    )
    parser.add_argument(
        "--expected-version",
        help="require project and artifact versions to match this value",
    )
    args = parser.parse_args(argv)
    if args.fast and (
        args.artifact_dir is not None or args.metadata_dir is not None or args.expected_version is not None
    ):
        parser.error("--artifact-dir, --metadata-dir, and --expected-version require the full or release gate")
    return args


def _validate_coverage(coverage_file: Path, output_path: Path) -> None:
    """Emit machine-readable counters and enforce CommCanary's exact policy."""

    _run(
        "coverage JSON",
        (
            PYTHON,
            "-m",
            "coverage",
            "json",
            f"--data-file={coverage_file}",
            "--fail-under=0",
            "-o",
            str(output_path),
        ),
    )
    _run(
        "coverage policy",
        (
            PYTHON,
            "-m",
            "tools.coverage_policy",
            "--report",
            str(output_path),
            "--policy",
            str(ROOT / "tools" / "coverage_policy.json"),
        ),
    )
    _run(
        "coverage report",
        (
            PYTHON,
            "-m",
            "coverage",
            "report",
            f"--data-file={coverage_file}",
            "--fail-under=0",
        ),
    )


def _validate_import_boundaries() -> None:
    _run(
        "import boundaries",
        (
            PYTHON,
            "-m",
            "tools.import_boundaries",
            "--source-root",
            "src",
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    include_artifacts = not args.fast
    try:
        _preflight(include_artifacts=include_artifacts)
        _validate_installed_source_mirror()
        _validate_repository_hygiene()

        ruff = shutil.which("ruff")
        mypy = shutil.which("mypy")
        if ruff is None or mypy is None:
            raise VerificationError("ruff or mypy disappeared after tool preflight")
        with tempfile.TemporaryDirectory(prefix="commcanary-check-") as raw_tmp:
            tmp = Path(raw_tmp)
            _run("ruff lint", (ruff, "check", "--no-cache", *SOURCE_DIRS))
            _run(
                "ruff format",
                (ruff, "format", "--check", "--no-cache", *SOURCE_DIRS),
            )
            _run(
                "mypy",
                (mypy, "--cache-dir", str(tmp / "mypy"), *MYPY_TARGETS),
            )
            _validate_import_boundaries()
            coverage_file = tmp / ".coverage"
            _run(
                "tests with coverage",
                (
                    PYTHON,
                    "-m",
                    "coverage",
                    "run",
                    f"--data-file={coverage_file}",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                ),
                unset_env=("PYTHONPATH",),
            )
            _validate_coverage(coverage_file, tmp / "coverage.json")

        _validate_json_files()
        _validate_shell_files()
        _validate_workflows()
        _validate_readme()
        if include_artifacts:
            artifact_dir = args.artifact_dir
            if artifact_dir is not None:
                artifact_dir = (ROOT / artifact_dir if not artifact_dir.is_absolute() else artifact_dir).resolve()
            metadata_dir = args.metadata_dir
            if metadata_dir is not None:
                metadata_dir = (ROOT / metadata_dir if not metadata_dir.is_absolute() else metadata_dir).resolve()
            _package_gate(
                release=args.release,
                reproducible=args.reproducible,
                artifact_dir=artifact_dir,
                metadata_dir=metadata_dir,
                expected_version=args.expected_version,
            )
    except (OSError, VerificationError) as exc:
        print(f"\nverification failed: {exc}", file=sys.stderr)
        return 1
    print("\nverification complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
