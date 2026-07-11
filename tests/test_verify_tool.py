from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import verify


def test_base_env_can_remove_python_import_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/checkout/src")
    monkeypatch.setenv("PYTHONHOME", "/custom/python")

    env = verify._base_env({"COMM_CANARY_TEST": "1"}, unset=("PYTHONPATH", "PYTHONHOME"))

    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env["COMM_CANARY_TEST"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_installed_source_mirror_rejects_stale_build_members(tmp_path: Path) -> None:
    source = tmp_path / "source"
    installed = tmp_path / "installed"
    source.mkdir()
    installed.mkdir()
    (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (installed / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (installed / "deleted_module.py").write_text("STALE = True\n", encoding="utf-8")

    with pytest.raises(verify.VerificationError, match=r"installed CommCanary.*extra=deleted_module\.py"):
        verify._compare_runtime_package_trees(source, installed)

    (installed / "deleted_module.py").unlink()
    verify._compare_runtime_package_trees(source, installed)


@pytest.mark.parametrize(
    "raw_path",
    (
        ".DS_Store",
        "docs/.DS_Store",
        "context.md",
        "repomix-output.xml",
        "src/commcanary/__pycache__/module.pyc",
        "src/commcanary.egg-info/PKG-INFO",
        ".benchmark-data/results.json",
        "release-metadata/SHA256SUMS",
        "experiments/rostam/results/run/result.json",
    ),
)
def test_repository_hygiene_rejects_private_and_generated_paths(raw_path: str) -> None:
    assert verify._forbidden_tracked_path(Path(raw_path))


def test_repository_hygiene_allows_reviewed_source_and_golden_fixtures() -> None:
    assert not verify._forbidden_tracked_path(Path("src/commcanary/schema.py"))
    assert not verify._forbidden_tracked_path(Path("tests/fixtures/experiments/golden/aggregate.json"))


def test_coverage_gate_emits_json_and_runs_authoritative_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_run(label, argv, **kwargs):
        calls.append((label, tuple(argv), kwargs))

    monkeypatch.setattr(verify, "_run", fake_run)
    data_file = tmp_path / ".coverage"
    json_file = tmp_path / "coverage.json"

    verify._validate_coverage(data_file, json_file)

    assert [label for label, _, _ in calls] == [
        "coverage JSON",
        "coverage policy",
        "coverage report",
    ]
    assert "--fail-under=0" in calls[0][1]
    assert str(json_file) in calls[0][1]
    assert calls[1][1] == (
        verify.PYTHON,
        "-m",
        "tools.coverage_policy",
        "--report",
        str(json_file),
        "--policy",
        str(verify.ROOT / "tools" / "coverage_policy.json"),
    )
    assert "--fail-under=0" in calls[2][1]


def test_canonical_gate_runs_the_ast_import_boundary_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(verify, "_run", lambda label, argv, **kwargs: calls.append((label, tuple(argv), kwargs)))

    verify._validate_import_boundaries()

    assert calls == [
        (
            "import boundaries",
            (verify.PYTHON, "-m", "tools.import_boundaries", "--source-root", "src"),
            {},
        )
    ]


def test_installed_wheel_gate_keeps_nested_runtime_suites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_root = tmp_path / "tests"
    (test_root / "compilation").mkdir(parents=True)
    (test_root / "benchmarks").mkdir()
    included_top_level = test_root / "test_runtime.py"
    included_nested = test_root / "compilation" / "test_compiler.py"
    excluded_tooling = test_root / "test_verify_tool.py"
    excluded_benchmark = test_root / "benchmarks" / "test_runner.py"
    for path in (included_top_level, included_nested, excluded_tooling, excluded_benchmark):
        path.write_text("def test_placeholder(): pass\n", encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    selected = verify._installed_wheel_test_paths()

    assert selected == [included_top_level, included_nested]


def test_readme_check_validates_links_and_command_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide", encoding="utf-8")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "demo.py").write_text("print('demo')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[Guide](docs/guide.md)\n\n```bash\ncommcanary compile examples/demo.py\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    verify._validate_readme()

    (tmp_path / "docs" / "guide.md").write_text("[Missing](missing.md)\n", encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="guide.md: missing local link"):
        verify._validate_readme()

    (tmp_path / "docs" / "guide.md").unlink()
    with pytest.raises(verify.VerificationError, match="docs/guide.md"):
        verify._validate_readme()


def test_json_check_rejects_invalid_utf8_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "broken.json").write_bytes(b'{"broken":')
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    with pytest.raises(verify.VerificationError, match="invalid JSON file"):
        verify._validate_json_files()


def test_json_check_includes_top_level_schema_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    (schemas / "broken.schema.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    with pytest.raises(verify.VerificationError, match="schemas/broken.schema.json"):
        verify._validate_json_files()


def test_packaged_schema_mirror_must_match_published_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = tmp_path / "schemas"
    packaged = tmp_path / "src" / "commcanary" / "schemas"
    published.mkdir()
    packaged.mkdir(parents=True)
    (published / "trace.schema.json").write_bytes(b'{"type":"object"}\n')
    (packaged / "trace.schema.json").write_bytes(b'{"type":"array"}\n')
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    with pytest.raises(verify.VerificationError, match=r"packaged schema mirror is stale:.*mismatched"):
        verify._validate_packaged_schema_mirror()


def test_artifact_hashes_and_copy_refuse_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    wheel = source / "commcanary-0.3.0-py3-none-any.whl"
    sdist = source / "commcanary-0.3.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    artifacts = verify.Artifacts(wheel=wheel, sdist=sdist)

    assert verify._artifact_hashes(artifacts) == {
        wheel.name: hashlib.sha256(b"wheel").hexdigest(),
        sdist.name: hashlib.sha256(b"sdist").hexdigest(),
    }

    destination = tmp_path / "release"
    verify._copy_artifacts(artifacts, destination)
    assert (destination / wheel.name).read_bytes() == b"wheel"
    assert (destination / sdist.name).read_bytes() == b"sdist"
    with pytest.raises(verify.VerificationError, match="already contains"):
        verify._copy_artifacts(artifacts, destination)


def test_artifact_normalization_removes_archive_timestamps(tmp_path: Path) -> None:
    archives = []
    for index, timestamp in enumerate((1_700_000_000, 1_710_000_000)):
        directory = tmp_path / str(index)
        directory.mkdir()
        wheel = directory / "package.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            info = zipfile.ZipInfo("package/__init__.py")
            info.date_time = (2023 + index, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, b"value = 1\n")

        sdist = directory / "package.tar.gz"
        with tarfile.open(sdist, "w:gz") as archive:
            info = tarfile.TarInfo("package-1/package/__init__.py")
            info.mtime = timestamp
            payload = b"value = 1\n"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

        verify._normalize_wheel(wheel, source_date_epoch=verify.SOURCE_DATE_EPOCH)
        verify._normalize_sdist(sdist, source_date_epoch=verify.SOURCE_DATE_EPOCH)
        archives.append(verify.Artifacts(wheel=wheel, sdist=sdist))

    assert verify._sha256(archives[0].wheel) == verify._sha256(archives[1].wheel)
    assert verify._sha256(archives[0].sdist) == verify._sha256(archives[1].sdist)


def test_artifact_inspection_matches_src_layout(tmp_path: Path) -> None:
    wheel = tmp_path / "commcanary-0.3.0-py3-none-any.whl"
    source_files = [
        path
        for path in (verify.ROOT / "src" / "commcanary").rglob("*")
        if path.is_file() and path.suffix in {".py", ".typed"}
    ]
    schema_files = verify._published_schema_files()
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in source_files:
            archive.write(path, path.relative_to(verify.ROOT / "src").as_posix())
        for path in schema_files:
            archive.write(path, f"commcanary/schemas/{path.name}")
        archive.writestr(
            "commcanary-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: commcanary\nVersion: 0.3.0\nRequires-Python: >=3.9\n\n",
        )

    sdist = tmp_path / "commcanary-0.3.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for relative in verify._release_repository_files():
            archive.add(verify.ROOT / relative, arcname=f"commcanary-0.3.0/{relative.as_posix()}")
        for path in schema_files:
            archive.add(path, arcname=f"commcanary-0.3.0/src/commcanary/schemas/{path.name}")

    verify._inspect_artifacts(verify.Artifacts(wheel=wheel, sdist=sdist), expected_version="0.3.0")

    with pytest.raises(verify.VerificationError, match="requested version"):
        verify._inspect_artifacts(verify.Artifacts(wheel=wheel, sdist=sdist), expected_version="9.9.9")

    wheel_without_marker = tmp_path / "missing-py-typed.whl"
    with zipfile.ZipFile(wheel_without_marker, "w") as archive:
        for path in source_files:
            if path.name != "py.typed":
                archive.write(path, path.relative_to(verify.ROOT / "src").as_posix())
        for path in schema_files:
            archive.write(path, f"commcanary/schemas/{path.name}")
        archive.writestr(
            "commcanary-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: commcanary\nVersion: 0.3.0\nRequires-Python: >=3.9\n\n",
        )
    with pytest.raises(verify.VerificationError, match=r"missing package files:.*commcanary/py\.typed"):
        verify._inspect_artifacts(
            verify.Artifacts(wheel=wheel_without_marker, sdist=sdist),
            expected_version="0.3.0",
        )

    wheel_without_schema = tmp_path / "missing-schema.whl"
    with zipfile.ZipFile(wheel_without_schema, "w") as archive:
        for path in source_files:
            archive.write(path, path.relative_to(verify.ROOT / "src").as_posix())
        for path in schema_files[1:]:
            archive.write(path, f"commcanary/schemas/{path.name}")
        archive.writestr(
            "commcanary-0.3.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: commcanary\nVersion: 0.3.0\nRequires-Python: >=3.9\n\n",
        )
    with pytest.raises(verify.VerificationError, match=rf"missing package files:.*{schema_files[0].name}"):
        verify._inspect_artifacts(
            verify.Artifacts(wheel=wheel_without_schema, sdist=sdist),
            expected_version="0.3.0",
        )


def test_stage_release_source_excludes_generated_package_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkout"
    package = root / "src" / "commcanary"
    package.mkdir(parents=True)
    schemas = root / "schemas"
    schemas.mkdir()
    published_schema = schemas / "example.schema.json"
    published_schema.write_text('{"$schema":"https://json-schema.org/draft/2020-12/schema"}\n', encoding="utf-8")
    (root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    (root / "LICENSE").write_text("license\n", encoding="utf-8")
    docs = root / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("guide\n", encoding="utf-8")
    (docs / ".DS_Store").write_bytes(b"generated")
    (package / "__init__.py").write_text("", encoding="utf-8")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"generated")
    egg_info = root / "src" / "commcanary.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated", encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", root)

    staged = tmp_path / "staged"
    verify._stage_release_source(staged)

    assert (staged / "src" / "commcanary" / "__init__.py").is_file()
    assert (staged / "docs" / "guide.md").is_file()
    assert not (staged / "docs" / ".DS_Store").exists()
    assert not (staged / "src" / "commcanary" / "__pycache__").exists()
    assert not (staged / "src" / "commcanary.egg-info").exists()
    assert (staged / "src" / "commcanary" / "schemas" / published_schema.name).read_bytes() == (
        published_schema.read_bytes()
    )


def test_installed_wheel_typing_check_invokes_strict_mypy_in_clean_downstream_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = workspace / "venv"
    python = environment / "bin" / "python"
    installed_package = environment / "lib" / "python3.12" / "site-packages" / "commcanary"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    (installed_package / "py.typed").write_text("", encoding="utf-8")
    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PYTHONPATH", "/checkout/src")
    monkeypatch.setenv("MYPYPATH", "/checkout/stubs")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if "-c" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=f"{installed_package}\n", stderr="")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="Success: no issues found in 1 source file\n",
            stderr=f"LOG:  Parsing {installed_package / '__init__.py'} (commcanary)\n",
        )

    monkeypatch.setattr(verify.subprocess, "run", fake_run)
    monkeypatch.setattr(verify.shutil, "which", lambda name: "/tools/mypy" if name == "mypy" else None)

    verify._test_installed_wheel_typing(python, workspace)

    assert len(calls) == 2
    for _, kwargs in calls:
        assert kwargs["cwd"] == str(workspace)
        assert "PYTHONHOME" not in kwargs["env"]
        assert "PYTHONPATH" not in kwargs["env"]
        assert "MYPYPATH" not in kwargs["env"]
    mypy_command = calls[1][0]
    assert mypy_command[0] == "/tools/mypy"
    assert "--strict" in mypy_command
    assert mypy_command[mypy_command.index("--python-version") + 1] == "3.9"
    assert mypy_command[mypy_command.index("--python-executable") + 1] == str(python)
    downstream_example = workspace / "downstream-typing" / "public_api_example.py"
    assert mypy_command[-1] == str(downstream_example)
    assert (
        downstream_example.read_bytes()
        == (verify.ROOT / "tests" / "fixtures" / "contracts" / "public_api_example.py").read_bytes()
    )


def test_installed_wheel_typing_check_requires_marker_before_mypy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = workspace / "venv"
    python = environment / "bin" / "python"
    installed_package = environment / "lib" / "python3.12" / "site-packages" / "commcanary"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=f"{installed_package}\n", stderr="")

    monkeypatch.setattr(verify.subprocess, "run", fake_run)

    with pytest.raises(verify.VerificationError, match="missing its PEP 561 marker"):
        verify._test_installed_wheel_typing(python, workspace)
    assert len(calls) == 1


def test_installed_wheel_typing_check_rejects_checkout_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = workspace / "venv"
    python = environment / "bin" / "python"
    installed_package = environment / "lib" / "python3.12" / "site-packages" / "commcanary"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    (installed_package / "py.typed").write_text("", encoding="utf-8")

    def fake_run(argv, **kwargs):
        if "-c" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=f"{installed_package}\n", stderr="")
        checkout_package = verify.ROOT / "src" / "commcanary"
        diagnostics = (
            f"LOG: Parsing {installed_package / '__init__.py'}\nLOG: Parsing {checkout_package / 'schema.py'}\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr=diagnostics)

    monkeypatch.setattr(verify.subprocess, "run", fake_run)
    monkeypatch.setattr(verify.shutil, "which", lambda name: "/tools/mypy" if name == "mypy" else None)

    with pytest.raises(verify.VerificationError, match="checkout source"):
        verify._test_installed_wheel_typing(python, workspace)


def test_fast_mode_rejects_artifact_output() -> None:
    with pytest.raises(SystemExit):
        verify._parse_args(["--fast", "--artifact-dir", "dist"])


def test_fast_mode_rejects_release_metadata_output() -> None:
    with pytest.raises(SystemExit):
        verify._parse_args(["--fast", "--metadata-dir", "release-metadata"])


def test_reproducible_mode_is_distinct_from_release_identity() -> None:
    args = verify._parse_args(["--reproducible"])

    assert args.reproducible is True
    assert args.release is False


def test_static_gate_includes_benchmark_package() -> None:
    assert "benchmarks" in verify.SOURCE_DIRS
    assert "benchmarks" in verify.MYPY_TARGETS
    assert "experiments/rostam" in verify.MYPY_TARGETS
    assert "experiments/rostam/harness" not in verify.MYPY_TARGETS


def test_release_source_policy_excludes_unregenerable_historical_paper() -> None:
    assert "paper" not in verify.RELEASE_SOURCE_DIRS
    assert not any(path.parts[0] == "paper" for path in verify._release_repository_files())
    assert "recursive-include paper" not in (verify.ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert verify._forbidden_sdist_member("commcanary-0.3.0/paper/draft.md")


def test_workflow_pin_policy_accepts_sha_version_comment_and_local_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    workflow = workflow_dir / "valid.yml"
    workflow.write_text(
        f'steps:\n  - uses: actions/checkout@{"a" * 40} # v4.3.1\n  - uses: "./.github/actions/local"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    verify._validate_workflow_action_pins([workflow])


@pytest.mark.parametrize(
    "uses_line, expected",
    [
        ("uses: actions/checkout@v4 # v4.3.1", "40-character"),
        (f"uses: actions/checkout@{'A' * 40} # v4.3.1", "lowercase"),
        (f"uses: actions/checkout@{'a' * 40}", "version comment"),
        (f"uses: actions/checkout@{'a' * 40} # release-4", "version comment"),
        ("uses: docker://alpine:3.20 # v3.20.0", "40-character"),
    ],
)
def test_workflow_pin_policy_rejects_mutable_or_unauditable_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uses_line: str,
    expected: str,
) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(f"steps:\n  - {uses_line}\n", encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    with pytest.raises(verify.VerificationError, match=expected):
        verify._validate_workflow_action_pins([workflow])


def test_workflow_pin_policy_runs_before_actionlint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "invalid.yml").write_text("steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    with pytest.raises(verify.VerificationError, match="workflow action pin policy"):
        verify._validate_workflows()


def test_package_gate_generates_metadata_for_same_tested_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "commcanary-0.3.0-py3-none-any.whl"
    sdist = tmp_path / "commcanary-0.3.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    artifacts = verify.Artifacts(wheel=wheel, sdist=sdist)
    seen = []
    seen_options = []
    copied = []

    monkeypatch.setattr(verify, "_build_artifacts", lambda *args, **kwargs: artifacts)
    monkeypatch.setattr(verify, "_inspect_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify, "_test_installed_wheel", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify, "_project_version", lambda: "0.3.0")
    monkeypatch.setattr(verify, "_copy_artifacts", lambda value, destination: copied.append(value))

    def write_metadata(value, destination, **kwargs):
        seen.append(value)
        seen_options.append(kwargs)
        return verify.release_metadata.ReleaseMetadataFiles(
            checksums=destination / "SHA256SUMS",
            inventory=destination / "release-inventory.json",
            sbom=destination / "commcanary-0.3.0.spdx.json",
        )

    monkeypatch.setattr(verify.release_metadata, "write_release_metadata", write_metadata)
    verify._package_gate(
        release=False,
        artifact_dir=tmp_path / "packages",
        metadata_dir=tmp_path / "metadata",
        expected_version="0.3.0",
    )

    assert len(seen) == 1
    assert seen[0].wheel is artifacts.wheel
    assert seen[0].sdist is artifacts.sdist
    assert seen_options[0]["expected_sha256"] == verify._artifact_hashes(artifacts)
    assert copied == [artifacts]


def test_release_changelog_requires_one_canonical_dated_heading(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## 0.3.0 - 2026-07-11\n", encoding="utf-8")
    verify._verify_release_changelog("0.3.0", changelog=changelog)

    changelog.write_text("# Changelog\n\n## 0.3.0 - Unreleased\n", encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="exactly one dated"):
        verify._verify_release_changelog("0.3.0", changelog=changelog)

    changelog.write_text("# Changelog\n\n## 0.3.0 - 2026-02-30\n", encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="date.*invalid"):
        verify._verify_release_changelog("0.3.0", changelog=changelog)


def test_release_package_gate_checks_changelog_before_build(monkeypatch: pytest.MonkeyPatch) -> None:
    checked = []
    monkeypatch.setattr(verify, "_verify_release_changelog", lambda version: checked.append(version))
    monkeypatch.setattr(verify, "_validate_release_source_state", lambda: None)
    monkeypatch.setattr(
        verify,
        "_build_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after identity check")),
    )

    with pytest.raises(RuntimeError, match="identity check"):
        verify._package_gate(
            release=True,
            artifact_dir=None,
            metadata_dir=None,
            expected_version="0.3.0",
        )
    assert checked == ["0.3.0"]


def test_release_source_state_requires_clean_head_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "README.md").write_text("release\n", encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify.shutil, "which", lambda name: "/usr/bin/git" if name == "git" else None)

    def clean_capture(argv, **kwargs):
        if "status" in argv:
            return ""
        return "README.md\0"

    monkeypatch.setattr(verify, "_capture", clean_capture)
    verify._validate_release_source_state()

    monkeypatch.setattr(verify, "_capture", lambda argv, **kwargs: "?? README.md\n" if "status" in argv else "")
    with pytest.raises(verify.VerificationError, match="clean Git worktree"):
        verify._validate_release_source_state()


def test_reproducible_package_gate_builds_twice_without_finalizing_changelog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "commcanary-0.3.0-py3-none-any.whl"
    sdist = tmp_path / "commcanary-0.3.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    artifacts = verify.Artifacts(wheel=wheel, sdist=sdist)
    builds = []

    def fake_build(workspace: Path, *, source_date_epoch: str) -> verify.Artifacts:
        builds.append((workspace, source_date_epoch))
        return artifacts

    monkeypatch.setattr(verify, "_build_artifacts", fake_build)
    monkeypatch.setattr(verify, "_inspect_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify, "_test_installed_wheel", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        verify,
        "_verify_release_changelog",
        lambda version: (_ for _ in ()).throw(AssertionError(version)),
    )

    verify._package_gate(
        release=False,
        reproducible=True,
        artifact_dir=None,
        metadata_dir=None,
        expected_version="0.3.0",
    )

    assert len(builds) == 2
