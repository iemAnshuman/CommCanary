from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import commcanary
import commcanary.formats as formats
from commcanary.formats import format_capabilities

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "contracts" / "public_api_example.py"
DOC_PATH = ROOT / "docs" / "formats" / "public-api.md"


def _documented_python_example() -> str:
    document = DOC_PATH.read_text(encoding="utf-8")
    marked = document.split("<!-- golden-example:start -->", 1)[1].split("<!-- golden-example:end -->", 1)[0]
    return marked.split("```python\n", 1)[1].rsplit("```", 1)[0]


def test_documented_public_type_example_is_literal_executable_fixture() -> None:
    source = FIXTURE_PATH.read_text(encoding="utf-8")
    assert _documented_python_example() == source
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, str(FIXTURE_PATH)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_top_level_and_format_module_exports_are_golden() -> None:
    assert commcanary.__all__ == [
        "CANARY_FORMAT",
        "CANONICAL_JSON_VERSION",
        "COMPARE_FORMAT",
        "DEFAULT_RESOURCE_LIMITS",
        "REPORT_FORMAT",
        "TRACE_FORMAT",
        "CommCanaryError",
        "FormatCapability",
        "JsonDict",
        "ResourceLimits",
        "SchemaError",
        "__version__",
        "compare_reports",
        "compile_trace",
        "format_capabilities",
        "load_json",
        "package_version",
        "replay_canary",
        "validate_canary",
        "validate_comparison",
        "validate_report",
        "validate_trace",
        "verify_canary_behavior",
        "verify_canary_fidelity",
        "verify_report_against_canary",
    ]
    assert formats.__all__ == [
        "ARTIFACT_PROVENANCE_ALGORITHM",
        "BEHAVIOR_VERIFICATION_FORMAT",
        "CANARY_FORMAT",
        "CANARY_INTEGRITY_PROFILE",
        "CANONICAL_JSON_VERSION",
        "COMPARE_FORMAT",
        "FIDELITY_VERIFICATION_FORMAT",
        "FORMAT_CAPABILITIES",
        "FormatCapability",
        "REPORT_FORMAT",
        "REPORT_VERIFICATION_FORMAT",
        "TRACE_FORMAT",
        "format_capabilities",
    ]


def test_capability_query_matches_published_schema_files_and_is_immutable() -> None:
    capabilities = format_capabilities()
    assert capabilities is format_capabilities()
    assert len(capabilities) == 7
    assert len({capability.artifact for capability in capabilities}) == 7
    assert len({capability.format_id for capability in capabilities}) == 7
    for capability in capabilities:
        schema_path = ROOT / capability.schema
        assert schema_path.is_file()
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["properties"]["format"]["const"] == capability.format_id
        assert capability.migrate is False
