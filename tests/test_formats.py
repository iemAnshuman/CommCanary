from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import commcanary
from commcanary.experimental import ddmin_ranking_reduction, isolated_collective_baseline_trace
from commcanary.formats import (
    BEHAVIOR_VERIFICATION_FORMAT,
    CANARY_FORMAT,
    CANONICAL_JSON_VERSION,
    COMPARE_FORMAT,
    FIDELITY_VERIFICATION_FORMAT,
    REPORT_FORMAT,
    REPORT_VERIFICATION_FORMAT,
    TRACE_FORMAT,
    format_capabilities,
)
from commcanary.version import SOURCE_TREE_VERSION, package_version


def test_format_capabilities_are_exact_unique_and_immutable() -> None:
    capabilities = format_capabilities()
    expected = {
        TRACE_FORMAT,
        CANARY_FORMAT,
        REPORT_FORMAT,
        COMPARE_FORMAT,
        FIDELITY_VERIFICATION_FORMAT,
        BEHAVIOR_VERIFICATION_FORMAT,
        REPORT_VERIFICATION_FORMAT,
    }

    assert isinstance(capabilities, tuple)
    assert {capability.format_id for capability in capabilities} == expected
    assert len({capability.artifact for capability in capabilities}) == len(capabilities)
    assert all(not capability.migrate for capability in capabilities)
    assert all(capability.schema.startswith("schemas/") for capability in capabilities)
    assert CANONICAL_JSON_VERSION == "commcanary.canonical-json.v1"

    with pytest.raises(FrozenInstanceError):
        capabilities[0].read = False  # type: ignore[misc]


def test_verification_outputs_are_write_only_without_semantic_validators() -> None:
    verification = [capability for capability in format_capabilities() if capability.artifact.endswith("verification")]

    assert len(verification) == 3
    assert all(not capability.read for capability in verification)
    assert all(capability.write for capability in verification)
    assert all(not capability.semantic_validator for capability in verification)


def test_public_api_is_explicit_and_version_comes_from_distribution_metadata() -> None:
    expected = {
        "CommCanaryError",
        "FormatCapability",
        "ResourceLimits",
        "SchemaError",
        "__version__",
        "compare_reports",
        "compile_trace",
        "format_capabilities",
        "load_json",
        "replay_canary",
        "validate_canary",
        "validate_comparison",
        "validate_report",
        "validate_trace",
        "verify_canary_behavior",
        "verify_canary_fidelity",
        "verify_report_against_canary",
    }

    assert expected.issubset(set(commcanary.__all__))
    assert commcanary.__version__ == package_version()
    assert package_version() == SOURCE_TREE_VERSION or package_version()[0].isdigit()
    assert "ddmin_ranking_reduction" not in commcanary.__all__
    assert callable(ddmin_ranking_reduction)
    assert callable(isolated_collective_baseline_trace)
