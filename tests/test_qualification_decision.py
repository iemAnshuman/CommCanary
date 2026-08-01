from __future__ import annotations

import copy

import pytest

from commcanary.artifacts import (
    qualification_observation_sha256,
    qualification_policy_sha256,
    qualification_verdict_sha256,
    validate_qualification_observation,
    validate_qualification_policy,
    validate_qualification_verdict,
)
from commcanary.errors import SchemaError
from commcanary.formats import QUALIFICATION_OBSERVATION_FORMAT, QUALIFICATION_POLICY_FORMAT
from commcanary.services import evaluate_qualification_observations


def _policy() -> dict:
    policy = {
        "format": QUALIFICATION_POLICY_FORMAT,
        "primary_metric": "program_median_us",
        "repetitions": {"warmup": 5, "minimum_measured": 20},
        "regression": {
            "relative_threshold_pct": 5.0,
            "absolute_threshold_us": 2.0,
            "threshold_combination": "larger_of_absolute_or_relative",
        },
        "uncertainty": {
            "method": "percentile_bootstrap_median_difference",
            "confidence": 0.95,
            "bootstrap_resamples": 1000,
            "seed": 7411,
        },
        "noise": {"max_relative_iqr_pct": 20.0},
        "environment": {
            "max_clock_variation_pct": 3.0,
            "require_same_gpu_count": True,
            "require_same_topology_class": True,
        },
        "outcome_policy": {
            "incomplete_measurement": "inconclusive",
            "unstable_measurement": "inconclusive",
            "environment_mismatch": "incomparable",
            "baseline_correctness_failure": "incomparable",
            "candidate_correctness_failure": "fail",
            "confidence_interval_crosses_threshold": "inconclusive",
        },
    }
    policy["policy_id"] = qualification_policy_sha256(policy)
    return policy


def _observation(
    policy: dict,
    *,
    role: str,
    samples: list[float],
    topology: str = "a100-pcie-4gpu",
    correctness: str = "pass",
    warmup: int = 5,
) -> dict:
    observation = {
        "format": QUALIFICATION_OBSERVATION_FORMAT,
        "request_id": "1" * 64,
        "materialization_id": "2" * 64,
        "policy_id": policy["policy_id"],
        "role": role,
        "metric": {"name": "program_median_us", "unit": "microseconds", "lower_is_better": True},
        "environment": {
            "gpu_count": 4,
            "topology_class": topology,
            "software_stack_sha256": ("3" if role == "baseline" else "4") * 64,
            "clock_variation_pct": 1.0,
        },
        "execution": {
            "executor": "commcanary-reference",
            "executor_version": "0.3.0",
            "hostname": "node0",
            "job_id": "123",
            "observed_at": "2026-08-01T00:00:00Z",
        },
        "measurement": {
            "warmup_count": warmup,
            "samples_us": samples,
            "discarded_attempts": 0,
            "correctness": {"status": correctness, "check_count": 32},
        },
        "claims": {
            "physical_execution": "observed",
            "executor_conformance": "unproven",
            "physical_decision": "not_interpreted",
            "producer_authenticity": "unsigned",
        },
    }
    observation["observation_id"] = qualification_observation_sha256(observation)
    return observation


def test_policy_and_observation_identities_fail_closed() -> None:
    policy = _policy()
    observation = _observation(policy, role="baseline", samples=[100.0] * 20)
    validate_qualification_policy(policy)
    validate_qualification_observation(observation)

    changed = copy.deepcopy(policy)
    changed["regression"]["relative_threshold_pct"] = 8.0
    with pytest.raises(SchemaError, match="policy_id does not match"):
        validate_qualification_policy(changed)

    changed_observation = copy.deepcopy(observation)
    changed_observation["measurement"]["samples_us"][0] = 101.0
    with pytest.raises(SchemaError, match="observation_id does not match"):
        validate_qualification_observation(changed_observation)


@pytest.mark.parametrize(
    ("candidate_samples", "expected"),
    [
        ([103.0] * 20, "pass"),
        ([110.0] * 20, "fail"),
        ([103.0] * 10 + [108.0] * 10, "inconclusive"),
    ],
)
def test_policy_produces_confidence_bounded_three_statistical_states(
    candidate_samples: list[float], expected: str
) -> None:
    policy = _policy()
    baseline = _observation(policy, role="baseline", samples=[100.0] * 20)
    candidate = _observation(policy, role="candidate", samples=candidate_samples)

    first = evaluate_qualification_observations(policy, baseline, candidate)
    second = evaluate_qualification_observations(policy, baseline, candidate)

    assert first == second
    assert first["verdict"] == expected
    assert first["statistics"]["acceptance_threshold_us"] == 5.0
    assert first["statistics"]["bootstrap_resamples"] == 1000
    assert first["verdict_id"] == qualification_verdict_sha256(first)
    validate_qualification_verdict(first)


def test_incomplete_noise_correctness_and_environment_have_explicit_outcomes() -> None:
    policy = _policy()
    baseline = _observation(policy, role="baseline", samples=[100.0] * 20)

    incomplete = _observation(policy, role="candidate", samples=[103.0] * 10)
    verdict = evaluate_qualification_observations(policy, baseline, incomplete)
    assert verdict["verdict"] == "inconclusive"
    assert verdict["reason_codes"] == ["candidate_measurements_incomplete"]
    assert verdict["statistics"]["bootstrap_resamples"] == 0

    bad_correctness = _observation(policy, role="candidate", samples=[103.0] * 20, correctness="fail")
    verdict = evaluate_qualification_observations(policy, baseline, bad_correctness)
    assert verdict["verdict"] == "fail"
    assert verdict["reason_codes"] == ["candidate_correctness_failed"]

    different_topology = _observation(
        policy,
        role="candidate",
        samples=[103.0] * 20,
        topology="a100-sxm-4gpu",
    )
    verdict = evaluate_qualification_observations(policy, baseline, different_topology)
    assert verdict["verdict"] == "incomparable"
    assert verdict["reason_codes"] == ["topology_class_mismatch"]


def test_wrong_policy_binding_and_roles_are_input_errors() -> None:
    policy = _policy()
    baseline = _observation(policy, role="baseline", samples=[100.0] * 20)
    candidate = _observation(policy, role="candidate", samples=[103.0] * 20)
    candidate["policy_id"] = "f" * 64
    candidate["observation_id"] = qualification_observation_sha256(candidate)
    with pytest.raises(SchemaError, match="not bound"):
        evaluate_qualification_observations(policy, baseline, candidate)

    wrong_role = _observation(policy, role="baseline", samples=[103.0] * 20)
    with pytest.raises(SchemaError, match="roles"):
        evaluate_qualification_observations(policy, baseline, wrong_role)
