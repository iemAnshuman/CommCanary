from __future__ import annotations

from commcanary.cli import main as cli_main
from commcanary.schema import load_json, write_json
from tests.test_qualification_decision import _observation, _policy


def test_policy_verification_and_four_state_evaluation_cli(tmp_path, capsys) -> None:
    policy = _policy()
    baseline = _observation(policy, role="baseline", samples=[100.0] * 20)
    passing = _observation(policy, role="candidate", samples=[103.0] * 20)
    failing = _observation(policy, role="candidate", samples=[110.0] * 20)
    policy_path = tmp_path / "policy.json"
    baseline_path = tmp_path / "baseline.json"
    passing_path = tmp_path / "passing.json"
    failing_path = tmp_path / "failing.json"
    verdict_path = tmp_path / "verdict.json"
    for path, document in (
        (policy_path, policy),
        (baseline_path, baseline),
        (passing_path, passing),
        (failing_path, failing),
    ):
        write_json(str(path), document)

    assert cli_main(["verify-policy", str(policy_path)]) == 0
    assert (
        cli_main(
            [
                "evaluate-qualification",
                str(policy_path),
                str(baseline_path),
                str(passing_path),
                "-o",
                str(verdict_path),
            ]
        )
        == 0
    )
    assert load_json(str(verdict_path))["verdict"] == "pass"
    verdict_path.unlink()
    assert (
        cli_main(
            [
                "evaluate-qualification",
                str(policy_path),
                str(baseline_path),
                str(failing_path),
                "-o",
                str(verdict_path),
            ]
        )
        == 1
    )
    assert load_json(str(verdict_path))["verdict"] == "fail"
    rendered = capsys.readouterr().out
    assert "VERDICT: PASS" in rendered
    assert "VERDICT: FAIL" in rendered
