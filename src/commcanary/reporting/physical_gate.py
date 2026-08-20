"""Deterministic CI presentations for physical gate results."""

from __future__ import annotations

import datetime
import html
import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Mapping, cast


def render_physical_gate_html(result: Mapping[str, Any]) -> str:
    rows = []
    for metric in result["metric_results"]:
        regression = metric["regression_pct"]
        rendered_regression = "undefined (zero baseline)" if regression is None else f"{float(regression):.3f}%"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(metric['name']))}</td>"
            f"<td>{float(metric['baseline_median']):.6g}</td>"
            f"<td>{float(metric['candidate_median']):.6g}</td>"
            f"<td>{html.escape(rendered_regression)}</td>"
            f"<td>{float(metric['regression_threshold_pct']):.6g}%</td>"
            f"<td>{html.escape(str(metric['status']).upper())}</td>"
            "</tr>"
        )
    issues = "".join(f"<li>{html.escape(str(issue))}</li>" for issue in result["issues"])
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CommCanary physical gate</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccd1d1; padding: .5rem; text-align: left; }}
    th {{ background: #f4f6f7; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>Physical gate: {outcome}</h1>
  <p>Bundle <code>{bundle_id}</code></p>
  <p>Runner <code>{runner}</code></p>
  <p>Executable <code>{executable}</code></p>
  <p>Baseline <code>{baseline_subject}</code></p>
  <p>Candidate <code>{candidate_subject}</code></p>
  <p>Baseline evidence <code>{baseline_evidence}</code></p>
  <p>Candidate evidence <code>{candidate_evidence}</code></p>
  <p>Baseline environment <code>{baseline_environment}</code></p>
  <p>Candidate environment <code>{candidate_environment}</code></p>
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Baseline median</th><th>Candidate median</th><th>Regression</th><th>Limit</th><th>Status</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Issues</h2>
  <ul>{issues}</ul>
</body>
</html>
""".format(
        outcome=html.escape(str(result["outcome"]).upper()),
        bundle_id=html.escape(str(result["bundle_id"])),
        runner=html.escape(str(result["runner_oci_digest"])),
        executable=html.escape(str(result["executable_sha256"])),
        baseline_subject=html.escape(str(result["baseline_subject_sha256"])),
        candidate_subject=html.escape(str(result["candidate_subject_sha256"])),
        baseline_evidence=html.escape(str(result["baseline_evidence_sha256"])),
        candidate_evidence=html.escape(str(result["candidate_evidence_sha256"])),
        baseline_environment=html.escape(str(result["baseline_environment_sha256"])),
        candidate_environment=html.escape(str(result["candidate_environment_sha256"])),
        rows="".join(rows),
        issues=issues or "<li>None</li>",
    )


def physical_gate_junit_bytes(result: Mapping[str, Any]) -> bytes:
    suite = ET.Element(
        "testsuite",
        {
            "name": "commcanary.physical_gate",
            "tests": str(len(result["metric_results"])),
            "failures": str(sum(row["status"] == "fail" for row in result["metric_results"])),
            "errors": "0",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    for row in result["metric_results"]:
        case = ET.SubElement(suite, "testcase", {"classname": "commcanary.physical_gate", "name": str(row["name"])})
        if result["outcome"] == "incomparable":
            skipped = ET.SubElement(case, "skipped", {"message": "physical gate evidence is incomparable"})
            skipped.text = "; ".join(str(issue) for issue in result["issues"])
        elif row["status"] == "fail":
            failure = ET.SubElement(case, "failure", {"message": "mandatory performance regression"})
            failure.text = json.dumps(row, sort_keys=True, allow_nan=False)
        elif row["status"] == "warn":
            output = ET.SubElement(case, "system-out")
            output.text = json.dumps(row, sort_keys=True, allow_nan=False)
    ET.indent(suite, space="  ")
    return cast(bytes, ET.tostring(suite, encoding="utf-8", xml_declaration=True)) + b"\n"


def physical_gate_sarif(result: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for metric in result["metric_results"]:
        if metric["status"] == "pass":
            continue
        regression = metric["regression_pct"]
        rendered_regression = "undefined (zero baseline)" if regression is None else f"{float(regression):.3f}%"
        rows.append(
            {
                "ruleId": f"commcanary.physical_gate.{metric['name']}",
                "level": "error" if metric["status"] == "fail" else "warning",
                "message": {
                    "text": (
                        f"{metric['name']} regressed {rendered_regression} "
                        f"beyond its {metric['regression_threshold_pct']}% policy threshold"
                    )
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": f"bundle:{result['bundle_id']}",
                            }
                        }
                    }
                ],
                "properties": {
                    "baselineMedian": metric["baseline_median"],
                    "candidateMedian": metric["candidate_median"],
                    "regressionPct": metric["regression_pct"],
                    "bundleId": result["bundle_id"],
                    "runnerOciDigest": result["runner_oci_digest"],
                    "executableSha256": result["executable_sha256"],
                    "baselineSubjectSha256": result["baseline_subject_sha256"],
                    "candidateSubjectSha256": result["candidate_subject_sha256"],
                    "baselineEvidenceSha256": result["baseline_evidence_sha256"],
                    "candidateEvidenceSha256": result["candidate_evidence_sha256"],
                },
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {"name": "CommCanary", "informationUri": "https://github.com/iemAnshuman/commcanary"}
                },
                "results": rows,
            }
        ],
    }


__all__ = ["physical_gate_junit_bytes", "physical_gate_sarif", "render_physical_gate_html"]
