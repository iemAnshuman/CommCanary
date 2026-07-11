from __future__ import annotations

import html
from typing import Any, Iterable, List, Mapping, Tuple

from ..artifacts.comparison import validate_comparison
from ..artifacts.io import SHAREABLE_HTML_POLICY, atomic_write_text
from ..artifacts.report import validate_report
from ..artifacts.wire import as_float


def write_report_html(path: str, report: Mapping[str, Any]) -> None:
    _write(path, render_report_html(report))


def write_compare_html(path: str, comparison: Mapping[str, Any]) -> None:
    _write(path, render_compare_html(comparison))


def render_report_html(report: Mapping[str, Any]) -> str:
    validate_report(report)
    metrics = report.get("metrics", {})
    workload = report.get("workload", {})
    backend = report.get("backend", {})
    samples = report.get("samples", [])
    canary_summary = report.get("canary_summary", {})
    fidelity = canary_summary.get("fidelity", {}) if isinstance(canary_summary, Mapping) else {}
    calibration = report.get("calibration", {})
    title = f"CommCanary Report - {workload.get('name', 'workload')}"
    cards = [
        ("Median", f"{as_float(metrics.get('median_us')):.1f} us"),
        ("P95", f"{as_float(metrics.get('p95_us')):.1f} us"),
        ("P99", f"{as_float(metrics.get('p99_us')):.1f} us"),
        ("Hidden", f"{as_float(metrics.get('communication_hidden_pct')):.1f}%"),
        ("Skew P95", f"{as_float(metrics.get('arrival_skew_p95_us')):.1f} us"),
        ("Events", str(int(as_float(metrics.get("count"))))),
    ]
    return _page(
        title,
        f"""
        <section class="hero">
          <div>
            <p class="eyebrow">CommCanary replay</p>
            <h1>{_esc(workload.get("name", "Unnamed workload"))}</h1>
            <p>{_esc(backend.get("label", "unknown backend"))} · {_esc(backend.get("mode", "unknown mode"))}</p>
          </div>
        </section>
        <section class="metric-grid">{"".join(_metric_card(label, value) for label, value in cards)}</section>
        <section class="panel">
          <h2>Exposed Latency Distribution</h2>
          {_histogram([as_float(sample.get("exposed_us")) for sample in samples])}
        </section>
        <section class="split">
          <div class="panel">
            <h2>By Phase</h2>
            {_breakdown_table(report.get("by_phase", []))}
          </div>
          <div class="panel">
            <h2>By Operation</h2>
            {_breakdown_table(report.get("by_op", []))}
          </div>
        </section>
        <section class="split">
          <div class="panel">
            <h2>Replay Settings</h2>
            {_kv_table(backend)}
          </div>
          <div class="panel">
            <h2>Canary Fidelity</h2>
            {_kv_table(fidelity) if isinstance(fidelity, Mapping) and fidelity else "<p>No fidelity metadata.</p>"}
          </div>
        </section>
        <section class="panel">
          <h2>Model Calibration</h2>
          {_kv_table(calibration) if isinstance(calibration, Mapping) and calibration else "<p>No observed latency signal was supplied.</p>"}
        </section>
        """,
    )


def render_compare_html(comparison: Mapping[str, Any]) -> str:
    validate_comparison(comparison)
    raw_verdict = str(comparison.get("verdict", "unknown"))
    verdict = raw_verdict if raw_verdict in {"pass", "warn", "fail"} else "warn"
    delta = comparison.get("delta", {})
    baseline = comparison.get("baseline", {}).get("metrics", {})
    candidate = comparison.get("candidate", {}).get("metrics", {})
    cards = [
        ("Verdict", verdict.upper()),
        ("Median Delta", _format_pct_delta(delta.get("median_pct"), delta.get("median_relative_status"))),
        ("P95 Delta", _format_pct_delta(delta.get("p95_pct"), delta.get("p95_relative_status"))),
        ("P99 Delta", _format_pct_delta(delta.get("p99_pct"), delta.get("p99_relative_status"))),
    ]
    rows = [
        ("Median", baseline.get("median_us"), candidate.get("median_us")),
        ("P95", baseline.get("p95_us"), candidate.get("p95_us")),
        ("P99", baseline.get("p99_us"), candidate.get("p99_us")),
        ("Hidden %", baseline.get("communication_hidden_pct"), candidate.get("communication_hidden_pct")),
        ("Arrival skew p95", baseline.get("arrival_skew_p95_us"), candidate.get("arrival_skew_p95_us")),
    ]
    reason_items = "".join(f"<li>{_esc(reason)}</li>" for reason in comparison.get("reasons", []))
    return _page(
        "CommCanary Compare",
        f"""
        <section class="hero {verdict}">
          <div>
            <p class="eyebrow">Baseline vs candidate</p>
            <h1>Comparison {_esc(verdict.upper())}</h1>
            <p>{_esc(comparison.get("created_at", ""))}</p>
          </div>
        </section>
        <section class="metric-grid">{"".join(_metric_card(label, value) for label, value in cards)}</section>
        <section class="split">
          <div class="panel">
            <h2>Metrics</h2>
            {_comparison_table(rows)}
          </div>
          <div class="panel">
            <h2>Reasons</h2>
            <ul class="reasons">{reason_items}</ul>
          </div>
        </section>
        <section class="split">
          <div class="panel">
            <h2>Phase Regressions</h2>
            {_regression_table(comparison.get("breakdown_delta", {}).get("by_phase", []))}
          </div>
          <div class="panel">
            <h2>Operation Regressions</h2>
            {_regression_table(comparison.get("breakdown_delta", {}).get("by_op", []))}
          </div>
        </section>
        """,
    )


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
  <title>{_esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172126;
      --muted: #5c6870;
      --line: #d9e0df;
      --paper: #fbfcfb;
      --panel: #ffffff;
      --teal: #167c80;
      --amber: #b56b15;
      --rose: #ad3856;
      --blue: #315f9f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
      letter-spacing: 0;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    .hero {{
      min-height: 210px;
      display: flex;
      align-items: end;
      padding: 28px;
      background: linear-gradient(135deg, #17343a, #1f675f 58%, #ac6e20);
      color: white;
      border-radius: 8px;
      margin-bottom: 18px;
    }}
    .hero.fail {{ background: linear-gradient(135deg, #3b2028, #9b344e 60%, #b56b15); }}
    .hero.warn {{ background: linear-gradient(135deg, #2f3327, #8a6817 62%, #167c80); }}
    .hero h1 {{ margin: 4px 0 6px; font-size: 3.8rem; line-height: 0.98; letter-spacing: 0; }}
    .hero p {{ margin: 0; color: rgba(255,255,255,0.82); }}
    .eyebrow {{ text-transform: uppercase; font-size: 0.78rem; font-weight: 700; letter-spacing: 0; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 28, 32, 0.04);
    }}
    .metric {{ padding: 14px; min-height: 88px; }}
    .metric .label {{ color: var(--muted); font-size: 0.82rem; }}
    .metric .value {{ font-size: 1.55rem; font-weight: 760; margin-top: 8px; overflow-wrap: anywhere; }}
    .panel {{ padding: 18px; margin-bottom: 16px; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 1.05rem; }}
    .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    .hist {{ width: 100%; height: 190px; display: block; }}
    .reasons {{ margin: 0; padding-left: 18px; }}
    .reasons li {{ margin: 0 0 8px; }}
    @media (max-width: 860px) {{
      main {{ padding: 14px; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .split {{ grid-template-columns: 1fr; }}
      .hero {{ min-height: 180px; }}
      .hero h1 {{ font-size: 2.35rem; }}
    }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>
"""


def _metric_card(label: str, value: str) -> str:
    return f'<div class="metric"><div class="label">{_esc(label)}</div><div class="value">{_esc(value)}</div></div>'


def _format_pct_delta(value: Any, status: Any = None) -> str:
    if value is None:
        if status == "new_nonzero_regression":
            return "new nonzero"
        return "undefined"
    return f"{as_float(value):+.1f}%"


def _histogram(values: Iterable[float], buckets: int = 28) -> str:
    data = [value for value in values if value >= 0.0]
    if not data:
        return (
            "<p>Samples unavailable; a distribution cannot be rendered. "
            "The reported count and quantiles remain available above.</p>"
        )
    minimum = min(data)
    maximum = max(data)
    span = max(0.001, maximum - minimum)
    counts = [0 for _ in range(buckets)]
    for value in data:
        index = min(buckets - 1, int((value - minimum) / span * buckets))
        counts[index] += 1
    max_count = max(counts) or 1
    width = 920
    height = 180
    gap = 3
    bar_width = (width - gap * (buckets - 1)) / buckets
    bars: List[str] = []
    for index, count in enumerate(counts):
        bar_height = count / max_count * (height - 32)
        x = index * (bar_width + gap)
        y = height - bar_height - 22
        color = "#167c80" if index < buckets * 0.75 else "#b56b15"
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="2" fill="{color}"/>'
        )
    return (
        f'<svg class="hist" viewBox="0 0 {width} {height}" role="img" aria-label="Latency histogram">'
        f'<line x1="0" y1="{height - 22}" x2="{width}" y2="{height - 22}" stroke="#d9e0df"/>'
        + "".join(bars)
        + f'<text x="0" y="{height - 4}" fill="#5c6870" font-size="13">{minimum:.1f} us</text>'
        + f'<text x="{width}" y="{height - 4}" text-anchor="end" fill="#5c6870" font-size="13">{maximum:.1f} us</text>'
        + "</svg>"
    )


def _breakdown_table(rows: Iterable[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{_esc(row.get('name', 'unknown'))}</td>"
            f"<td>{int(as_float(row.get('count')))}</td>"
            f"<td>{as_float(row.get('median_us')):.1f}</td>"
            f"<td>{as_float(row.get('p99_us')):.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Name</th><th>Count</th><th>Median us</th><th>P99 us</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _regression_table(rows: Iterable[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{_esc(row.get('name', 'unknown'))}</td>"
            f"<td>{_esc(_format_pct_delta(row.get('median_pct'), row.get('median_relative_status')))}</td>"
            f"<td>{_esc(_format_pct_delta(row.get('p95_pct'), row.get('p95_relative_status')))}</td>"
            f"<td>{_esc(_format_pct_delta(row.get('p99_pct'), row.get('p99_relative_status')))}</td>"
            "</tr>"
        )
    if not body:
        return "<p>No breakdown data.</p>"
    return (
        "<table><thead><tr><th>Name</th><th>Median Δ</th><th>P95 Δ</th><th>P99 Δ</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _comparison_table(rows: Iterable[Tuple[Any, Any, Any]]) -> str:
    body = []
    for label, baseline, candidate in rows:
        body.append(
            f"<tr><td>{_esc(label)}</td><td>{as_float(baseline):.2f}</td><td>{as_float(candidate):.2f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _kv_table(mapping: Mapping[str, Any]) -> str:
    body = []
    for key, value in mapping.items():
        body.append(f"<tr><td>{_esc(key)}</td><td>{_esc(value)}</td></tr>")
    return "<table><tbody>" + "".join(body) + "</tbody></table>"


def _write(path: str, content: str) -> None:
    atomic_write_text(path, content, policy=SHAREABLE_HTML_POLICY)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)
