"""HTML presentation for decision-fidelity reports."""

from __future__ import annotations

import html
from typing import Any, List, Mapping

from ..artifacts.io import SHAREABLE_HTML_POLICY, atomic_write_text


def write_fidelity_html(path: str, report: Mapping[str, Any]) -> None:
    atomic_write_text(path, render_fidelity_html(report), policy=SHAREABLE_HTML_POLICY)


def render_fidelity_html(report: Mapping[str, Any]) -> str:
    proxies = report["proxies"]
    by_agreement = sorted(
        proxies,
        key=lambda row: (
            -float(row["ranking_quality"]["pairwise_decision_agreement"]),
            str(row["name"]),
        ),
    )
    decision_rows: List[str] = []
    for row in proxies:
        interval = row["regret_at_1_confidence_interval"]
        disaster = row["disaster_detection"]
        shortlists = {int(item["k"]): item for item in row["regret_at_k"]}
        decision_rows.append(
            "<tr>"
            f"<td>{int(row['regret_rank'])}</td>"
            f"<td>{_esc(row['name'])}</td>"
            f"<td><code>{_esc(row['recommended_configuration'])}</code></td>"
            f'<td class="number headline">{float(row["regret_at_1_pct"]):.2f}%</td>'
            f'<td class="number">{float(interval["lower_pct"]):.2f}% to {float(interval["upper_pct"]):.2f}%</td>'
            f'<td class="number">{float(shortlists[2]["regret_pct"]):.2f}%</td>'
            f'<td class="number">{float(shortlists[3]["regret_pct"]):.2f}%</td>'
            f"<td>{'YES' if disaster['in_bottom_quartile'] else 'NO'}</td>"
            "</tr>"
        )
    agreement_rows: List[str] = []
    for index, row in enumerate(by_agreement, start=1):
        quality = row["ranking_quality"]
        tau = quality["kendall_tau_b"]
        rendered_tau = "undefined" if tau is None else f"{float(tau):.3f}"
        agreement_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{_esc(row['name'])}</td>"
            f'<td class="number">{float(quality["pairwise_decision_agreement_pct"]):.1f}%</td>'
            f'<td class="number">{rendered_tau}</td>'
            f'<td class="number">{float(row["regret_at_1_pct"]):.2f}%</td>'
            "</tr>"
        )
    divergence = bool(report["rankings_diverge"])
    divergence_title = "Decision and ranking metrics disagree" if divergence else "Decision and ranking metrics agree"
    divergence_copy = (
        "The proxy order changes when scored by regret instead of pairwise agreement. "
        "Regret measures the cost of the configuration the proxy recommends."
        if divergence
        else "The proxy order is the same under regret and pairwise agreement for these inputs."
    )
    reference = report["reference"]
    optimum = reference["true_optimum"]
    worst = reference["true_worst"]
    bootstrap = report["method"]["bootstrap"]
    tie = report["method"]["tie_tolerance"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
  <title>CommCanary fidelity report</title>
  <style>
    :root {{ color-scheme: light; --ink: #172126; --muted: #5c6870; --line: #d9e0df;
      --paper: #f6f8f7; --panel: #fff; --teal: #167c80; --amber: #b56b15; --rose: #ad3856; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: var(--paper); color: var(--ink); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    .hero {{ padding: 30px; color: white; border-radius: 8px; margin-bottom: 18px;
      background: linear-gradient(135deg, #17343a, #1f675f 58%, #ac6e20); }}
    .hero.diverges {{ background: linear-gradient(135deg, #39232b, #9b344e 58%, #b56b15); }}
    .hero h1 {{ margin: 4px 0 10px; font-size: 2.8rem; line-height: 1; }}
    .hero p {{ max-width: 780px; margin: 0; color: rgba(255,255,255,.86); }}
    .eyebrow {{ text-transform: uppercase; font-size: .78rem; font-weight: 750; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
      padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(15,28,32,.04); overflow-x: auto; }}
    .panel h2 {{ margin: 0 0 6px; font-size: 1.2rem; }}
    .panel > p {{ color: var(--muted); margin: 0 0 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .91rem; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    .number {{ font-variant-numeric: tabular-nums; }}
    .headline {{ color: var(--rose); font-weight: 800; }}
    code {{ overflow-wrap: anywhere; white-space: normal; }}
    .facts {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }}
    .fact {{ border-left: 3px solid var(--teal); padding-left: 12px; }}
    .fact span {{ display: block; color: var(--muted); font-size: .82rem; }}
    @media (max-width: 760px) {{ main {{ padding: 14px; }} .hero h1 {{ font-size: 2rem; }}
      .facts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <section class="hero{" diverges" if divergence else ""}">
      <div class="eyebrow">Decision fidelity</div>
      <h1>{_esc(divergence_title)}</h1>
      <p>{_esc(divergence_copy)}</p>
    </section>
    <section class="panel">
      <h2>Reference</h2>
      <div class="facts">
        <div class="fact"><span>Measurement set</span>{_esc(reference["name"])} ({_esc(reference["metric"])})</div>
        <div class="fact"><span>True optimum</span><code>{_esc(optimum["configuration"])}</code> at {float(optimum["value"]):.6g}</div>
        <div class="fact"><span>Worst configuration</span><code>{_esc(worst["configuration"])}</code> at {float(worst["value"]):.6g}</div>
        <div class="fact"><span>Configurations</span>{int(reference["configuration_count"])}</div>
      </div>
    </section>
    <section class="panel">
      <h2>Decision metrics: ranked by regret at 1</h2>
      <p>Lower regret is better. Zero means the proxy selected the reference optimum.</p>
      <table>
        <thead><tr><th>Rank</th><th>Proxy</th><th>Recommended configuration</th><th>Regret @ 1</th><th>Bootstrap interval</th><th>Regret @ 2</th><th>Regret @ 3</th><th>Disaster detected</th></tr></thead>
        <tbody>{"".join(decision_rows)}</tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Ranking-quality metrics: ranked by pairwise agreement</h2>
      <p>Pairwise agreement and Kendall tau-b describe ranking quality. They do not measure the cost of the proxy's recommendation.</p>
      <table>
        <thead><tr><th>Rank</th><th>Proxy</th><th>Pairwise agreement</th><th>Kendall tau-b</th><th>Regret @ 1</th></tr></thead>
        <tbody>{"".join(agreement_rows)}</tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Method</h2>
      <p>Point estimates are medians. Pairwise ties use each pair's maximum configuration IQR with an absolute floor of {float(tie["absolute_floor"]):.6g}. Regret intervals use {int(bootstrap["resamples"])} seeded percentile-bootstrap resamples at {float(bootstrap["confidence"]) * 100.0:.1f}% confidence.</p>
    </section>
  </main>
</body>
</html>
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


__all__ = ["render_fidelity_html", "write_fidelity_html"]
