#!/usr/bin/env python3
"""Generate deterministic monochrome TikZ figures from committed paper evidence.

Figures use the document's own serif font and black and gray only, so they
print and photocopy like the text around them. Each figure is drawn at its
final width; the manuscript includes them without rescaling, which keeps
figure text the same size in every figure.
"""

from __future__ import annotations

import json
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = SOURCE_ROOT / "evidence" / "verified_metrics.json"
UNCERTAINTY_PATH = SOURCE_ROOT / "evidence" / "trusted_join_uncertainty.json"
FIGURE_DIR = SOURCE_ROOT / "figures"
PROXY_ORDER = ("W-micro", "W-canary", "W-canary-overlap", "W-shared-overlap")


def _tex(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _write(name: str, content: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def _publication_path() -> str:
    stages = (
        ("Frozen campaign", "manifest, policy"),
        ("Immutable attempts", "terminal records"),
        ("Explicit selection", "IDs and hashes"),
        ("Completeness", "expected = selected"),
        ("Bound archive", "raw evidence"),
        ("Publication", "tables, figures"),
    )
    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,>=latex,font=\footnotesize,",
        r"  stage/.style={draw,minimum width=2.5cm,minimum height=0.95cm,align=center,inner sep=3pt}]",
    ]
    for index, (title, detail) in enumerate(stages):
        lines.append(rf"\node[stage] (s{index}) at ({index * 2.78:.2f},0) {{{title}\\{{\scriptsize {detail}}}}};")
    for index in range(len(stages) - 1):
        lines.append(rf"\draw[->] (s{index}) -- (s{index + 1});")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _agreement_intervals(uncertainty: dict[str, object]) -> str:
    """Agreeing pairs out of 28, point and 95% bootstrap interval, per proxy."""

    everything = uncertainty["all_configurations"]
    assert isinstance(everything, dict)
    agreement = everything["agreement"]
    assert isinstance(agreement, dict)
    scale = 0.155  # cm per pair; 28 pairs span 4.34 cm, leaving room for labels
    lines = [rf"\begin{{tikzpicture}}[x={scale}cm,y=0.55cm,font=\footnotesize]"]
    top = len(PROXY_ORDER) - 0.4
    for tick in range(0, 29, 4):
        lines.append(rf"\draw[gray!40] ({tick},-0.6) -- ({tick},{top:.1f});")
        lines.append(rf"\node[below] at ({tick},-0.6) {{{tick}}};")
    lines.append(r"\draw (0,-0.6) -- (28,-0.6);")
    for index, name in enumerate(PROXY_ORDER):
        row = agreement[name]
        assert isinstance(row, dict)
        low, high = (float(value) for value in row["interval_95"])
        point = float(row["point"])
        y = len(PROXY_ORDER) - 1 - index
        lines.extend(
            [
                rf"\node[anchor=east] at (-0.8,{y}) {{\texttt{{{_tex(name)}}}}};",
                rf"\draw[line width=0.8pt] ({low},{y}) -- ({high},{y});",
                rf"\draw[line width=0.8pt] ({low},{y - 0.16:.2f}) -- ({low},{y + 0.16:.2f});",
                rf"\draw[line width=0.8pt] ({high},{y - 0.16:.2f}) -- ({high},{y + 0.16:.2f});",
                rf"\fill ({point},{y}) circle (1.8pt);",
            ]
        )
    lines.append(r"\node at (14,-1.75) {Pairs agreeing with \texttt{W-full} (of 28)};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _inconclusive_gate(metrics: dict[str, object]) -> str:
    gate = metrics["exact_work_gate"]
    assert isinstance(gate, dict)
    intervals = gate["representative_normalized_intervals"]
    iqr = gate["tree_ll_relative_iqr_pct"]
    assert isinstance(intervals, list) and isinstance(iqr, dict)
    rows = len(intervals)
    top = rows - 0.45
    lines = [
        r"\begin{tikzpicture}[font=\footnotesize]",
        r"\begin{scope}[x=0.42cm,y=0.5cm]",
        rf"\fill[gray!15] (-1,-0.55) rectangle (1,{top:.2f});",
        rf"\draw[dashed] (-1,-0.55) -- (-1,{top:.2f});",
        rf"\draw[dashed] (1,-0.55) -- (1,{top:.2f});",
    ]
    for tick in range(-6, 8, 2):
        lines.append(rf"\node[below] at ({tick},-0.55) {{{tick}}};")
        lines.append(rf"\draw ({tick},-0.55) -- ({tick},-0.43);")
    lines.append(r"\draw (-6,-0.55) -- (7.4,-0.55);")
    for index, item in enumerate(intervals):
        assert isinstance(item, dict)
        y = rows - 1 - index
        low, high, point = float(item["low"]), float(item["high"]), float(item["point"])
        lines.extend(
            [
                rf"\node[anchor=east] at (-6.4,{y}) {{{_tex(str(item['label']))}}};",
                rf"\draw[line width=0.8pt] ({low:.4f},{y}) -- ({high:.4f},{y});",
                rf"\draw[line width=0.8pt] ({low:.4f},{y - 0.17:.2f}) -- ({low:.4f},{y + 0.17:.2f});",
                rf"\draw[line width=0.8pt] ({high:.4f},{y - 0.17:.2f}) -- ({high:.4f},{y + 0.17:.2f});",
                rf"\fill ({point:.4f},{y}) circle (1.8pt);",
            ]
        )
    lines.extend(
        [
            r"\node at (0.7,-2.1) {(a) Pair difference / tie threshold};",
            r"\end{scope}",
            r"\begin{scope}[xshift=5.7cm,x=0.052cm,y=0.5cm]",
            r"\draw (0,-0.55) -- (62,-0.55);",
        ]
    )
    for tick in range(0, 61, 20):
        lines.append(rf"\node[below] at ({tick},-0.55) {{{tick}}};")
        lines.append(rf"\draw ({tick},-0.55) -- ({tick},-0.43);")
    lines.append(rf"\draw[dashed] (20,-0.55) -- (20,{top:.2f});")
    lines.append(rf"\node[anchor=south] at (20,{top:.2f}) {{20\% limit}};")
    names = (("source", "source"), ("exact work", "exact_work"), ("stratified", "stratified"))
    for index, (label, key) in enumerate(names):
        y = rows - 2 - index * 1.5
        value = float(iqr[key])
        lines.extend(
            [
                rf"\node[anchor=east] at (-2,{y}) {{{label}}};",
                rf"\fill[gray!60] (0,{y - 0.3:.2f}) rectangle ({value:.4f},{y + 0.3:.2f});",
                rf"\node[anchor=west] at ({value + 1:.4f},{y}) {{{value:.1f}\%}};",
            ]
        )
    lines.extend(
        [
            r"\node at (30,-2.1) {(b) Relative IQR of \texttt{tree-ll} (\%)};",
            r"\end{scope}",
            r"\end{tikzpicture}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    uncertainty = json.loads(UNCERTAINTY_PATH.read_text(encoding="utf-8"))
    _write("figure1-publication-path.tex", _publication_path())
    _write("figure2-agreement.tex", _agreement_intervals(uncertainty))
    _write("figure3-inconclusive.tex", _inconclusive_gate(metrics))
    for stale in ("figure3-kendall.tex", "figure4-inconclusive.tex"):
        (FIGURE_DIR / stale).unlink(missing_ok=True)
    print(f"generated 3 figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
