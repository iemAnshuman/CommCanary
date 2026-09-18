#!/usr/bin/env python3
"""Generate deterministic TikZ vector figures from verified paper metrics."""

from __future__ import annotations

import json
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = SOURCE_ROOT / "evidence" / "verified_metrics.json"
FIGURE_DIR = SOURCE_ROOT / "figures"


def _tex_label(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _write(name: str, content: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def _publication_path() -> str:
    return r"""
\begin{tikzpicture}[
  x=1cm,y=1cm,>=latex,
  box/.style={draw=paperblue,rounded corners=2pt,minimum width=2.05cm,
    minimum height=1.05cm,align=center,fill=paperblue!3,font=\sffamily\scriptsize},
  label/.style={font=\sffamily\tiny,text=papergray,align=center}
]
\node[box] (a) at (0,0) {\textbf{Frozen campaign}\\manifest and policy};
\node[box] (b) at (2.35,0) {\textbf{Immutable attempts}\\terminal records};
\node[box] (c) at (4.70,0) {\textbf{Explicit selection}\\IDs and hashes};
\node[box] (d) at (7.05,0) {\textbf{Completeness verdict}\\expected equals selected};
\node[box] (e) at (9.40,0) {\textbf{Hash-bound archive}\\raw evidence};
\node[box] (f) at (11.75,0) {\textbf{Publication}\\tables and figures};
\foreach \x/\y in {a/b,b/c,c/d,d/e,e/f} {\draw[->,paperblue] (\x) -- (\y);}
\node[label,text=paperorange] at (5.875,-0.95)
  {Any missing identity, hash mismatch, incomplete selection, or absent archive stops the claim path.};
\node[font=\sffamily\small\bfseries,text=paperblue] at (5.875,-1.55)
  {Fail closed: publication requires every boundary to pass.};
\end{tikzpicture}
"""


def _bars(metrics: dict[str, object], key: str, maximum: float, axis_label: str) -> str:
    rows = metrics["trusted_comparison"]["representations"]
    assert isinstance(rows, list)
    colors = ["paperblue!75", "paperblue", "paperorange", "papergreen"]
    x_scale = 0.075 if maximum > 1.0 else 7.5
    ticks = [float(value) for value in range(0, 101, 20)]
    if maximum <= 1.0:
        ticks = [value / 5.0 for value in range(0, 6)]
    lines = [
        rf"\begin{{tikzpicture}}[x={x_scale:.3f}cm,y=0.72cm,font=\sffamily\scriptsize]",
        rf"\draw[->,papergray] (0,-0.65) -- ({maximum + 5:.1f},-0.65);",
    ]
    if maximum <= 1.0:
        lines[1] = rf"\draw[->,papergray] (0,-0.65) -- ({maximum + 0.05:.2f},-0.65);"
    for tick in ticks:
        tick_label = f"{tick:.1f}" if maximum <= 1.0 else f"{int(tick)}"
        lines.append(
            rf"\draw[papergray!30] ({tick},-0.55) -- ({tick},3.55) "
            rf"node[below=2pt,text=papergray] {{{tick_label}}};"
        )
    for index, (row, color) in enumerate(zip(rows, colors)):
        assert isinstance(row, dict)
        y = 3 - index
        value = float(row[key])
        label = _tex_label(str(row["id"]))
        suffix = f"{int(row['agreement_count'])}/28" if key == "agreement_pct" else f"{value:.3f}"
        label_coordinate = -2.0 if maximum > 1.0 else -0.02
        lines.append(
            rf"\node[anchor=east,text=paperblue] at "
            rf"({label_coordinate:.3f},{y}) {{{label}}};"
        )
        lines.append(rf"\fill[{color}] (0,{y - 0.22:.2f}) rectangle ({value:.6f},{y + 0.22:.2f});")
        label_offset = 1.2 if maximum > 1.0 else 0.02
        lines.append(
            rf"\node[anchor=west,text=paperblue] at "
            rf"({value + label_offset:.6f},{y}) {{{suffix}}};"
        )
    lines.append(rf"\node[text=paperblue] at ({maximum / 2:.3f},-1.35) {{{_tex_label(axis_label)}}};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def _inconclusive_gate(metrics: dict[str, object]) -> str:
    gate = metrics["exact_work_gate"]
    assert isinstance(gate, dict)
    intervals = gate["representative_normalized_intervals"]
    iqr = gate["tree_ll_relative_iqr_pct"]
    assert isinstance(intervals, list) and isinstance(iqr, dict)
    lines = [
        r"\begin{tikzpicture}[x=0.64cm,y=0.67cm,font=\sffamily\tiny]",
        r"\begin{scope}[xshift=0cm]",
        r"\fill[paperblue!7] (-1,-0.55) rectangle (1,5.55);",
        r"\draw[dashed,papergray] (-1,-0.55) -- (-1,5.55);",
        r"\draw[dashed,papergray] (1,-0.55) -- (1,5.55);",
        r"\draw[papergray] (0,-0.55) -- (0,5.55);",
        r"\node[text=papergray] at (-2.5,6.0) {second faster boundary};",
        r"\node[text=papergray] at (2.5,6.0) {first faster boundary};",
    ]
    for index, item in enumerate(intervals):
        assert isinstance(item, dict)
        y = 5 - index
        low = float(item["low"])
        high = float(item["high"])
        point = float(item["point"])
        label = _tex_label(str(item["label"]))
        lines.extend(
            [
                rf"\node[anchor=east,text=paperblue] at (-5.1,{y}) {{{label}}};",
                rf"\draw[paperblue,thick] ({low:.6f},{y}) -- ({high:.6f},{y});",
                rf"\draw[paperblue] ({low:.6f},{y - 0.12:.2f}) -- ({low:.6f},{y + 0.12:.2f});",
                rf"\draw[paperblue] ({high:.6f},{y - 0.12:.2f}) -- ({high:.6f},{y + 0.12:.2f});",
                rf"\fill[paperblue] ({point:.6f},{y}) circle (2pt);",
            ]
        )
    for tick in range(-4, 7, 2):
        lines.append(
            rf"\draw[papergray!35] ({tick},-0.45) -- ({tick},5.45);"
            rf"\node[below,text=papergray] at ({tick},-0.55) {{{tick}}};"
        )
    lines.extend(
        [
            r"\node[text=paperblue] at (0,-1.25) {Exact-work difference / pair tie threshold};",
            r"\end{scope}",
            r"\begin{scope}[xshift=9.6cm,x=0.09cm]",
            r"\draw[->,papergray] (0,-0.55) -- (64,-0.55);",
            r"\draw[dashed,paperorange] (20,-0.55) -- (20,3.45);",
            r"\node[text=paperorange,anchor=west] at (21,3.65) {frozen 20\% limit};",
        ]
    )
    names = [("source", 2), ("exact work", 1), ("stratified", 0)]
    colors = ["paperblue!75", "papergreen", "paperorange"]
    for (name, y), color in zip(names, colors):
        key = name.replace(" ", "_")
        value = float(iqr[key])
        lines.extend(
            [
                rf"\node[anchor=east,text=paperblue] at (-3,{y}) {{{name}}};",
                rf"\fill[{color}] (0,{y - 0.22:.2f}) rectangle ({value:.6f},{y + 0.22:.2f});",
                rf"\node[anchor=west,text=paperblue] at ({value + 1.2:.6f},{y}) {{{value:.1f}\%}};",
            ]
        )
    for tick in range(0, 61, 10):
        lines.append(
            rf"\draw[papergray!25] ({tick},-0.45) -- ({tick},3.3);"
            rf"\node[below,text=papergray] at ({tick},-0.55) {{{tick}}};"
        )
    lines.extend(
        [
            r"\node[text=paperblue] at (30,-1.25) {Relative IQR for tree LL (\%)};",
            r"\end{scope}",
            r"\end{tikzpicture}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    _write("figure1-publication-path.tex", _publication_path())
    _write(
        "figure2-agreement.tex",
        _bars(metrics, "agreement_pct", 100.0, "Pairwise agreement with W-full (%)"),
    )
    _write(
        "figure3-kendall.tex",
        _bars(metrics, "kendall_tau_b", 1.0, "Kendall tau-b against W-full"),
    )
    _write("figure4-inconclusive.tex", _inconclusive_gate(metrics))
    print(f"generated 4 figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
