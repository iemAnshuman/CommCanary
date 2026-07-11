from __future__ import annotations

import copy
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from commcanary.html_report import render_compare_html, render_report_html

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


class StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.starts: Counter[str] = Counter()
        self.ends: Counter[str] = Counter()
        self.meta: List[Dict[str, str]] = []
        self.text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.starts[tag] += 1
        if tag == "meta":
            self.meta.append({key: value or "" for key, value in attrs})

    def handle_endtag(self, tag: str) -> None:
        self.ends[tag] += 1

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _fixture(name: str) -> dict:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _parse(document: str) -> StructureParser:
    parser = StructureParser()
    parser.feed(document)
    parser.close()
    return parser


def test_report_html_is_structural_self_contained_and_does_not_invent_samples() -> None:
    report = _fixture("report.valid.json")
    report.pop("samples", None)
    report["workload"] = {"name": '<script>alert("trace")</script>'}

    parser = _parse(render_report_html(report))
    combined_text = " ".join(parser.text)

    assert parser.starts["html"] == parser.ends["html"] == 1
    assert parser.starts["main"] == parser.ends["main"] == 1
    assert parser.starts["script"] == 0
    assert '<script>alert("trace")</script>' in combined_text
    assert "Samples unavailable" in combined_text
    assert "reported count and quantiles" in combined_text
    csp = [meta for meta in parser.meta if meta.get("http-equiv") == "Content-Security-Policy"]
    assert len(csp) == 1
    assert "default-src 'none'" in csp[0]["content"]


def test_comparison_html_escapes_untrusted_reasons_and_has_expected_structure() -> None:
    comparison = copy.deepcopy(_fixture("comparison.valid.json"))
    comparison["reasons"] = ["<img src=x onerror=alert(1)>"]

    parser = _parse(render_compare_html(comparison))
    combined_text = " ".join(parser.text)

    assert parser.starts["img"] == 0
    assert "<img src=x onerror=alert(1)>" in combined_text
    assert parser.starts["table"] >= 1
    assert parser.starts["section"] >= 2
