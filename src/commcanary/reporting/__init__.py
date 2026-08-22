"""Presentation adapters for validated CommCanary artifacts."""

from .fidelity import render_fidelity_html, write_fidelity_html
from .html import render_compare_html, render_report_html, write_compare_html, write_report_html
from .physical_gate import physical_gate_junit_bytes, physical_gate_sarif, render_physical_gate_html

__all__ = [
    "physical_gate_junit_bytes",
    "physical_gate_sarif",
    "render_compare_html",
    "render_fidelity_html",
    "render_physical_gate_html",
    "render_report_html",
    "write_compare_html",
    "write_fidelity_html",
    "write_report_html",
]
