"""Compatibility facade for the historical :mod:`commcanary.html_report` path.

New code may import the same presentation API from :mod:`commcanary.reporting`.
"""

from .reporting import render_compare_html, render_report_html, write_compare_html, write_report_html

__all__ = [
    "render_compare_html",
    "render_report_html",
    "write_compare_html",
    "write_report_html",
]
