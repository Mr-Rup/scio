"""Renderers for Rich terminal tables, Markdown, LaTeX, and interactive notebook HTML cards."""

from __future__ import annotations

import html
from typing import Any


def format_html_card(
    title: str,
    subtitle: str,
    metrics: list[tuple[str, str]],
    badge: tuple[str, str] | None = None,
    details: list[tuple[str, str]] | None = None,
    warnings: list[str] | None = None,
) -> str:
    """Generate a self-contained, stylish HTML card for Jupyter notebook display (`_repr_html_`).

    Parameters
    ----------
    title : str
        Primary card title (e.g. test name or dataset profile).
    subtitle : str
        Secondary subtitle (e.g. "target ~ group").
    metrics : list[tuple[str, str]]
        High-level key-value metric pairs (e.g. [("Statistic", "t = 3.48"), ("p-value", "p = 0.001")]).
    badge : tuple[str, str] | None
        Optional (badge_text, badge_color_hex) (e.g. ("Significant", "#10b981")).
    details : list[tuple[str, str]] | None
        Extended key-value rows.
    warnings : list[str] | None
        List of warning/audit alert strings.
    """
    badge_html = ""
    if badge:
        text, color = badge
        badge_html = f"""<span style="
            background: {color}22;
            color: {color};
            border: 1px solid {color}55;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        ">{html.escape(text)}</span>"""

    metrics_html = "".join(
        f"""
        <div style="flex: 1; min-width: 120px; background: rgba(125, 125, 125, 0.06); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(125, 125, 125, 0.12);">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.7; margin-bottom: 2px;">{html.escape(label)}</div>
            <div style="font-size: 15px; font-weight: 600; font-family: monospace;">{html.escape(val)}</div>
        </div>
        """
        for label, val in metrics
    )

    details_html = ""
    if details:
        rows = "".join(
            f"""
            <tr style="border-bottom: 1px solid rgba(125, 125, 125, 0.1);">
                <td style="padding: 6px 8px; font-weight: 500; opacity: 0.8; width: 40%;">{html.escape(k)}</td>
                <td style="padding: 6px 8px; font-family: monospace;">{html.escape(v)}</td>
            </tr>
            """
            for k, v in details
        )
        details_html = f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 12px;">
            <tbody>{rows}</tbody>
        </table>
        """

    warnings_html = ""
    if warnings:
        warn_items = "".join(
            f"""<li style="margin-bottom: 3px;">{html.escape(w)}</li>""" for w in warnings
        )
        warnings_html = f"""
        <div style="margin-top: 12px; padding: 8px 12px; background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; border-radius: 4px; font-size: 12px; color: #b45309;">
            <div style="font-weight: 600; margin-bottom: 4px;">⚠️ Statistical & Assumption Audits</div>
            <ul style="margin: 0; padding-left: 18px;">{warn_items}</ul>
        </div>
        """

    return f"""
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        border: 1px solid rgba(125, 125, 125, 0.25);
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
        max-width: 680px;
        background: transparent;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <div>
                <div style="font-size: 16px; font-weight: 700; line-height: 1.2;">{html.escape(title)}</div>
                <div style="font-size: 12px; opacity: 0.7; margin-top: 3px;">{html.escape(subtitle)}</div>
            </div>
            {badge_html}
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            {metrics_html}
        </div>
        {details_html}
        {warnings_html}
    </div>
    """


def format_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Format tabular data into standard GitHub-flavored Markdown table."""
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str_headers[i]), max((len(r[i]) for r in str_rows), default=0))
        for i in range(len(str_headers))
    ]
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(str_headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(len(str_headers))) + "-|"
    data_lines = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(r))) + " |" for r in str_rows
    ]
    return "\n".join([header_line, sep_line, *data_lines])


def format_latex_table(
    headers: list[str], rows: list[list[Any]], caption: str | None = None
) -> str:
    """Format tabular data into a publication-ready LaTeX table."""
    alignments = "l" * len(headers)
    header_str = " & ".join(headers) + r" \\ \hline"
    row_strs = [" & ".join(str(cell) for cell in row) + r" \\" for row in rows]
    caption_str = f"\\caption{{{caption}}}\n" if caption else ""
    return (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        f"{caption_str}"
        f"\\begin{{tabular}}{{{alignments}}}\n"
        "\\hline\n"
        f"{header_str}\n" + "\n".join(row_strs) + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
