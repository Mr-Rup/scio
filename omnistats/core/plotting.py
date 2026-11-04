"""Dual-backend statistical visualization engine (matplotlib + zero-dependency SVG)."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from omnistats.core.results.eda import EDAReport
    from omnistats.core.results.inference import ScreenReport, TestResult
    from omnistats.core.results.regression import RegressionResult


@dataclass(frozen=True)
class SVGFigure:
    """Lightweight standalone SVG figure container with Jupyter & file export support."""

    svg: str
    width: int = 640
    height: int = 360

    def _repr_svg_(self) -> str:
        return self.svg

    def _repr_html_(self) -> str:
        return self.svg

    def __str__(self) -> str:
        return self.svg

    def save(self, path: str | Path) -> Path:
        """Write the SVG figure to disk."""
        target = Path(path)
        target.write_text(self.svg, encoding="utf-8")
        return target


def _use_matplotlib(ax: Any, backend: str) -> bool:
    """Determine whether to render with matplotlib or fallback to native SVG."""
    if ax is not None or backend == "matplotlib":
        return True
    if backend == "svg":
        return False
    try:
        import matplotlib.pyplot  # noqa: F401

        return True
    except ImportError:
        return False


def _svg_interval_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    estimates: list[float],
    lows: list[float],
    highs: list[float],
    sigs: list[bool],
    null_val: float = 0.0,
    x_label: str = "Estimate",
) -> SVGFigure:
    """Render a horizontal forest/interval chart as a self-contained SVGFigure."""
    n = max(1, len(labels))
    width = 640
    row_h = 38
    top_pad = 72
    bot_pad = 50
    left_pad = 160
    right_pad = 40
    height = top_pad + n * row_h + bot_pad
    plot_w = width - left_pad - right_pad

    all_vals = [null_val, *estimates, *lows, *highs]
    finite_vals = [v for v in all_vals if np.isfinite(v)] or [0.0, 1.0]
    v_min = min(finite_vals)
    v_max = max(finite_vals)
    span = v_max - v_min if v_max > v_min else 1.0
    v_min -= 0.12 * span
    v_max += 0.12 * span
    span = v_max - v_min

    def x_pos(val: float) -> float:
        return left_pad + ((val - v_min) / span) * plot_w

    null_x = x_pos(null_val)
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;">',
        f'<text x="20" y="30" font-size="15" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
        f'<text x="20" y="50" font-size="12" fill="#64748b">{html.escape(subtitle)}</text>',
        f'<line x1="{left_pad}" y1="{top_pad - 8}" x2="{left_pad}" y2="{height - bot_pad}" stroke="#cbd5e1" stroke-width="1"/>',
        f'<line x1="{left_pad}" y1="{height - bot_pad}" x2="{width - right_pad}" y2="{height - bot_pad}" stroke="#cbd5e1" stroke-width="1"/>',
        f'<line x1="{null_x:.1f}" y1="{top_pad - 8}" x2="{null_x:.1f}" y2="{height - bot_pad}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4"/>',
    ]

    for tick_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tv = v_min + tick_frac * span
        tx = left_pad + tick_frac * plot_w
        elements.append(
            f'<text x="{tx:.1f}" y="{height - bot_pad + 18}" font-size="11" fill="#64748b" text-anchor="middle">{tv:.3g}</text>'
        )

    for i, (lbl, est, lo, hi, sig) in enumerate(
        zip(labels, estimates, lows, highs, sigs, strict=False)
    ):
        cy = top_pad + i * row_h + row_h / 2.0
        color = "#10b981" if sig else "#3b82f6"
        elements.append(
            f'<text x="{left_pad - 10}" y="{cy + 4:.1f}" font-size="12" fill="#1e293b" text-anchor="end">{html.escape(lbl[:22])}</text>'
        )
        if np.isfinite(lo) and np.isfinite(hi):
            x1, x2 = x_pos(lo), x_pos(hi)
            elements.append(
                f'<line x1="{x1:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{cy:.1f}" stroke="{color}" stroke-width="2.5"/>'
            )
            elements.append(
                f'<line x1="{x1:.1f}" y1="{cy - 5:.1f}" x2="{x1:.1f}" y2="{cy + 5:.1f}" stroke="{color}" stroke-width="2"/>'
            )
            elements.append(
                f'<line x1="{x2:.1f}" y1="{cy - 5:.1f}" x2="{x2:.1f}" y2="{cy + 5:.1f}" stroke="{color}" stroke-width="2"/>'
            )
        if np.isfinite(est):
            cx = x_pos(est)
            elements.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
            )

    elements.append(
        f'<text x="{left_pad + plot_w / 2:.1f}" y="{height - 12}" font-size="12" font-weight="600" fill="#334155" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    elements.append("</svg>")
    return SVGFigure(svg="\n".join(elements), width=width, height=height)


def plot_test_result(result: TestResult, ax: Any = None, backend: str = "auto") -> Any:
    """Plot effect size or test statistic with confidence interval for a TestResult."""
    has_eff = result.effect is not None and np.isfinite(result.effect)
    est = float(result.effect) if has_eff and result.effect is not None else float(result.stat)
    ci = result.effect_ci if (has_eff and result.effect_ci is not None) else result.ci
    lo = float(ci[0]) if ci is not None else est
    hi = float(ci[1]) if ci is not None else est
    metric_label = result.metric if (has_eff and result.metric) else "Test Statistic"
    null_val = 1.0 if metric_label and "Odds Ratio" in metric_label else 0.0

    if not _use_matplotlib(ax, backend):
        return _svg_interval_chart(
            title=result.test_name,
            subtitle=result.apa(),
            labels=[f"{result.y_name} ~ {result.x_name}"],
            estimates=[est],
            lows=[lo],
            highs=[hi],
            sigs=[result.sig],
            null_val=null_val,
            x_label=f"{metric_label} ({int(result.ci_level * 100)}% CI)",
        )

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6.5, 2.8))
    color = "#10b981" if result.sig else "#3b82f6"
    ax.axvline(null_val, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax.errorbar(
        [est],
        [0],
        xerr=[[max(0.0, est - lo)], [max(0.0, hi - est)]],
        fmt="o",
        color=color,
        capsize=5,
        markersize=7,
        linewidth=2,
    )
    ax.set_yticks([0])
    ax.set_yticklabels([f"{result.y_name} ~ {result.x_name}"])
    ax.set_xlabel(f"{metric_label} ({int(result.ci_level * 100)}% CI)")
    ax.set_title(f"{result.test_name}\n{result.apa()}", fontsize=10)
    return ax


def plot_regression_forest(result: RegressionResult, ax: Any = None, backend: str = "auto") -> Any:
    """Plot a coefficient forest plot (estimates and 95% CIs) for a RegressionResult."""
    df = result.coef_df
    non_int = df[df["term"] != "Intercept"] if len(df) > 1 else df
    labels = [str(t) for t in non_int["term"].tolist()]
    estimates = [float(v) for v in non_int["coef"].tolist()]
    lo_col = "ci_low" if "ci_low" in non_int.columns else "ci_lower"
    hi_col = "ci_high" if "ci_high" in non_int.columns else "ci_upper"
    lows = [float(v) for v in non_int[lo_col].tolist()]
    highs = [float(v) for v in non_int[hi_col].tolist()]
    sigs = [float(p) < result.alpha for p in non_int["p_value"].tolist()]

    if not _use_matplotlib(ax, backend):
        return _svg_interval_chart(
            title=f"{result.model_type}: {result.formula}",
            subtitle=result.apa(),
            labels=labels,
            estimates=estimates,
            lows=lows,
            highs=highs,
            sigs=sigs,
            null_val=0.0,
            x_label="Coefficient Estimate (95% CI)",
        )

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(7.0, max(3.0, 0.6 * len(labels) + 1.5)))
    y_pos = np.arange(len(labels))
    ax.axvline(0.0, color="#94a3b8", linestyle="--", linewidth=1.2)
    for i, (est, lo, hi, sig) in enumerate(zip(estimates, lows, highs, sigs, strict=False)):
        color = "#10b981" if sig else "#64748b"
        ax.errorbar(
            [est],
            [i],
            xerr=[[max(0.0, est - lo)], [max(0.0, hi - est)]],
            fmt="o",
            color=color,
            capsize=4,
            markersize=6,
            linewidth=2,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient Estimate (95% CI)")
    ax.set_title(f"{result.model_type}: {result.formula}", fontsize=11)
    return ax


def plot_screen_report(report: ScreenReport, ax: Any = None, backend: str = "auto") -> Any:
    """Plot FDR discovery significance (-log10 q-value) across screened features."""
    df = report.leaderboard_df.head(15)
    labels = [str(f) for f in df["feature"].tolist()] if not df.empty else ["(no features)"]
    neg_log_q = (
        [float(-np.log10(max(float(q), 1e-16))) for q in df["p_adj"].tolist()]
        if not df.empty
        else [0.0]
    )
    sigs = [bool(s) for s in df["significant_fdr"].tolist()] if not df.empty else [False]
    thresh = float(-np.log10(report.alpha))

    if not _use_matplotlib(ax, backend):
        return _svg_interval_chart(
            title=f"Discovery Screening: {report.target}",
            subtitle=f"Benjamini-Hochberg FDR Threshold (α = {report.alpha})",
            labels=labels,
            estimates=neg_log_q,
            lows=[0.0] * len(labels),
            highs=neg_log_q,
            sigs=sigs,
            null_val=thresh,
            x_label="-log10(FDR q-value)",
        )

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(7.0, max(3.0, 0.5 * len(labels) + 1.5)))
    colors = ["#10b981" if s else "#94a3b8" for s in sigs]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, neg_log_q, color=colors, height=0.6)
    ax.axvline(thresh, color="#ef4444", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(FDR q-value)")
    ax.set_title(f"Discovery Screening: {report.target}")
    return ax


def plot_eda_report(report: EDAReport, ax: Any = None, backend: str = "auto") -> Any:
    """Plot column completeness (%) across profiled dataset columns."""
    labels = list(report.profiles.keys())[:20] or ["(empty)"]
    completeness = (
        [100.0 - report.profiles[k].missing_pct for k in labels] if report.profiles else [100.0]
    )
    sigs = [c >= 95.0 for c in completeness]

    if not _use_matplotlib(ax, backend):
        return _svg_interval_chart(
            title=f"Dataset Completeness (Health Score: {report.health_score:.1f}/100)",
            subtitle=f"{report.n_rows:,} rows × {report.n_cols} columns",
            labels=labels,
            estimates=completeness,
            lows=[0.0] * len(labels),
            highs=completeness,
            sigs=sigs,
            null_val=95.0,
            x_label="Completeness (%)",
        )

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(7.0, max(3.0, 0.45 * len(labels) + 1.5)))
    y_pos = np.arange(len(labels))
    colors = ["#10b981" if s else "#f59e0b" for s in sigs]
    ax.barh(y_pos, completeness, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 105.0)
    ax.set_xlabel("Completeness (%)")
    ax.set_title(f"EDA Completeness (Health: {report.health_score:.1f}/100)")
    return ax
