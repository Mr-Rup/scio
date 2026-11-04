"""Plotting integration (`omnistats[plot]`): raincloud plots, forest plots, and diagnostics."""

from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from omnistats.core.dataframe import ensure_pandas_df
from omnistats.core.errors import StatError
from omnistats.core.plotting import (
    SVGFigure,
    _use_matplotlib,
    plot_eda_report,
    plot_regression_forest,
    plot_screen_report,
    plot_test_result,
)

_PALETTE = ("#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899")


def plot_raincloud(
    df: pd.DataFrame | Any,
    x: str,
    y: str,
    ax: Any = None,
    backend: str = "auto",
) -> Any:
    """Render a Raincloud Plot (half-violin KDE + median/IQR boxplot + jittered points).

    Supports both `matplotlib` (`omnistats[plot]`) and zero-dependency `SVGFigure` (`backend="svg"`).
    """
    pdf = ensure_pandas_df(df)
    if x not in pdf.columns or y not in pdf.columns:
        raise KeyError(f"Columns '{x}' and '{y}' must exist in DataFrame.")

    # Auto-orient if x is continuous and y is categorical
    col_grp, col_val = (
        (y, x)
        if pd.api.types.is_numeric_dtype(pdf[x]) and not pd.api.types.is_numeric_dtype(pdf[y])
        else (x, y)
    )
    sub = pdf[[col_grp, col_val]].dropna()
    if sub.empty:
        raise StatError(f"No valid observations for raincloud plot of '{col_val}' by '{col_grp}'.")

    groups: list[tuple[str, np.ndarray]] = []
    for k, grp in sub.groupby(col_grp, observed=False):
        arr = pd.to_numeric(grp[col_val], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(arr) > 0:
            groups.append((str(k), arr))

    if not groups:
        raise StatError("No numeric group observations available for raincloud plot.")

    if not _use_matplotlib(ax, backend):
        return _svg_raincloud(groups, col_grp, col_val)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(7.5, max(3.5, 1.1 * len(groups) + 1.2)))

    rng = np.random.default_rng(42)
    for idx, (_, arr) in enumerate(groups):
        color = _PALETTE[idx % len(_PALETTE)]
        if len(arr) >= 3 and np.std(arr) > 0:
            kde = stats.gaussian_kde(arr)
            grid = np.linspace(float(np.min(arr)), float(np.max(arr)), 80)
            dens = kde(grid)
            dens = dens / max(float(np.max(dens)), 1e-12) * 0.35
            ax.fill_between(grid, idx, idx - dens, color=color, alpha=0.35)
            ax.plot(grid, idx - dens, color=color, linewidth=1.5)
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        ax.plot([q25, q75], [idx + 0.08, idx + 0.08], color="#0f172a", linewidth=4)
        ax.scatter([med], [idx + 0.08], color="#ffffff", edgecolor="#0f172a", zorder=4, s=30)
        jitter = rng.uniform(0.14, 0.32, size=len(arr))
        ax.scatter(arr, idx + jitter, color=color, alpha=0.65, s=16)

    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels([g[0] for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel(col_val)
    ax.set_ylabel(col_grp)
    ax.set_title(f"Raincloud Distribution: {col_val} by {col_grp}")
    return ax


def _svg_raincloud(
    groups: list[tuple[str, np.ndarray]],
    col_grp: str,
    col_val: str,
) -> SVGFigure:
    """Render a self-contained SVG Raincloud plot."""
    width = 660
    row_h = 76
    top_pad = 64
    bot_pad = 48
    left_pad = 130
    right_pad = 36
    height = top_pad + len(groups) * row_h + bot_pad
    plot_w = width - left_pad - right_pad

    all_v = np.concatenate([g[1] for g in groups])
    v_min, v_max = float(np.min(all_v)), float(np.max(all_v))
    span = v_max - v_min if v_max > v_min else 1.0
    v_min -= 0.08 * span
    v_max += 0.08 * span
    span = v_max - v_min

    def x_pos(val: float) -> float:
        return left_pad + ((val - v_min) / span) * plot_w

    rng = np.random.default_rng(42)
    elems: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="font-family: -apple-system, sans-serif; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;">',
        f'<text x="20" y="32" font-size="15" font-weight="700" fill="#0f172a">{html.escape(f"Raincloud Distribution: {col_val} by {col_grp}")}</text>',
        f'<line x1="{left_pad}" y1="{top_pad - 10}" x2="{left_pad}" y2="{height - bot_pad}" stroke="#cbd5e1" stroke-width="1"/>',
        f'<line x1="{left_pad}" y1="{height - bot_pad}" x2="{width - right_pad}" y2="{height - bot_pad}" stroke="#cbd5e1" stroke-width="1"/>',
    ]

    for i, (label, arr) in enumerate(groups):
        color = _PALETTE[i % len(_PALETTE)]
        base_y = top_pad + i * row_h + 36.0
        elems.append(
            f'<text x="{left_pad - 12}" y="{base_y + 4:.1f}" font-size="12" font-weight="600" fill="#1e293b" text-anchor="end">{html.escape(label[:18])}</text>'
        )
        if len(arr) >= 3 and np.std(arr) > 0:
            kde = stats.gaussian_kde(arr)
            grid = np.linspace(float(np.min(arr)), float(np.max(arr)), 48)
            dens = kde(grid)
            dens = dens / max(float(np.max(dens)), 1e-12) * 26.0
            pts = [f"{x_pos(float(grid[0])):.1f},{base_y:.1f}"]
            for gx, dy in zip(grid, dens, strict=False):
                pts.append(f"{x_pos(float(gx)):.1f},{base_y - float(dy):.1f}")
            pts.append(f"{x_pos(float(grid[-1])):.1f},{base_y:.1f}")
            elems.append(
                f'<polygon points="{" ".join(pts)}" fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="1.5"/>'
            )

        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        elems.append(
            f'<line x1="{x_pos(float(q25)):.1f}" y1="{base_y + 6:.1f}" x2="{x_pos(float(q75)):.1f}" y2="{base_y + 6:.1f}" stroke="#0f172a" stroke-width="4" stroke-linecap="round"/>'
        )
        elems.append(
            f'<circle cx="{x_pos(float(med)):.1f}" cy="{base_y + 6:.1f}" r="3.5" fill="#ffffff" stroke="#0f172a" stroke-width="1.5"/>'
        )
        sample_pts = arr if len(arr) <= 80 else rng.choice(arr, size=80, replace=False)
        jitters = rng.uniform(13.0, 25.0, size=len(sample_pts))
        for val, jit in zip(sample_pts, jitters, strict=False):
            elems.append(
                f'<circle cx="{x_pos(float(val)):.1f}" cy="{base_y + float(jit):.1f}" r="2.5" fill="{color}" fill-opacity="0.65"/>'
            )

    elems.append(
        f'<text x="{left_pad + plot_w / 2:.1f}" y="{height - 14}" font-size="12" font-weight="600" fill="#334155" text-anchor="middle">{html.escape(col_val)}</text>'
    )
    elems.append("</svg>")
    return SVGFigure(svg="\n".join(elems), width=width, height=height)


__all__ = [
    "SVGFigure",
    "plot_eda_report",
    "plot_raincloud",
    "plot_regression_forest",
    "plot_screen_report",
    "plot_test_result",
]
