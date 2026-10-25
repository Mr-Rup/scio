"""Immutable, typed result containers with dual notebook/terminal rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pyscio.core.base import BaseResult
from pyscio.core.fmt import format_html_card, format_markdown_table

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult


@dataclass(frozen=True)
class CheckResult:
    """Statistical assumption check result (e.g. Shapiro-Wilk normality, Levene homoscedasticity)."""

    name: str
    stat: float
    p: float
    passed: bool
    details: str = ""
    clt_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stat": self.stat,
            "p": self.p,
            "passed": self.passed,
            "details": self.details,
            "clt_note": self.clt_note,
        }


@dataclass(frozen=True)
class TestResult(BaseResult):
    """Complete hypothesis test container with assumptions, effect sizes, and confidence intervals."""

    __test__ = False

    test_name: str
    x_name: str
    y_name: str
    stat: float
    p: float
    p_display: str
    dof: float | tuple[float, ...] | None = None
    ci: tuple[float, float] | None = None
    ci_level: float = 0.95
    sig: bool = False
    alpha: float = 0.05
    effect: float | None = None
    metric: str | None = None
    magnitude: str | None = None
    checks: list[CheckResult] = field(default_factory=list)
    posthoc: pd.DataFrame | None = None
    warns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "x_name": self.x_name,
            "y_name": self.y_name,
            "stat": self.stat,
            "p": self.p,
            "p_display": self.p_display,
            "dof": self.dof,
            "ci": self.ci,
            "ci_level": self.ci_level,
            "sig": self.sig,
            "alpha": self.alpha,
            "effect": self.effect,
            "metric": self.metric,
            "magnitude": self.magnitude,
            "checks": [c.to_dict() for c in self.checks],
            "posthoc": self.posthoc.to_dict(orient="records") if self.posthoc is not None else None,
            "warns": list(self.warns),
        }

    def to_df(self) -> pd.DataFrame:
        data = {
            "test": [self.test_name],
            "target": [self.y_name],
            "group": [self.x_name],
            "statistic": [self.stat],
            "p_value": [self.p],
            "df": [str(self.dof) if self.dof is not None else "-"],
            "significant": [self.sig],
            "effect_size": [self.effect if self.effect is not None else np.nan],
            "effect_metric": [self.metric if self.metric is not None else "-"],
            "magnitude": [self.magnitude if self.magnitude is not None else "-"],
            "ci_low": [self.ci[0] if self.ci else np.nan],
            "ci_high": [self.ci[1] if self.ci else np.nan],
        }
        return pd.DataFrame(data)

    def summary(self, format: str = "text") -> str:
        if format == "markdown":
            headers = ["Metric", "Value"]
            rows = [
                ["Test", self.test_name],
                ["Target ~ Group", f"{self.y_name} ~ {self.x_name}"],
                ["Statistic", f"{self.stat:.4f}"],
                ["p-value", self.p_display],
                ["Significant (α=0.05)", "Yes" if self.sig else "No"],
                ["Effect Size", f"{self.effect:.4f} ({self.metric}, {self.magnitude})" if self.effect else "-"],
                ["Confidence Interval", f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]" if self.ci else "-"],
            ]
            return format_markdown_table(headers, rows)

        # Default rich / text summary
        lines = [
            f"Test: {self.test_name}",
            f"Comparison: {self.y_name} ~ {self.x_name}",
            f"Statistic: {self.stat:.5g}, {self.p_display}",
            f"Significance: {'Significant' if self.sig else 'Not Significant'} (alpha={self.alpha})",
        ]
        if self.dof is not None:
            lines.append(f"Degrees of Freedom: {self.dof}")
        if self.ci is not None:
            lines.append(f"{int(self.ci_level * 100)}% CI: [{self.ci[0]:.5g}, {self.ci[1]:.5g}]")
        if self.effect is not None:
            lines.append(f"Effect Size: {self.metric} = {self.effect:.4g} ({self.magnitude})")
        if self.checks:
            lines.append("Assumption Checks:")
            for c in self.checks:
                status = "PASSED" if c.passed else "VIOLATED"
                lines.append(f"  - {c.name}: {status} (p={c.p:.4g}) {c.details}")
        if self.warns:
            lines.append("Audits & Warnings:")
            for w in self.warns:
                lines.append(f"  ⚠️ {w}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        badge = ("Significant", "#10b981") if self.sig else ("Not Significant", "#6b7280")
        metrics = [
            ("Statistic", f"{self.stat:.4f}"),
            ("p-value", self.p_display),
        ]
        if self.effect is not None and self.metric:
            metrics.append((self.metric, f"{self.effect:.3f} ({self.magnitude})"))
        if self.ci is not None:
            metrics.append((f"{int(self.ci_level*100)}% CI", f"[{self.ci[0]:.3f}, {self.ci[1]:.3f}]"))

        details = [
            ("Comparison", f"{self.y_name} ~ {self.x_name}"),
            ("Degrees of Freedom", str(self.dof) if self.dof is not None else "-"),
        ]
        for c in self.checks:
            icon = "✓" if c.passed else "✗"
            details.append((f"Assumption: {c.name}", f"{icon} p={c.p:.4g} {c.details}"))

        return format_html_card(
            title=self.test_name,
            subtitle=f"{self.y_name} across {self.x_name}",
            metrics=metrics,
            badge=badge,
            details=details,
            warnings=self.warns,
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.panel import Panel
        from rich.table import Table

        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")

        table.add_row("Comparison", f"{self.y_name} ~ {self.x_name}")
        table.add_row("Statistic", f"{self.stat:.5g} ({self.p_display})")
        if self.dof is not None:
            table.add_row("Degrees of Freedom", str(self.dof))
        if self.ci:
            table.add_row(f"{int(self.ci_level*100)}% Confidence Interval", f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]")
        if self.effect is not None:
            table.add_row("Effect Size", f"{self.metric} = {self.effect:.4f} ({self.magnitude})")

        for c in self.checks:
            style = "green" if c.passed else "yellow"
            table.add_row(f"Assumption: {c.name}", f"[{style}]{'PASSED' if c.passed else 'VIOLATED'}[/{style}] (p={c.p:.4g})")

        for w in self.warns:
            table.add_row("Audit Alert", f"[yellow]⚠️ {w}[/yellow]")

        border_style = "green" if self.sig else "blue"
        yield Panel(table, title=f"[bold]{self.test_name}[/bold]", border_style=border_style)


@dataclass(frozen=True)
class ColProfile(BaseResult):
    """Detailed profile for a single column/variable."""

    name: str
    role: str
    dtype: str
    n_total: int
    n_missing: int
    missing_pct: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    iqr: float | None = None
    skew: float | None = None
    kurt: float | None = None
    top_values: list[tuple[Any, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "n_total": self.n_total,
            "n_missing": self.n_missing,
            "missing_pct": self.missing_pct,
            "n_unique": self.n_unique,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "iqr": self.iqr,
            "skew": self.skew,
            "kurt": self.kurt,
            "top_values": self.top_values,
        }

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def summary(self, format: str = "text") -> str:
        lines = [
            f"Column: {self.name} ({self.role}, {self.dtype})",
            f"Count: {self.n_total} (Missing: {self.n_missing}, {self.missing_pct:.1f}%)",
            f"Cardinality: {self.n_unique} unique",
        ]
        if self.mean is not None:
            lines.append(f"Mean ± Std: {self.mean:.4g} ± {self.std:.4g}")
        if self.median is not None:
            lines.append(f"Median (IQR): {self.median:.4g} ({self.iqr:.4g})")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        metrics = [
            ("Role", self.role),
            ("Unique", str(self.n_unique)),
            ("Missing", f"{self.missing_pct:.1f}%"),
        ]
        if self.mean is not None:
            metrics.append(("Mean", f"{self.mean:.4g}"))
        return format_html_card(
            title=f"Column Profile: {self.name}",
            subtitle=f"Data Type: {self.dtype} | Total: {self.n_total}",
            metrics=metrics,
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.panel import Panel
        yield Panel(self.summary(), title=f"Column: {self.name}")


@dataclass(frozen=True)
class EDAReport(BaseResult):
    """Complete structural EDA and data quality assessment report."""

    n_rows: int
    n_cols: int
    profiles: dict[str, ColProfile]
    health_score: float
    quality_alerts: list[str] = field(default_factory=list)
    correlations: pd.DataFrame | None = None
    missingness: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "health_score": self.health_score,
            "quality_alerts": list(self.quality_alerts),
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
        }

    def to_df(self) -> pd.DataFrame:
        records = [p.to_dict() for p in self.profiles.values()]
        return pd.DataFrame(records)

    def to_html(self, path: str) -> None:
        """Export standalone interactive HTML report to disk."""
        content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>pyscio EDA Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 40px auto; max-width: 960px; color: #1e293b; background: #f8fafc; }}
        .header {{ background: white; padding: 24px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 24px; }}
        .card {{ background: white; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 16px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: 13px; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>pyscio Exploratory Data Analysis</h1>
        <p>Rows: <b>{self.n_rows}</b> | Columns: <b>{self.n_cols}</b> | Health Score: <b>{self.health_score:.1f}/100</b></p>
    </div>
    <div class="card">
        <h2>Column Breakdown</h2>
        {self.to_df().to_html(classes="table", index=False)}
    </div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def summary(self, format: str = "text") -> str:
        lines = [
            f"Dataset Dimensions: {self.n_rows} rows x {self.n_cols} columns",
            f"Data Health Score: {self.health_score:.1f}/100",
        ]
        if self.quality_alerts:
            lines.append("Quality Alerts:")
            for a in self.quality_alerts:
                lines.append(f"  ⚠️ {a}")
        lines.append(f"Columns Profiled: {len(self.profiles)}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        badge_color = "#10b981" if self.health_score >= 80 else ("#f59e0b" if self.health_score >= 60 else "#ef4444")
        return format_html_card(
            title="Exploratory Data Analysis Report",
            subtitle=f"{self.n_rows:,} rows × {self.n_cols} columns",
            metrics=[
                ("Health Score", f"{self.health_score:.1f}/100"),
                ("Columns", str(self.n_cols)),
                ("Observations", f"{self.n_rows:,}"),
            ],
            badge=(f"Health: {self.health_score:.0f}%", badge_color),
            warnings=self.quality_alerts,
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.panel import Panel
        yield Panel(self.summary(), title="pyscio EDA Report", border_style="cyan")


@dataclass(frozen=True)
class ScreenReport(BaseResult):
    """Ranked multi-feature hypothesis screening report with Benjamini-Hochberg FDR control."""

    target: str
    n_features: int
    alpha: float
    results: list[TestResult]
    leaderboard_df: pd.DataFrame

    def leaderboard(self) -> pd.DataFrame:
        """Return the ranked feature discovery leaderboard."""
        return self.leaderboard_df

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "n_discoveries": int(self.leaderboard_df["significant_fdr"].sum()) if "significant_fdr" in self.leaderboard_df else 0,
            "results": [r.to_dict() for r in self.results],
        }

    def to_df(self) -> pd.DataFrame:
        return self.leaderboard_df

    def summary(self, format: str = "text") -> str:
        n_sig = int(self.leaderboard_df["significant_fdr"].sum()) if "significant_fdr" in self.leaderboard_df else 0
        return (
            f"Discovery Screening Target: {self.target}\n"
            f"Features Screened: {self.n_features}\n"
            f"Discoveries (FDR q < {self.alpha}): {n_sig} / {self.n_features}\n\n"
            f"Leaderboard:\n{self.leaderboard_df.head(10).to_string(index=False)}"
        )

    def _repr_html_(self) -> str:
        n_sig = int(self.leaderboard_df["significant_fdr"].sum()) if "significant_fdr" in self.leaderboard_df else 0
        metrics = [
            ("Target", self.target),
            ("Features", str(self.n_features)),
            ("Discoveries", f"{n_sig} (FDR q < {self.alpha})"),
        ]
        return format_html_card(
            title="Multi-Feature Discovery Screening",
            subtitle=f"Outcome: {self.target} | Benjamini-Hochberg FDR",
            metrics=metrics,
            badge=(f"{n_sig} Significant", "#10b981" if n_sig > 0 else "#6b7280"),
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.panel import Panel
        yield Panel(self.summary(), title=f"Screening Report: {self.target}", border_style="magenta")
