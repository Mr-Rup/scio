"""Inference result containers: CheckResult, TestResult, and ScreenReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from omnistats.core.base import BaseResult
from omnistats.core.fmt import format_html_card, format_markdown_table
from omnistats.core.plotting import plot_screen_report, plot_test_result
from omnistats.core.results.apa import format_test_apa

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
    effect_ci: tuple[float, float] | None = None
    checks: list[CheckResult] = field(default_factory=list)
    posthoc: pd.DataFrame | None = None
    warns: list[str] = field(default_factory=list)

    def apa(self) -> str:
        """Return an APA 7th Edition statistical report string."""
        return format_test_apa(self)

    def plot(self, ax: Any = None, backend: str = "auto") -> Any:
        """Plot point estimate and confidence interval via matplotlib or SVGFigure."""
        return plot_test_result(self, ax=ax, backend=backend)

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
            "effect_ci": self.effect_ci,
            "checks": [c.to_dict() for c in self.checks],
            "posthoc": self.posthoc.to_dict(orient="records") if self.posthoc is not None else None,
            "warns": list(self.warns),
        }

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
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
                "effect_ci_low": [self.effect_ci[0] if self.effect_ci else np.nan],
                "effect_ci_high": [self.effect_ci[1] if self.effect_ci else np.nan],
                "ci_low": [self.ci[0] if self.ci else np.nan],
                "ci_high": [self.ci[1] if self.ci else np.nan],
            }
        )

    def summary(self, format: str = "text") -> str:
        if format == "apa":
            return self.apa()
        if format == "markdown":
            eff_str = (
                f"{self.effect:.4f} ({self.metric}, {self.magnitude})"
                if self.effect is not None
                else "-"
            )
            if self.effect is not None and self.effect_ci is not None:
                eff_str += f" CI[{self.effect_ci[0]:.4f}, {self.effect_ci[1]:.4f}]"
            rows = [
                ["Test", self.test_name],
                ["Target ~ Group", f"{self.y_name} ~ {self.x_name}"],
                ["Statistic", f"{self.stat:.4f}"],
                ["p-value", self.p_display],
                ["Significant (α=0.05)", "Yes" if self.sig else "No"],
                ["Effect Size", eff_str],
                [
                    "Confidence Interval",
                    f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]" if self.ci else "-",
                ],
            ]
            return format_markdown_table(["Metric", "Value"], rows)

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
            eff_line = f"Effect Size: {self.metric} = {self.effect:.4g} ({self.magnitude})"
            if self.effect_ci is not None:
                eff_line += f", {int(self.ci_level * 100)}% CI [{self.effect_ci[0]:.4g}, {self.effect_ci[1]:.4g}]"
            lines.append(eff_line)
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
        metrics = [("Statistic", f"{self.stat:.4f}"), ("p-value", self.p_display)]
        if self.effect is not None and self.metric:
            metrics.append((self.metric, f"{self.effect:.3f} ({self.magnitude})"))
        if self.ci is not None:
            metrics.append(
                (f"{int(self.ci_level * 100)}% CI", f"[{self.ci[0]:.3f}, {self.ci[1]:.3f}]")
            )
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
            table.add_row(
                f"{int(self.ci_level * 100)}% Confidence Interval",
                f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
            )
        if self.effect is not None:
            table.add_row("Effect Size", f"{self.metric} = {self.effect:.4f} ({self.magnitude})")
        for c in self.checks:
            style = "green" if c.passed else "yellow"
            table.add_row(
                f"Assumption: {c.name}",
                f"[{style}]{'PASSED' if c.passed else 'VIOLATED'}[/{style}] (p={c.p:.4g})",
            )
        for w in self.warns:
            table.add_row("Audit Alert", f"[yellow]⚠️ {w}[/yellow]")
        yield Panel(
            table,
            title=f"[bold]{self.test_name}[/bold]",
            border_style="green" if self.sig else "blue",
        )


@dataclass(frozen=True)
class ScreenReport(BaseResult):
    """Ranked multi-feature hypothesis screening report with Benjamini-Hochberg FDR control."""

    target: str
    n_features: int
    alpha: float
    results: list[TestResult]
    leaderboard_df: pd.DataFrame

    def leaderboard(self) -> pd.DataFrame:
        return self.leaderboard_df

    def plot(self, ax: Any = None, backend: str = "auto") -> Any:
        """Plot FDR discovery significance (-log10 q-value) across screened features."""
        return plot_screen_report(self, ax=ax, backend=backend)

    def to_dict(self) -> dict[str, Any]:
        n_sig = (
            int(self.leaderboard_df["significant_fdr"].sum())
            if "significant_fdr" in self.leaderboard_df
            else 0
        )
        return {
            "target": self.target,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "n_discoveries": n_sig,
            "results": [r.to_dict() for r in self.results],
        }

    def to_df(self) -> pd.DataFrame:
        return self.leaderboard_df

    def summary(self, format: str = "text") -> str:
        n_sig = (
            int(self.leaderboard_df["significant_fdr"].sum())
            if "significant_fdr" in self.leaderboard_df
            else 0
        )
        return (
            f"Discovery Screening Target: {self.target}\n"
            f"Features Screened: {self.n_features}\n"
            f"Discoveries (FDR q < {self.alpha}): {n_sig} / {self.n_features}\n\n"
            f"Leaderboard:\n{self.leaderboard_df.head(10).to_string(index=False)}"
        )

    def _repr_html_(self) -> str:
        n_sig = (
            int(self.leaderboard_df["significant_fdr"].sum())
            if "significant_fdr" in self.leaderboard_df
            else 0
        )
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

        yield Panel(
            self.summary(), title=f"Screening Report: {self.target}", border_style="magenta"
        )
