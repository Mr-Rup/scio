"""Regression result container for OLS, ANCOVA, and Logistic GLM models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pyscio.core.base import BaseResult
from pyscio.core.fmt import format_html_card
from pyscio.core.plotting import plot_regression_forest
from pyscio.core.results.apa import format_regression_apa

if TYPE_CHECKING:
    import pandas as pd
    from rich.console import Console, ConsoleOptions, RenderResult


@dataclass(frozen=True)
class RegressionResult(BaseResult):
    """Multivariate regression result container (OLS / ANCOVA / Logistic GLM)."""

    model_type: str
    formula: str
    y_name: str
    n_obs: int
    df_model: int
    df_resid: int
    stat: float
    p: float
    p_display: str
    r_squared: float
    adj_r_squared: float
    aic: float
    bic: float
    coef_df: pd.DataFrame
    robust_se: str = "HC3"
    alpha: float = 0.05
    warns: list[str] = field(default_factory=list)

    def apa(self) -> str:
        """Return an APA 7th Edition statistical summary string."""
        return format_regression_apa(self)

    def plot(self, ax: Any = None, backend: str = "auto") -> Any:
        """Plot a coefficient forest plot (estimates and 95% CIs) via matplotlib or SVGFigure."""
        return plot_regression_forest(self, ax=ax, backend=backend)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "formula": self.formula,
            "y_name": self.y_name,
            "n_obs": self.n_obs,
            "df_model": self.df_model,
            "df_resid": self.df_resid,
            "stat": self.stat,
            "p": self.p,
            "p_display": self.p_display,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "aic": self.aic,
            "bic": self.bic,
            "robust_se": self.robust_se,
            "coefficients": self.coef_df.to_dict(orient="records"),
            "warns": list(self.warns),
        }

    def to_df(self) -> pd.DataFrame:
        return self.coef_df

    def summary(self, format: str = "text") -> str:
        if format == "apa":
            return self.apa()
        lines = [
            f"Model: {self.model_type} ({self.robust_se} SE)",
            f"Formula: {self.formula}",
            f"Observations: {self.n_obs} (df_model={self.df_model}, df_resid={self.df_resid})",
            f"Fit: R² = {self.r_squared:.4f}, Adj R² = {self.adj_r_squared:.4f}, AIC = {self.aic:.2f}, BIC = {self.bic:.2f}",
            f"Omnibus Statistic: {self.stat:.4f} ({self.p_display})",
            "",
            "Coefficients:",
            self.coef_df.to_string(index=False),
        ]
        if self.warns:
            lines.append("\nDiagnostics & Warnings:")
            for w in self.warns:
                lines.append(f"  ⚠️ {w}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        sig = self.p < self.alpha
        badge = ("Significant Fit", "#10b981") if sig else ("Non-Significant", "#6b7280")
        metrics = [
            ("R² / Pseudo-R²", f"{self.r_squared:.4f}"),
            ("Adj. R²", f"{self.adj_r_squared:.4f}"),
            ("Omnibus p", self.p_display),
            ("N", str(self.n_obs)),
        ]
        return format_html_card(
            title=f"{self.model_type}: {self.formula}",
            subtitle=f"Standard Errors: {self.robust_se} | AIC: {self.aic:.1f} | BIC: {self.bic:.1f}",
            metrics=metrics,
            badge=badge,
            warnings=self.warns,
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.panel import Panel

        yield Panel(self.summary(), title=f"[bold]{self.model_type}[/bold]", border_style="green")
