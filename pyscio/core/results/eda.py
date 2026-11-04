"""EDA result containers: ColProfile and EDAReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pyscio.core.base import BaseResult
from pyscio.core.fmt import format_html_card
from pyscio.core.plotting import plot_eda_report

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult


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

    def plot(self, ax: Any = None, backend: str = "auto") -> Any:
        """Plot column completeness (%) across profiled dataset columns."""
        return plot_eda_report(self, ax=ax, backend=backend)

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
        badge_color = (
            "#10b981"
            if self.health_score >= 80
            else ("#f59e0b" if self.health_score >= 60 else "#ef4444")
        )
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

    def save_html(self, path: str | Path) -> None:
        """Export standalone HTML report to disk."""
        target = Path(path)
        target.write_text(self._repr_html_(), encoding="utf-8")

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.panel import Panel

        yield Panel(self.summary(), title="pyscio EDA Report", border_style="cyan")
