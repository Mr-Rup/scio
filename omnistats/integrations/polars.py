"""Polars and Narwhals DataFrame integration (`pl_df.omni` and `pl_df.scio`)."""

from __future__ import annotations

from typing import Any

from omnistats.core.dataframe import ensure_pandas_df, is_dataframe_like


class PolarsOmniNamespace:
    """One-tap statistical and EDA namespace registered under `pl.DataFrame.omni` and `.scio`."""

    def __init__(self, df: Any) -> None:
        self._df = df

    def to_pandas(self) -> Any:
        """Convert the underlying Polars DataFrame to a Pandas DataFrame."""
        return ensure_pandas_df(self._df)

    def eda(self, **kwargs: Any) -> Any:
        """Run complete exploratory data analysis on the Polars DataFrame."""
        from omnistats.eda import profile

        return profile(self.to_pandas(), **kwargs)

    def compare(self, x: str, y: str | None = None, **kwargs: Any) -> Any:
        """Run autonomous hypothesis testing on the Polars DataFrame."""
        from omnistats.stat import compare

        return compare(self.to_pandas(), x=x, y=y, **kwargs)

    def screen(self, y: str, **kwargs: Any) -> Any:
        """Screen candidate features against target outcome `y` with FDR control."""
        from omnistats.stat import screen

        return screen(self.to_pandas(), y=y, **kwargs)

    def ols(self, formula: str, **kwargs: Any) -> Any:
        """Fit Ordinary Least Squares (OLS) regression with HC3 robust standard errors."""
        from omnistats.models import ols

        return ols(self.to_pandas(), formula=formula, **kwargs)

    def logistic(self, formula: str, **kwargs: Any) -> Any:
        """Fit binary Logistic Regression via Newton-Raphson IRLS."""
        from omnistats.models import logistic

        return logistic(self.to_pandas(), formula=formula, **kwargs)

    def poisson(self, formula: str, **kwargs: Any) -> Any:
        """Fit Poisson log-linear GLM via IRLS with QMLE HC3 sandwich standard errors."""
        from omnistats.models import poisson

        return poisson(self.to_pandas(), formula=formula, **kwargs)

    def anova_table(self, formula: str, **kwargs: Any) -> Any:
        """Compute Type II or Type III Factorial ANOVA / ANCOVA table."""
        from omnistats.models import anova_table

        return anova_table(self.to_pandas(), formula=formula, **kwargs)

    def mixedlm(self, formula: str, **kwargs: Any) -> Any:
        """Fit a Linear Mixed-Effects Model (Random Intercepts) via REML/ML."""
        from omnistats.models import mixedlm

        return mixedlm(self.to_pandas(), formula=formula, **kwargs)

    def coxph(self, formula: str, **kwargs: Any) -> Any:
        """Fit a Cox Proportional Hazards model via Breslow partial likelihood."""
        from omnistats.models import coxph

        return coxph(self.to_pandas(), formula=formula, **kwargs)

    def ate(self, outcome: str, treatment: str, covariates: list[str], **kwargs: Any) -> Any:
        """Estimate causal Average Treatment Effect (ATE) via Doubly Robust AIPW or IPTW."""
        from omnistats.models import ate

        return ate(
            self.to_pandas(), outcome=outcome, treatment=treatment, covariates=covariates, **kwargs
        )


PolarsScioNamespace = PolarsOmniNamespace


def register_polars_namespace() -> None:
    """Register the `omni` and `scio` namespaces on `polars.DataFrame` if Polars is installed."""
    try:
        import polars as pl

        if not hasattr(pl.DataFrame, "omni"):
            pl.api.register_dataframe_namespace("omni")(PolarsOmniNamespace)
        if not hasattr(pl.DataFrame, "scio"):
            pl.api.register_dataframe_namespace("scio")(PolarsOmniNamespace)
    except (ImportError, AttributeError):
        pass


__all__ = [
    "PolarsOmniNamespace",
    "PolarsScioNamespace",
    "ensure_pandas_df",
    "is_dataframe_like",
    "register_polars_namespace",
]
