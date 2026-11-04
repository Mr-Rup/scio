"""Pandas DataFrame accessor binding for one-tap pyscio operations (`df.scio`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class ScioAccessor:
    """One-tap statistical and EDA accessor registered under `df.scio`."""

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj: pd.DataFrame = pandas_obj

    def eda(self, **kwargs: Any) -> Any:
        """Run complete exploratory data analysis on the DataFrame."""
        from pyscio.eda import profile

        return profile(self._obj, **kwargs)

    def compare(self, x: str, y: str | None = None, **kwargs: Any) -> Any:
        """Run autonomous hypothesis testing comparing target variable `y` across groups `x` or formula."""
        from pyscio.stat import compare

        return compare(self._obj, x=x, y=y, **kwargs)

    def screen(self, y: str, **kwargs: Any) -> Any:
        """Screen all candidate features against target outcome `y` with FDR control."""
        from pyscio.stat import screen

        return screen(self._obj, y=y, **kwargs)

    def ols(self, formula: str, **kwargs: Any) -> Any:
        """Fit Ordinary Least Squares (OLS) regression with HC3 robust standard errors."""
        from pyscio.models import ols

        return ols(self._obj, formula=formula, **kwargs)

    def logistic(self, formula: str, **kwargs: Any) -> Any:
        """Fit binary Logistic Regression via Newton-Raphson IRLS."""
        from pyscio.models import logistic

        return logistic(self._obj, formula=formula, **kwargs)

    def poisson(self, formula: str, **kwargs: Any) -> Any:
        """Fit Poisson log-linear GLM via IRLS with QMLE HC3 sandwich standard errors."""
        from pyscio.models import poisson

        return poisson(self._obj, formula=formula, **kwargs)

    def anova_table(self, formula: str, **kwargs: Any) -> Any:
        """Compute Type II or Type III Factorial ANOVA / ANCOVA table."""
        from pyscio.models import anova_table

        return anova_table(self._obj, formula=formula, **kwargs)

    def mixedlm(self, formula: str, **kwargs: Any) -> Any:
        """Fit a Linear Mixed-Effects Model (Random Intercepts) via REML/ML."""
        from pyscio.models import mixedlm

        return mixedlm(self._obj, formula=formula, **kwargs)

    def coxph(self, formula: str, **kwargs: Any) -> Any:
        """Fit a Cox Proportional Hazards model via Breslow partial likelihood."""
        from pyscio.models import coxph

        return coxph(self._obj, formula=formula, **kwargs)

    def ate(self, outcome: str, treatment: str, covariates: list[str], **kwargs: Any) -> Any:
        """Estimate causal Average Treatment Effect (ATE) via Doubly Robust AIPW or IPTW."""
        from pyscio.models import ate

        return ate(self._obj, outcome=outcome, treatment=treatment, covariates=covariates, **kwargs)

    def covariate_balance(self, treatment: str, covariates: list[str], **kwargs: Any) -> Any:
        """Compute unweighted and propensity-weighted Standardized Mean Differences (SMD)."""
        from pyscio.models import covariate_balance

        return covariate_balance(self._obj, treatment=treatment, covariates=covariates, **kwargs)

    def plot_raincloud(self, x: str, y: str, **kwargs: Any) -> Any:
        """Render a Raincloud distribution plot comparing `y` across `x`."""
        from pyscio.integrations.plotting import plot_raincloud

        return plot_raincloud(self._obj, x=x, y=y, **kwargs)


def register_accessor() -> None:
    """Register the `scio` accessor on pandas.DataFrame and polars.DataFrame if available."""
    try:
        import pandas as pd

        if not hasattr(pd.DataFrame, "scio"):
            pd.api.extensions.register_dataframe_accessor("scio")(ScioAccessor)
    except (ImportError, AttributeError):
        pass

    from pyscio.integrations.polars import register_polars_namespace

    register_polars_namespace()
