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
        from pyscio.stat import ols

        return ols(self._obj, formula=formula, **kwargs)

    def logistic(self, formula: str, **kwargs: Any) -> Any:
        """Fit binary Logistic Regression via Newton-Raphson IRLS."""
        from pyscio.stat import logistic

        return logistic(self._obj, formula=formula, **kwargs)


def register_accessor() -> None:
    """Register the `scio` accessor on pandas.DataFrame if pandas is available."""
    try:
        import pandas as pd

        if not hasattr(pd.DataFrame, "scio"):
            pd.api.extensions.register_dataframe_accessor("scio")(ScioAccessor)
    except (ImportError, AttributeError):
        pass
