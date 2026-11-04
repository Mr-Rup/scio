"""DataFrame, Polars/Narwhals, and plotting ecosystem integrations for omnistats."""

from __future__ import annotations

from omnistats.integrations.pandas import ScioAccessor, register_accessor
from omnistats.integrations.plotting import (
    SVGFigure,
    plot_eda_report,
    plot_raincloud,
    plot_regression_forest,
    plot_screen_report,
    plot_test_result,
)
from omnistats.integrations.polars import (
    PolarsScioNamespace,
    ensure_pandas_df,
    is_dataframe_like,
    register_polars_namespace,
)

__all__ = [
    "PolarsScioNamespace",
    "SVGFigure",
    "ScioAccessor",
    "ensure_pandas_df",
    "is_dataframe_like",
    "plot_eda_report",
    "plot_raincloud",
    "plot_regression_forest",
    "plot_screen_report",
    "plot_test_result",
    "register_accessor",
    "register_polars_namespace",
]
