"""Universal hypothesis testing entry point with autonomous assumption routing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from pyscio.core.errors import MissingDataError, StatError
from pyscio.core.results import TestResult
from pyscio.eda.roles import Role, classify_series
from pyscio.stat.router import (
    route_contingency,
    route_independent_two_sample,
    route_k_sample,
    route_one_sample,
    route_paired_two_sample,
)


def compare(
    data: pd.DataFrame | np.ndarray | Sequence[float] | None = None,
    x: str | np.ndarray | Sequence[float] | None = None,
    y: str | np.ndarray | Sequence[float] | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    alternative: str = "two-sided",
    mu: float = 0.0,
    **kwargs: Any,
) -> TestResult:
    """Run autonomous hypothesis testing across groups, conditions, or against reference parameters.

    Automatically resolves data types, validates statistical assumptions (normality, sample size,
    variance, homoscedasticity, Cochran's rule), applies the Large-N Central Limit Theorem guard,
    routes to the mathematically optimal test, calculates standardized effect sizes, and executes
    post-hoc multiple comparisons when omnibus K-group tests are significant.

    Parameters
    ----------
    data : pd.DataFrame | array-like | None
        Input tabular DataFrame, 2D contingency table, or 1D sample vector.
    x : str | array-like | None
        Independent grouping variable name or first data sample vector.
    y : str | array-like | None
        Dependent outcome variable name or second data sample vector.
    alpha : float
        Significance threshold (default 0.05).
    paired : bool
        Whether the comparison is between paired / repeated-measures samples (default False).
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', or 'greater' (default 'two-sided').
    mu : float
        Hypothesized population mean for one-sample tests (default 0.0).

    Returns
    -------
    TestResult
        Complete typed result container with assumptions, effect size, confidence interval,
        and optional post-hoc table.
    """
    # -------------------------------------------------------------------------
    # Case 1: Pandas DataFrame Input
    # -------------------------------------------------------------------------
    if isinstance(data, pd.DataFrame):
        df = data

        # Normalize x and y column arguments
        col_x = x if isinstance(x, str) else None
        col_y = y if isinstance(y, str) else None

        if col_x is None and col_y is None:
            raise StatError(
                "At least one column name (x or y) must be specified when passing a DataFrame."
            )

        # 1-Sample test: df[col] vs mu
        if col_x is not None and col_y is None:
            if col_x not in df.columns:
                raise KeyError(f"Column '{col_x}' not found in DataFrame.")
            series = df[col_x].dropna()
            return route_one_sample(
                series.to_numpy(),
                popmean=mu,
                x_name=col_x,
                alpha=alpha,
                alternative=alternative,
            )

        if col_x is None and col_y is not None:
            if col_y not in df.columns:
                raise KeyError(f"Column '{col_y}' not found in DataFrame.")
            series = df[col_y].dropna()
            return route_one_sample(
                series.to_numpy(),
                popmean=mu,
                x_name=col_y,
                alpha=alpha,
                alternative=alternative,
            )

        assert col_x is not None and col_y is not None
        if col_x not in df.columns:
            raise KeyError(f"Column '{col_x}' not found in DataFrame.")
        if col_y not in df.columns:
            raise KeyError(f"Column '{col_y}' not found in DataFrame.")

        role_x = classify_series(df[col_x])
        role_y = classify_series(df[col_y])

        # Paired test between two numeric columns in same DataFrame
        if paired:
            sub = df[[col_x, col_y]].dropna()
            if len(sub) == 0:
                raise MissingDataError(
                    f"No non-null paired observations between '{col_x}' and '{col_y}'."
                )
            return route_paired_two_sample(
                sub[col_x].to_numpy(dtype=np.float64),
                sub[col_y].to_numpy(dtype=np.float64),
                x_name=col_x,
                y_name=col_y,
                alpha=alpha,
                alternative=alternative,
            )

        # Categorical vs Categorical: 2D Contingency Table
        if role_x in (Role.CATEGORICAL, Role.BINARY, Role.DISCRETE) and role_y in (
            Role.CATEGORICAL,
            Role.BINARY,
            Role.DISCRETE,
        ):
            # If both are numeric discrete with many values, defer to group comparison
            if not (pd.api.types.is_numeric_dtype(df[col_y]) and df[col_y].nunique() > 10):
                crosstab = pd.crosstab(df[col_x], df[col_y])
                return route_contingency(crosstab, x_name=col_x, y_name=col_y, alpha=alpha)

        # Grouping comparison: target outcome `y` grouped by independent variable `x`
        # If user passed x=target (numeric) and y=group (categorical), auto-orient
        if pd.api.types.is_numeric_dtype(df[col_x]) and not pd.api.types.is_numeric_dtype(
            df[col_y]
        ):
            col_x, col_y = col_y, col_x

        sub = df[[col_x, col_y]].dropna()
        if len(sub) == 0:
            raise MissingDataError(f"No non-null observations for '{col_x}' and '{col_y}'.")

        grouped = [
            group[col_y].to_numpy(dtype=np.float64)
            for _, group in sub.groupby(col_x, observed=False)
        ]
        group_labels = [str(k) for k, _ in sub.groupby(col_x, observed=False)]

        if len(grouped) < 2:
            raise StatError(
                f"Grouping column '{col_x}' must contain at least 2 distinct groups, got {len(grouped)}."
            )

        if len(grouped) == 2:
            return route_independent_two_sample(
                grouped[0],
                grouped[1],
                x_name=f"{col_x} ({group_labels[0]} vs {group_labels[1]})",
                y_name=col_y,
                alpha=alpha,
                alternative=alternative,
            )

        return route_k_sample(
            grouped,
            group_names=group_labels,
            x_name=col_x,
            y_name=col_y,
            alpha=alpha,
        )

    # -------------------------------------------------------------------------
    # Case 2: Direct Array / Matrix / Vector Input
    # -------------------------------------------------------------------------
    # If data is 2D numpy array, treat as contingency table
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return route_contingency(data, alpha=alpha)

    # Extract sample A and sample B
    sample_a = data if data is not None else x
    sample_b = y

    if sample_a is None:
        raise StatError("No input data or sample vectors provided to compare().")

    # One-sample test: sample_a vs mu
    if sample_b is None:
        return route_one_sample(
            sample_a,  # type: ignore[arg-type]
            popmean=mu,
            x_name="sample",
            alpha=alpha,
            alternative=alternative,
        )

    # Two-sample test: sample_a vs sample_b
    if paired:
        return route_paired_two_sample(
            sample_a,  # type: ignore[arg-type]
            sample_b,  # type: ignore[arg-type]
            x_name="sample_a",
            y_name="sample_b",
            alpha=alpha,
            alternative=alternative,
        )

    return route_independent_two_sample(
        sample_a,  # type: ignore[arg-type]
        sample_b,  # type: ignore[arg-type]
        x_name="group_a vs group_b",
        y_name="measurement",
        alpha=alpha,
        alternative=alternative,
    )
