"""Universal hypothesis testing entry point with autonomous assumption routing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from pyscio.core.dataframe import ensure_pandas_df, is_dataframe_like
from pyscio.core.errors import MissingDataError, StatError
from pyscio.core.results import TestResult
from pyscio.core.roles import Role, classify_series
from pyscio.functional.effects import interpret_effect
from pyscio.models import _clean_var_name, coxph, extract_formula_vars, logistic, mixedlm, ols
from pyscio.stat.router import (
    route_contingency,
    route_correlation,
    route_independent_two_sample,
    route_k_sample,
    route_one_sample,
    route_paired_k_sample,
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
    formula: str | None = None,
    **kwargs: Any,
) -> TestResult:
    """Run autonomous hypothesis testing across groups, conditions, formulas, or reference parameters.

    Automatically resolves data types, validates statistical assumptions (normality, sample size,
    variance, homoscedasticity, Cochran's rule), applies the Large-N Central Limit Theorem and
    distributional pathology guards, routes to the mathematically optimal test, calculates
    standardized effect sizes with exact Non-Central t confidence intervals, and executes
    post-hoc multiple comparisons when omnibus K-group tests are significant.

    Supports Wilkinson formula strings (e.g. ``compare(df, "y ~ x")`` or
    ``compare(df, "y ~ x1 + x2 + C(group)")``).
    """
    # -------------------------------------------------------------------------
    # Case 1: Tabular DataFrame Input (Pandas, Polars, PyArrow, Narwhals, dict)
    # -------------------------------------------------------------------------
    if is_dataframe_like(data):
        df = ensure_pandas_df(data)

        # Detect Wilkinson formula in `formula` or `x`
        active_formula = (
            formula if formula is not None else (x if isinstance(x, str) and "~" in x else None)
        )
        if active_formula is not None:
            if active_formula.strip().startswith("Surv("):
                cox_res = coxph(df, active_formula, alpha=alpha)
                return TestResult(
                    test_name=cox_res.model_type,
                    x_name=cox_res.formula.split("~", 1)[1].strip(),
                    y_name=cox_res.y_name,
                    stat=cox_res.stat,
                    p=cox_res.p,
                    p_display=cox_res.p_display,
                    dof=float(cox_res.df_model),
                    ci=None,
                    ci_level=1.0 - alpha,
                    sig=cox_res.p < alpha,
                    alpha=alpha,
                    effect=cox_res.r_squared,
                    effect_ci=None,
                    metric="Harrell's C-index",
                    magnitude=interpret_effect("cohen", cox_res.r_squared),
                    checks=[],
                    posthoc=cox_res.coef_df,
                    warns=cox_res.warns,
                )
            if "|" in active_formula:
                reg_res = mixedlm(df, active_formula, alpha=alpha)
                return TestResult(
                    test_name=reg_res.model_type,
                    x_name=reg_res.formula.split("~", 1)[1].strip(),
                    y_name=reg_res.y_name,
                    stat=reg_res.stat,
                    p=reg_res.p,
                    p_display=reg_res.p_display,
                    dof=(float(reg_res.df_model), float(reg_res.df_resid)),
                    ci=None,
                    ci_level=1.0 - alpha,
                    sig=reg_res.p < alpha,
                    alpha=alpha,
                    effect=reg_res.r_squared,
                    effect_ci=None,
                    metric="Marginal R²",
                    magnitude=interpret_effect("eta", reg_res.r_squared),
                    checks=[],
                    posthoc=reg_res.coef_df,
                    warns=reg_res.warns,
                )
            lhs, rhs_terms = extract_formula_vars(active_formula)
            non_intercept = [t for t in rhs_terms if t not in ("1", "0", "-1")]
            if len(non_intercept) == 1 and ":" not in non_intercept[0]:
                raw_x, _ = _clean_var_name(non_intercept[0])
                if raw_x in df.columns and lhs in df.columns:
                    role_rx = classify_series(df[raw_x])
                    role_ly = classify_series(df[lhs])
                    # If single categorical/binary predictor or contingency, use standard router
                    if role_rx in (Role.CATEGORICAL, Role.BINARY) or role_ly in (
                        Role.CATEGORICAL,
                        Role.BINARY,
                    ):
                        x = raw_x
                        y = lhs
                        active_formula = None

            if active_formula is not None:
                role_lhs = classify_series(df[lhs]) if lhs in df.columns else Role.NUMERIC
                if role_lhs == Role.BINARY and df[lhs].dropna().nunique() == 2:
                    reg_res = logistic(df, active_formula, alpha=alpha)
                    eff_metric = "McFadden Pseudo-R²"
                else:
                    reg_res = ols(df, active_formula, robust_se="HC3", alpha=alpha)
                    eff_metric = "R-squared (R²)"
                return TestResult(
                    test_name=reg_res.model_type,
                    x_name=" + ".join(non_intercept),
                    y_name=reg_res.y_name,
                    stat=reg_res.stat,
                    p=reg_res.p,
                    p_display=reg_res.p_display,
                    dof=(float(reg_res.df_model), float(reg_res.df_resid)),
                    ci=None,
                    ci_level=1.0 - alpha,
                    sig=reg_res.p < alpha,
                    alpha=alpha,
                    effect=reg_res.r_squared,
                    effect_ci=None,
                    metric=eff_metric,
                    magnitude=interpret_effect("eta", reg_res.r_squared),
                    checks=[],
                    posthoc=reg_res.coef_df,
                    warns=reg_res.warns,
                )

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

        # Paired test between two numeric columns in same DataFrame (wide format)
        if (
            paired
            and pd.api.types.is_numeric_dtype(df[col_x])
            and pd.api.types.is_numeric_dtype(df[col_y])
        ):
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

        # Continuous Numeric vs Continuous Numeric: Correlation Test (Pearson r / Spearman rho)
        if role_x == Role.NUMERIC and role_y == Role.NUMERIC:
            sub = df[[col_x, col_y]].dropna()
            if len(sub) < 3:
                raise MissingDataError(
                    f"Insufficient paired observations (N={len(sub)}) between '{col_x}' and '{col_y}'."
                )
            return route_correlation(
                sub[col_x].to_numpy(dtype=np.float64),
                sub[col_y].to_numpy(dtype=np.float64),
                x_name=col_x,
                y_name=col_y,
                alpha=alpha,
            )

        # Categorical vs Categorical: 2D Contingency Table
        if (
            role_x in (Role.CATEGORICAL, Role.BINARY, Role.DISCRETE)
            and role_y in (Role.CATEGORICAL, Role.BINARY, Role.DISCRETE)
            and not (pd.api.types.is_numeric_dtype(df[col_y]) and df[col_y].nunique() > 10)
        ):
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
            if paired:
                return route_paired_two_sample(
                    grouped[0],
                    grouped[1],
                    x_name=f"{col_x} ({group_labels[0]} vs {group_labels[1]})",
                    y_name=col_y,
                    alpha=alpha,
                    alternative=alternative,
                )
            return route_independent_two_sample(
                grouped[0],
                grouped[1],
                x_name=f"{col_x} ({group_labels[0]} vs {group_labels[1]})",
                y_name=col_y,
                alpha=alpha,
                alternative=alternative,
            )

        if paired:
            return route_paired_k_sample(
                grouped,
                group_names=group_labels,
                x_name=col_x,
                y_name=col_y,
                alpha=alpha,
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

    # Extract sample A and sample B (supports both compare(a, b) and compare(x=a, y=b))
    if data is not None and x is not None and y is None:
        sample_a: Any = data
        sample_b: Any = x
    else:
        sample_a = data if data is not None else x
        sample_b = y

    if sample_a is None:
        raise StatError("No input data or sample vectors provided to compare().")

    # One-sample test: sample_a vs mu
    if sample_b is None:
        return route_one_sample(
            sample_a,
            popmean=mu,
            x_name="sample",
            alpha=alpha,
            alternative=alternative,
        )

    # Two-sample test: sample_a vs sample_b
    if paired:
        return route_paired_two_sample(
            sample_a,
            sample_b,
            x_name="sample_a",
            y_name="sample_b",
            alpha=alpha,
            alternative=alternative,
        )

    return route_independent_two_sample(
        sample_a,
        sample_b,
        x_name="group_a vs group_b",
        y_name="measurement",
        alpha=alpha,
        alternative=alternative,
    )
