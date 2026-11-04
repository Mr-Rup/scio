"""Relationships: Pearson/Spearman correlations, Cramér's V, VIF multicollinearity, and null co-occurrence."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from omnistats.core.roles import Role, classify
from omnistats.functional.effects import cramers_v


def correlations(
    df: pd.DataFrame, method: Literal["pearson", "spearman"] = "pearson"
) -> pd.DataFrame:
    """Compute pairwise numeric correlation matrix (Pearson or Spearman)."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame()
    return num_df.corr(method=method)


def cramers_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Cramér's V association matrix for categorical and binary columns."""
    roles = classify(df)
    cat_cols = [
        col for col, r in roles.items() if r in (Role.CATEGORICAL, Role.BINARY, Role.DISCRETE)
    ]
    k = len(cat_cols)
    if k < 2:
        return pd.DataFrame()

    matrix = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        matrix[i, i] = 1.0
        for j in range(i + 1, k):
            ct = pd.crosstab(df[cat_cols[i]], df[cat_cols[j]])
            v = cramers_v(ct.to_numpy(), bias_correction=True)
            matrix[i, j] = v
            matrix[j, i] = v

    return pd.DataFrame(matrix, index=cat_cols, columns=cat_cols)


def vif(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) to detect multicollinearity among numeric features."""
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif", "severity"])

    cols = list(num_df.columns)
    vifs = []

    # Calculate VIF via R^2 of each feature regressed on all others
    for col in cols:
        y = num_df[col].to_numpy()
        x_cols = [c for c in cols if c != col]
        X = num_df[x_cols].to_numpy()
        # Add intercept
        X_design = np.column_stack([np.ones(len(X)), X])
        try:
            # Ordinary least squares
            beta, residuals, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if len(residuals) > 0 and ss_tot > 0:
                ss_res = float(residuals[0])
                r2 = max(0.0, 1.0 - (ss_res / ss_tot))
            else:
                y_pred = X_design @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                r2 = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            val = float(1.0 / (1.0 - r2)) if r2 < 0.99999 else 1000.0
        except Exception:
            val = 1.0

        severity = "High" if val > 10.0 else ("Moderate" if val > 5.0 else "Low")
        vifs.append({"feature": col, "vif": round(val, 2), "severity": severity})

    return pd.DataFrame(vifs).sort_values("vif", ascending=False).reset_index(drop=True)


def missingness_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute null correlation matrix (co-occurrence of missingness between features)."""
    null_indicators = df.isna().astype(float)
    # Filter columns that have at least some variance in missingness
    varying_cols = [col for col in null_indicators.columns if null_indicators[col].std() > 0]
    if len(varying_cols) < 2:
        return pd.DataFrame()
    return null_indicators[varying_cols].corr()
