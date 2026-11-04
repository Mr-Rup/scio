"""Data health scoring engine and automated quality rules audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from omnistats.core.roles import Role, classify


def audit_quality(df: pd.DataFrame) -> tuple[float, list[str]]:
    """Audit tabular data quality, generate alerts, and compute a health score (0 to 100).

    Returns
    -------
    tuple[float, list[str]]
        (health_score, quality_alerts)
    """
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        return 0.0, ["Dataset is completely empty."]

    score = 100.0
    alerts: list[str] = []
    roles = classify(df)

    # 1. Missingness Audit
    total_cells = n_rows * n_cols
    total_missing = int(df.isna().sum().sum())
    missing_ratio = total_missing / total_cells

    if missing_ratio > 0.0:
        # Scale penalty up to 25 points based on missing cell percentage
        score -= min(25.0, missing_ratio * 100.0 * 0.5)
        alerts.append(
            f"Missing Values: {total_missing:,} cells ({missing_ratio * 100:.1f}%) are null."
        )

    for col in df.columns:
        col_missing = int(df[col].isna().sum())
        col_ratio = col_missing / n_rows
        if col_ratio == 1.0:
            score -= 10.0
            alerts.append(f"Dead Column: '{col}' is 100% missing.")
        elif col_ratio > 0.4:
            score -= 3.0
            alerts.append(f"High Missingness: '{col}' has {col_ratio * 100:.1f}% missing values.")

    # 2. Constant / Zero-Variance Columns
    for col, role in roles.items():
        if role == Role.CONSTANT:
            score -= 5.0
            alerts.append(f"Constant Feature: '{col}' contains only 1 unique value.")

    # 3. Extreme Skewness / Outliers in Numeric Features
    for col, role in roles.items():
        if role == Role.NUMERIC:
            valid_vals = df[col].dropna()
            if len(valid_vals) >= 20:
                skew = float(stats.skew(valid_vals, bias=False))
                if abs(skew) > 3.0:
                    score -= 2.0
                    alerts.append(
                        f"Extreme Skewness: '{col}' has skewness={skew:.2f}; non-parametric or log-transform recommended."
                    )

    # 4. Duplicate Rows
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        dup_ratio = n_duplicates / n_rows
        score -= min(10.0, dup_ratio * 50.0)
        alerts.append(
            f"Duplicate Rows: {n_duplicates:,} duplicate rows detected ({dup_ratio * 100:.1f}%)."
        )

    health_score = float(np.clip(score, 0.0, 100.0))
    return health_score, alerts
