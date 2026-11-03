"""Vectorized dataset profiling and column summary extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pyscio.core.results import ColProfile, EDAReport
from pyscio.core.roles import Role, classify
from pyscio.eda.io import read
from pyscio.eda.quality import audit_quality
from pyscio.eda.rel import correlations, missingness_correlation


def profile_column(s: pd.Series, role: Role) -> ColProfile:
    """Generate a single vectorized ColumnProfile container."""
    n_total = len(s)
    n_missing = int(s.isna().sum())
    missing_pct = float((n_missing / n_total) * 100.0) if n_total > 0 else 0.0

    valid = s.dropna()
    n_unique = int(valid.nunique())

    # Top-5 values frequency breakdown
    top_counts = valid.value_counts().head(5)
    top_values = [(k, int(v)) for k, v in top_counts.items()]

    mean: float | None = None
    std: float | None = None
    median: float | None = None
    iqr: float | None = None
    skew: float | None = None
    kurt: float | None = None

    if role in (Role.NUMERIC, Role.DISCRETE) and len(valid) > 0:
        numeric_vals = pd.to_numeric(valid, errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(numeric_vals) > 0:
            mean = float(np.mean(numeric_vals))
            std = float(np.std(numeric_vals, ddof=1)) if len(numeric_vals) > 1 else 0.0
            median = float(np.median(numeric_vals))
            q25, q75 = np.percentile(numeric_vals, [25, 75])
            iqr = float(q75 - q25)
            if len(numeric_vals) >= 3:
                skew = float(stats.skew(numeric_vals, bias=False))
            if len(numeric_vals) >= 4:
                kurt = float(stats.kurtosis(numeric_vals, bias=False))

    return ColProfile(
        name=str(s.name),
        role=str(role),
        dtype=str(s.dtype),
        n_total=n_total,
        n_missing=n_missing,
        missing_pct=missing_pct,
        n_unique=n_unique,
        mean=mean,
        std=std,
        median=median,
        iqr=iqr,
        skew=skew,
        kurt=kurt,
        top_values=top_values,
    )


def profile(
    data: pd.DataFrame | str | Path,
    **kwargs: Any,
) -> EDAReport:
    """Run complete exploratory data analysis, quality audit, and statistical profiling.

    Parameters
    ----------
    data : pd.DataFrame | str | Path
        A DataFrame or file path to ingest.
    **kwargs : Any
        Passed to `pyscio.eda.read()` if data is a filepath.

    Returns
    -------
    EDAReport
        Comprehensive report containing column profiles, quality score, and correlations.
    """
    df = read(data, **kwargs) if not isinstance(data, pd.DataFrame) else data
    n_rows, n_cols = df.shape

    roles = classify(df)
    profiles = {col: profile_column(df[col], roles[col]) for col in df.columns}

    health_score, alerts = audit_quality(df)
    corr_df = correlations(df) if n_cols > 1 else None
    miss_df = missingness_correlation(df) if n_cols > 1 else None

    return EDAReport(
        n_rows=n_rows,
        n_cols=n_cols,
        profiles=profiles,
        health_score=health_score,
        quality_alerts=alerts,
        correlations=corr_df,
        missingness=miss_df,
    )
