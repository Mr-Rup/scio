"""Stateless assumption checks: Normality (Shapiro-Wilk, D'Agostino), Homoscedasticity, Outliers."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats

from pyscio.core.guards import clean_1d


class NormalityResult(NamedTuple):
    """Result of a normality test."""

    stat: float
    p: float
    is_normal: bool
    test_used: str
    skewness: float
    excess_kurtosis: float


class HomoscedasticityResult(NamedTuple):
    """Result of an equal variance / homoscedasticity test."""

    stat: float
    p: float
    is_homoscedastic: bool
    test_used: str


def shapiro(x: np.ndarray | list[float], alpha: float = 0.05) -> NormalityResult:
    """Perform Shapiro-Wilk test for normality ($N \\le 5000$).

    Parameters
    ----------
    x : array-like
        1D sample data.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    NormalityResult
        NamedTuple with test statistic, p-value, is_normal boolean, and sample moments.
    """
    arr = clean_1d(x, name="x")
    res = stats.shapiro(arr)
    stat, p = float(res.statistic), float(res.pvalue)
    skew = float(stats.skew(arr, bias=False)) if len(arr) > 2 else 0.0
    kurt = float(stats.kurtosis(arr, bias=False)) if len(arr) > 3 else 0.0
    return NormalityResult(
        stat=stat,
        p=p,
        is_normal=p >= alpha,
        test_used="Shapiro-Wilk",
        skewness=skew,
        excess_kurtosis=kurt,
    )


def dagostino(x: np.ndarray | list[float], alpha: float = 0.05) -> NormalityResult:
    """Perform D'Agostino-Pearson omnibus test ($K^2$) for normality based on skewness and kurtosis.

    Requires $N \\ge 20$. If $N < 20$, falls back cleanly to Shapiro-Wilk.
    """
    arr = clean_1d(x, name="x")
    if len(arr) < 20:
        return shapiro(arr, alpha=alpha)

    res = stats.normaltest(arr)
    stat, p = float(res.statistic), float(res.pvalue)
    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, bias=False))
    return NormalityResult(
        stat=stat,
        p=p,
        is_normal=p >= alpha,
        test_used="D'Agostino's K²",
        skewness=skew,
        excess_kurtosis=kurt,
    )


def check_normality(x: np.ndarray | list[float], alpha: float = 0.05) -> NormalityResult:
    """Automatically select the optimal normality test: Shapiro-Wilk for $N \\le 5000$, D'Agostino for $N > 5000$."""
    arr = clean_1d(x, name="x")
    if len(arr) <= 5000:
        return shapiro(arr, alpha=alpha)
    return dagostino(arr, alpha=alpha)


def levene(*groups: np.ndarray | list[float], center: str = "median", alpha: float = 0.05) -> HomoscedasticityResult:
    """Perform Levene's test for equal variances across 2 or more groups.

    Center='median' computes Brown-Forsythe version, robust to non-normality.
    """
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    if len(cleaned) < 2:
        raise ValueError("Levene test requires at least 2 groups.")

    res = stats.levene(*cleaned, center=center)
    stat, p = float(res.statistic), float(res.pvalue)
    name = "Brown-Forsythe (Levene median)" if center == "median" else "Levene"
    return HomoscedasticityResult(
        stat=stat,
        p=p,
        is_homoscedastic=p >= alpha,
        test_used=name,
    )


def bartlett(*groups: np.ndarray | list[float], alpha: float = 0.05) -> HomoscedasticityResult:
    """Perform Bartlett's test for homoscedasticity across normally distributed groups."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    if len(cleaned) < 2:
        raise ValueError("Bartlett test requires at least 2 groups.")

    res = stats.bartlett(*cleaned)
    stat, p = float(res.statistic), float(res.pvalue)
    return HomoscedasticityResult(
        stat=stat,
        p=p,
        is_homoscedastic=p >= alpha,
        test_used="Bartlett",
    )


def iqr_bounds(x: np.ndarray | list[float], k: float = 1.5) -> tuple[float, float, int]:
    """Calculate Tukey fences $[Q_1 - k \\cdot IQR, Q_3 + k \\cdot IQR]$ and count outliers."""
    arr = clean_1d(x, name="x")
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = q75 - q25
    lower_bound = float(q25 - k * iqr)
    upper_bound = float(q75 + k * iqr)
    outliers_count = int(np.sum((arr < lower_bound) | (arr > upper_bound)))
    return lower_bound, upper_bound, outliers_count
