"""Stateless assumption checks: Normality (Shapiro-Wilk, D'Agostino), Homoscedasticity, Outliers."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats

from omnistats.core.errors import StatError
from omnistats.core.guards import clean_1d


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
    """Perform Shapiro-Wilk test for normality ($N \\le 5000$)."""
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


def levene(
    *groups: np.ndarray | list[float], center: str = "median", alpha: float = 0.05
) -> HomoscedasticityResult:
    """Perform Levene's test for equal variances across 2 or more groups.

    Center='median' computes Brown-Forsythe version, robust to non-normality.
    """
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    if len(cleaned) < 2:
        raise StatError("Levene test requires at least 2 groups.")

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
        raise StatError("Bartlett test requires at least 2 groups.")

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


def sarle_bimodality(
    x: np.ndarray | list[float],
    *,
    skewness: float | None = None,
    excess_kurtosis: float | None = None,
) -> float:
    """Compute Sarle's Bimodality Coefficient (BC).

    $$BC = \\frac{\\gamma_1^2 + 1}{\\gamma_2 + 3 \\cdot \\frac{(n - 1)^2}{(n - 2)(n - 3)}}$$

    Values exceeding $5/9 \\approx 0.555$ indicate bimodal or multimodal distributions.
    """
    arr = clean_1d(x, name="x")
    n = len(arr)
    if n < 4:
        return 0.0
    skew = float(stats.skew(arr, bias=False)) if skewness is None else float(skewness)
    kurt = (
        float(stats.kurtosis(arr, bias=False))
        if excess_kurtosis is None
        else float(excess_kurtosis)
    )
    denom = kurt + 3.0 * ((n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
    if denom <= 1e-12:
        return 0.0
    return float((skew**2 + 1.0) / denom)


def check_pathology(
    x: np.ndarray | list[float],
    tie_threshold: float = 0.30,
    tail_ratio_max: float = 2.2,
    bimodality_max: float = 0.555,
    *,
    skewness: float | None = None,
    excess_kurtosis: float | None = None,
) -> tuple[bool, list[str]]:
    """Detect distributional pathologies where moment-based CLT and Levene guards fail."""
    arr = clean_1d(x, name="x")
    n = len(arr)
    if n < 4:
        return False, []

    reasons: list[str] = []

    # 1. Point-mass / zero-inflation check
    _, counts = np.unique(arr, return_counts=True)
    max_tie_prop = float(np.max(counts) / n)
    if max_tie_prop >= tie_threshold:
        reasons.append(
            f"Point-mass/zero-inflation ({max_tie_prop * 100:.1f}% tied at a single value)"
        )

    # 2. Heavy-tailed / infinite-variance check (SD vs Gaussian IQR scale & extreme outer fences)
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    sd = float(np.std(arr, ddof=1))
    if iqr > 1e-12:
        normal_iqr_sd = iqr / 1.3489795003921634
        tail_ratio = sd / normal_iqr_sd
        extreme_low = q25 - 3.0 * iqr
        extreme_high = q75 + 3.0 * iqr
        extreme_prop = float(np.mean((arr < extreme_low) | (arr > extreme_high)))
        if tail_ratio > tail_ratio_max or extreme_prop > 0.025:
            reasons.append(
                f"Heavy-tailed/outlier regime (tail scale ratio={tail_ratio:.2f}, extreme outliers={extreme_prop * 100:.1f}%)"
            )

    # 3. Bimodal / multimodal check via Sarle's BC (reuses precomputed moments when supplied)
    kurt = (
        float(stats.kurtosis(arr, bias=False))
        if excess_kurtosis is None
        else float(excess_kurtosis)
    )
    bc = sarle_bimodality(arr, skewness=skewness, excess_kurtosis=kurt)
    if bc > bimodality_max and kurt < -0.5:
        reasons.append(f"Bimodal/multimodal regime (Sarle's BC={bc:.3f} > {bimodality_max})")

    return len(reasons) > 0, reasons
