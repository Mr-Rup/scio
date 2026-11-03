"""Shared type aliases and lightweight NamedTuple return containers for functional kernels."""

from __future__ import annotations

from typing import Literal, NamedTuple

Alternative = Literal["two-sided", "less", "greater"]


class TTestResult(NamedTuple):
    """Lightweight return tuple for Student's, Welch's, Paired, One-Sample, and Yuen t-tests."""

    stat: float
    p: float
    dof: float
    ci: tuple[float, float]
    mean_diff: float
    se_diff: float


class ANOVAResult(NamedTuple):
    """Lightweight return tuple for One-Way and Welch's ANOVA."""

    stat: float
    p: float
    df_between: float
    df_within: float


class NonParametricResult(NamedTuple):
    """Lightweight return tuple for Mann-Whitney U, Wilcoxon, Kruskal-Wallis, and Permutation tests."""

    stat: float
    p: float


class ChiSqResult(NamedTuple):
    """Lightweight return tuple for Pearson's Chi-Squared contingency test."""

    stat: float
    p: float
    dof: int


class FisherResult(NamedTuple):
    """Lightweight return tuple for Fisher's Exact 2x2 contingency test."""

    stat: float
    p: float


class NormalityResult(NamedTuple):
    """Lightweight return tuple for Shapiro-Wilk and D'Agostino-Pearson normality checks."""

    stat: float
    p: float
    is_normal: bool
    skewness: float
    excess_kurtosis: float
    test_used: str


class HomoscedasticityResult(NamedTuple):
    """Lightweight return tuple for Levene / Brown-Forsythe and Bartlett variance checks."""

    stat: float
    p: float
    is_homoscedastic: bool
    test_used: str


class OddsRatioResult(NamedTuple):
    """Lightweight return tuple for 2x2 Odds Ratio with Woolf / Haldane-Anscombe CI."""

    or_val: float
    ci: tuple[float, float]
    se_log: float


class TOSTResult(NamedTuple):
    """Lightweight return tuple for Two One-Sided Tests (TOST) equivalence kernel."""

    stat: float
    p: float
    dof: float
    ci: tuple[float, float]
    mean_diff: float
    se_diff: float
    low: float
    high: float


class CorrelationResult(NamedTuple):
    """Lightweight return tuple for Pearson and Spearman correlation kernels with CI."""

    r: float
    stat: float
    p: float
    dof: float
    ci: tuple[float, float]
    method: str
