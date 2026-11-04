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


class RMAnovaResult(NamedTuple):
    """Lightweight return tuple for One-Way Repeated-Measures ANOVA with Greenhouse-Geisser correction."""

    stat: float
    p: float
    p_gg: float
    eps_gg: float
    df_num: float
    df_den: float
    partial_eta_sq: float


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
    """Lightweight return tuple for Pearson, Spearman, Kendall, and Partial correlation kernels."""

    r: float
    stat: float
    p: float
    dof: float
    ci: tuple[float, float]
    method: str


class BayesFactorResult(NamedTuple):
    """Lightweight return tuple for Rouder's JZS Bayes Factor t-test."""

    bf10: float
    bf01: float
    t_stat: float
    dof: float
    r_scale: float
    evidence: str


class SurvivalCurveResult(NamedTuple):
    """Lightweight return tuple for Kaplan-Meier product-limit survival curve estimation."""

    times: tuple[float, ...]
    survival_prob: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]
    n_at_risk: tuple[int, ...]
    n_events: tuple[int, ...]
    median_survival: float


class LogRankResult(NamedTuple):
    """Lightweight return tuple for Mantel-Cox Log-Rank survival comparison."""

    stat: float
    p: float
    dof: int
    observed: tuple[float, ...]
    expected: tuple[float, ...]
    hazard_ratio: float


class StationarityResult(NamedTuple):
    """Lightweight return tuple for Time-Series ADF stationarity and Ljung-Box autocorrelation tests."""

    stat: float
    p: float
    lags: int
    n_obs: int
    is_stationary: bool
    method: str
