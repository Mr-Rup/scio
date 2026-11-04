"""Pure, stateless numerical kernels for hypothesis testing."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import stats

from omnistats.core.guards import check_groups, check_paired, check_sample_size, clean_1d


class TTestResult(NamedTuple):
    stat: float
    p: float
    dof: float
    ci: tuple[float, float]
    mean_diff: float
    se_diff: float


class NonParametricResult(NamedTuple):
    stat: float
    p: float


class ANOVAResult(NamedTuple):
    stat: float
    p: float
    df_between: float
    df_within: float


class ChiSqResult(NamedTuple):
    stat: float
    p: float
    dof: int
    expected: np.ndarray


class FisherResult(NamedTuple):
    stat: float
    p: float


def student_t(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alternative: str = "two-sided",
    conf_level: float = 0.95,
) -> TTestResult:
    """Student's two-sample t-test assuming equal variances.

    Parameters
    ----------
    a, b : array-like
        Independent samples.
    alternative : str
        'two-sided', 'less', or 'greater'.
    conf_level : float
        Confidence interval level (default 0.95).
    """
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    check_groups(g1, g2)

    n1, n2 = len(g1), len(g2)
    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
    dof = float(n1 + n2 - 2)

    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / dof
    se = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    diff = m1 - m2
    stat = diff / se if se > 0 else 0.0

    if alternative == "two-sided":
        p = 2.0 * float(stats.t.sf(abs(stat), df=dof))
    elif alternative == "less":
        p = float(stats.t.cdf(stat, df=dof))
    else:
        p = float(stats.t.sf(stat, df=dof))

    t_crit = float(stats.t.ppf(1.0 - (1.0 - conf_level) / 2.0, df=dof))
    ci = (diff - t_crit * se, diff + t_crit * se)

    return TTestResult(stat=stat, p=p, dof=dof, ci=ci, mean_diff=diff, se_diff=se)


def welch_t(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alternative: str = "two-sided",
    conf_level: float = 0.95,
) -> TTestResult:
    """Welch's two-sample t-test for unequal variances (Satterthwaite approximation)."""
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    check_groups(g1, g2)

    n1, n2 = len(g1), len(g2)
    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))

    se1 = v1 / n1
    se2 = v2 / n2
    se = math.sqrt(se1 + se2)
    diff = m1 - m2
    stat = diff / se if se > 0 else 0.0

    # Satterthwaite-Welch degrees of freedom
    denom = (se1**2 / (n1 - 1)) + (se2**2 / (n2 - 1))
    dof = (se1 + se2) ** 2 / denom if denom > 0 else 1.0

    if alternative == "two-sided":
        p = 2.0 * float(stats.t.sf(abs(stat), df=dof))
    elif alternative == "less":
        p = float(stats.t.cdf(stat, df=dof))
    else:
        p = float(stats.t.sf(stat, df=dof))

    t_crit = float(stats.t.ppf(1.0 - (1.0 - conf_level) / 2.0, df=dof))
    ci = (diff - t_crit * se, diff + t_crit * se)

    return TTestResult(stat=stat, p=p, dof=dof, ci=ci, mean_diff=diff, se_diff=se)


def paired_t(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alternative: str = "two-sided",
    conf_level: float = 0.95,
) -> TTestResult:
    """Paired samples t-test."""
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    check_paired(g1, g2, check_diff_var=True)

    diffs = g1 - g2
    n = len(diffs)
    dof = float(n - 1)
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    se = sd_diff / math.sqrt(n)
    stat = mean_diff / se if se > 0 else 0.0

    if alternative == "two-sided":
        p = 2.0 * float(stats.t.sf(abs(stat), df=dof))
    elif alternative == "less":
        p = float(stats.t.cdf(stat, df=dof))
    else:
        p = float(stats.t.sf(stat, df=dof))

    t_crit = float(stats.t.ppf(1.0 - (1.0 - conf_level) / 2.0, df=dof))
    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)

    return TTestResult(stat=stat, p=p, dof=dof, ci=ci, mean_diff=mean_diff, se_diff=se)


def one_sample_t(
    a: np.ndarray | list[float],
    popmean: float = 0.0,
    alternative: str = "two-sided",
    conf_level: float = 0.95,
) -> TTestResult:
    """One-sample t-test comparing sample mean against popmean."""
    g = clean_1d(a, name="sample")
    check_sample_size(g, min_n=3)

    n = len(g)
    dof = float(n - 1)
    m = float(np.mean(g))
    s = float(np.std(g, ddof=1))
    se = s / math.sqrt(n)
    diff = m - popmean
    stat = diff / se if se > 0 else 0.0

    if alternative == "two-sided":
        p = 2.0 * float(stats.t.sf(abs(stat), df=dof))
    elif alternative == "less":
        p = float(stats.t.cdf(stat, df=dof))
    else:
        p = float(stats.t.sf(stat, df=dof))

    t_crit = float(stats.t.ppf(1.0 - (1.0 - conf_level) / 2.0, df=dof))
    ci = (m - t_crit * se, m + t_crit * se)

    return TTestResult(stat=stat, p=p, dof=dof, ci=ci, mean_diff=diff, se_diff=se)


def mann_whitney(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alternative: str = "two-sided",
) -> NonParametricResult:
    """Mann-Whitney U test (Wilcoxon rank-sum test) for independent samples."""
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    check_groups(g1, g2)

    res = stats.mannwhitneyu(g1, g2, alternative=alternative)
    return NonParametricResult(stat=float(res.statistic), p=float(res.pvalue))


def wilcoxon(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float] | None = None,
    alternative: str = "two-sided",
) -> NonParametricResult:
    """Wilcoxon signed-rank test for paired samples or one-sample difference from zero."""
    g1 = clean_1d(a, name="sample_a")
    if b is not None:
        g2 = clean_1d(b, name="sample_b")
        check_paired(g1, g2, check_diff_var=False)
        res = stats.wilcoxon(g1, g2, alternative=alternative)
    else:
        res = stats.wilcoxon(g1, alternative=alternative)
    return NonParametricResult(stat=float(res.statistic), p=float(res.pvalue))


def anova(*groups: np.ndarray | list[float]) -> ANOVAResult:
    """One-Way ANOVA test across 2 or more groups."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    if len(cleaned) < 2:
        raise ValueError("ANOVA requires at least 2 groups.")

    k = len(cleaned)
    n_total = sum(len(g) for g in cleaned)
    df_between = float(k - 1)
    df_within = float(n_total - k)

    res = stats.f_oneway(*cleaned)
    return ANOVAResult(
        stat=float(res.statistic),
        p=float(res.pvalue),
        df_between=df_between,
        df_within=df_within,
    )


def welch_anova(*groups: np.ndarray | list[float]) -> ANOVAResult:
    """Welch's ANOVA for testing equality of means under heteroscedasticity."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 2:
        raise ValueError("Welch ANOVA requires at least 2 groups.")

    ns = np.array([len(g) for g in cleaned], dtype=float)
    means = np.array([np.mean(g) for g in cleaned], dtype=float)
    vars_ = np.array([np.var(g, ddof=1) for g in cleaned], dtype=float)

    weights = ns / vars_
    w_sum = np.sum(weights)
    grand_mean = np.sum(weights * means) / w_sum

    numerator = np.sum(weights * (means - grand_mean) ** 2) / (k - 1)
    lambda_factor = np.sum((1.0 - weights / w_sum) ** 2 / (ns - 1)) / (k**2 - 1)
    denominator = 1.0 + 2.0 * (k - 2) * lambda_factor / 3.0

    stat = float(numerator / denominator)
    df_between = float(k - 1)
    df_within = float(1.0 / (3.0 * lambda_factor)) if lambda_factor > 0 else 1.0
    p = float(stats.f.sf(stat, dfn=df_between, dfd=df_within))

    return ANOVAResult(stat=stat, p=p, df_between=df_between, df_within=df_within)


def kruskal(*groups: np.ndarray | list[float]) -> NonParametricResult:
    """Kruskal-Wallis H-test across 2 or more groups."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    if len(cleaned) < 2:
        raise ValueError("Kruskal-Wallis test requires at least 2 groups.")

    res = stats.kruskal(*cleaned)
    return NonParametricResult(stat=float(res.statistic), p=float(res.pvalue))


def chisq(observed: np.ndarray | list[list[float]], correct: bool = True) -> ChiSqResult:
    """Pearson's Chi-squared test on a contingency table."""
    obs = np.asarray(observed, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[0] < 2 or obs.shape[1] < 2:
        raise ValueError("Chi-squared contingency test requires a 2D matrix of shape at least 2x2.")

    res = stats.chi2_contingency(obs, correction=correct)
    return ChiSqResult(
        stat=float(res.statistic),
        p=float(res.pvalue),
        dof=int(res.dof),
        expected=res.expected_freq,
    )


def fisher(table: np.ndarray | list[list[float]], alternative: str = "two-sided") -> FisherResult:
    """Fisher's exact test for a 2x2 contingency table."""
    arr = np.asarray(table, dtype=np.int64)
    if arr.shape != (2, 2):
        raise ValueError("Fisher exact test requires a 2x2 matrix.")

    res = stats.fisher_exact(arr, alternative=alternative)
    return FisherResult(stat=float(res.statistic), p=float(res.pvalue))


def mcnemar(
    table: np.ndarray | list[list[int | float]],
    *,
    exact: bool | None = None,
    correction: bool = True,
) -> NonParametricResult:
    """McNemar's test of marginal homogeneity for paired 2x2 contingency tables."""
    arr = np.asarray(table, dtype=np.float64)
    if arr.shape != (2, 2):
        raise ValueError("McNemar test requires a 2x2 contingency table.")

    b = float(arr[0, 1])
    c = float(arr[1, 0])
    n_disc = round(b + c)
    if n_disc == 0:
        return NonParametricResult(stat=0.0, p=1.0)

    use_exact = (n_disc < 25) if exact is None else exact
    if use_exact:
        k_min = round(min(b, c))
        p_val = float(stats.binomtest(k_min, n=n_disc, p=0.5, alternative="two-sided").pvalue)
        return NonParametricResult(stat=float(k_min), p=p_val)

    diff = abs(b - c)
    num = max(0.0, diff - 1.0) ** 2 if correction else diff**2
    chi2_stat = float(num / (b + c))
    p_val = float(stats.chi2.sf(chi2_stat, df=1))
    return NonParametricResult(stat=chi2_stat, p=p_val)


def friedman(*groups: np.ndarray | list[float]) -> NonParametricResult:
    """Friedman's non-parametric repeated-measures test across K >= 3 matched conditions."""
    cleaned = [clean_1d(g, name=f"condition_{i}") for i, g in enumerate(groups)]
    if len(cleaned) < 3:
        raise ValueError("Friedman test requires at least 3 matched conditions.")
    n0 = len(cleaned[0])
    if any(len(g) != n0 for g in cleaned):
        raise ValueError("All conditions in Friedman test must have identical sample lengths.")
    if n0 < 2:
        raise ValueError("Friedman test requires at least 2 blocks/subjects.")

    res = stats.friedmanchisquare(*cleaned)
    return NonParametricResult(stat=float(res.statistic), p=float(res.pvalue))
