"""Effect sizes with O(N log N) Cliff's delta and standardized magnitude classification."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import optimize, stats

from pyscio.core.guards import check_groups, check_paired, clean_1d


class OddsRatioResult(NamedTuple):
    or_val: float
    ci: tuple[float, float]
    se_log: float


def _nct_ci(
    t_obs: float,
    df: float,
    scale_factor: float,
    d_est: float,
    conf_level: float = 0.95,
) -> tuple[float, float]:
    """Invert the non-central t-distribution for exact standardized effect size CIs."""
    alpha = 1.0 - conf_level
    p_upper_tail = 1.0 - alpha / 2.0
    p_lower_tail = alpha / 2.0

    # For large degrees of freedom (df > 200), non-central t converges to analytical normal SE
    if df > 200.0 or not np.isfinite(t_obs):
        z_crit = float(stats.norm.ppf(p_upper_tail))
        se_d = math.sqrt((1.0 / (scale_factor**2)) + (d_est**2) / (2.0 * df))
        return (float(d_est - z_crit * se_d), float(d_est + z_crit * se_d))

    # Bracket width based on normal approximation
    spread = max(10.0, abs(t_obs) * 3.0 + 12.0)

    def obj_low(ncp: float) -> float:
        return float(stats.nct.cdf(t_obs, df, ncp) - p_upper_tail)

    def obj_high(ncp: float) -> float:
        return float(stats.nct.cdf(t_obs, df, ncp) - p_lower_tail)

    try:
        ncp_low = float(optimize.brentq(obj_low, t_obs - spread, t_obs + spread, xtol=1e-6))
        ncp_high = float(optimize.brentq(obj_high, t_obs - spread, t_obs + spread, xtol=1e-6))
        return (float(ncp_low / scale_factor), float(ncp_high / scale_factor))
    except (ValueError, RuntimeError):
        z_crit = float(stats.norm.ppf(p_upper_tail))
        se_d = math.sqrt((1.0 / (scale_factor**2)) + (d_est**2) / (2.0 * df))
        return (float(d_est - z_crit * se_d), float(d_est + z_crit * se_d))


def cohen_d(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Compute Cohen's d effect size for two independent samples.

    $$d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{pooled}}$$
    """
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    check_groups(g1, g2)

    n1, n2 = len(g1), len(g2)
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((np.mean(g1) - np.mean(g2)) / sp) if sp > 0 else 0.0


def cohen_d_ci(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float] | None = None,
    conf_level: float = 0.95,
    paired: bool = False,
    popmean: float = 0.0,
) -> tuple[float, float]:
    """Compute exact Non-Central t (NCT) confidence interval for Cohen's d.

    Supports two-sample independent, two-sample paired, and one-sample designs.
    """
    g1 = clean_1d(a, name="sample_a")
    if b is None:
        # One-sample Cohen's d CI
        n = len(g1)
        df = float(n - 1)
        sd = float(np.std(g1, ddof=1))
        d_est = float((np.mean(g1) - popmean) / sd) if sd > 0 else 0.0
        scale = math.sqrt(n)
        t_obs = d_est * scale
        return _nct_ci(t_obs, df, scale, d_est, conf_level=conf_level)

    g2 = clean_1d(b, name="sample_b")
    if paired:
        check_paired(g1, g2, check_diff_var=True)
        diff = g1 - g2
        n = len(diff)
        df = float(n - 1)
        sd_diff = float(np.std(diff, ddof=1))
        d_est = float(np.mean(diff) / sd_diff) if sd_diff > 0 else 0.0
        scale = math.sqrt(n)
        t_obs = d_est * scale
        return _nct_ci(t_obs, df, scale, d_est, conf_level=conf_level)

    check_groups(g1, g2)
    n1, n2 = len(g1), len(g2)
    df = float(n1 + n2 - 2)
    d_est = cohen_d(g1, g2)
    n_eff = (n1 * n2) / (n1 + n2)
    scale = math.sqrt(n_eff)
    t_obs = d_est * scale
    return _nct_ci(t_obs, df, scale, d_est, conf_level=conf_level)


def hedges_g(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Compute Hedges' g (Cohen's d with small-sample bias correction factor $J$)."""
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    d = cohen_d(g1, g2)
    df = len(g1) + len(g2) - 2
    # Hedges (1981) small-sample approximation
    j = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    return float(d * j)


def hedges_g_ci(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    conf_level: float = 0.95,
) -> tuple[float, float]:
    """Compute exact Non-Central t confidence interval for Hedges' g."""
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    df = len(g1) + len(g2) - 2
    j = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    d_low, d_high = cohen_d_ci(g1, g2, conf_level=conf_level)
    return (float(d_low * j), float(d_high * j))


def cliffs_d(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Compute Cliff's delta (d) in $O((N_1 + N_2) \\log(N_1 + N_2))$ time via Mann-Whitney U identity.

    $$\\delta = \\frac{2 U}{n_1 n_2} - 1$$

    This eliminates naive $O(N_1 N_2)$ loops and executes instantaneously on large datasets.
    """
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    check_groups(g1, g2)

    n1, n2 = len(g1), len(g2)
    res = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    u = float(res.statistic)
    delta = (2.0 * u) / (n1 * n2) - 1.0
    return float(np.clip(delta, -1.0, 1.0))


def cliffs_d_ci(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    conf_level: float = 0.95,
    *,
    delta: float | None = None,
) -> tuple[float, float]:
    """Compute confidence interval for Cliff's delta via hyperbolic tangent z-transformation."""
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    n1, n2 = len(g1), len(g2)
    d_val = cliffs_d(g1, g2) if delta is None else float(delta)
    # Clamp slightly away from +/-1 to prevent atanh singularity
    d_clamped = float(np.clip(d_val, -0.9999, 0.9999))
    # Asymptotic variance of Cliff's delta under general alternative
    var_d = ((1.0 - d_clamped**2) ** 2) * (n1 + n2 + 1.0) / (3.0 * n1 * n2)
    se_z = math.sqrt(var_d) / (1.0 - d_clamped**2)
    z = math.atanh(d_clamped)
    z_crit = float(stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))
    return (float(math.tanh(z - z_crit * se_z)), float(math.tanh(z + z_crit * se_z)))


def rank_biserial(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Compute rank-biserial correlation $r_{rb}$ for Mann-Whitney U."""
    return cliffs_d(a, b)


def eta_sq(*groups: np.ndarray | list[float]) -> float:
    """Compute Eta-squared (\\eta^2) effect size for One-Way ANOVA ($SS_{between} / SS_{total}$)."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    if len(cleaned) < 2:
        return 0.0

    all_data = np.concatenate(cleaned)
    grand_mean = float(np.mean(all_data))
    ss_total = float(np.sum((all_data - grand_mean) ** 2))
    if ss_total == 0:
        return 0.0

    ss_between = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in cleaned))
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def eps_sq(*groups: np.ndarray | list[float]) -> float:
    """Compute Epsilon-squared (\\epsilon^2) effect size for Kruskal-Wallis."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    n = sum(len(g) for g in cleaned)
    if k < 2 or n <= k:
        return 0.0

    h_res = stats.kruskal(*cleaned)
    h = float(h_res.statistic)
    eps = (h - k + 1) / (n - k)
    return float(np.clip(eps, 0.0, 1.0))


def kendalls_w(*groups: np.ndarray | list[float]) -> float:
    """Compute Kendall's coefficient of concordance $W \\in [0, 1]$ for Friedman's test."""
    cleaned = [clean_1d(g, name=f"condition_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 3:
        return 0.0
    n = len(cleaned[0])
    if n < 2 or any(len(g) != n for g in cleaned):
        return 0.0
    chi2_f = float(stats.friedmanchisquare(*cleaned).statistic)
    w = chi2_f / (n * (k - 1.0))
    return float(np.clip(w, 0.0, 1.0))


def cramers_v(table: np.ndarray | list[list[float]], bias_correction: bool = True) -> float:
    """Compute Cramér's V with optional Bergsma (2013) bias correction for contingency tables."""
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0

    n = np.sum(arr)
    if n == 0:
        return 0.0

    chi2 = float(stats.chi2_contingency(arr, correction=False).statistic)
    r, c = arr.shape

    if not bias_correction:
        v = math.sqrt(chi2 / (n * (min(r, c) - 1)))
        return float(np.clip(v, 0.0, 1.0))

    # Bergsma (2013) bias correction
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    c_corr = c - ((c - 1) ** 2) / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    if denom <= 0:
        return 0.0

    v = math.sqrt(phi2_corr / denom)
    return float(np.clip(v, 0.0, 1.0))


def odds_ratio(
    table: np.ndarray | list[list[int | float]], conf_level: float = 0.95
) -> OddsRatioResult:
    """Compute Odds Ratio with 95% Confidence Interval for a 2x2 table.

    Applies Haldane-Anscombe 0.5 correction if zero cells are detected.
    """
    arr = np.asarray(table, dtype=np.float64)
    if arr.shape != (2, 2):
        raise ValueError("Odds ratio requires a 2x2 contingency table.")

    if np.any(arr == 0):
        # Haldane-Anscombe 0.5 continuity correction
        arr = arr + 0.5

    a, b = arr[0, 0], arr[0, 1]
    c, d = arr[1, 0], arr[1, 1]

    or_val = float((a * d) / (b * c))
    se_log = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z_crit = float(stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))
    log_or = math.log(or_val)
    ci = (math.exp(log_or - z_crit * se_log), math.exp(log_or + z_crit * se_log))

    return OddsRatioResult(or_val=or_val, ci=ci, se_log=se_log)


def interpret_effect(metric: str, value: float) -> str:
    """Classify effect size magnitude based on standard scientific conventions."""
    abs_v = abs(value)
    name = metric.lower()

    if "cohen" in name or "hedges" in name:
        if abs_v < 0.2:
            return "negligible"
        if abs_v < 0.5:
            return "small"
        if abs_v < 0.8:
            return "medium"
        return "large"

    if "cliff" in name:
        # Romano et al. (2006)
        if abs_v < 0.147:
            return "negligible"
        if abs_v < 0.33:
            return "small"
        if abs_v < 0.474:
            return "medium"
        return "large"

    if "eta" in name:
        if abs_v < 0.01:
            return "negligible"
        if abs_v < 0.06:
            return "small"
        if abs_v < 0.14:
            return "medium"
        return "large"

    if "cram" in name:
        if abs_v < 0.1:
            return "negligible"
        if abs_v < 0.3:
            return "small"
        if abs_v < 0.5:
            return "medium"
        return "large"

    # Default fallback
    if abs_v < 0.2:
        return "negligible"
    if abs_v < 0.5:
        return "small"
    if abs_v < 0.8:
        return "medium"
    return "large"
