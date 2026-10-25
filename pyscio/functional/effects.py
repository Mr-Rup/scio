"""Effect sizes with O(N log N) Cliff's delta and standardized magnitude classification."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import stats

from pyscio.core.guards import check_groups, clean_1d


class OddsRatioResult(NamedTuple):
    or_val: float
    ci: tuple[float, float]
    se_log: float


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


def hedges_g(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Compute Hedges' g (Cohen's d with small-sample bias correction factor $J$)."""
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    d = cohen_d(g1, g2)
    df = len(g1) + len(g2) - 2
    # Hedges (1981) small-sample approximation
    j = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    return float(d * j)


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


def odds_ratio(table: np.ndarray | list[list[int | float]], conf_level: float = 0.95) -> OddsRatioResult:
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
