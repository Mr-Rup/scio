"""Multiple testing corrections (Benjamini-Hochberg FDR, Bonferroni, Holm) and power analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def fdr_bh(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg False Discovery Rate (FDR) procedure.

    Calculates monotonically non-decreasing adjusted p-values ($q$-values)
    and returns a boolean discovery mask.

    Parameters
    ----------
    p_values : Sequence[float] | np.ndarray
        Array of raw unadjusted p-values.
    alpha : float
        FDR discovery threshold (default 0.05).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (reject_mask, q_values), both preserving the original input order.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)
    if m == 1:
        return np.array([p[0] <= alpha], dtype=bool), np.clip(p, 0.0, 1.0)

    # Sort p-values ascending
    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]

    # Calculate raw q-values: q_i = p_i * (m / rank)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    raw_q = sorted_p * (m / ranks)

    # Enforce monotonicity backwards: q_i = min(q_i, q_{i+1})
    q_mono = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_clamped = np.clip(q_mono, 0.0, 1.0)

    # Restore original input order
    orig_idx = np.argsort(sort_idx)
    q_values = q_clamped[orig_idx]
    reject = q_values <= alpha

    return reject, q_values


def bonferroni(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni Family-Wise Error Rate (FWER) single-step correction."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    p_adj = np.clip(p * m, 0.0, 1.0)
    reject = p_adj <= alpha
    return reject, p_adj


def holm(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down FWER procedure."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)
    if m == 1:
        return np.array([p[0] <= alpha], dtype=bool), np.clip(p, 0.0, 1.0)

    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]

    multipliers = np.arange(m, 0, -1, dtype=np.float64)
    raw_adj = sorted_p * multipliers

    # Enforce monotonicity forward: adj_i = max(adj_i, adj_{i-1})
    adj_mono = np.maximum.accumulate(raw_adj)
    adj_clamped = np.clip(adj_mono, 0.0, 1.0)

    orig_idx = np.argsort(sort_idx)
    p_adj = adj_clamped[orig_idx]
    reject = p_adj <= alpha

    return reject, p_adj


def power_t(n: int, d: float, alpha: float = 0.05) -> float:
    """Compute post-hoc statistical power for a two-sample t-test with equal group sizes $n$.

    Parameters
    ----------
    n : int
        Sample size per group ($n_1 = n_2 = n$).
    d : float
        Standardized effect size (Cohen's d).
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    float
        Statistical power in [0.0, 1.0].
    """
    if n < 2 or abs(d) == 0.0:
        return alpha

    dof = 2 * n - 2
    # Non-centrality parameter delta
    delta = d * np.sqrt(n / 2.0)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=dof)

    # Power under non-central t-distribution
    power = float(
        1.0 - stats.nct.cdf(t_crit, df=dof, nc=delta) + stats.nct.cdf(-t_crit, df=dof, nc=delta)
    )
    return float(np.clip(power, 0.0, 1.0))


def power_anova(k: int, n_per_group: int, eta_squared: float, alpha: float = 0.05) -> float:
    """Compute statistical power for One-Way ANOVA across $k$ groups of size $n$."""
    if k < 2 or n_per_group < 2 or eta_squared <= 0.0:
        return alpha

    # Cohen's f = sqrt(eta^2 / (1 - eta^2))
    eta_clamped = min(eta_squared, 0.999)
    f_stat = np.sqrt(eta_clamped / (1.0 - eta_clamped))

    total_n = k * n_per_group
    df_between = k - 1
    df_within = total_n - k

    # Non-centrality parameter lambda
    nc_lambda = total_n * (f_stat**2)
    f_crit = stats.f.ppf(1.0 - alpha, dfn=df_between, dfd=df_within)

    power = float(1.0 - stats.ncf.cdf(f_crit, dfn=df_between, dfd=df_within, nc=nc_lambda))
    return float(np.clip(power, 0.0, 1.0))
