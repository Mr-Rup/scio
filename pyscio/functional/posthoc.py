"""Pairwise post-hoc tests: Tensor-broadcasted Tukey HSD, Games-Howell, and Dunn's test."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from pyscio.core.guards import clean_1d


def tukey(
    *groups: np.ndarray | list[float],
    names: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Tukey's Honest Significant Difference (HSD) pairwise test for equal variance ANOVA.

    Returns a DataFrame with columns: group1, group2, diff, p_adj, ci_low, ci_high, reject.
    """
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 2:
        raise ValueError("Tukey HSD requires at least 2 groups.")

    group_labels = np.asarray(
        names if names and len(names) == k else [f"Group_{i + 1}" for i in range(k)],
        dtype=object,
    )
    res = stats.tukey_hsd(*cleaned)
    ci = res.confidence_interval(confidence_level=1.0 - alpha)

    means = np.array([np.mean(g) for g in cleaned], dtype=np.float64)
    idx_i, idx_j = np.triu_indices(k, k=1)

    diffs = means[idx_i] - means[idx_j]
    p_vals = np.asarray(res.pvalue[idx_i, idx_j], dtype=np.float64)
    ci_lows = np.asarray(ci.low[idx_i, idx_j], dtype=np.float64)
    ci_highs = np.asarray(ci.high[idx_i, idx_j], dtype=np.float64)

    return pd.DataFrame(
        {
            "group1": group_labels[idx_i],
            "group2": group_labels[idx_j],
            "diff": diffs,
            "p_adj": p_vals,
            "ci_low": ci_lows,
            "ci_high": ci_highs,
            "reject": p_vals < alpha,
        }
    )


def games_howell(
    *groups: np.ndarray | list[float],
    names: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Vectorized Games-Howell post-hoc test for unequal variances and sample sizes.

    Uses tensor-broadcasted Welch-Satterthwaite degrees of freedom and Studentized Range distribution.
    """
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 2:
        raise ValueError("Games-Howell requires at least 2 groups.")

    group_labels = np.asarray(
        names if names and len(names) == k else [f"Group_{i + 1}" for i in range(k)],
        dtype=object,
    )

    means = np.array([np.mean(g) for g in cleaned], dtype=np.float64)
    vars_ = np.array([np.var(g, ddof=1) for g in cleaned], dtype=np.float64)
    ns = np.array([len(g) for g in cleaned], dtype=np.float64)
    vn = vars_ / ns

    # Upper-triangle pairwise broadcasting over all K*(K-1)/2 pairs
    idx_i, idx_j = np.triu_indices(k, k=1)

    m_diff = means[idx_i] - means[idx_j]
    se2 = vn[idx_i] + vn[idx_j]
    se = np.sqrt(se2)

    # Vectorized Welch-Satterthwaite degrees of freedom
    denom = (vn[idx_i] ** 2) / (ns[idx_i] - 1.0) + (vn[idx_j] ** 2) / (ns[idx_j] - 1.0)
    df = np.where(denom > 0.0, (se2**2) / denom, 1.0)

    # Vectorized Studentized range statistic q = t * sqrt(2)
    t_stat = np.where(se > 0.0, np.abs(m_diff) / se, 0.0)
    q_stat = t_stat * math.sqrt(2.0)

    p_vals = np.asarray(stats.studentized_range.sf(q_stat, k, df), dtype=np.float64)
    q_crit = np.asarray(stats.studentized_range.ppf(1.0 - alpha, k, df), dtype=np.float64)

    margin = (q_crit / math.sqrt(2.0)) * se
    ci_low = m_diff - margin
    ci_high = m_diff + margin

    return pd.DataFrame(
        {
            "group1": group_labels[idx_i],
            "group2": group_labels[idx_j],
            "diff": m_diff,
            "p_adj": p_vals,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reject": p_vals < alpha,
        }
    )


def dunn(
    *groups: np.ndarray | list[float],
    names: list[str] | None = None,
    p_adjust: str = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Vectorized Dunn's non-parametric pairwise test following Kruskal-Wallis."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 2:
        raise ValueError("Dunn test requires at least 2 groups.")

    group_labels = np.asarray(
        names if names and len(names) == k else [f"Group_{i + 1}" for i in range(k)],
        dtype=object,
    )

    # Pool and compute global ranks
    all_data = np.concatenate(cleaned)
    ranks = stats.rankdata(all_data)

    ns = np.array([len(g) for g in cleaned], dtype=np.float64)
    n_total = float(np.sum(ns))

    # Vectorized segment mean ranks via reduceat
    offsets = np.zeros(k, dtype=np.intp)
    offsets[1:] = np.cumsum([len(g) for g in cleaned[:-1]])
    rank_sums = np.add.reduceat(ranks, offsets)
    mean_ranks = rank_sums / ns

    # Vectorized tie correction
    _, counts = np.unique(ranks, return_counts=True)
    tied = counts[counts > 1].astype(np.float64)
    ties_sum = float(np.sum(tied**3 - tied)) if len(tied) > 0 else 0.0
    tie_corr = 1.0 - (ties_sum / (n_total**3 - n_total)) if n_total > 1.0 else 1.0

    # Upper-triangle pairwise broadcasting
    idx_i, idx_j = np.triu_indices(k, k=1)
    r_diff = mean_ranks[idx_i] - mean_ranks[idx_j]
    base_var = (n_total * (n_total + 1.0) / 12.0) * tie_corr
    se = np.sqrt(base_var * (1.0 / ns[idx_i] + 1.0 / ns[idx_j]))
    z = np.where(se > 0.0, np.abs(r_diff) / se, 0.0)
    raw_p_values = 2.0 * np.asarray(stats.norm.sf(z), dtype=np.float64)

    # Multiple testing adjustment
    from pyscio.functional.adjust import bonferroni, fdr_bh, holm

    if p_adjust == "fdr_bh":
        _, p_adjs = fdr_bh(raw_p_values, alpha=alpha)
    elif p_adjust == "bonferroni":
        _, p_adjs = bonferroni(raw_p_values, alpha=alpha)
    else:
        _, p_adjs = holm(raw_p_values, alpha=alpha)

    return pd.DataFrame(
        {
            "group1": group_labels[idx_i],
            "group2": group_labels[idx_j],
            "rank_diff": r_diff,
            "p_raw": raw_p_values,
            "p_adj": p_adjs.astype(np.float64),
            "reject": p_adjs < alpha,
        }
    )
