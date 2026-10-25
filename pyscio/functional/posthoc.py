"""Pairwise post-hoc tests: Tukey HSD, Games-Howell, and Dunn's test."""

from __future__ import annotations

import itertools
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

    group_labels = names if names and len(names) == k else [f"Group_{i+1}" for i in range(k)]
    res = stats.tukey_hsd(*cleaned)

    records = []
    pairs = list(itertools.combinations(range(k), 2))
    for i, j in pairs:
        diff = float(np.mean(cleaned[i]) - np.mean(cleaned[j]))
        p_val = float(res.pvalue[i, j])
        # confidence interval at 1 - alpha
        ci = res.confidence_interval(confidence_level=1.0 - alpha)
        ci_low = float(ci.low[i, j])
        ci_high = float(ci.high[i, j])

        records.append({
            "group1": group_labels[i],
            "group2": group_labels[j],
            "diff": diff,
            "p_adj": p_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reject": p_val < alpha,
        })

    return pd.DataFrame(records)


def games_howell(
    *groups: np.ndarray | list[float],
    names: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Games-Howell post-hoc test for unequal variances and sample sizes.

    Uses Welch-Satterthwaite degrees of freedom and Studentized Range distribution.
    """
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 2:
        raise ValueError("Games-Howell requires at least 2 groups.")

    group_labels = names if names and len(names) == k else [f"Group_{i+1}" for i in range(k)]

    means = [float(np.mean(g)) for g in cleaned]
    vars_ = [float(np.var(g, ddof=1)) for g in cleaned]
    ns = [len(g) for g in cleaned]

    records = []
    pairs = list(itertools.combinations(range(k), 2))
    for i, j in pairs:
        m_diff = means[i] - means[j]
        se2 = (vars_[i] / ns[i]) + (vars_[j] / ns[j])
        se = math.sqrt(se2)

        # Welch-Satterthwaite df
        denom = ((vars_[i] / ns[i]) ** 2 / (ns[i] - 1)) + ((vars_[j] / ns[j]) ** 2 / (ns[j] - 1))
        df = (se2 ** 2) / denom if denom > 0 else 1.0

        # Studentized range statistic q = t * sqrt(2)
        t = abs(m_diff) / se if se > 0 else 0.0
        q = t * math.sqrt(2.0)
        p_val = float(stats.studentized_range.sf(q, k, df))

        # Critical q for CI
        q_crit = float(stats.studentized_range.ppf(1.0 - alpha, k, df))
        margin = (q_crit / math.sqrt(2.0)) * se
        ci_low = m_diff - margin
        ci_high = m_diff + margin

        records.append({
            "group1": group_labels[i],
            "group2": group_labels[j],
            "diff": m_diff,
            "p_adj": p_val,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reject": p_val < alpha,
        })

    return pd.DataFrame(records)


def dunn(
    *groups: np.ndarray | list[float],
    names: list[str] | None = None,
    p_adjust: str = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Dunn's non-parametric pairwise test following Kruskal-Wallis with multiple testing adjustment."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 2:
        raise ValueError("Dunn test requires at least 2 groups.")

    group_labels = names if names and len(names) == k else [f"Group_{i+1}" for i in range(k)]

    # Pool and compute ranks
    all_data = np.concatenate(cleaned)
    ranks = stats.rankdata(all_data)

    ns = [len(g) for g in cleaned]
    n_total = sum(ns)

    # Compute mean ranks
    mean_ranks = []
    idx = 0
    for n in ns:
        mean_ranks.append(float(np.mean(ranks[idx : idx + n])))
        idx += n

    # Tie correction
    _, counts = np.unique(ranks, return_counts=True)
    ties_sum = sum(c**3 - c for c in counts if c > 1)
    tie_corr = 1.0 - (ties_sum / (n_total**3 - n_total)) if n_total > 1 else 1.0

    pairs = list(itertools.combinations(range(k), 2))
    raw_p_values = []
    diffs = []

    for i, j in pairs:
        r_diff = mean_ranks[i] - mean_ranks[j]
        se = math.sqrt((n_total * (n_total + 1) / 12.0) * tie_corr * (1.0 / ns[i] + 1.0 / ns[j]))
        z = abs(r_diff) / se if se > 0 else 0.0
        p = 2.0 * float(stats.norm.sf(z))
        diffs.append(r_diff)
        raw_p_values.append(p)

    # Adjust p-values
    from pyscio.functional.adjust import fdr_bh, holm

    if p_adjust == "fdr_bh":
        _, p_adjs = fdr_bh(raw_p_values, alpha=alpha)
    else:
        _, p_adjs = holm(raw_p_values, alpha=alpha)

    records = []
    for idx_pair, (i, j) in enumerate(pairs):
        records.append({
            "group1": group_labels[i],
            "group2": group_labels[j],
            "rank_diff": diffs[idx_pair],
            "p_raw": raw_p_values[idx_pair],
            "p_adj": float(p_adjs[idx_pair]),
            "reject": p_adjs[idx_pair] < alpha,
        })

    return pd.DataFrame(records)
