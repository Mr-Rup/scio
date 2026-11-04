"""One-Way Repeated-Measures ANOVA kernel with Greenhouse-Geisser sphericity adjustment."""

from __future__ import annotations

import numpy as np
from scipy import stats

from omnistats.core.errors import SampleSizeError, StatError, ZeroVarError
from omnistats.core.guards import clean_1d
from omnistats.core.types import RMAnovaResult


def rm_anova(*groups: np.ndarray | list[float]) -> RMAnovaResult:
    """Compute One-Way Repeated-Measures ANOVA with Greenhouse-Geisser sphericity correction.

    Parameters
    ----------
    *groups : array-like
        K >= 3 matched 1D arrays of identical length N (subjects).

    Returns
    -------
    RMAnovaResult
        NamedTuple with F statistic, unadjusted p, Greenhouse-Geisser p_gg,
        sphericity epsilon eps_gg, df_num, df_den, and partial_eta_sq.
    """
    cleaned = [clean_1d(g, name=f"condition_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 3:
        raise StatError("Repeated-measures ANOVA requires at least 3 matched conditions.")

    n = len(cleaned[0])
    if any(len(g) != n for g in cleaned):
        raise SampleSizeError(
            "All repeated-measures conditions must have identical sample lengths."
        )
    if n < 3:
        raise SampleSizeError("Repeated-measures ANOVA requires at least 3 subjects.")

    # Y matrix of shape (N, K): rows = subjects, cols = conditions
    y_mat = np.column_stack(cleaned)
    grand_mean = float(np.mean(y_mat))
    subj_means = np.mean(y_mat, axis=1, keepdims=True)
    cond_means = np.mean(y_mat, axis=0, keepdims=True)

    ss_cond = float(n * np.sum((cond_means - grand_mean) ** 2))
    resid_mat = y_mat - subj_means - cond_means + grand_mean
    ss_error = float(np.sum(resid_mat**2))

    df_num = float(k - 1)
    df_den = float((n - 1) * (k - 1))

    if ss_error <= 1e-15:
        raise ZeroVarError("Residual error variance across repeated measures is zero.")

    ms_cond = ss_cond / df_num
    ms_error = ss_error / df_den
    f_stat = float(ms_cond / ms_error)
    p_unadj = float(stats.f.sf(f_stat, dfn=df_num, dfd=df_den))
    partial_eta_sq = float(np.clip(ss_cond / (ss_cond + ss_error), 0.0, 1.0))

    # Sample covariance across conditions (K x K) and double-centered sphericity matrix S*
    cov_k = np.cov(y_mat, rowvar=False, ddof=1)
    row_means = np.mean(cov_k, axis=1, keepdims=True)
    col_means = np.mean(cov_k, axis=0, keepdims=True)
    cov_grand = float(np.mean(cov_k))
    s_star = cov_k - row_means - col_means + cov_grand

    tr_s = float(np.trace(s_star))
    tr_s2 = float(np.sum(s_star**2))
    lower_bound = 1.0 / (k - 1.0)
    if tr_s2 > 1e-15:
        eps_raw = (tr_s**2) / ((k - 1.0) * tr_s2)
        eps_gg = float(np.clip(eps_raw, lower_bound, 1.0))
    else:
        eps_gg = 1.0

    p_gg = float(stats.f.sf(f_stat, dfn=eps_gg * df_num, dfd=eps_gg * df_den))

    return RMAnovaResult(
        stat=f_stat,
        p=p_unadj,
        p_gg=p_gg,
        eps_gg=eps_gg,
        df_num=df_num,
        df_den=df_den,
        partial_eta_sq=partial_eta_sq,
    )
