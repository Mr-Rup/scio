"""Linear Mixed-Effects Models (LMM) via Profiled REML / ML with QR whitening."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from omnistats.core.dataframe import ensure_pandas_df
from omnistats.core.errors import MissingDataError, StatError
from omnistats.core.guards import clamp_p
from omnistats.core.results import RegressionResult
from omnistats.models.formula import parse_formula

_RE_RANDOM_INTERCEPT = re.compile(r"\(\s*1\s*\|\s*([A-Za-z0-9_]+)\s*\)")


def _extract_random_group(formula: str, groups: str | None) -> tuple[str, str]:
    """Extract fixed-effects formula and grouping variable name from formula or `groups` arg."""
    match = _RE_RANDOM_INTERCEPT.search(formula)
    if match:
        group_col = match.group(1).strip()
        cleaned_rhs = _RE_RANDOM_INTERCEPT.sub("", formula)
        lhs_raw, rhs_raw = cleaned_rhs.split("~", 1)
        tokens = [t.strip() for t in rhs_raw.split("+") if t.strip()]
        fixed_formula = f"{lhs_raw.strip()} ~ {' + '.join(tokens) if tokens else '1'}"
        return fixed_formula, group_col
    if groups is not None and groups.strip():
        return formula, groups.strip()
    raise StatError(
        "Mixed model requires either a random-intercept term in the formula "
        "(e.g. 'y ~ x + (1 | subject)') or the `groups='subject'` argument."
    )


def mixedlm(
    df: pd.DataFrame | Any,
    formula: str,
    groups: str | None = None,
    reml: bool = True,
    alpha: float = 0.05,
) -> RegressionResult:
    """Fit a Linear Mixed-Effects Model (Random Intercepts) via exact Profiled REML or ML.

    Supports Wilkinson random-effect syntax ``"y ~ x1 + x2 + (1 | cluster)"`` or an explicit
    ``groups="cluster"`` parameter. Computes fixed-effect estimates, Satterthwaite degrees of
    freedom, Intraclass Correlation Coefficient (ICC), and Nakagawa's marginal ($R^2_m$) and
    conditional ($R^2_c$) $R^2$.
    """
    pdf = ensure_pandas_df(df)
    fixed_formula, group_col = _extract_random_group(formula, groups)

    if group_col not in pdf.columns:
        raise KeyError(f"Grouping variable '{group_col}' not found in DataFrame.")

    from omnistats.models.formula import _clean_var_name, extract_formula_vars

    lhs, rhs_terms = extract_formula_vars(fixed_formula)
    if lhs not in pdf.columns:
        raise KeyError(f"Dependent variable '{lhs}' not found in DataFrame.")
    ref_cols = [lhs, group_col]
    for term in rhs_terms:
        if term in ("1", "0", "-1"):
            continue
        for sp in (p.strip() for p in term.split(":")):
            cname, _ = _clean_var_name(sp)
            if cname not in pdf.columns:
                raise KeyError(f"Predictor variable '{cname}' not found in DataFrame.")
            if cname not in ref_cols:
                ref_cols.append(cname)

    clean_df = pdf[ref_cols].dropna().reset_index(drop=True)
    lhs, y, design_names, x_mat = parse_formula(clean_df, fixed_formula)
    group_codes, group_uniques = pd.factorize(clean_df[group_col], sort=True)
    n_clusters = len(group_uniques)
    n, p = x_mat.shape

    if n_clusters < 2:
        raise StatError(
            f"Grouping column '{group_col}' must have at least 2 clusters (got {n_clusters})."
        )
    if n <= p + 1:
        raise MissingDataError(f"Insufficient observations (N={n}) for {p} fixed parameters.")

    cluster_indices = [np.flatnonzero(group_codes == c) for c in range(n_clusters)]
    cluster_sizes = np.array([len(idx) for idx in cluster_indices], dtype=np.float64)

    def _whiten_and_qr(theta: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        x_w = x_mat.copy()
        y_w = y.copy()
        for idx in cluster_indices:
            ni = float(len(idx))
            shrink = 1.0 - 1.0 / np.sqrt(1.0 + ni * theta)
            x_w[idx] -= shrink * np.mean(x_w[idx], axis=0)
            y_w[idx] -= shrink * float(np.mean(y_w[idx]))
        q_mat, r_mat = np.linalg.qr(x_w, mode="reduced")
        beta = np.linalg.solve(r_mat, q_mat.T @ y_w)
        resid_w = y_w - x_w @ beta
        rss_w = max(float(np.sum(resid_w**2)), 1e-15)
        log_det_xvx = float(2.0 * np.sum(np.log(np.maximum(np.abs(np.diag(r_mat)), 1e-15))))
        return beta, r_mat, rss_w, log_det_xvx

    def _neg2_loglik(log_theta: float) -> float:
        theta = float(np.exp(log_theta))
        _, _, rss_w, log_det_xvx = _whiten_and_qr(theta)
        log_det_v = float(np.sum(np.log(1.0 + cluster_sizes * theta)))
        if reml:
            df_eff = float(n - p)
            sig2 = rss_w / df_eff
            return float(df_eff * np.log(2.0 * np.pi * sig2) + df_eff + log_det_v + log_det_xvx)
        sig2 = rss_w / float(n)
        return float(n * np.log(2.0 * np.pi * sig2) + n + log_det_v)

    opt = optimize.minimize_scalar(_neg2_loglik, bounds=(-18.0, 10.0), method="bounded")
    theta_hat = float(np.exp(opt.x))
    if _neg2_loglik(-25.0) <= opt.fun:
        theta_hat = 0.0

    beta, r_mat, rss_w, _ = _whiten_and_qr(theta_hat)
    df_eff = float(n - p) if reml else float(n)
    sigma2_eps = rss_w / df_eff
    sigma2_u = theta_hat * sigma2_eps
    icc = float(sigma2_u / (sigma2_u + sigma2_eps)) if (sigma2_u + sigma2_eps) > 0 else 0.0

    r_inv = np.linalg.inv(r_mat)
    cov_beta = sigma2_eps * (r_inv @ r_inv.T)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-15))
    t_stats = beta / se

    # Satterthwaite-style effective degrees of freedom (between M-1 and N-p)
    df_satt = max(float(n_clusters - 1), float(n - p) * (1.0 - 0.5 * icc))
    p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=df_satt)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_satt))
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se

    # Nakagawa & Schielzeth (2013) Marginal and Conditional R^2
    var_fixed = float(np.var(x_mat @ beta, ddof=1))
    var_total = var_fixed + sigma2_u + sigma2_eps
    r2_marginal = float(var_fixed / var_total) if var_total > 0 else 0.0
    r2_conditional = float((var_fixed + sigma2_u) / var_total) if var_total > 0 else 0.0

    # Omnibus Wald F-test on non-intercept fixed effects
    if p > 1:
        f_stat = float(np.mean(t_stats[1:] ** 2))
        f_p = float(stats.f.sf(f_stat, dfn=p - 1, dfd=df_satt))
    else:
        f_stat = float(t_stats[0] ** 2)
        f_p = float(p_vals[0])

    neg2ll = _neg2_loglik(float(np.log(max(theta_hat, 1e-12))))
    k_params = p + 2
    aic = float(neg2ll + 2.0 * k_params)
    bic = float(neg2ll + np.log(n) * k_params)

    coef_df = pd.DataFrame(
        {
            "term": design_names,
            "coef": beta,
            "std_err": se,
            "t": t_stats,
            "p_value": p_vals,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
        }
    )

    warns = [
        f"Random Intercept ({group_col}, M={n_clusters} clusters): "
        f"σ²_u = {sigma2_u:.4g}, σ²_e = {sigma2_eps:.4g}, ICC = {icc:.3f} "
        f"(Marginal R² = {r2_marginal:.3f}, Conditional R² = {r2_conditional:.3f})."
    ]
    p_val, p_disp = clamp_p(f_p)

    return RegressionResult(
        model_type=f"Linear Mixed Model ({'REML' if reml else 'ML'})",
        formula=f"{fixed_formula} + (1 | {group_col})",
        y_name=lhs,
        n_obs=n,
        df_model=max(1, p - 1),
        df_resid=round(df_satt),
        stat=f_stat,
        p=p_val,
        p_display=p_disp,
        r_squared=r2_marginal,
        adj_r_squared=r2_conditional,
        aic=aic,
        bic=bic,
        coef_df=coef_df,
        robust_se="REML-GLS" if reml else "ML-GLS",
        alpha=alpha,
        warns=warns,
    )
