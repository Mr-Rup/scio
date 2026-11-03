"""Ordinary Least Squares (OLS) regression with HC0-HC3 heteroscedasticity-robust SEs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from pyscio.core.errors import StatError
from pyscio.core.guards import clamp_p
from pyscio.core.results import RegressionResult
from pyscio.models.formula import parse_formula


def ols(
    df: pd.DataFrame,
    formula: str,
    robust_se: str | None = "HC3",
    alpha: float = 0.05,
) -> RegressionResult:
    """Fit Ordinary Least Squares (OLS) regression with MacKinnon-White HC3 robust standard errors."""
    y_name, y, term_names, x_mat = parse_formula(df, formula)
    n, p = x_mat.shape
    if n <= p:
        raise StatError(f"Insufficient observations (N={n}) to estimate {p} regression parameters.")

    df_resid = float(n - p)
    has_intercept = "Intercept" in term_names
    df_model = float(p - 1 if has_intercept else p)

    beta, _, rank, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    if rank < p:
        raise StatError(
            f"Design matrix is rank-deficient (rank={rank} < {p} columns); check for collinearity."
        )

    y_hat = x_mat @ beta
    resid = y - y_hat

    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y)) ** 2)) if has_intercept else float(np.sum(y**2))

    r2 = float(np.clip(1.0 - rss / tss, 0.0, 1.0)) if tss > 0 else 0.0
    adj_r2 = (
        float(1.0 - (1.0 - r2) * ((n - (1 if has_intercept else 0)) / df_resid))
        if df_resid > 0
        else 0.0
    )

    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    sigma2 = rss / df_resid

    se_mode = robust_se.upper() if robust_se is not None else "NONROBUST"
    if se_mode in ("HC0", "HC1", "HC2", "HC3"):
        h_diag = np.einsum("ij,ij->i", x_mat @ xtx_inv, x_mat)
        if se_mode == "HC3":
            omega = (resid**2) / (np.maximum(1.0 - h_diag, 1e-6) ** 2)
        elif se_mode == "HC2":
            omega = (resid**2) / np.maximum(1.0 - h_diag, 1e-6)
        elif se_mode == "HC1":
            omega = (n / df_resid) * (resid**2)
        else:
            omega = resid**2
        meat = x_mat.T @ (x_mat * omega[:, None])
        cov_beta = xtx_inv @ meat @ xtx_inv
        model_label = f"OLS Regression ({se_mode} Robust SE)"
    else:
        cov_beta = sigma2 * xtx_inv
        model_label = "OLS Regression (Classical SE)"

    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 1e-30))
    t_stats = beta / se_beta
    p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=df_resid)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_resid))
    ci_low = beta - t_crit * se_beta
    ci_high = beta + t_crit * se_beta

    if df_model > 0 and rss > 0:
        ess = max(0.0, tss - rss)
        f_stat = float((ess / df_model) / (rss / df_resid))
        f_p_raw = float(stats.f.sf(f_stat, df_model, df_resid))
    else:
        f_stat = 0.0
        f_p_raw = 1.0
    f_p, f_p_disp = clamp_p(f_p_raw)

    sigma2_mle = max(rss / n, 1e-30)
    ll = -0.5 * n * (math.log(2.0 * math.pi) + math.log(sigma2_mle) + 1.0)
    k_params = p + 1
    aic = float(2.0 * k_params - 2.0 * ll)
    bic = float(math.log(n) * k_params - 2.0 * ll)

    coef_df = pd.DataFrame(
        {
            "term": term_names,
            "coef": beta.astype(float),
            "std_err": se_beta.astype(float),
            "t_stat": t_stats.astype(float),
            "p_value": p_vals.astype(float),
            "ci_low": ci_low.astype(float),
            "ci_high": ci_high.astype(float),
        }
    )

    return RegressionResult(
        model_type=model_label,
        formula=formula,
        y_name=y_name,
        n_obs=n,
        df_model=int(df_model),
        df_resid=int(df_resid),
        stat=f_stat,
        p=f_p,
        p_display=f_p_disp,
        r_squared=r2,
        adj_r_squared=adj_r2,
        aic=aic,
        bic=bic,
        coef_df=coef_df,
        robust_se=robust_se if robust_se is not None else "Non-Robust",
        alpha=alpha,
    )
