"""Ordinary Least Squares (OLS) regression with QR decomposition and HC0-HC3 robust SEs."""

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
    robust_se: str | bool | None = "HC3",
    alpha: float = 0.05,
    hac_lags: int | None = None,
) -> RegressionResult:
    """Fit Ordinary Least Squares (OLS) regression via QR decomposition with HC0-HC3 or Newey-West HAC SEs."""
    y_name, y, term_names, x_mat = parse_formula(df, formula)
    n, p = x_mat.shape
    if n <= p:
        raise StatError(f"Insufficient observations (N={n}) to estimate {p} regression parameters.")

    df_resid = float(n - p)
    has_intercept = "Intercept" in term_names
    df_model = float(p - 1 if has_intercept else p)

    # Reduced QR decomposition: X = Q R, where Q is (n, p) orthonormal and R is (p, p) upper triangular
    q_mat, r_mat = np.linalg.qr(x_mat, mode="reduced")
    diag_r = np.abs(np.diag(r_mat))
    tol = max(n, p) * np.finfo(np.float64).eps * (float(np.max(diag_r)) if len(diag_r) > 0 else 1.0)
    rank = int(np.sum(diag_r > tol))
    if rank < p:
        raise StatError(
            f"Design matrix is rank-deficient (rank={rank} < {p} columns); check for collinearity."
        )

    r_inv = np.linalg.inv(r_mat)
    beta = r_inv @ (q_mat.T @ y)
    xtx_inv = r_inv @ r_inv.T

    y_hat = q_mat @ (q_mat.T @ y)
    resid = y - y_hat

    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y)) ** 2)) if has_intercept else float(np.sum(y**2))

    r2 = float(np.clip(1.0 - rss / tss, 0.0, 1.0)) if tss > 0 else 0.0
    adj_r2 = (
        float(1.0 - (1.0 - r2) * ((n - (1 if has_intercept else 0)) / df_resid))
        if df_resid > 0
        else 0.0
    )

    sigma2 = rss / df_resid

    if isinstance(robust_se, bool):
        se_mode = "HC3" if robust_se else "NONROBUST"
    elif robust_se is None:
        se_mode = "NONROBUST"
    else:
        se_mode = robust_se.upper()

    if se_mode in ("HC0", "HC1", "HC2", "HC3"):
        # Leverage h_ii = sum_j Q_{ij}^2 in O(N p) without materializing N x N hat matrix
        h_diag = np.sum(q_mat**2, axis=1)
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
        se_label = se_mode
    elif se_mode in ("HAC", "NEWEY-WEST", "NW"):
        max_l = (
            max(1, int(hac_lags))
            if hac_lags is not None
            else max(1, int(4.0 * ((n / 100.0) ** (2.0 / 9.0))))
        )
        max_l = min(max_l, n - 2)
        u_mat = x_mat * resid[:, None]
        meat = u_mat.T @ u_mat
        for j in range(1, max_l + 1):
            wj = 1.0 - j / float(max_l + 1)
            gamma_j = u_mat[j:].T @ u_mat[:-j]
            meat = meat + wj * (gamma_j + gamma_j.T)
        meat = (n / df_resid) * meat
        cov_beta = xtx_inv @ meat @ xtx_inv
        model_label = f"OLS Regression (Newey-West HAC SE, L={max_l})"
        se_label = f"HAC(L={max_l})"
    else:
        cov_beta = sigma2 * xtx_inv
        model_label = "OLS Regression (Classical SE)"
        se_label = "Non-Robust"

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
        robust_se=se_label,
        alpha=alpha,
    )
