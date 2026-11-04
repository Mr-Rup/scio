"""Count and Overdispersed Generalized Linear Models (Poisson IRLS with QMLE HC3 SEs)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import special, stats

from pyscio.core.errors import StatError
from pyscio.core.guards import clamp_p
from pyscio.core.results import RegressionResult
from pyscio.models.formula import parse_formula


def poisson(
    df: pd.DataFrame,
    formula: str,
    *,
    robust_se: str | bool = "HC3",
    alpha: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> RegressionResult:
    """Fit Poisson log-linear GLM via Newton-Raphson IRLS with QMLE robust sandwich standard errors.

    Automatically audits Pearson dispersion ($\\hat{\\phi} = \\chi^2_P / df_{\\text{resid}}$) and reports
    Incidence Rate Ratios ($\\text{IRR} = \\exp(\\hat{\\beta})$).
    """
    y_name, y, term_names, x_mat = parse_formula(df, formula)
    if np.any(y < 0) or not np.all(np.isclose(y, np.round(y))):
        raise StatError(
            f"Poisson regression outcome '{y_name}' must contain non-negative integer counts."
        )

    n, p = x_mat.shape
    if n <= p:
        raise StatError(f"Insufficient observations (N={n}) to estimate {p} Poisson parameters.")

    df_resid = int(n - p)
    has_intercept = "Intercept" in term_names
    df_model = int(p - 1 if has_intercept else p)

    # Initialize beta so mu_0 = max(y, 0.5)
    beta = np.zeros(p, dtype=np.float64)
    if has_intercept:
        beta[0] = math.log(max(float(np.mean(y)), 1e-3))

    converged = False
    warns: list[str] = []

    for _ in range(max_iter):
        eta = np.clip(x_mat @ beta, -20.0, 20.0)
        mu = np.exp(eta)
        w = np.maximum(mu, 1e-10)
        z = eta + (y - mu) / w

        wx = x_mat * w[:, None]
        fisher_info = x_mat.T @ wx
        rhs = x_mat.T @ (w * z)
        try:
            beta_new = np.linalg.solve(fisher_info, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(fisher_info) @ rhs

        if float(np.max(np.abs(beta_new - beta))) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    if not converged:
        warns.append(f"Poisson IRLS did not fully converge within {max_iter} iterations.")

    eta = np.clip(x_mat @ beta, -20.0, 20.0)
    mu = np.maximum(np.exp(eta), 1e-12)
    resid_raw = y - mu

    # Log-likelihood: sum(y * eta - mu - gammaln(y + 1))
    ll_model = float(np.sum(y * eta - mu - special.gammaln(y + 1.0)))
    mu_null = max(float(np.mean(y)), 1e-12)
    ll_null = float(np.sum(y * math.log(mu_null) - mu_null - special.gammaln(y + 1.0)))

    pseudo_r2 = float(np.clip(1.0 - (ll_model / ll_null), 0.0, 1.0)) if ll_null < 0 else 0.0
    adj_r2 = float(1.0 - ((ll_model - p) / ll_null)) if ll_null < 0 else 0.0

    lr_stat = float(max(0.0, 2.0 * (ll_model - ll_null))) if df_model > 0 else 0.0
    lr_p_raw = float(stats.chi2.sf(lr_stat, df=df_model)) if df_model > 0 else 1.0
    lr_p, lr_p_disp = clamp_p(lr_p_raw)

    aic = float(2.0 * p - 2.0 * ll_model)
    bic = float(math.log(n) * p - 2.0 * ll_model)

    # Pearson dispersion diagnostic
    pearson_chi2 = float(np.sum((resid_raw**2) / mu))
    dispersion = pearson_chi2 / max(df_resid, 1)
    if dispersion > 1.5:
        warns.append(
            f"Overdispersion detected (Pearson dispersion ratio = {dispersion:.2f} > 1.5). "
            "QMLE robust sandwich standard errors are active to protect inference."
        )

    bread = np.linalg.pinv(x_mat.T @ (x_mat * mu[:, None]))
    se_mode = (
        ("HC3" if robust_se else "NONROBUST") if isinstance(robust_se, bool) else robust_se.upper()
    )

    if se_mode in ("HC0", "HC1", "HC2", "HC3"):
        sqrt_w = np.sqrt(mu)
        wx_half = x_mat * sqrt_w[:, None]
        h_diag = np.einsum("ij,ij->i", wx_half @ bread, wx_half)
        if se_mode == "HC3":
            omega = (resid_raw**2) / (np.maximum(1.0 - h_diag, 1e-6) ** 2)
        elif se_mode == "HC2":
            omega = (resid_raw**2) / np.maximum(1.0 - h_diag, 1e-6)
        elif se_mode == "HC1":
            omega = (n / max(df_resid, 1)) * (resid_raw**2)
        else:
            omega = resid_raw**2
        meat = x_mat.T @ (x_mat * omega[:, None])
        cov_beta = bread @ meat @ bread
        se_label = se_mode
    else:
        cov_beta = bread
        se_label = "Non-Robust (MLE)"

    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 1e-30))
    z_stats = beta / se_beta
    p_vals = 2.0 * stats.norm.sf(np.abs(z_stats))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_low = beta - z_crit * se_beta
    ci_high = beta + z_crit * se_beta
    irr = np.exp(np.clip(beta, -50.0, 50.0))

    coef_df = pd.DataFrame(
        {
            "term": term_names,
            "coef": beta.astype(float),
            "std_err": se_beta.astype(float),
            "z_stat": z_stats.astype(float),
            "p_value": p_vals.astype(float),
            "ci_low": ci_low.astype(float),
            "ci_high": ci_high.astype(float),
            "irr": irr.astype(float),
        }
    )

    return RegressionResult(
        model_type="Poisson Log-Linear GLM",
        formula=formula,
        y_name=y_name,
        n_obs=n,
        df_model=df_model,
        df_resid=df_resid,
        stat=lr_stat,
        p=lr_p,
        p_display=lr_p_disp,
        r_squared=pseudo_r2,
        adj_r_squared=adj_r2,
        aic=aic,
        bic=bic,
        coef_df=coef_df,
        robust_se=se_label,
        alpha=alpha,
        warns=warns,
    )
