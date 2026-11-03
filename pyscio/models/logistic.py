"""Binary Logistic Regression via Newton-Raphson Iteratively Reweighted Least Squares (IRLS)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from pyscio.core.errors import StatError
from pyscio.core.guards import clamp_p
from pyscio.core.results import RegressionResult
from pyscio.models.formula import parse_formula


def logistic(
    df: pd.DataFrame,
    formula: str,
    alpha: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> RegressionResult:
    """Fit binary Logistic Regression via Newton-Raphson IRLS."""
    y_name, y, term_names, x_mat = parse_formula(df, formula)
    unique_y = np.unique(y)
    if len(unique_y) != 2:
        raise StatError(
            f"Logistic regression requires a binary outcome variable (got {len(unique_y)} distinct values)."
        )
    if not np.array_equal(unique_y, np.array([0.0, 1.0])):
        y = (y == unique_y[1]).astype(np.float64)

    n, p = x_mat.shape
    if n <= p:
        raise StatError(
            f"Insufficient observations (N={n}) to estimate {p} logistic regression parameters."
        )

    beta = np.zeros(p, dtype=np.float64)
    warns: list[str] = []
    converged = False

    for _ in range(max_iter):
        eta = np.clip(x_mat @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-8)

        grad = x_mat.T @ (y - mu)
        xtwx = x_mat.T @ (x_mat * w[:, None])
        try:
            step = np.linalg.solve(xtwx, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(xtwx) @ grad

        beta_new = beta + step
        if float(np.max(np.abs(step))) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    if not converged:
        warns.append(
            f"IRLS did not converge within {max_iter} iterations; check for quasi-complete separation."
        )

    eta = np.clip(x_mat @ beta, -30.0, 30.0)
    mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    w = np.maximum(mu * (1.0 - mu), 1e-8)
    cov_beta = np.linalg.pinv(x_mat.T @ (x_mat * w[:, None]))

    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 1e-30))
    z_stats = beta / se_beta
    p_vals = 2.0 * stats.norm.sf(np.abs(z_stats))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_low = beta - z_crit * se_beta
    ci_high = beta + z_crit * se_beta
    or_vals = np.exp(np.clip(beta, -50.0, 50.0))

    ll_full = float(np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)))
    p_bar = float(np.clip(np.mean(y), 1e-12, 1.0 - 1e-12))
    ll_null = float(n * (p_bar * math.log(p_bar) + (1.0 - p_bar) * math.log(1.0 - p_bar)))

    pseudo_r2 = float(np.clip(1.0 - (ll_full / ll_null), 0.0, 1.0)) if ll_null < 0 else 0.0
    has_intercept = "Intercept" in term_names
    df_model = float(p - 1 if has_intercept else p)
    df_resid = float(n - p)

    lrt_stat = max(0.0, 2.0 * (ll_full - ll_null))
    lrt_p_raw = float(stats.chi2.sf(lrt_stat, df=df_model)) if df_model > 0 else 1.0
    lrt_p, lrt_p_disp = clamp_p(lrt_p_raw)

    aic = float(2.0 * p - 2.0 * ll_full)
    bic = float(math.log(n) * p - 2.0 * ll_full)

    coef_df = pd.DataFrame(
        {
            "term": term_names,
            "coef": beta.astype(float),
            "or_val": or_vals.astype(float),
            "std_err": se_beta.astype(float),
            "t_stat": z_stats.astype(float),
            "p_value": p_vals.astype(float),
            "ci_low": ci_low.astype(float),
            "ci_high": ci_high.astype(float),
        }
    )

    return RegressionResult(
        model_type="Logistic Regression (IRLS)",
        formula=formula,
        y_name=y_name,
        n_obs=n,
        df_model=int(df_model),
        df_resid=int(df_resid),
        stat=lrt_stat,
        p=lrt_p,
        p_display=lrt_p_disp,
        r_squared=pseudo_r2,
        adj_r_squared=pseudo_r2,
        aic=aic,
        bic=bic,
        coef_df=coef_df,
        robust_se="IRLS Fisher",
        alpha=alpha,
        warns=warns,
    )
