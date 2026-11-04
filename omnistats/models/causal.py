"""Causal Inference Engine: Doubly Robust AIPW, IPTW, and Standardized Mean Difference (SMD) balance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from omnistats.core.dataframe import ensure_pandas_df
from omnistats.core.errors import MissingDataError, StatError
from omnistats.core.guards import clamp_p
from omnistats.core.results import CheckResult, TestResult
from omnistats.functional.effects import interpret_effect
from omnistats.models.formula import parse_formula


def _fit_propensity_logit(x_mat: np.ndarray, t_vec: np.ndarray, max_iter: int = 35) -> np.ndarray:
    """Fit binary logistic regression via IRLS with ridge stabilization to estimate P(T=1 | X)."""
    _n, p = x_mat.shape
    beta = np.zeros(p, dtype=np.float64)
    for _ in range(max_iter):
        eta = np.clip(x_mat @ beta, -25.0, 25.0)
        prob = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(prob * (1.0 - prob), 1e-6)
        grad = x_mat.T @ (t_vec - prob) - 1e-4 * beta
        hess = (x_mat.T * w) @ x_mat + np.eye(p) * 1e-4
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    eta_final = np.clip(x_mat @ beta, -25.0, 25.0)
    return np.asarray(1.0 / (1.0 + np.exp(-eta_final)), dtype=np.float64)


def covariate_balance(
    df: pd.DataFrame | Any,
    treatment: str,
    covariates: list[str],
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute unweighted and propensity-weighted Standardized Mean Differences (SMD) across covariates.

    An absolute weighted $|\text{SMD}| < 0.10$ indicates well-balanced observational treatment arms.
    """
    pdf = ensure_pandas_df(df)
    formula = f"{treatment} ~ {' + '.join(covariates)}"
    _, t_vec, design_names, x_mat = parse_formula(pdf, formula)
    t_bin = (t_vec > 0.5).astype(np.float64)

    mask1 = t_bin == 1.0
    mask0 = t_bin == 0.0
    if int(np.sum(mask1)) < 2 or int(np.sum(mask0)) < 2:
        raise StatError(
            "Both treatment (T=1) and control (T=0) arms require at least 2 observations."
        )

    if weights is None:
        ps = np.clip(_fit_propensity_logit(x_mat, t_bin), 0.02, 0.98)
        w_vec = np.where(mask1, 1.0 / ps, 1.0 / (1.0 - ps))
    else:
        w_vec = np.asarray(weights, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(design_names):
        if name == "Intercept":
            continue
        col = x_mat[:, idx]
        x1, x0 = col[mask1], col[mask0]
        m1_raw, m0_raw = float(np.mean(x1)), float(np.mean(x0))
        v1_raw, v0_raw = float(np.var(x1, ddof=1)), float(np.var(x0, ddof=1))
        pooled_sd = np.sqrt(max((v1_raw + v0_raw) / 2.0, 1e-12))
        smd_unweighted = float((m1_raw - m0_raw) / pooled_sd)

        w1, w0 = w_vec[mask1], w_vec[mask0]
        m1_w = float(np.average(x1, weights=w1))
        m0_w = float(np.average(x0, weights=w0))
        smd_weighted = float((m1_w - m0_w) / pooled_sd)

        rows.append(
            {
                "covariate": name,
                "mean_treated": m1_raw,
                "mean_control": m0_raw,
                "smd_unweighted": smd_unweighted,
                "smd_weighted": smd_weighted,
                "balanced": abs(smd_weighted) < 0.10,
            }
        )

    return pd.DataFrame(rows)


def ate(
    df: pd.DataFrame | Any,
    outcome: str,
    treatment: str,
    covariates: list[str],
    method: str = "aipw",
    trim_ps: float = 0.02,
    alpha: float = 0.05,
) -> TestResult:
    """Estimate causal Average Treatment Effect (ATE) via Doubly Robust AIPW or IPTW.

    Parameters
    ----------
    df : DataFrame-like
        Observational dataset.
    outcome : str
        Target outcome variable $Y$.
    treatment : str
        Binary treatment indicator $T \\in \\{0, 1\\}$.
    covariates : list[str]
        Confounding baseline covariates $X$ to adjust for.
    method : {"aipw", "iptw"}, default "aipw"
        ``"aipw"`` uses Robins-Rotnitzky-Zhao Doubly Robust Augmented Inverse Probability Weighting;
        ``"iptw"`` uses Horvitz-Thompson / Hajek Inverse Probability of Treatment Weighting.
    trim_ps : float, default 0.02
        Propensity score clipping bound $[\\text{trim\\_ps}, 1 - \\text{trim\\_ps}]$ for overlap protection.
    """
    pdf = ensure_pandas_df(df)
    if outcome not in pdf.columns or treatment not in pdf.columns:
        raise KeyError(f"Outcome '{outcome}' and treatment '{treatment}' must exist in DataFrame.")
    if not covariates:
        raise StatError("At least one confounding covariate must be provided to `ate()`.")

    # Build aligned complete-case frame
    clean_df = pdf[[outcome, treatment, *covariates]].dropna().reset_index(drop=True)
    if len(clean_df) < 10:
        raise MissingDataError(
            f"Insufficient complete observations (N={len(clean_df)}) for causal ATE."
        )

    y_vec = pd.to_numeric(clean_df[outcome], errors="coerce").to_numpy(dtype=np.float64)
    _, t_raw, _, x_mat = parse_formula(clean_df, f"{treatment} ~ {' + '.join(covariates)}")
    t_vec = (t_raw > 0.5).astype(np.float64)
    n = len(y_vec)

    mask1 = t_vec == 1.0
    mask0 = t_vec == 0.0
    n1, n0 = int(np.sum(mask1)), int(np.sum(mask0))
    if n1 < 3 or n0 < 3:
        raise StatError(
            f"Treatment arms require at least 3 observations each (got n1={n1}, n0={n0})."
        )

    ps_raw = _fit_propensity_logit(x_mat, t_vec)
    n_clipped = int(np.sum((ps_raw < trim_ps) | (ps_raw > (1.0 - trim_ps))))
    ps = np.clip(ps_raw, trim_ps, 1.0 - trim_ps)
    ipw_weights = np.where(mask1, 1.0 / ps, 1.0 / (1.0 - ps))

    method_clean = method.lower().strip()
    if method_clean == "aipw":
        # Outcome regressions mu_1(X) and mu_0(X) via ridge-stabilized QR/lstsq
        beta1 = np.linalg.lstsq(x_mat[mask1], y_vec[mask1], rcond=None)[0]
        beta0 = np.linalg.lstsq(x_mat[mask0], y_vec[mask0], rcond=None)[0]
        mu1_hat = x_mat @ beta1
        mu0_hat = x_mat @ beta0

        # Efficient influence function psi_i for Doubly Robust AIPW
        psi1 = mu1_hat + (t_vec * (y_vec - mu1_hat)) / ps
        psi0 = mu0_hat + ((1.0 - t_vec) * (y_vec - mu0_hat)) / (1.0 - ps)
        psi = psi1 - psi0
        ate_est = float(np.mean(psi))
        se_ate = float(np.std(psi, ddof=1) / np.sqrt(n))
        test_label = "Doubly Robust AIPW Causal ATE"
    elif method_clean == "iptw":
        # Hajek stabilized IPTW estimator and influence function
        w1 = t_vec / ps
        w0 = (1.0 - t_vec) / (1.0 - ps)
        mu1_iptw = float(np.sum(w1 * y_vec) / np.sum(w1))
        mu0_iptw = float(np.sum(w0 * y_vec) / np.sum(w0))
        ate_est = mu1_iptw - mu0_iptw
        psi = w1 * (y_vec - mu1_iptw) - w0 * (y_vec - mu0_iptw)
        se_ate = float(np.std(psi, ddof=1) / np.sqrt(n))
        test_label = "IPTW Causal ATE (Hajek)"
    else:
        raise StatError(f"Unknown causal estimator '{method}'; choose 'aipw' or 'iptw'.")

    se_ate = max(se_ate, 1e-15)
    z_stat = float(ate_est / se_ate)
    p_raw = float(2.0 * stats.norm.sf(abs(z_stat)))
    p_val, p_disp = clamp_p(p_raw)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci = (float(ate_est - z_crit * se_ate), float(ate_est + z_crit * se_ate))

    # Standardized ATE (Cohen's d scale)
    sd_y = float(np.std(y_vec, ddof=1))
    std_ate = float(ate_est / sd_y) if sd_y > 0 else 0.0
    std_ate_ci = (float(ci[0] / sd_y), float(ci[1] / sd_y)) if sd_y > 0 else (0.0, 0.0)

    bal_df = covariate_balance(
        clean_df, treatment=treatment, covariates=covariates, weights=ipw_weights
    )
    max_smd_before = float(bal_df["smd_unweighted"].abs().max()) if not bal_df.empty else 0.0
    max_smd_after = float(bal_df["smd_weighted"].abs().max()) if not bal_df.empty else 0.0
    bal_passed = max_smd_after < 0.10

    checks = [
        CheckResult(
            name="Propensity Covariate Balance (|SMD| < 0.10)",
            stat=max_smd_after,
            p=1.0 if bal_passed else 0.01,
            passed=bal_passed,
            details=f"max |SMD| reduced from {max_smd_before:.3f} to {max_smd_after:.3f}",
        )
    ]

    warns: list[str] = []
    if n_clipped > 0:
        warns.append(
            f"Positivity guard clipped {n_clipped}/{n} propensity scores to [{trim_ps}, {1.0 - trim_ps:.2f}]."
        )
    if not bal_passed:
        warns.append(
            f"Residual covariate imbalance detected after weighting (max |SMD| = {max_smd_after:.3f} >= 0.10)."
        )

    return TestResult(
        test_name=test_label,
        x_name=f"{treatment} | {', '.join(covariates)}",
        y_name=outcome,
        stat=z_stat,
        p=p_val,
        p_display=p_disp,
        dof=float(n - 1),
        ci=ci,
        ci_level=1.0 - alpha,
        sig=p_val < alpha,
        alpha=alpha,
        effect=std_ate,
        metric="Standardized ATE (d)",
        magnitude=interpret_effect("cohen", std_ate),
        effect_ci=std_ate_ci,
        checks=checks,
        posthoc=bal_df,
        warns=warns,
    )
