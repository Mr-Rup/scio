"""Cox Proportional Hazards Regression via Breslow Partial Likelihood and Harrell's C-index."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from omnistats.core.dataframe import ensure_pandas_df
from omnistats.core.errors import MissingDataError, StatError
from omnistats.core.guards import clamp_p
from omnistats.core.results import RegressionResult
from omnistats.models.formula import parse_formula

_RE_SURV_LHS = re.compile(r"^Surv\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)$")


def _parse_cox_formula(formula: str, event: str | None) -> tuple[str, str, str]:
    """Return (time_col, event_col, rhs_formula_with_time_lhs)."""
    if "~" not in formula:
        raise StatError(f"Cox formula must contain '~': '{formula}'")
    lhs_raw, rhs_raw = formula.split("~", 1)
    lhs = lhs_raw.strip()
    m = _RE_SURV_LHS.match(lhs)
    if m:
        t_col, e_col = m.group(1).strip(), m.group(2).strip()
        return t_col, e_col, f"{t_col} ~ {rhs_raw.strip()}"
    if event is not None and event.strip():
        return lhs, event.strip(), f"{lhs} ~ {rhs_raw.strip()}"
    raise StatError(
        "Cox model requires either 'Surv(time, event) ~ x1 + x2' syntax or `event='col'` parameter."
    )


def _harrell_c_index(time: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> float:
    """Compute Harrell's Concordance Index (C-statistic) for right-censored survival predictions."""
    n = len(time)
    concordant = 0.0
    comparable = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        longer = time > time[i]
        ties_event = (time == time[i]) & (event == 0)
        mask = longer | ties_event
        n_comp = int(np.sum(mask))
        if n_comp == 0:
            continue
        comparable += float(n_comp)
        r_diff = risk_score[i] - risk_score[mask]
        concordant += float(np.sum(r_diff > 1e-12) + 0.5 * np.sum(np.abs(r_diff) <= 1e-12))
    return float(concordant / comparable) if comparable > 0 else 0.5


def coxph(
    df: pd.DataFrame | Any,
    formula: str,
    event: str | None = None,
    alpha: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> RegressionResult:
    """Fit a Cox Proportional Hazards model via Breslow partial likelihood Newton-Raphson.

    Supports ``"Surv(time, status) ~ age + C(treatment)"`` or ``coxph(df, "time ~ age", event="status")``.
    Computes log-hazard coefficients, Hazard Ratios ($\\text{HR} = \\exp(\\beta)$), 95% CIs,
    Likelihood Ratio $\\chi^2$ omnibus test, and Harrell's Concordance Index ($C$-index).
    """
    pdf = ensure_pandas_df(df)
    t_col, e_col, std_formula = _parse_cox_formula(formula, event)
    if t_col not in pdf.columns or e_col not in pdf.columns:
        raise KeyError(f"Survival columns '{t_col}' and '{e_col}' must exist in DataFrame.")

    from omnistats.models.formula import _clean_var_name, extract_formula_vars

    _, rhs_terms = extract_formula_vars(std_formula)
    ref_cols = [t_col, e_col]
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
    _, t_raw, term_names, x_full = parse_formula(clean_df, std_formula)
    e_raw = (pd.to_numeric(clean_df[e_col], errors="coerce").to_numpy(dtype=np.float64) > 0).astype(
        np.int64
    )

    # Cox proportional hazards has no intercept (absorbed into baseline hazard h_0(t))
    keep_idx = [i for i, name in enumerate(term_names) if name != "Intercept"]
    if not keep_idx:
        raise StatError("Cox model requires at least one non-intercept predictor.")
    names = [term_names[i] for i in keep_idx]
    x_mat = x_full[:, keep_idx]
    n, p = x_mat.shape
    n_events = int(np.sum(e_raw))

    if n <= p or n_events < 2:
        raise MissingDataError(
            f"Insufficient survival observations (N={n}, events={n_events}) for {p} covariates."
        )
    if np.any(t_raw < 0):
        raise StatError("Survival follow-up times must be non-negative.")

    # Center & scale covariates for numerical stability during Newton-Raphson
    x_mean = np.mean(x_mat, axis=0)
    x_scale = np.where(np.std(x_mat, axis=0) > 1e-12, np.std(x_mat, axis=0), 1.0)
    x_std = (x_mat - x_mean) / x_scale

    # Sort observations by time ascending; risk set for time t_i is all j with t_j >= t_i
    order = np.argsort(t_raw, kind="mergesort")
    t_sorted = t_raw[order]
    e_sorted = e_raw[order]
    x_sorted = x_std[order]

    # Map each observation i to the first index in sorted array with time == t_sorted[i]
    # so tied times share the exact same risk set R(t_i)
    unique_t, first_idx = np.unique(t_sorted, return_index=True)
    time_to_first = dict(zip(unique_t.tolist(), first_idx.tolist(), strict=False))
    risk_start = np.array([time_to_first[float(tv)] for tv in t_sorted], dtype=np.int64)

    def _breslow_loglik_grad_hess(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        eta = np.clip(x_sorted @ b, -30.0, 30.0)
        w = np.exp(eta)
        # Suffix sums from right to left give sum over j >= r
        s0_rev = np.cumsum(w[::-1])[::-1]
        wx = x_sorted * w[:, None]
        s1_rev = np.cumsum(wx[::-1], axis=0)[::-1]

        s0_at_i = np.maximum(s0_rev[risk_start], 1e-15)
        s1_at_i = s1_rev[risk_start]
        x_bar = s1_at_i / s0_at_i[:, None]

        ev_mask = e_sorted == 1
        ll = float(np.sum(eta[ev_mask] - np.log(s0_at_i[ev_mask])))
        grad = np.sum(x_sorted[ev_mask] - x_bar[ev_mask], axis=0)

        # Second derivative (negative information matrix)
        w_xx = np.einsum("ni,nj,n->nij", x_sorted, x_sorted, w)
        s2_rev = np.cumsum(w_xx[::-1], axis=0)[::-1]
        s2_at_i = s2_rev[risk_start]
        info = np.zeros((p, p), dtype=np.float64)
        for idx in np.flatnonzero(ev_mask):
            v_i = s2_at_i[idx] / s0_at_i[idx] - np.outer(x_bar[idx], x_bar[idx])
            info += v_i
        return ll, grad, info

    beta_std = np.zeros(p, dtype=np.float64)
    ll_null, _, _ = _breslow_loglik_grad_hess(beta_std)

    for _ in range(max_iter):
        _ll_cur, grad, info = _breslow_loglik_grad_hess(beta_std)
        reg_info = info + np.eye(p) * 1e-8
        try:
            step = np.linalg.solve(reg_info, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(reg_info, grad, rcond=None)[0]
        beta_std = beta_std + step
        if float(np.max(np.abs(step))) < tol:
            break

    ll_final, _, info_final = _breslow_loglik_grad_hess(beta_std)
    cov_std = np.linalg.pinv(info_final + np.eye(p) * 1e-10)

    # Transform back to original covariate scale
    beta = beta_std / x_scale
    cov_beta = cov_std / np.outer(x_scale, x_scale)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-15))
    z_stats = beta / se
    p_vals = 2.0 * stats.norm.sf(np.abs(z_stats))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_low = beta - z_crit * se
    ci_high = beta + z_crit * se

    hr = np.exp(np.clip(beta, -20.0, 20.0))
    hr_lo = np.exp(np.clip(ci_low, -20.0, 20.0))
    hr_hi = np.exp(np.clip(ci_high, -20.0, 20.0))

    lr_stat = max(0.0, 2.0 * (ll_final - ll_null))
    lr_p_raw = float(stats.chi2.sf(lr_stat, df=p))
    lr_p, lr_p_disp = clamp_p(lr_p_raw)

    c_index = _harrell_c_index(t_sorted, e_sorted, x_sorted @ beta_std)
    r2_cox = float(1.0 - np.exp(-lr_stat / float(n)))
    aic = float(2.0 * p - 2.0 * ll_final)
    bic = float(np.log(max(n_events, 1)) * p - 2.0 * ll_final)

    coef_df = pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "hazard_ratio": hr,
            "std_err": se,
            "z": z_stats,
            "p_value": p_vals,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "hr_ci_low": hr_lo,
            "hr_ci_high": hr_hi,
        }
    )

    warns = [
        f"Events: {n_events}/{n} ({100.0 * n_events / n:.1f}%), "
        f"Harrell's C-index = {c_index:.3f}, Cox-Snell R² = {r2_cox:.3f}."
    ]

    return RegressionResult(
        model_type="Cox Proportional Hazards (Breslow)",
        formula=f"Surv({t_col}, {e_col}) ~ {' + '.join(names)}",
        y_name=f"Surv({t_col}, {e_col})",
        n_obs=n,
        df_model=p,
        df_resid=max(1, n_events - p),
        stat=lr_stat,
        p=lr_p,
        p_display=lr_p_disp,
        r_squared=c_index,
        adj_r_squared=r2_cox,
        aic=aic,
        bic=bic,
        coef_df=coef_df,
        robust_se="Partial-MLE",
        alpha=alpha,
        warns=warns,
    )
