"""Wilkinson formula parser, Ordinary Least Squares (HC3), and IRLS Logistic Regression."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import stats

from pyscio.core.errors import MissingDataError, StatError
from pyscio.core.guards import clamp_p
from pyscio.core.results import RegressionResult


def extract_formula_vars(formula: str) -> tuple[str, list[str]]:
    """Extract raw LHS outcome name and RHS term tokens from a Wilkinson formula string."""
    if "~" not in formula:
        raise StatError(f"Formula must contain '~' separating outcome and predictors: '{formula}'")

    lhs_raw, rhs_raw = formula.split("~", 1)
    lhs = lhs_raw.strip()
    if not lhs:
        raise StatError(f"Formula is missing dependent outcome variable on LHS: '{formula}'")

    raw_tokens = [tok.strip() for tok in rhs_raw.split("+") if tok.strip()]
    if not raw_tokens:
        raise StatError(f"Formula is missing predictor terms on RHS: '{formula}'")

    # Expand 'a * b' shorthand into 'a', 'b', 'a:b'
    expanded_terms: list[str] = []
    for tok in raw_tokens:
        if "*" in tok:
            parts = [p.strip() for p in tok.split("*") if p.strip()]
            for p in parts:
                if p not in expanded_terms:
                    expanded_terms.append(p)
            if len(parts) >= 2:
                inter = ":".join(parts)
                if inter not in expanded_terms:
                    expanded_terms.append(inter)
        else:
            if tok not in expanded_terms:
                expanded_terms.append(tok)

    return lhs, expanded_terms


def _clean_var_name(token: str) -> tuple[str, bool]:
    """Return (column_name, force_categorical) from token like 'C(group)' or 'age'."""
    m = re.match(r"^C\(\s*([^)]+?)\s*\)$", token)
    if m:
        return m.group(1).strip(), True
    return token.strip(), False


def parse_formula(
    df: pd.DataFrame,
    formula: str,
) -> tuple[str, np.ndarray, list[str], np.ndarray]:
    """Parse a Wilkinson formula ('y ~ x1 + C(x2) + x1:x2') into design matrix X and response y.

    Performs listwise deletion of missing observations across all referenced columns,
    automatically dummy-encodes categorical columns (`drop_first=True`), prepends an
    Intercept column, and constructs interaction product columns.
    """
    lhs, rhs_terms = extract_formula_vars(formula)

    # Identify all underlying DataFrame columns required
    referenced_cols: list[str] = []
    if lhs not in df.columns:
        raise KeyError(f"Dependent variable '{lhs}' not found in DataFrame.")
    referenced_cols.append(lhs)

    for term in rhs_terms:
        if term in ("1", "0", "-1"):
            continue
        sub_parts = [p.strip() for p in term.split(":")]
        for sp in sub_parts:
            col_name, _ = _clean_var_name(sp)
            if col_name not in df.columns:
                raise KeyError(f"Predictor variable '{col_name}' not found in DataFrame.")
            if col_name not in referenced_cols:
                referenced_cols.append(col_name)

    sub_df = df[referenced_cols].dropna().copy()
    if len(sub_df) < 3:
        raise MissingDataError(
            f"Insufficient complete observations (N={len(sub_df)}) after dropping missing values."
        )

    # Extract response vector y
    if pd.api.types.is_numeric_dtype(sub_df[lhs]) and not isinstance(
        sub_df[lhs].dtype, pd.BooleanDtype
    ):
        y_vec = sub_df[lhs].to_numpy(dtype=np.float64)
    else:
        # Encode binary/categorical LHS into 0.0 / 1.0
        codes, uniques = pd.factorize(sub_df[lhs], sort=True)
        if len(uniques) != 2:
            raise StatError(
                f"Non-numeric dependent variable '{lhs}' must be binary (got {len(uniques)} levels)."
            )
        y_vec = codes.astype(np.float64)

    include_intercept = True
    term_columns: dict[str, list[tuple[str, np.ndarray]]] = {}
    design_names: list[str] = []
    design_cols: list[np.ndarray] = []

    def _build_atomic(atom: str) -> list[tuple[str, np.ndarray]]:
        if atom in term_columns:
            return term_columns[atom]
        col_name, force_cat = _clean_var_name(atom)
        series = sub_df[col_name]
        is_cat = (
            force_cat
            or not pd.api.types.is_numeric_dtype(series)
            or isinstance(series.dtype, pd.BooleanDtype)
        )
        res: list[tuple[str, np.ndarray]] = []
        if is_cat:
            dummies = pd.get_dummies(series, drop_first=True, dtype=np.float64)
            for lvl in dummies.columns:
                label = f"{col_name}[T.{lvl}]"
                res.append((label, dummies[lvl].to_numpy(dtype=np.float64)))
        else:
            res.append((col_name, series.to_numpy(dtype=np.float64)))
        term_columns[atom] = res
        return res

    for term in rhs_terms:
        if term in ("0", "-1"):
            include_intercept = False
            continue
        if term == "1":
            include_intercept = True
            continue

        if ":" in term:
            parts = [p.strip() for p in term.split(":")]
            part_blocks = [_build_atomic(p) for p in parts]
            # Cartesian product of columns for interaction terms
            current = part_blocks[0]
            for nxt in part_blocks[1:]:
                combined: list[tuple[str, np.ndarray]] = []
                for name_a, vec_a in current:
                    for name_b, vec_b in nxt:
                        combined.append((f"{name_a}:{name_b}", vec_a * vec_b))
                current = combined
            for cname, cvec in current:
                if cname not in design_names:
                    design_names.append(cname)
                    design_cols.append(cvec)
        else:
            for cname, cvec in _build_atomic(term):
                if cname not in design_names:
                    design_names.append(cname)
                    design_cols.append(cvec)

    if include_intercept:
        design_names.insert(0, "Intercept")
        design_cols.insert(0, np.ones(len(sub_df), dtype=np.float64))

    if not design_cols:
        raise StatError(f"Formula '{formula}' produced an empty design matrix.")

    x_mat = np.column_stack(design_cols)
    return lhs, y_vec, design_names, x_mat


def ols(
    df: pd.DataFrame,
    formula: str,
    robust_se: str | None = "HC3",
    alpha: float = 0.05,
) -> RegressionResult:
    """Fit Ordinary Least Squares (OLS) regression with MacKinnon-White HC3 robust standard errors.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing outcome and predictor variables.
    formula : str
        Wilkinson formula string (e.g., ``"y ~ x1 + x2 + C(group) + x1:x2"``).
    robust_se : str | None
        Covariance estimator: ``"HC3"`` (default, MacKinnon & White jackknife-equivalent
        heteroscedasticity-robust SE) or ``None`` / ``"nonrobust"`` for classical OLS SEs.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    RegressionResult
        Rich regression result container with coefficient table, $R^2$, Adj-$R^2$, AIC, BIC,
        and omnibus $F$-test.
    """
    y_name, y, term_names, x_mat = parse_formula(df, formula)
    n, p = x_mat.shape
    if n <= p:
        raise StatError(f"Insufficient observations (N={n}) to estimate {p} regression parameters.")

    df_resid = float(n - p)
    has_intercept = "Intercept" in term_names
    df_model = float(p - 1 if has_intercept else p)

    # Solve normal equations via SVD least squares
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

    if robust_se is not None and robust_se.upper() == "HC3":
        # Hat matrix diagonal h_ii = sum_j (X @ (X^T X)^-1)_ij * X_ij
        h_diag = np.einsum("ij,ij->i", x_mat @ xtx_inv, x_mat)
        denom = np.maximum(1.0 - h_diag, 1e-6) ** 2
        omega = (resid**2) / denom
        meat = x_mat.T @ (x_mat * omega[:, None])
        cov_beta = xtx_inv @ meat @ xtx_inv
        model_label = "OLS Regression (HC3 Robust SE)"
    else:
        cov_beta = sigma2 * xtx_inv
        model_label = "OLS Regression (Classical SE)"

    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 1e-30))
    t_stats = beta / se_beta
    p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=df_resid)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_resid))
    ci_low = beta - t_crit * se_beta
    ci_high = beta + t_crit * se_beta

    # Omnibus F-statistic
    if df_model > 0 and rss > 0:
        ess = max(0.0, tss - rss)
        f_stat = float((ess / df_model) / (rss / df_resid))
        f_p_raw = float(stats.f.sf(f_stat, df_model, df_resid))
    else:
        f_stat = 0.0
        f_p_raw = 1.0
    f_p, f_p_disp = clamp_p(f_p_raw)

    # Log-likelihood, AIC, BIC
    sigma2_mle = max(rss / n, 1e-30)
    ll = -0.5 * n * (math.log(2.0 * math.pi) + math.log(sigma2_mle) + 1.0)
    k_params = p + 1  # coefficients + error variance
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


def logistic(
    df: pd.DataFrame,
    formula: str,
    alpha: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> RegressionResult:
    """Fit binary Logistic Regression via Newton-Raphson Iteratively Reweighted Least Squares (IRLS).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a binary outcome variable and predictors.
    formula : str
        Wilkinson formula string (e.g., ``"outcome ~ age + C(treatment)"``).
    alpha : float
        Significance threshold (default 0.05).
    max_iter : int
        Maximum Newton-Raphson IRLS iterations (default 50).
    tol : float
        Convergence tolerance on step norm $||\\Delta \\beta||_\\infty$ (default 1e-6).

    Returns
    -------
    RegressionResult
        Rich regression result container with log-odds coefficients, Odds Ratios (``or_val``),
        Wald $z$-statistics, McFadden's pseudo-$R^2$, AIC, BIC, and Likelihood Ratio Test.
    """
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

        # Gradient and Fisher information (Hessian)
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

    # Log-likelihood of full model vs null intercept-only model
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
