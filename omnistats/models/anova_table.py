"""Factorial ANOVA and ANCOVA tables (Type II & Type III Sums of Squares) via QR projections."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from omnistats.core.errors import StatError
from omnistats.models.formula import _clean_var_name, extract_formula_vars, parse_formula


def _rss_qr(x_sub: np.ndarray, y: np.ndarray) -> float:
    """Compute Residual Sum of Squares (RSS) of y projected onto column space of x_sub via QR."""
    if x_sub.shape[1] == 0:
        return float(np.sum(y**2))
    q_mat, _ = np.linalg.qr(x_sub, mode="reduced")
    resid = y - q_mat @ (q_mat.T @ y)
    return float(np.sum(resid**2))


def _canonical_term_key(term: str) -> tuple[str, ...]:
    """Convert a formula term like 'C(A):B' or design column 'A[T.1]:B' to sorted atomic factor names."""
    atoms: list[str] = []
    for part in term.split(":"):
        stripped = re.sub(r"\[T\..*?\]$", "", part.strip())
        col_name, _ = _clean_var_name(stripped)
        atoms.append(col_name)
    return tuple(atoms)


def anova_table(
    df: pd.DataFrame,
    formula: str,
    *,
    typ: int = 2,
) -> pd.DataFrame:
    """Compute Type II or Type III Factorial ANOVA / ANCOVA table with partial eta-squared and omega-squared.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    formula : str
        Wilkinson formula string (e.g. ``"y ~ C(A) * C(B) + covar"``).
    typ : int, default 2
        Sum of Squares type: ``2`` (Type II, hierarchical marginality) or ``3`` (Type III, partial).

    Returns
    -------
    pd.DataFrame
        ANOVA table with columns ``["term", "sum_sq", "df", "F", "p_value", "partial_eta_sq", "omega_sq"]``.
    """
    if typ not in (2, 3):
        raise StatError(f"Unsupported ANOVA Sum of Squares type '{typ}'; choose typ=2 or typ=3.")

    _, rhs_terms = extract_formula_vars(formula)
    _, y, design_names, x_mat = parse_formula(df, formula)
    n, p = x_mat.shape
    if n <= p:
        raise StatError(f"Insufficient observations (N={n}) for ANOVA model with {p} parameters.")

    df_resid = float(n - p)
    rss_full = _rss_qr(x_mat, y)
    ms_resid = rss_full / df_resid if df_resid > 0 else 0.0
    tss = float(np.sum((y - np.mean(y)) ** 2))

    # Map each non-intercept design column to its canonical factor tuple
    col_keys: list[tuple[str, ...] | None] = [
        None if name == "Intercept" else _canonical_term_key(name) for name in design_names
    ]

    model_terms: list[tuple[str, tuple[str, ...], list[int]]] = []
    for raw_t in rhs_terms:
        if raw_t in ("0", "1", "-1"):
            continue
        t_key = _canonical_term_key(raw_t)
        cols = [idx for idx, ck in enumerate(col_keys) if ck == t_key]
        if cols:
            model_terms.append((raw_t, t_key, cols))

    intercept_cols = [idx for idx, name in enumerate(design_names) if name == "Intercept"]

    rows: list[dict[str, object]] = []
    for raw_t, t_key, t_cols in model_terms:
        t_set = set(t_key)
        if typ == 2:
            # Include all terms S whose factor set does NOT contain t_set
            base_cols = list(intercept_cols)
            for _, s_key, s_cols in model_terms:
                if s_key == t_key:
                    continue
                if not t_set.issubset(set(s_key)):
                    base_cols.extend(s_cols)
            full_sub_cols = sorted(set(base_cols + t_cols))
            red_sub_cols = sorted(set(base_cols))
            ss_term = max(
                0.0, _rss_qr(x_mat[:, red_sub_cols], y) - _rss_qr(x_mat[:, full_sub_cols], y)
            )
        else:
            # Type III: drop only t_cols from full design matrix
            keep_cols = [i for i in range(p) if i not in t_cols]
            ss_term = max(0.0, _rss_qr(x_mat[:, keep_cols], y) - rss_full)

        df_term = float(len(t_cols))
        ms_term = ss_term / df_term
        f_stat = float(ms_term / ms_resid) if ms_resid > 1e-15 else 0.0
        p_val = float(stats.f.sf(f_stat, dfn=df_term, dfd=df_resid)) if ms_resid > 1e-15 else 1.0
        p_eta2 = float(ss_term / (ss_term + rss_full)) if (ss_term + rss_full) > 0 else 0.0
        omega2 = (
            float(max(0.0, (ss_term - df_term * ms_resid) / (tss + ms_resid)))
            if (tss + ms_resid) > 0
            else 0.0
        )

        rows.append(
            {
                "term": raw_t,
                "sum_sq": ss_term,
                "df": df_term,
                "F": f_stat,
                "p_value": p_val,
                "partial_eta_sq": p_eta2,
                "omega_sq": omega2,
            }
        )

    rows.append(
        {
            "term": "Residual",
            "sum_sq": rss_full,
            "df": df_resid,
            "F": np.nan,
            "p_value": np.nan,
            "partial_eta_sq": np.nan,
            "omega_sq": np.nan,
        }
    )

    return pd.DataFrame(rows)
