"""Wilkinson formula parser for OLS, ANCOVA, and GLM design matrix construction."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from pyscio.core.dataframe import ensure_pandas_df
from pyscio.core.errors import MissingDataError, StatError


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
        elif tok not in expanded_terms:
            expanded_terms.append(tok)

    return lhs, expanded_terms


def _clean_var_name(token: str) -> tuple[str, bool]:
    """Return (column_name, force_categorical) from token like 'C(group)' or 'age'."""
    m = re.match(r"^C\(\s*([^)]+?)\s*\)$", token)
    if m:
        return m.group(1).strip(), True
    return token.strip(), False


def parse_formula(
    df: pd.DataFrame | Any,
    formula: str,
) -> tuple[str, np.ndarray, list[str], np.ndarray]:
    """Parse a Wilkinson formula ('y ~ x1 + C(x2) + x1:x2') into design matrix X and response y."""
    df = ensure_pandas_df(df)
    lhs, rhs_terms = extract_formula_vars(formula)

    referenced_cols: list[str] = []
    if lhs not in df.columns:
        raise KeyError(f"Dependent variable '{lhs}' not found in DataFrame.")
    referenced_cols.append(lhs)

    for term in rhs_terms:
        if term in ("1", "0", "-1"):
            continue
        for sp in (p.strip() for p in term.split(":")):
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

    if pd.api.types.is_numeric_dtype(sub_df[lhs]) and not isinstance(
        sub_df[lhs].dtype, pd.BooleanDtype
    ):
        y_vec = sub_df[lhs].to_numpy(dtype=np.float64)
    else:
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
                res.append((f"{col_name}[T.{lvl}]", dummies[lvl].to_numpy(dtype=np.float64)))
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
