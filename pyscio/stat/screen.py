"""Multi-feature discovery screening against target outcome with Benjamini-Hochberg FDR control."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pyscio.core.errors import ScioError
from pyscio.core.results import ScreenReport, TestResult
from pyscio.core.roles import Role, classify
from pyscio.functional.adjust import fdr_bh
from pyscio.stat.compare import compare


def screen(
    df: pd.DataFrame,
    y: str,
    alpha: float = 0.05,
    features: list[str] | None = None,
    **kwargs: Any,
) -> ScreenReport:
    """Screen multiple candidate features against target outcome variable `y` with FDR control.

    Iterates through candidate predictor features, runs autonomous hypothesis testing via
    `compare()`, collects unadjusted p-values, applies Benjamini-Hochberg False Discovery
    Rate (FDR) control, and returns a ranked discovery leaderboard.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing target and features.
    y : str
        Target dependent variable name.
    alpha : float
        FDR discovery threshold (default 0.05).
    features : list[str] | None
        Subset of feature column names to screen. If None, automatically selects all valid
        columns excluding constants, identifiers, and the target column itself.

    Returns
    -------
    ScreenReport
        Typed report containing the ranked feature leaderboard, discovery counts,
        and all underlying TestResult containers.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"screen() expects a pandas DataFrame, got {type(df).__name__}")

    if y not in df.columns:
        raise KeyError(f"Target column '{y}' not found in DataFrame columns: {list(df.columns)}")

    # Classify all columns to exclude ID and CONSTANT columns by default
    roles = classify(df)

    if features is None:
        candidate_features = [
            col for col in df.columns if col != y and roles.get(col) not in (Role.ID, Role.CONSTANT)
        ]
    else:
        candidate_features = [col for col in features if col in df.columns and col != y]

    if not candidate_features:
        empty_df = pd.DataFrame(
            columns=[
                "feature",
                "test",
                "statistic",
                "p_value",
                "p_adj",
                "significant_fdr",
                "effect_size",
                "effect_metric",
                "magnitude",
            ]
        )
        return ScreenReport(
            target=y,
            n_features=0,
            alpha=alpha,
            results=[],
            leaderboard_df=empty_df,
        )

    results: list[TestResult] = []
    records: list[dict[str, Any]] = []

    for feat in candidate_features:
        try:
            res = compare(df, x=feat, y=y, alpha=alpha, **kwargs)
            results.append(res)
            records.append(
                {
                    "feature": feat,
                    "test": res.test_name,
                    "statistic": res.stat,
                    "p_value": res.p,
                    "effect_size": res.effect if res.effect is not None else np.nan,
                    "effect_metric": res.metric if res.metric is not None else "-",
                    "magnitude": res.magnitude if res.magnitude is not None else "-",
                    "significant_raw": res.sig,
                }
            )
        except (ScioError, ValueError, KeyError):
            # Skip invalid, unanalyzable, or constant feature slices
            continue

    if not records:
        empty_df = pd.DataFrame(
            columns=[
                "feature",
                "test",
                "statistic",
                "p_value",
                "p_adj",
                "significant_fdr",
                "effect_size",
                "effect_metric",
                "magnitude",
            ]
        )
        return ScreenReport(
            target=y,
            n_features=len(candidate_features),
            alpha=alpha,
            results=[],
            leaderboard_df=empty_df,
        )

    # Benjamini-Hochberg False Discovery Rate correction
    raw_p_values = [r["p_value"] for r in records]
    reject_mask, q_values = fdr_bh(raw_p_values, alpha=alpha)

    for i, r in enumerate(records):
        r["p_adj"] = float(q_values[i])
        r["significant_fdr"] = bool(reject_mask[i])

    leaderboard_df = pd.DataFrame(records)

    # Sort leaderboard: significant first, then lowest p_adj, then highest effect size
    leaderboard_df["abs_effect"] = leaderboard_df["effect_size"].abs().fillna(0.0)
    leaderboard_df = (
        leaderboard_df.sort_values(
            by=["significant_fdr", "p_adj", "abs_effect"],
            ascending=[False, True, False],
        )
        .drop(columns=["abs_effect"])
        .reset_index(drop=True)
    )

    # Reorder columns
    ordered_cols = [
        "feature",
        "test",
        "statistic",
        "p_value",
        "p_adj",
        "significant_fdr",
        "effect_size",
        "effect_metric",
        "magnitude",
    ]
    leaderboard_df = leaderboard_df[[c for c in ordered_cols if c in leaderboard_df.columns]]

    return ScreenReport(
        target=y,
        n_features=len(candidate_features),
        alpha=alpha,
        results=results,
        leaderboard_df=leaderboard_df,
    )
