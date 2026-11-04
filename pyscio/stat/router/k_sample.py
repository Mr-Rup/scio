"""Autonomous router for K-group (ANOVA / Welch / Kruskal / RM-ANOVA / Friedman) and 2D contingency tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pyscio.core.errors import StatError
from pyscio.core.guards import check_sample_size, check_variance, clamp_p, clean_1d
from pyscio.core.results import CheckResult, TestResult
from pyscio.functional.checks import levene
from pyscio.functional.effects import (
    cramers_v,
    eps_sq,
    eta_sq,
    interpret_effect,
    kendalls_w,
    odds_ratio,
)
from pyscio.functional.posthoc import dunn, games_howell, tukey
from pyscio.functional.rm import rm_anova
from pyscio.functional.tests import anova, chisq, fisher, friedman, kruskal, welch_anova
from pyscio.stat.router.normality import evaluate_normality_with_clt


def route_k_sample(
    groups: Sequence[np.ndarray | list[float]],
    group_names: Sequence[str] | None = None,
    x_name: str = "group",
    y_name: str = "target",
    alpha: float = 0.05,
) -> TestResult:
    """Route K >= 3 independent groups through the ANOVA / Kruskal-Wallis lattice with auto post-hoc."""
    cleaned = [clean_1d(g, name=f"group_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 3:
        raise StatError("route_k_sample requires k >= 3 groups.")

    names = (
        list(group_names)
        if group_names and len(group_names) == k
        else [f"Group_{i + 1}" for i in range(k)]
    )

    for i, arr in enumerate(cleaned):
        check_sample_size(arr, min_n=3, name=names[i])
        check_variance(arr, name=names[i])

    checks: list[CheckResult] = []
    warns: list[str] = []

    all_normal = True
    for i, arr in enumerate(cleaned):
        norm, chk = evaluate_normality_with_clt(arr, name=names[i], alpha=alpha)
        checks.append(chk)
        if not norm:
            all_normal = False

    posthoc_df: pd.DataFrame | None = None

    if all_normal:
        homo_res = levene(*cleaned, center="median", alpha=alpha)
        checks.append(
            CheckResult(
                name="Homoscedasticity (Levene / Brown-Forsythe)",
                stat=homo_res.stat,
                p=homo_res.p,
                passed=homo_res.is_homoscedastic,
                details=f"p={homo_res.p:.4g} ({'equal variance' if homo_res.is_homoscedastic else 'unequal variance'})",
            )
        )

        if homo_res.is_homoscedastic:
            a_res = anova(*cleaned)
            test_name = "One-Way ANOVA"
            stat_val = a_res.stat
            p_val, p_display = clamp_p(a_res.p)
            dof: float | tuple[float, ...] | None = (a_res.df_between, a_res.df_within)
            eff_val = eta_sq(*cleaned)
            metric = "Eta-squared (η²)"
            magnitude = interpret_effect("eta", eff_val)
            if p_val < alpha:
                posthoc_df = tukey(*cleaned, names=names, alpha=alpha)
        else:
            w_res = welch_anova(*cleaned)
            test_name = "Welch's ANOVA"
            stat_val = w_res.stat
            p_val, p_display = clamp_p(w_res.p)
            dof = (w_res.df_between, w_res.df_within)
            eff_val = eta_sq(*cleaned)
            metric = "Eta-squared (η²)"
            magnitude = interpret_effect("eta", eff_val)
            if p_val < alpha:
                posthoc_df = games_howell(*cleaned, names=names, alpha=alpha)
    else:
        kw_res = kruskal(*cleaned)
        test_name = "Kruskal-Wallis H Test"
        stat_val = kw_res.stat
        p_val, p_display = clamp_p(kw_res.p)
        dof = float(k - 1)
        eff_val = eps_sq(*cleaned)
        metric = "Epsilon-squared (ε²)"
        magnitude = interpret_effect("eta", eff_val)
        if p_val < alpha:
            posthoc_df = dunn(*cleaned, names=names, p_adjust="fdr_bh", alpha=alpha)

    return TestResult(
        test_name=test_name,
        x_name=x_name,
        y_name=y_name,
        stat=stat_val,
        p=p_val,
        p_display=p_display,
        dof=dof,
        ci=None,
        ci_level=1.0 - alpha,
        sig=p_val < alpha,
        alpha=alpha,
        effect=eff_val,
        metric=metric,
        magnitude=magnitude,
        checks=checks,
        posthoc=posthoc_df,
        warns=warns,
    )


def route_paired_k_sample(
    groups: Sequence[np.ndarray | list[float]],
    group_names: Sequence[str] | None = None,
    x_name: str = "condition",
    y_name: str = "target",
    alpha: float = 0.05,
) -> TestResult:
    """Route K >= 3 repeated-measures conditions via RM-ANOVA (Greenhouse-Geisser) or Friedman's test."""
    cleaned = [clean_1d(g, name=f"condition_{i}") for i, g in enumerate(groups)]
    k = len(cleaned)
    if k < 3:
        raise StatError("route_paired_k_sample requires k >= 3 matched conditions.")

    n0 = len(cleaned[0])
    if any(len(g) != n0 for g in cleaned):
        raise StatError("All repeated-measures conditions must have identical sample lengths.")
    check_sample_size(cleaned[0], min_n=3, name="subjects")

    names = (
        list(group_names)
        if group_names and len(group_names) == k
        else [f"Condition_{i + 1}" for i in range(k)]
    )

    checks: list[CheckResult] = []
    all_normal = True
    for i, arr in enumerate(cleaned):
        norm, chk = evaluate_normality_with_clt(arr, name=names[i], alpha=alpha)
        checks.append(chk)
        if not norm:
            all_normal = False

    if all_normal:
        rm_res = rm_anova(*cleaned)
        checks.append(
            CheckResult(
                name="Sphericity (Greenhouse-Geisser ε)",
                stat=rm_res.eps_gg,
                p=rm_res.p_gg,
                passed=rm_res.eps_gg >= 0.75,
                details=f"ε_GG={rm_res.eps_gg:.3f} (df adjusted: {rm_res.eps_gg * rm_res.df_num:.2f}, {rm_res.eps_gg * rm_res.df_den:.2f})",
            )
        )
        stat_val = rm_res.stat
        p_val, p_display = clamp_p(rm_res.p_gg)
        dof: float | tuple[float, ...] = (
            round(rm_res.eps_gg * rm_res.df_num, 2),
            round(rm_res.eps_gg * rm_res.df_den, 2),
        )
        eff_val = rm_res.partial_eta_sq
        metric = "Partial Eta-squared (η²_p)"
        magnitude = interpret_effect("eta", eff_val)
        test_name = "One-Way Repeated-Measures ANOVA (Greenhouse-Geisser)"
        posthoc_df = games_howell(*cleaned, names=names, alpha=alpha) if p_val < alpha else None
    else:
        f_res = friedman(*cleaned)
        stat_val = f_res.stat
        p_val, p_display = clamp_p(f_res.p)
        dof = float(k - 1)
        eff_val = kendalls_w(*cleaned)
        metric = "Kendall's W"
        magnitude = interpret_effect("cram", eff_val)
        test_name = "Friedman Repeated-Measures Test"
        posthoc_df = (
            dunn(*cleaned, names=names, p_adjust="fdr_bh", alpha=alpha) if p_val < alpha else None
        )

    return TestResult(
        test_name=test_name,
        x_name=x_name,
        y_name=y_name,
        stat=stat_val,
        p=p_val,
        p_display=p_display,
        dof=dof,
        ci=None,
        ci_level=1.0 - alpha,
        sig=p_val < alpha,
        alpha=alpha,
        effect=eff_val,
        metric=metric,
        magnitude=magnitude,
        checks=checks,
        posthoc=posthoc_df,
        warns=[],
    )


def route_contingency(
    table_or_df: pd.DataFrame | np.ndarray,
    x_name: str = "variable_1",
    y_name: str = "variable_2",
    alpha: float = 0.05,
) -> TestResult:
    """Route 2D contingency table through Cochran's rule (Fisher's exact vs Pearson's Chi-squared)."""
    table = np.asarray(table_or_df, dtype=np.float64)
    if table.ndim != 2:
        raise StatError(f"Contingency table must be 2D, got shape {table.shape}")

    r, c = table.shape
    if r < 2 or c < 2:
        raise StatError(f"Contingency table must be at least 2x2, got {r}x{c}")

    checks: list[CheckResult] = []
    warns: list[str] = []

    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    grand_total = table.sum()
    expected = (row_sums @ col_sums) / grand_total

    low_expected_cells = int(np.sum(expected < 5.0))
    pct_low = (low_expected_cells / expected.size) * 100.0

    checks.append(
        CheckResult(
            name="Cochran's Rule (Expected Cell Counts >= 5)",
            stat=float(expected.min()),
            p=1.0,
            passed=low_expected_cells == 0,
            details=f"{low_expected_cells}/{expected.size} cells ({pct_low:.1f}%) < 5 (min={expected.min():.2f})",
        )
    )

    if r == 2 and c == 2 and low_expected_cells > 0:
        f_res = fisher(table, alternative="two-sided")
        test_name = "Fisher's Exact Test"
        stat_val = f_res.stat
        p_val, p_display = clamp_p(f_res.p)
        dof: float | None = None

        or_res = odds_ratio(table, conf_level=1.0 - alpha)
        eff_val: float | None = or_res.or_val
        metric: str | None = "Odds Ratio"
        ci: tuple[float, float] | None = or_res.ci
        magnitude: str | None = None
    else:
        if low_expected_cells > 0:
            warns.append(
                f"{low_expected_cells} cells have expected frequencies < 5. "
                "Chi-squared asymptotic approximation may be unreliable."
            )
        cs_res = chisq(table)
        test_name = "Pearson's Chi-Squared Test"
        stat_val = cs_res.stat
        p_val, p_display = clamp_p(cs_res.p)
        dof = float(cs_res.dof)
        ci = None

        eff_val = cramers_v(table)
        metric = "Cramér's V"
        magnitude = interpret_effect("cram", eff_val)

    return TestResult(
        test_name=test_name,
        x_name=x_name,
        y_name=y_name,
        stat=stat_val,
        p=p_val,
        p_display=p_display,
        dof=dof,
        ci=ci,
        ci_level=1.0 - alpha,
        sig=p_val < alpha,
        alpha=alpha,
        effect=eff_val,
        metric=metric,
        magnitude=magnitude,
        checks=checks,
        posthoc=None,
        warns=warns,
    )
