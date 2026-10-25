"""Autonomous statistical router and hierarchical assumption lattice."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pyscio.core.guards import (
    check_groups,
    check_low_power,
    check_paired,
    check_sample_size,
    check_variance,
    clamp_p,
    clean_1d,
)
from pyscio.core.results import CheckResult, TestResult
from pyscio.functional.checks import check_normality, levene
from pyscio.functional.effects import (
    cliffs_d,
    cohen_d,
    cramers_v,
    eps_sq,
    eta_sq,
    hedges_g,
    interpret_effect,
    odds_ratio,
)
from pyscio.functional.posthoc import dunn, games_howell, tukey
from pyscio.functional.tests import (
    anova,
    chisq,
    fisher,
    kruskal,
    mann_whitney,
    one_sample_t,
    paired_t,
    student_t,
    welch_anova,
    welch_t,
    wilcoxon,
)


def evaluate_normality_with_clt(
    arr: np.ndarray,
    name: str = "sample",
    alpha: float = 0.05,
) -> tuple[bool, CheckResult]:
    """Check normality with the Large-N Central Limit Theorem (CLT) guard.

    If sample size N >= 500, formal normality tests (Shapiro-Wilk, D'Agostino) have excessive power
    and reject normality even for micro-deviations. If |skewness| < 1.0 and |excess kurtosis| < 2.0,
    the sampling distribution of the mean is asymptotically normal by the CLT.
    """
    n_res = check_normality(arr, alpha=alpha)
    n = len(arr)
    skew = abs(n_res.skewness)
    kurt = abs(n_res.excess_kurtosis)

    clt_holds = n >= 500 and skew < 1.0 and kurt < 2.0

    if n_res.is_normal:
        passed = True
        details = f"{n_res.test_used}: p={n_res.p:.4g} >= {alpha}"
        clt_note = None
    elif clt_holds:
        passed = True
        details = (
            f"{n_res.test_used} rejected (p={n_res.p:.4g}), but N={n:,} >= 500, "
            f"skewness={n_res.skewness:.2f}, kurtosis={n_res.excess_kurtosis:.2f}"
        )
        clt_note = "Central Limit Theorem holds (N >= 500, moderate skew/kurtosis); parametric test robust"
    else:
        passed = False
        details = (
            f"{n_res.test_used} rejected (p={n_res.p:.4g} < {alpha}, "
            f"skewness={n_res.skewness:.2f}, kurtosis={n_res.excess_kurtosis:.2f})"
        )
        clt_note = None

    check = CheckResult(
        name=f"Normality ({name})",
        stat=n_res.stat,
        p=n_res.p,
        passed=passed,
        details=details,
        clt_note=clt_note,
    )
    return passed, check


def route_independent_two_sample(
    g1: np.ndarray | list[float],
    g2: np.ndarray | list[float],
    x_name: str = "group",
    y_name: str = "target",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Route two independent samples through the normality, CLT, and homoscedasticity lattice."""
    arr1 = clean_1d(g1, name="group 1")
    arr2 = clean_1d(g2, name="group 2")
    check_groups(arr1, arr2)
    check_sample_size(arr1, min_n=3, name="group 1")
    check_sample_size(arr2, min_n=3, name="group 2")
    check_variance(arr1, name="group 1")
    check_variance(arr2, name="group 2")

    checks: list[CheckResult] = []
    warns: list[str] = []

    # Normality checks with CLT guard
    norm1, check1 = evaluate_normality_with_clt(arr1, name="group 1", alpha=alpha)
    norm2, check2 = evaluate_normality_with_clt(arr2, name="group 2", alpha=alpha)
    checks.extend([check1, check2])

    # Statistical power check
    total_n = len(arr1) + len(arr2)
    pwr_warn = check_low_power(total_n)
    if pwr_warn is not None:
        warns.append(pwr_warn)

    is_parametric = norm1 and norm2

    if is_parametric:
        # Check homoscedasticity
        homo_res = levene(arr1, arr2, center="median", alpha=alpha)
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
            test_res = student_t(arr1, arr2, alternative=alternative, conf_level=1.0 - alpha)
            test_name = "Student's Two-Sample t-Test"
        else:
            test_res = welch_t(arr1, arr2, alternative=alternative, conf_level=1.0 - alpha)
            test_name = "Welch's Two-Sample t-Test"

        stat_val = test_res.stat
        p_val, p_display = clamp_p(test_res.p)
        dof: float | tuple[float, ...] | None = test_res.dof
        ci = test_res.ci

        # Small sample size bias correction (Hedges' g) vs Cohen's d
        if total_n < 50:
            eff_val = hedges_g(arr1, arr2)
            metric = "Hedges' g"
        else:
            eff_val = cohen_d(arr1, arr2)
            metric = "Cohen's d"
        magnitude = interpret_effect(metric, eff_val)

    else:
        # Non-parametric: Mann-Whitney U test
        test_res_np = mann_whitney(arr1, arr2, alternative=alternative)
        test_name = "Mann-Whitney U Test"
        stat_val = test_res_np.stat
        p_val, p_display = clamp_p(test_res_np.p)
        dof = None
        ci = None

        eff_val = cliffs_d(arr1, arr2)
        metric = "Cliff's delta"
        magnitude = interpret_effect("cliff", eff_val)

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


def route_paired_two_sample(
    g1: np.ndarray | list[float],
    g2: np.ndarray | list[float],
    x_name: str = "condition_1",
    y_name: str = "condition_2",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Route paired/repeated-measures samples through the difference normality lattice."""
    arr1 = clean_1d(g1, name="sample_1")
    arr2 = clean_1d(g2, name="sample_2")
    check_paired(arr1, arr2, check_diff_var=False)
    diff = arr1 - arr2
    check_sample_size(diff, min_n=3, name="differences")

    checks: list[CheckResult] = []
    warns: list[str] = []

    norm_diff, check_diff = evaluate_normality_with_clt(diff, name="differences", alpha=alpha)
    checks.append(check_diff)

    pwr_warn = check_low_power(len(diff))
    if pwr_warn is not None:
        warns.append(pwr_warn)

    if norm_diff:
        test_res = paired_t(arr1, arr2, alternative=alternative, conf_level=1.0 - alpha)
        test_name = "Paired Samples t-Test"
        stat_val = test_res.stat
        p_val, p_display = clamp_p(test_res.p)
        dof = test_res.dof
        ci = test_res.ci

        std_diff = float(np.std(diff, ddof=1))
        eff_val = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0
        metric = "Cohen's d (paired)"
        magnitude = interpret_effect("cohen", eff_val)
    else:
        test_res_np = wilcoxon(arr1, arr2, alternative=alternative)
        test_name = "Wilcoxon Signed-Rank Test"
        stat_val = test_res_np.stat
        p_val, p_display = clamp_p(test_res_np.p)
        dof = None
        ci = None

        eff_val = cliffs_d(arr1, arr2)
        metric = "Rank-biserial r"
        magnitude = interpret_effect("cliff", eff_val)

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


def route_one_sample(
    g: np.ndarray | list[float],
    popmean: float = 0.0,
    x_name: str = "sample",
    y_name: str = "mu_0",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Route single sample against hypothetical population mean mu_0."""
    arr = clean_1d(g, name=x_name)
    check_sample_size(arr, min_n=3, name=x_name)
    check_variance(arr, name=x_name)

    checks: list[CheckResult] = []
    warns: list[str] = []

    norm_res, check_res = evaluate_normality_with_clt(arr, name=x_name, alpha=alpha)
    checks.append(check_res)

    if norm_res:
        test_res = one_sample_t(
            arr, popmean=popmean, alternative=alternative, conf_level=1.0 - alpha
        )
        test_name = "One-Sample t-Test"
        stat_val = test_res.stat
        p_val, p_display = clamp_p(test_res.p)
        dof = test_res.dof
        ci = test_res.ci

        s_val = float(np.std(arr, ddof=1))
        eff_val = float((np.mean(arr) - popmean) / s_val) if s_val > 0 else 0.0
        metric = "Cohen's d (one-sample)"
        magnitude = interpret_effect("cohen", eff_val)
    else:
        test_res_np = wilcoxon(arr - popmean, alternative=alternative)
        test_name = "Wilcoxon Signed-Rank Test (One-Sample)"
        stat_val = test_res_np.stat
        p_val, p_display = clamp_p(test_res_np.p)
        dof = None
        ci = None
        eff_val = None
        metric = None
        magnitude = None

    return TestResult(
        test_name=test_name,
        x_name=x_name,
        y_name=f"{y_name}={popmean}",
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
        raise ValueError("route_k_sample requires k >= 3 groups.")

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

    # Normality across all groups
    all_normal = True
    for i, arr in enumerate(cleaned):
        norm, chk = evaluate_normality_with_clt(arr, name=names[i], alpha=alpha)
        checks.append(chk)
        if not norm:
            all_normal = False

    posthoc_df: pd.DataFrame | None = None

    if all_normal:
        # Check homoscedasticity across all k groups
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
            # One-Way ANOVA
            a_res = anova(*cleaned)
            test_name = "One-Way ANOVA"
            stat_val = a_res.stat
            p_val, p_display = clamp_p(a_res.p)
            dof: float | tuple[float, ...] | None = (a_res.df_between, a_res.df_within)
            eff_val = eta_sq(*cleaned)
            metric = "Eta-squared (η²)"
            magnitude = interpret_effect("eta", eff_val)

            # Auto trigger Tukey HSD if omnibus test is significant
            if p_val < alpha:
                posthoc_df = tukey(*cleaned, names=names, alpha=alpha)
        else:
            # Welch's ANOVA for unequal variances
            w_res = welch_anova(*cleaned)
            test_name = "Welch's ANOVA"
            stat_val = w_res.stat
            p_val, p_display = clamp_p(w_res.p)
            dof = (w_res.df_between, w_res.df_within)
            eff_val = eta_sq(*cleaned)
            metric = "Eta-squared (η²)"
            magnitude = interpret_effect("eta", eff_val)

            # Auto trigger Games-Howell if omnibus test is significant
            if p_val < alpha:
                posthoc_df = games_howell(*cleaned, names=names, alpha=alpha)
    else:
        # Non-parametric Kruskal-Wallis H test
        kw_res = kruskal(*cleaned)
        test_name = "Kruskal-Wallis H Test"
        stat_val = kw_res.stat
        p_val, p_display = clamp_p(kw_res.p)
        dof = float(k - 1)
        eff_val = eps_sq(*cleaned)
        metric = "Epsilon-squared (ε²)"
        magnitude = interpret_effect("eta", eff_val)

        # Auto trigger Dunn's test with FDR BH correction if significant
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


def route_contingency(
    table_or_df: pd.DataFrame | np.ndarray,
    x_name: str = "variable_1",
    y_name: str = "variable_2",
    alpha: float = 0.05,
) -> TestResult:
    """Route 2D contingency table through Cochran's rule (Fisher's exact vs Pearson's Chi-squared)."""
    table = np.asarray(table_or_df, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError(f"Contingency table must be 2D, got shape {table.shape}")

    r, c = table.shape
    if r < 2 or c < 2:
        raise ValueError(f"Contingency table must be at least 2x2, got {r}x{c}")

    checks: list[CheckResult] = []
    warns: list[str] = []

    # Compute expected frequencies under independence
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
        # Fisher's Exact Test
        f_res = fisher(table, alternative="two-sided")
        test_name = "Fisher's Exact Test"
        stat_val = f_res.stat
        p_val, p_display = clamp_p(f_res.p)
        dof = None

        or_res = odds_ratio(table, conf_level=1.0 - alpha)
        eff_val: float | None = or_res.or_val
        metric: str | None = "Odds Ratio"
        ci: tuple[float, float] | None = or_res.ci
        magnitude: str | None = None
    else:
        # Pearson's Chi-squared test of independence
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
