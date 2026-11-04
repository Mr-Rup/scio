"""Autonomous router for one-sample, two-sample (independent & paired), and correlation designs."""

from __future__ import annotations

import numpy as np

from omnistats.core.constants import SMALL_SAMPLE_HEDGES_N
from omnistats.core.guards import (
    check_groups,
    check_low_power,
    check_paired,
    check_sample_size,
    check_variance,
    clamp_p,
    clean_1d,
)
from omnistats.core.results import CheckResult, TestResult
from omnistats.functional.checks import levene
from omnistats.functional.correlation import pearson_r, spearman_r
from omnistats.functional.effects import (
    cliffs_d,
    cliffs_d_ci,
    cohen_d,
    cohen_d_ci,
    hedges_g,
    hedges_g_ci,
    interpret_effect,
)
from omnistats.functional.tests import (
    mann_whitney,
    one_sample_t,
    paired_t,
    student_t,
    welch_t,
    wilcoxon,
)
from omnistats.stat.router.normality import evaluate_normality_with_clt


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

    norm1, check1 = evaluate_normality_with_clt(arr1, name="group 1", alpha=alpha)
    norm2, check2 = evaluate_normality_with_clt(arr2, name="group 2", alpha=alpha)
    checks.extend([check1, check2])

    total_n = len(arr1) + len(arr2)
    pwr_warn = check_low_power(total_n)
    if pwr_warn is not None:
        warns.append(pwr_warn)

    if norm1 and norm2:
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
        ci: tuple[float, float] | None = test_res.ci

        if total_n < SMALL_SAMPLE_HEDGES_N:
            eff_val: float = hedges_g(arr1, arr2)
            eff_ci: tuple[float, float] | None = hedges_g_ci(arr1, arr2, conf_level=1.0 - alpha)
            metric: str = "Hedges' g"
        else:
            eff_val = cohen_d(arr1, arr2)
            eff_ci = cohen_d_ci(arr1, arr2, conf_level=1.0 - alpha)
            metric = "Cohen's d"
        magnitude: str = interpret_effect(metric, eff_val)
    else:
        test_res_np = mann_whitney(arr1, arr2, alternative=alternative)
        test_name = "Mann-Whitney U Test"
        stat_val = test_res_np.stat
        p_val, p_display = clamp_p(test_res_np.p)
        dof = None
        ci = None
        eff_val = cliffs_d(arr1, arr2)
        eff_ci = cliffs_d_ci(arr1, arr2, conf_level=1.0 - alpha, delta=eff_val)
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
        effect_ci=eff_ci,
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
        dof: float | None = test_res.dof
        ci: tuple[float, float] | None = test_res.ci

        std_diff = float(np.std(diff, ddof=1))
        eff_val = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0
        eff_ci: tuple[float, float] | None = (
            cohen_d_ci(arr1, arr2, conf_level=1.0 - alpha, paired=True)
            if std_diff > 0
            else (0.0, 0.0)
        )
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
        eff_ci = cliffs_d_ci(arr1, arr2, conf_level=1.0 - alpha, delta=eff_val)
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
        effect_ci=eff_ci,
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
        dof: float | None = test_res.dof
        ci: tuple[float, float] | None = test_res.ci
        s_val = float(np.std(arr, ddof=1))
        eff_f = float((np.mean(arr) - popmean) / s_val) if s_val > 0 else 0.0
        eff_val: float | None = eff_f
        eff_ci: tuple[float, float] | None = (
            cohen_d_ci(arr, popmean=popmean, conf_level=1.0 - alpha) if s_val > 0 else (0.0, 0.0)
        )
        metric: str | None = "Cohen's d (one-sample)"
        magnitude: str | None = interpret_effect("cohen", eff_f)
    else:
        test_res_np = wilcoxon(arr - popmean, alternative=alternative)
        test_name = "Wilcoxon Signed-Rank Test (One-Sample)"
        stat_val = test_res_np.stat
        p_val, p_display = clamp_p(test_res_np.p)
        dof = None
        ci = None
        eff_val = None
        eff_ci = None
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
        effect_ci=eff_ci,
        metric=metric,
        magnitude=magnitude,
        checks=checks,
        posthoc=None,
        warns=warns,
    )


def route_correlation(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    x_name: str = "x",
    y_name: str = "y",
    alpha: float = 0.05,
) -> TestResult:
    """Route two continuous variables through normality & pathology checks to Pearson r or Spearman rho."""
    arr_x = clean_1d(x, name=x_name)
    arr_y = clean_1d(y, name=y_name)
    check_paired(arr_x, arr_y, min_n=3, check_diff_var=False)
    check_variance(arr_x, name=x_name)
    check_variance(arr_y, name=y_name)

    norm_x, chk_x = evaluate_normality_with_clt(arr_x, name=x_name, alpha=alpha)
    norm_y, chk_y = evaluate_normality_with_clt(arr_y, name=y_name, alpha=alpha)
    checks = [chk_x, chk_y]
    warns: list[str] = []

    pwr_warn = check_low_power(len(arr_x))
    if pwr_warn is not None:
        warns.append(pwr_warn)

    if norm_x and norm_y:
        c_res = pearson_r(arr_x, arr_y, conf_level=1.0 - alpha)
        test_name = "Pearson Correlation Test"
    else:
        c_res = spearman_r(arr_x, arr_y, conf_level=1.0 - alpha)
        test_name = "Spearman Rank Correlation Test"

    p_val, p_display = clamp_p(c_res.p)
    return TestResult(
        test_name=test_name,
        x_name=x_name,
        y_name=y_name,
        stat=c_res.stat,
        p=p_val,
        p_display=p_display,
        dof=c_res.dof,
        ci=c_res.ci,
        ci_level=1.0 - alpha,
        sig=p_val < alpha,
        alpha=alpha,
        effect=c_res.r,
        effect_ci=c_res.ci,
        metric=c_res.method,
        magnitude=interpret_effect("cliff", c_res.r),
        checks=checks,
        posthoc=None,
        warns=warns,
    )
