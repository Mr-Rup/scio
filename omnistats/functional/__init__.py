"""omnistats.functional: Pure, stateless numerical kernels for hypothesis testing.

Canonical usage:
    import omnistats.functional as F

    stat, p, dof, ci, _, _ = F.welch_t(group_a, group_b)
    delta = F.cliffs_d(group_a, group_b)  # O(N log N)
    reject, q_vals = F.fdr_bh(p_values)
"""

from __future__ import annotations

from omnistats.core.types import (
    BayesFactorResult,
    CorrelationResult,
    LogRankResult,
    RMAnovaResult,
    StationarityResult,
    SurvivalCurveResult,
    TOSTResult,
)
from omnistats.functional.adjust import (
    bonferroni,
    fdr_bh,
    holm,
    power_anova,
    power_t,
    solve_power_t,
)
from omnistats.functional.bayes import bayes_factor_t, interpret_bf
from omnistats.functional.checks import (
    HomoscedasticityResult,
    NormalityResult,
    bartlett,
    check_normality,
    check_pathology,
    dagostino,
    iqr_bounds,
    levene,
    sarle_bimodality,
    shapiro,
)
from omnistats.functional.correlation import (
    kendall_tau,
    partial_corr,
    pearson_r,
    spearman_r,
)
from omnistats.functional.effects import (
    OddsRatioResult,
    cliffs_d,
    cliffs_d_ci,
    cohen_d,
    cohen_d_ci,
    cramers_v,
    eps_sq,
    eta_sq,
    hedges_g,
    hedges_g_ci,
    interpret_effect,
    kendalls_w,
    odds_ratio,
    rank_biserial,
)
from omnistats.functional.posthoc import (
    dunn,
    games_howell,
    tukey,
)
from omnistats.functional.resample import (
    bootstrap_ci,
    permutation_test,
    tost,
)
from omnistats.functional.rm import rm_anova
from omnistats.functional.robust import (
    huber_location,
    trimmed_mean,
    winsorized_mean,
    winsorized_var,
    yuen_t,
)
from omnistats.functional.survival import kaplan_meier, logrank_test
from omnistats.functional.tests import (
    ANOVAResult,
    ChiSqResult,
    FisherResult,
    NonParametricResult,
    TTestResult,
    anova,
    chisq,
    fisher,
    friedman,
    kruskal,
    mann_whitney,
    mcnemar,
    one_sample_t,
    paired_t,
    student_t,
    welch_anova,
    welch_t,
    wilcoxon,
)
from omnistats.functional.timeseries import adf_test, durbin_watson, ljung_box

__all__ = [
    "ANOVAResult",
    "BayesFactorResult",
    "ChiSqResult",
    "CorrelationResult",
    "FisherResult",
    "HomoscedasticityResult",
    "LogRankResult",
    "NonParametricResult",
    "NormalityResult",
    "OddsRatioResult",
    "RMAnovaResult",
    "StationarityResult",
    "SurvivalCurveResult",
    "TOSTResult",
    "TTestResult",
    "adf_test",
    "anova",
    "bartlett",
    "bayes_factor_t",
    "bonferroni",
    "bootstrap_ci",
    "check_normality",
    "check_pathology",
    "chisq",
    "cliffs_d",
    "cliffs_d_ci",
    "cohen_d",
    "cohen_d_ci",
    "cramers_v",
    "dagostino",
    "dunn",
    "durbin_watson",
    "eps_sq",
    "eta_sq",
    "fdr_bh",
    "fisher",
    "friedman",
    "games_howell",
    "hedges_g",
    "hedges_g_ci",
    "holm",
    "huber_location",
    "interpret_bf",
    "interpret_effect",
    "iqr_bounds",
    "kaplan_meier",
    "kendall_tau",
    "kendalls_w",
    "kruskal",
    "levene",
    "ljung_box",
    "logrank_test",
    "mann_whitney",
    "mcnemar",
    "odds_ratio",
    "one_sample_t",
    "paired_t",
    "partial_corr",
    "pearson_r",
    "permutation_test",
    "power_anova",
    "power_t",
    "rank_biserial",
    "rm_anova",
    "sarle_bimodality",
    "shapiro",
    "solve_power_t",
    "spearman_r",
    "student_t",
    "tost",
    "trimmed_mean",
    "tukey",
    "welch_anova",
    "welch_t",
    "wilcoxon",
    "winsorized_mean",
    "winsorized_var",
    "yuen_t",
]
