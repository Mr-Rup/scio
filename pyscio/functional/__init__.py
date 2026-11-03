"""pyscio.functional: Pure, stateless numerical kernels for hypothesis testing.

Canonical usage:
    import pyscio.functional as F

    stat, p, dof, ci, _, _ = F.welch_t(group_a, group_b)
    delta = F.cliffs_d(group_a, group_b)  # O(N log N)
    reject, q_vals = F.fdr_bh(p_values)
"""

from __future__ import annotations

from pyscio.core.types import CorrelationResult, TOSTResult
from pyscio.functional.adjust import (
    bonferroni,
    fdr_bh,
    holm,
    power_anova,
    power_t,
    solve_power_t,
)
from pyscio.functional.checks import (
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
from pyscio.functional.correlation import (
    kendall_tau,
    partial_corr,
    pearson_r,
    spearman_r,
)
from pyscio.functional.effects import (
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
from pyscio.functional.posthoc import (
    dunn,
    games_howell,
    tukey,
)
from pyscio.functional.resample import (
    bootstrap_ci,
    permutation_test,
    tost,
)
from pyscio.functional.robust import (
    huber_location,
    trimmed_mean,
    winsorized_mean,
    winsorized_var,
    yuen_t,
)
from pyscio.functional.tests import (
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

__all__ = [
    "ANOVAResult",
    "ChiSqResult",
    "CorrelationResult",
    "FisherResult",
    "HomoscedasticityResult",
    "NonParametricResult",
    "NormalityResult",
    "OddsRatioResult",
    "TOSTResult",
    "TTestResult",
    "anova",
    "bartlett",
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
    "interpret_effect",
    "iqr_bounds",
    "kendall_tau",
    "kendalls_w",
    "kruskal",
    "levene",
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
