"""pyscio.functional: Pure, stateless numerical kernels for hypothesis testing.

Canonical usage:
    import pyscio.functional as F

    stat, p, dof, ci, _, _ = F.welch_t(group_a, group_b)
    delta = F.cliffs_d(group_a, group_b)  # O(N log N)
    reject, q_vals = F.fdr_bh(p_values)
"""

from __future__ import annotations

from pyscio.functional.adjust import (
    bonferroni,
    fdr_bh,
    holm,
    power_anova,
    power_t,
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
    kruskal,
    mann_whitney,
    one_sample_t,
    paired_t,
    student_t,
    welch_anova,
    welch_t,
    wilcoxon,
)

__all__ = [
    # Continuous parametric tests
    "student_t",
    "welch_t",
    "paired_t",
    "one_sample_t",
    "anova",
    "welch_anova",
    "TTestResult",
    "ANOVAResult",
    # Non-parametric tests
    "mann_whitney",
    "wilcoxon",
    "kruskal",
    "NonParametricResult",
    # Categorical contingency tests
    "chisq",
    "fisher",
    "ChiSqResult",
    "FisherResult",
    # Assumption checks & pathology guards
    "shapiro",
    "dagostino",
    "check_normality",
    "levene",
    "bartlett",
    "iqr_bounds",
    "sarle_bimodality",
    "check_pathology",
    "NormalityResult",
    "HomoscedasticityResult",
    # Effect sizes & exact CIs
    "cohen_d",
    "cohen_d_ci",
    "hedges_g",
    "hedges_g_ci",
    "cliffs_d",
    "cliffs_d_ci",
    "rank_biserial",
    "eta_sq",
    "eps_sq",
    "cramers_v",
    "odds_ratio",
    "interpret_effect",
    "OddsRatioResult",
    # Post-hoc pairwise tests
    "tukey",
    "games_howell",
    "dunn",
    # Resampling & equivalence
    "permutation_test",
    "bootstrap_ci",
    "tost",
    # Robust statistics
    "trimmed_mean",
    "winsorized_mean",
    "winsorized_var",
    "huber_location",
    "yuen_t",
    # Multiple testing & power
    "fdr_bh",
    "bonferroni",
    "holm",
    "power_t",
    "power_anova",
]
