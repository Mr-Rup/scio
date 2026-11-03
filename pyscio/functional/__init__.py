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
from pyscio.functional.correlation import pearson_r, spearman_r
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
    # Categorical contingency tests
    "chisq",
    "cliffs_d",
    "cliffs_d_ci",
    # Effect sizes & exact CIs
    "cohen_d",
    "cohen_d_ci",
    "cramers_v",
    "dagostino",
    "dunn",
    "eps_sq",
    "eta_sq",
    # Multiple testing & power
    "fdr_bh",
    "fisher",
    "games_howell",
    "hedges_g",
    "hedges_g_ci",
    "holm",
    "huber_location",
    "interpret_effect",
    "iqr_bounds",
    "kruskal",
    "levene",
    # Non-parametric tests
    "mann_whitney",
    "odds_ratio",
    "one_sample_t",
    "paired_t",
    # Resampling, correlation & equivalence
    "pearson_r",
    "permutation_test",
    "power_anova",
    "power_t",
    "rank_biserial",
    "sarle_bimodality",
    # Assumption checks & pathology guards
    "shapiro",
    "spearman_r",
    # Continuous parametric tests
    "student_t",
    "tost",
    # Robust statistics
    "trimmed_mean",
    # Post-hoc pairwise tests
    "tukey",
    "welch_anova",
    "welch_t",
    "wilcoxon",
    "winsorized_mean",
    "winsorized_var",
    "yuen_t",
]
