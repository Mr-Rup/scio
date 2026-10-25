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
    dagostino,
    iqr_bounds,
    levene,
    shapiro,
)
from pyscio.functional.effects import (
    OddsRatioResult,
    cliffs_d,
    cohen_d,
    cramers_v,
    eps_sq,
    eta_sq,
    hedges_g,
    interpret_effect,
    odds_ratio,
    rank_biserial,
)
from pyscio.functional.posthoc import (
    dunn,
    games_howell,
    tukey,
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
    # Assumption checks
    "shapiro",
    "dagostino",
    "check_normality",
    "levene",
    "bartlett",
    "iqr_bounds",
    "NormalityResult",
    "HomoscedasticityResult",
    # Effect sizes
    "cohen_d",
    "hedges_g",
    "cliffs_d",
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
    # Multiple testing & power
    "fdr_bh",
    "bonferroni",
    "holm",
    "power_t",
    "power_anova",
]
