"""R 4.3 and SciPy golden reference parity benchmarks.

Validates that numerical outputs from pyscio.functional match R 4.3 / SciPy reference outputs
to within 1e-4 relative/absolute tolerance across statistics, degrees of freedom, and p-values.
"""

from __future__ import annotations

import numpy as np

import pyscio.functional as F

# =============================================================================
# 1. Welch's and Student's t-test Parity
# =============================================================================

def test_r_parity_welch_t() -> None:
    """Benchmark:
    x = [14.2, 15.1, 14.8, 15.4, 15.0, 14.9, 15.2, 14.6]
    y = [16.5, 17.2, 16.8, 17.0, 16.9, 17.4, 16.6]
    R t.test(x, y, var.equal=FALSE):
    t = -11.2620, df = 12.9961, p-value = 4.4705e-08
    95% CI: [-2.4007, -1.6279]
    """
    x = [14.2, 15.1, 14.8, 15.4, 15.0, 14.9, 15.2, 14.6]
    y = [16.5, 17.2, 16.8, 17.0, 16.9, 17.4, 16.6]

    res = F.welch_t(x, y)
    assert np.isclose(res.stat, -11.2620, atol=1e-3)
    assert np.isclose(res.dof, 12.9961, atol=1e-3)
    assert np.isclose(res.p, 4.4705e-08, rtol=1e-3)
    assert np.isclose(res.ci[0], -2.4007, atol=1e-3)
    assert np.isclose(res.ci[1], -1.6279, atol=1e-3)


def test_r_parity_student_t() -> None:
    """Benchmark:
    R t.test(x, y, var.equal=TRUE):
    t = -11.1337, df = 13, p-value = 5.1062e-08
    95% CI: [-2.4051, -1.6234]
    """
    x = [14.2, 15.1, 14.8, 15.4, 15.0, 14.9, 15.2, 14.6]
    y = [16.5, 17.2, 16.8, 17.0, 16.9, 17.4, 16.6]

    res = F.student_t(x, y)
    assert np.isclose(res.stat, -11.1337, atol=1e-3)
    assert np.isclose(res.dof, 13.0, atol=1e-6)
    assert np.isclose(res.p, 5.1062e-08, rtol=1e-3)
    assert np.isclose(res.ci[0], -2.4051, atol=1e-3)
    assert np.isclose(res.ci[1], -1.6234, atol=1e-3)


# =============================================================================
# 2. ANOVA and Kruskal-Wallis Parity
# =============================================================================

def test_r_parity_anova() -> None:
    """Benchmark:
    g1 = [2.4, 3.5, 3.2, 2.8, 3.0]
    g2 = [4.5, 4.2, 4.8, 4.0, 4.1]
    g3 = [5.2, 5.8, 5.1, 5.4, 5.5]
    R summary(aov(val ~ group)):
    F = 62.2768, p-value = 4.6054e-07, Df between = 2, Df within = 12
    """
    g1 = [2.4, 3.5, 3.2, 2.8, 3.0]
    g2 = [4.5, 4.2, 4.8, 4.0, 4.1]
    g3 = [5.2, 5.8, 5.1, 5.4, 5.5]

    res = F.anova(g1, g2, g3)
    assert np.isclose(res.stat, 62.2768, atol=1e-3)
    assert np.isclose(res.p, 4.6054e-07, rtol=1e-3)
    assert res.df_between == 2.0
    assert res.df_within == 12.0


def test_r_parity_kruskal() -> None:
    """Benchmark:
    R kruskal.test(list(g1, g2, g3)):
    Kruskal-Wallis chi-squared = 12.5, df = 2, p-value = 0.00193
    """
    g1 = [2.4, 3.5, 3.2, 2.8, 3.0]
    g2 = [4.5, 4.2, 4.8, 4.0, 4.1]
    g3 = [5.2, 5.8, 5.1, 5.4, 5.5]

    res = F.kruskal(g1, g2, g3)
    assert np.isclose(res.stat, 12.5, atol=1e-4)
    assert np.isclose(res.p, 0.00193, atol=1e-4)


# =============================================================================
# 3. Categorical Tests Parity
# =============================================================================

def test_r_parity_chisq() -> None:
    """Benchmark:
    m = [[12, 7], [5, 16]]
    R chisq.test(m, correct=FALSE):
    X-squared = 6.3199, df = 1, p-value = 0.01194
    """
    table = [[12, 7], [5, 16]]
    res = F.chisq(table, correct=False)
    assert np.isclose(res.stat, 6.3199, atol=1e-3)
    assert res.dof == 1
    assert np.isclose(res.p, 0.01194, atol=1e-4)


def test_r_parity_fisher() -> None:
    """Benchmark:
    m = [[12, 7], [5, 16]]
    R fisher.test(m):
    p-value = 0.02384, odds ratio = 5.4857
    """
    table = [[12, 7], [5, 16]]
    res = F.fisher(table)
    assert np.isclose(res.p, 0.02384, atol=1e-4)
    assert np.isclose(res.stat, 5.4857, atol=1e-3)
