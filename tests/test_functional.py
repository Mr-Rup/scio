"""Unit and mathematical accuracy tests for pyscio.functional kernels."""

from __future__ import annotations

import numpy as np

import pyscio.functional as F

# =============================================================================
# 1. Assumption Checks Tests
# =============================================================================

def test_normality_checks() -> None:
    rng = np.random.default_rng(42)
    normal_data = rng.normal(loc=0.0, scale=1.0, size=100)
    exp_data = rng.exponential(scale=2.0, size=100)

    # Shapiro-Wilk
    res_norm = F.shapiro(normal_data)
    assert res_norm.is_normal is True
    assert res_norm.test_used == "Shapiro-Wilk"

    res_exp = F.shapiro(exp_data)
    assert res_exp.is_normal is False

    # D'Agostino K^2
    res_dag = F.dagostino(normal_data)
    assert res_dag.test_used == "D'Agostino's K²"
    assert res_dag.is_normal is True

    # Adaptive router check_normality
    res_adapt = F.check_normality(normal_data)
    assert res_adapt.test_used == "Shapiro-Wilk"  # N <= 5000


def test_homoscedasticity_checks() -> None:
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 50)
    g2 = rng.normal(0, 1, 50)
    g3 = rng.normal(0, 4, 50)  # High variance

    # Equal variances
    lev_equal = F.levene(g1, g2)
    assert lev_equal.is_homoscedastic is True

    # Unequal variances
    lev_unequal = F.levene(g1, g3)
    assert lev_unequal.is_homoscedastic is False

    # Bartlett
    bart = F.bartlett(g1, g2)
    assert bart.is_homoscedastic is True


def test_iqr_bounds() -> None:
    data = [10, 11, 12, 10, 11, 12, 100]  # 100 is an extreme outlier
    low, high, outliers = F.iqr_bounds(data)
    assert low < 10
    assert high < 100
    assert outliers == 1


# =============================================================================
# 2. Hypothesis Testing Kernels Tests
# =============================================================================

def test_student_and_welch_t() -> None:
    rng = np.random.default_rng(42)
    g1 = rng.normal(10, 2, 40)
    g2 = rng.normal(13, 2, 40)

    # Student's t
    st_res = F.student_t(g1, g2)
    assert st_res.stat < 0
    assert st_res.p < 0.001
    assert st_res.dof == 78.0
    assert st_res.ci[0] < st_res.ci[1]

    # Welch's t
    w_res = F.welch_t(g1, g2)
    assert w_res.stat < 0
    assert w_res.p < 0.001
    assert w_res.dof > 0


def test_paired_and_one_sample_t() -> None:
    before = [10.0, 12.0, 11.5, 9.8, 10.5]
    after = [12.0, 14.2, 13.0, 11.9, 12.8]

    p_res = F.paired_t(before, after)
    assert p_res.dof == 4.0
    assert p_res.p < 0.01

    one_res = F.one_sample_t(before, popmean=10.0)
    assert one_res.dof == 4.0
    assert one_res.p > 0.05


def test_nonparametric_mann_whitney_and_wilcoxon() -> None:
    g1 = [1, 2, 3, 4, 5]
    g2 = [6, 7, 8, 9, 10]

    mw = F.mann_whitney(g1, g2)
    assert mw.stat == 0.0
    assert mw.p < 0.02

    w = F.wilcoxon([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert w.p < 0.07


def test_anova_and_welch_anova() -> None:
    g1 = [10, 11, 12, 10, 11]
    g2 = [15, 16, 17, 15, 16]
    g3 = [20, 21, 22, 20, 21]

    a_res = F.anova(g1, g2, g3)
    assert a_res.stat > 10.0
    assert a_res.p < 0.001
    assert a_res.df_between == 2.0
    assert a_res.df_within == 12.0

    w_res = F.welch_anova(g1, g2, g3)
    assert w_res.stat > 10.0
    assert w_res.p < 0.001


def test_kruskal() -> None:
    g1 = [1, 2, 3]
    g2 = [10, 11, 12]
    g3 = [20, 21, 22]
    k_res = F.kruskal(g1, g2, g3)
    assert k_res.stat > 5.0
    assert k_res.p < 0.05


def test_chisq_and_fisher() -> None:
    table = [[10, 20], [20, 10]]
    c_res = F.chisq(table)
    assert c_res.stat > 0
    assert c_res.dof == 1

    f_res = F.fisher(table)
    assert f_res.p < 0.05


# =============================================================================
# 3. Effect Size Kernels Tests (including O(N log N) Cliff's delta)
# =============================================================================

def test_cohen_d_and_hedges_g() -> None:
    g1 = [10.0, 11.0, 12.0, 10.0, 11.0]
    g2 = [13.0, 14.0, 15.0, 13.0, 14.0]

    d = F.cohen_d(g1, g2)
    assert d < -1.5  # Large negative difference

    g = F.hedges_g(g1, g2)
    # Hedges g is slightly smaller in absolute value due to small-sample correction
    assert abs(g) < abs(d)


def test_cliffs_d_fast_identity() -> None:
    """Verify O(N log N) Cliff's delta identity: delta = (2U / (n1*n2)) - 1."""
    g1 = [1.0, 2.0, 3.0]
    g2 = [4.0, 5.0, 6.0]

    # All g1 are smaller than all g2 => delta = -1.0
    d_neg = F.cliffs_d(g1, g2)
    assert np.isclose(d_neg, -1.0)

    # Identical arrays => delta = 0.0
    d_zero = F.cliffs_d(g1, g1)
    assert np.isclose(d_zero, 0.0)

    # All g2 are larger than g1 => delta = 1.0
    d_pos = F.cliffs_d(g2, g1)
    assert np.isclose(d_pos, 1.0)


def test_eta_sq_and_cramers_v() -> None:
    g1 = [1, 2, 3]
    g2 = [10, 11, 12]
    eta = F.eta_sq(g1, g2)
    assert 0.8 < eta <= 1.0

    table = [[50, 10], [10, 50]]
    v = F.cramers_v(table)
    assert 0.5 < v <= 1.0


def test_odds_ratio() -> None:
    table = [[40, 10], [10, 40]]
    or_res = F.odds_ratio(table)
    assert np.isclose(or_res.or_val, 16.0)
    assert or_res.ci[0] < 16.0 < or_res.ci[1]


def test_interpret_effect() -> None:
    assert F.interpret_effect("cohen_d", 0.1) == "negligible"
    assert F.interpret_effect("cohen_d", 0.35) == "small"
    assert F.interpret_effect("cohen_d", 0.65) == "medium"
    assert F.interpret_effect("cohen_d", 0.95) == "large"


# =============================================================================
# 4. Post-Hoc Pairwise Tests
# =============================================================================

def test_tukey_hsd() -> None:
    g1 = [10, 11, 12, 10, 11]
    g2 = [15, 16, 17, 15, 16]
    g3 = [20, 21, 22, 20, 21]

    df_tukey = F.tukey(g1, g2, g3, names=["A", "B", "C"])
    assert len(df_tukey) == 3
    assert "reject" in df_tukey.columns
    assert df_tukey["reject"].all()  # All pairs are strongly different


def test_games_howell() -> None:
    g1 = [10, 11, 12, 10, 11]
    g2 = [20, 22, 24, 21, 23]
    df_gh = F.games_howell(g1, g2, names=["Control", "Dose"])
    assert len(df_gh) == 1
    assert df_gh["reject"].iloc[0]


def test_dunn_test() -> None:
    g1 = [1, 2, 3, 2, 1]
    g2 = [10, 12, 11, 10, 12]
    g3 = [20, 21, 22, 20, 21]
    df_dunn = F.dunn(g1, g2, g3, p_adjust="fdr_bh")
    assert len(df_dunn) == 3
    assert "p_adj" in df_dunn.columns


# =============================================================================
# 5. Multiple Testing & Power Analysis Tests
# =============================================================================

def test_fdr_bh() -> None:
    raw_p = [0.001, 0.01, 0.04, 0.08, 0.50]
    reject, q = F.fdr_bh(raw_p, alpha=0.05)
    assert len(reject) == 5
    assert len(q) == 5
    # BH q-values must be monotonically non-decreasing for ordered p
    sorted_q = q[np.argsort(raw_p)]
    assert np.all(np.diff(sorted_q) >= -1e-15)


def test_bonferroni_and_holm() -> None:
    raw_p = [0.01, 0.02, 0.05]
    _, bonf = F.bonferroni(raw_p)
    assert np.isclose(bonf[0], 0.03)

    _, h = F.holm(raw_p)
    assert h[0] <= h[1] <= h[2]


def test_power_t_and_anova() -> None:
    # High sample size + medium effect => high power
    pow_t = F.power_t(n=100, d=0.5, alpha=0.05)
    assert 0.85 < pow_t < 1.0

    pow_a = F.power_anova(k=3, n_per_group=50, eta_squared=0.15, alpha=0.05)
    assert 0.85 < pow_a < 1.0
