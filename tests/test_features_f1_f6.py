"""Tests for Phase R8 & F1-F9 features: correlations, McNemar, RM-ANOVA/Friedman, Factorial ANOVA, Poisson GLM, Bayes Factors, APA, and solve_power."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import omnistats as sc
import omnistats.functional as F
from omnistats.core.errors import StatError


def test_kendall_tau_and_partial_corr() -> None:
    rng = np.random.default_rng(42)
    z = rng.normal(0.0, 1.0, 120)
    # x and y are correlated purely through confounder z
    x = 2.0 * z + rng.normal(0.0, 0.4, 120)
    y = 2.0 * z + rng.normal(0.0, 0.4, 120)

    tau_res = F.kendall_tau(x, y)
    assert tau_res.r > 0.65
    assert tau_res.p < 1e-10
    assert tau_res.ci[0] < tau_res.r < tau_res.ci[1]

    raw_r = F.pearson_r(x, y)
    part_r = F.partial_corr(x, y, z, method="pearson")
    part_rho = F.partial_corr(x, y, z, method="spearman")
    assert raw_r.r > 0.85
    assert abs(part_r.r) < 0.25
    assert abs(part_rho.r) < 0.25
    assert part_r.dof == 117.0


def test_mcnemar_exact_and_asymptotic() -> None:
    small_table = [[40, 10], [2, 35]]
    res_exact = F.mcnemar(small_table)
    assert res_exact.stat == 2.0
    assert res_exact.p < 0.05

    large_table = [[100, 35], [10, 90]]
    res_chi2 = F.mcnemar(large_table, exact=False, correction=True)
    assert np.isclose(res_chi2.stat, 12.8)
    assert res_chi2.p < 0.001


def test_rm_anova_and_friedman_paired_k_sample_routing() -> None:
    rng = np.random.default_rng(77)
    subj_baseline = rng.normal(50.0, 5.0, 25)
    c1 = subj_baseline + rng.normal(0.0, 1.0, 25)
    c2 = subj_baseline + 4.0 + rng.normal(0.0, 1.0, 25)
    c3 = subj_baseline + 9.0 + rng.normal(0.0, 1.0, 25)

    # 1. Normal repeated measures -> routes to One-Way Repeated-Measures ANOVA (Greenhouse-Geisser)
    rm_res = F.rm_anova(c1, c2, c3)
    assert rm_res.p_gg < 1e-6
    assert 0.5 <= rm_res.eps_gg <= 1.0
    assert rm_res.partial_eta_sq > 0.7

    df_norm = pd.DataFrame(
        {
            "condition": ["T0"] * 25 + ["T1"] * 25 + ["T2"] * 25,
            "score": np.concatenate([c1, c2, c3]),
        }
    )
    routed_rm = sc.compare(df_norm, x="condition", y="score", paired=True)
    assert routed_rm.test_name == "One-Way Repeated-Measures ANOVA (Greenhouse-Geisser)"
    assert routed_rm.sig is True
    assert routed_rm.metric == "Partial Eta-squared (η²_p)"
    assert routed_rm.posthoc is not None

    # 2. Skewed / non-normal repeated measures -> routes to Friedman Repeated-Measures Test
    s1 = rng.exponential(scale=1.0, size=30)
    s2 = s1 + rng.exponential(scale=3.0, size=30)
    s3 = s1 + rng.exponential(scale=8.0, size=30)
    df_skew = pd.DataFrame(
        {
            "condition": ["T0"] * 30 + ["T1"] * 30 + ["T2"] * 30,
            "score": np.concatenate([s1, s2, s3]),
        }
    )
    routed_fr = sc.compare(df_skew, x="condition", y="score", paired=True)
    assert routed_fr.test_name == "Friedman Repeated-Measures Test"
    assert routed_fr.sig is True
    assert routed_fr.metric == "Kendall's W"
    assert routed_fr.posthoc is not None


def test_factorial_anova_table_type2_and_type3() -> None:
    rng = np.random.default_rng(101)
    n = 120
    a = np.repeat(["A1", "A2"], n // 2)
    b = np.tile(["B1", "B2", "B3"], n // 3)
    y = (
        10.0
        + np.where(a == "A2", 4.0, 0.0)
        + np.where(b == "B2", 2.0, np.where(b == "B3", 5.0, 0.0))
        + rng.normal(0.0, 1.0, n)
    )
    df = pd.DataFrame({"y": y, "A": a, "B": b})

    tab2 = sc.anova_table(df, "y ~ A * B", typ=2)
    tab3 = df.scio.anova_table("y ~ A * B", typ=3)
    assert list(tab2["term"]) == ["A", "B", "A:B", "Residual"]
    assert tab2.loc[tab2["term"] == "A", "p_value"].iloc[0] < 1e-6
    assert tab2.loc[tab2["term"] == "B", "p_value"].iloc[0] < 1e-6
    assert tab2.loc[tab2["term"] == "A:B", "p_value"].iloc[0] > 0.05
    assert len(tab3) == 4


def test_poisson_glm_and_overdispersion_audit() -> None:
    rng = np.random.default_rng(303)
    n = 150
    x = rng.uniform(0.0, 2.0, n)
    lam = np.exp(0.5 + 0.8 * x)
    counts = rng.poisson(lam)
    df = pd.DataFrame({"counts": counts, "x": x})

    res = sc.poisson(df, "counts ~ x", robust_se="HC3")
    assert res.model_type == "Poisson Log-Linear GLM"
    assert "irr" in res.coef_df.columns
    slope_coef = float(res.coef_df.loc[res.coef_df["term"] == "x", "coef"].iloc[0])
    slope_irr = float(res.coef_df.loc[res.coef_df["term"] == "x", "irr"].iloc[0])
    assert abs(slope_coef - 0.8) < 0.15
    assert np.isclose(slope_irr, np.exp(slope_coef))

    # Inject severe overdispersion (multiply counts by 5) -> triggers overdispersion warning
    df_over = pd.DataFrame({"counts": counts * 5, "x": x})
    res_over = df_over.scio.poisson("counts ~ x")
    assert any("Overdispersion detected" in w for w in res_over.warns)


def test_jzs_bayes_factor_t() -> None:
    rng = np.random.default_rng(505)
    # Null is true (zero difference) -> BF01 > 3.0 (moderate evidence for H0)
    null_a = rng.normal(0.0, 1.0, 50)
    null_a = null_a - np.mean(null_a)
    null_b = rng.normal(0.0, 1.0, 50)
    null_b = null_b - np.mean(null_b)
    bf_null = F.bayes_factor_t(null_a, null_b)
    assert bf_null.bf01 > 3.0
    assert bf_null.evidence == "moderate evidence for H0"

    # Strong difference -> BF10 > 100 (extreme evidence for H1)
    alt_b = null_b + 1.5
    bf_alt = F.bayes_factor_t(null_a, alt_b)
    assert bf_alt.bf10 > 100.0
    assert bf_alt.evidence == "extreme evidence for H1"


def test_apa_7th_edition_reporting() -> None:
    rng = np.random.default_rng(19)
    a = rng.normal(10.0, 1.5, 40)
    b = rng.normal(13.0, 1.5, 40)

    res = sc.compare(a, b)
    apa_str = res.apa()
    assert "t(" in apa_str
    assert "p < .001" in apa_str
    assert "d = " in apa_str
    assert "95% CI [" in apa_str
    assert res.summary(format="apa") == apa_str

    df = pd.DataFrame({"x": a, "y": 2.5 * a + rng.normal(0.0, 1.0, 40)})
    ols_res = sc.ols(df, "y ~ x")
    ols_apa = ols_res.apa()
    assert "F(1, 38)" in ols_apa
    assert "R² =" in ols_apa
    assert ols_res.summary(format="apa") == ols_apa


def test_bidirectional_power_solver() -> None:
    n_req = sc.solve_power(effect=0.5, power=0.80, alpha=0.05)
    assert n_req == 64.0

    pwr_64 = sc.solve_power(effect=0.5, n=64, alpha=0.05)
    pwr_63 = sc.solve_power(effect=0.5, n=63, alpha=0.05)
    assert pwr_64 >= 0.80
    assert pwr_63 < 0.80

    d_req = sc.solve_power(n=64, power=0.80, alpha=0.05)
    assert np.isclose(d_req, 0.5, atol=0.01)

    with pytest.raises(StatError):
        sc.solve_power(effect=0.5, n=64, power=0.80)
