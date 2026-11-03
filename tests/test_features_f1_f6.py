"""Tests for Phase R8 & F1-F6 features: Kendall tau, partial correlation, McNemar, Friedman, APA, and solve_power."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyscio as sc
import pyscio.functional as F
from pyscio.core.errors import StatError


def test_kendall_tau_and_partial_corr() -> None:
    rng = np.random.default_rng(42)
    z = rng.normal(0.0, 1.0, 120)
    # x and y are correlated purely through confounder z
    x = 2.0 * z + rng.normal(0.0, 0.4, 120)
    y = 2.0 * z + rng.normal(0.0, 0.4, 120)

    # Kendall tau-b detects strong monotonic association
    tau_res = F.kendall_tau(x, y)
    assert tau_res.r > 0.65
    assert tau_res.p < 1e-10
    assert tau_res.ci[0] < tau_res.r < tau_res.ci[1]

    # Marginal Pearson correlation is high (~0.92), but partial correlation controlling for z is ~0
    raw_r = F.pearson_r(x, y)
    part_r = F.partial_corr(x, y, z, method="pearson")
    part_rho = F.partial_corr(x, y, z, method="spearman")
    assert raw_r.r > 0.85
    assert abs(part_r.r) < 0.25
    assert abs(part_rho.r) < 0.25
    assert part_r.dof == 117.0


def test_mcnemar_exact_and_asymptotic() -> None:
    # Small discordant count (b + c = 12 < 25) -> exact binomial
    small_table = [[40, 10], [2, 35]]
    res_exact = F.mcnemar(small_table)
    assert res_exact.stat == 2.0
    assert res_exact.p < 0.05

    # Large discordant count (b + c = 45 >= 25) -> continuity-corrected chi-square
    large_table = [[100, 35], [10, 90]]
    res_chi2 = F.mcnemar(large_table, exact=False, correction=True)
    # (|35 - 10| - 1)^2 / 45 = 24^2 / 45 = 576 / 45 = 12.8
    assert np.isclose(res_chi2.stat, 12.8)
    assert res_chi2.p < 0.001


def test_friedman_and_paired_k_sample_routing() -> None:
    rng = np.random.default_rng(77)
    subj_baseline = rng.normal(50.0, 5.0, 25)
    c1 = subj_baseline + rng.normal(0.0, 1.0, 25)
    c2 = subj_baseline + 4.0 + rng.normal(0.0, 1.0, 25)
    c3 = subj_baseline + 9.0 + rng.normal(0.0, 1.0, 25)

    f_res = F.friedman(c1, c2, c3)
    w_val = F.kendalls_w(c1, c2, c3)
    assert f_res.p < 1e-6
    assert 0.7 < w_val <= 1.0

    df = pd.DataFrame(
        {
            "condition": ["T0"] * 25 + ["T1"] * 25 + ["T2"] * 25,
            "score": np.concatenate([c1, c2, c3]),
        }
    )
    routed = sc.compare(df, x="condition", y="score", paired=True)
    assert routed.test_name == "Friedman Repeated-Measures Test"
    assert routed.sig is True
    assert routed.metric == "Kendall's W"
    assert routed.posthoc is not None
    assert len(routed.posthoc) == 3


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
    # Solve for sample size n per group for d=0.5, power=0.80, alpha=0.05 -> n = 64
    n_req = sc.solve_power(effect=0.5, power=0.80, alpha=0.05)
    assert n_req == 64.0

    # Verify forward power at n=64 is >= 0.80 and at n=63 is < 0.80
    pwr_64 = sc.solve_power(effect=0.5, n=64, alpha=0.05)
    pwr_63 = sc.solve_power(effect=0.5, n=63, alpha=0.05)
    assert pwr_64 >= 0.80
    assert pwr_63 < 0.80

    # Solve for minimum detectable effect size d given n=64, power=0.80
    d_req = sc.solve_power(n=64, power=0.80, alpha=0.05)
    assert np.isclose(d_req, 0.5, atol=0.01)

    with pytest.raises(StatError):
        sc.solve_power(effect=0.5, n=64, power=0.80)
