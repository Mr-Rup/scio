"""Phase 7 and Argon Audit Pillars verification suite for pyscio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyscio as sc
import pyscio.functional as F
from pyscio.core.errors import StatError
from pyscio.core.results import RegressionResult, TestResult


class TestPillar1PathologyGuards:
    """Verify distributional pathology guards block false-positive CLT overrides at N >= 500."""

    def test_bimodal_mixture_blocks_clt(self) -> None:
        rng = np.random.default_rng(42)
        # Symmetric bimodal mixture: 0.5 * N(-5, 1) + 0.5 * N(5, 1), N = 600
        # Skewness ~ 0.0, Excess Kurtosis ~ -1.85 (would pass |skew|<1 and |kurt|<2 without BC guard)
        g1 = np.concatenate([rng.normal(-5.0, 1.0, 300), rng.normal(5.0, 1.0, 300)])
        g2 = np.concatenate([rng.normal(-4.5, 1.0, 300), rng.normal(5.5, 1.0, 300)])

        bc = F.sarle_bimodality(g1)
        assert bc > 0.555

        has_path, reasons = F.check_pathology(g1)
        assert has_path is True
        assert any("bimodal" in r.lower() for r in reasons)

        res = sc.compare(g1, g2)
        assert res.test_name == "Mann-Whitney U Test"
        assert res.checks[0].passed is False
        assert "pathology" in res.checks[0].details

    def test_zero_inflated_point_mass_blocks_clt(self) -> None:
        rng = np.random.default_rng(101)
        # 45% zeros + uniform/normal positive values, N = 600
        zeros = np.zeros(270)
        pos1 = rng.normal(3.0, 0.8, 330)
        pos2 = rng.normal(3.5, 0.8, 330)
        g1 = np.concatenate([zeros, pos1])
        g2 = np.concatenate([zeros, pos2])

        has_path, reasons = F.check_pathology(g1)
        assert has_path is True
        assert any("point-mass" in r.lower() for r in reasons)

        res = sc.compare(g1, g2)
        assert res.test_name == "Mann-Whitney U Test"

    def test_heavy_tailed_outlier_contamination_blocks_clt(self) -> None:
        rng = np.random.default_rng(2024)
        # Base normal N(0, 1) with 4% extreme symmetric outliers (+/- 12), N = 600
        base1 = rng.normal(0.0, 1.0, 570)
        outliers1 = np.array([12.0, -12.0] * 15)
        g1 = np.concatenate([base1, outliers1])
        g2 = rng.normal(0.5, 1.0, 600)

        has_path, reasons = F.check_pathology(g1)
        assert has_path is True
        assert any("heavy-tailed" in r.lower() for r in reasons)

        res = sc.compare(g1, g2)
        assert res.test_name == "Mann-Whitney U Test"


class TestPillar2ExactEffectSizeCIs:
    """Verify exact Non-Central t (NCT) CIs for Cohen's d, Hedges' g, and Cliff's delta."""

    def test_cohen_d_and_hedges_g_nct_ci(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(10.0, 2.0, 20)
        b = rng.normal(8.0, 2.0, 20)

        d_val = F.cohen_d(a, b)
        d_ci = F.cohen_d_ci(a, b, conf_level=0.95)
        assert d_ci[0] < d_val < d_ci[1]
        assert d_ci[0] > 0.0  # clear separation

        g_val = F.hedges_g(a, b)
        g_ci = F.hedges_g_ci(a, b, conf_level=0.95)
        assert g_ci[0] < g_val < g_ci[1]
        # Hedges' g shrinks Cohen's d toward zero by J
        assert abs(g_val) < abs(d_val)
        assert abs(g_ci[1]) < abs(d_ci[1])

        # Check TestResult integration
        res = sc.compare(a, b)
        assert res.effect_ci is not None
        assert np.isclose(res.effect_ci[0], g_ci[0])
        assert np.isclose(res.effect_ci[1], g_ci[1])

    def test_paired_and_one_sample_effect_ci(self) -> None:
        rng = np.random.default_rng(19)
        pre = rng.normal(50.0, 5.0, 30)
        post = pre - rng.normal(3.0, 2.0, 30)

        res_paired = sc.compare(pre, post, paired=True)
        assert res_paired.effect_ci is not None
        assert res_paired.effect is not None
        assert res_paired.effect_ci[0] < res_paired.effect < res_paired.effect_ci[1]

        res_one = sc.compare(pre, mu=45.0)
        assert res_one.effect_ci is not None
        assert res_one.effect is not None
        assert res_one.effect_ci[0] < res_one.effect < res_one.effect_ci[1]

    def test_cliffs_delta_ci(self) -> None:
        rng = np.random.default_rng(99)
        a = rng.exponential(scale=5.0, size=40) + 2.0
        b = rng.exponential(scale=1.0, size=40)

        delta = F.cliffs_d(a, b)
        ci = F.cliffs_d_ci(a, b, conf_level=0.95)
        assert -1.0 <= ci[0] < delta < ci[1] <= 1.0

        res = sc.compare(a, b)
        assert res.test_name == "Mann-Whitney U Test"
        assert res.effect_ci is not None
        assert np.isclose(res.effect_ci[0], ci[0])
        assert np.isclose(res.effect_ci[1], ci[1])


class TestPillar3TensorPostHoc:
    """Verify tensor-broadcasted Tukey, Games-Howell, and Dunn post-hoc algorithms."""

    def test_tensor_posthoc_k10_groups(self) -> None:
        rng = np.random.default_rng(123)
        # 10 groups -> 45 pairwise comparisons
        groups = [rng.normal(loc=i * 0.8, scale=1.0 + 0.2 * i, size=25) for i in range(10)]
        names = [f"G{i}" for i in range(10)]

        gh_df = F.games_howell(*groups, names=names)
        assert len(gh_df) == 45
        assert list(gh_df.columns) == [
            "group1",
            "group2",
            "diff",
            "p_adj",
            "ci_low",
            "ci_high",
            "reject",
        ]
        # G0 vs G9 should be highly significant with negative diff
        row_0_9 = gh_df[(gh_df["group1"] == "G0") & (gh_df["group2"] == "G9")].iloc[0]
        assert row_0_9["diff"] < -5.0
        assert bool(row_0_9["reject"]) is True

        dunn_df = F.dunn(*groups, names=names, p_adjust="fdr_bh")
        assert len(dunn_df) == 45
        row_dunn = dunn_df[(dunn_df["group1"] == "G0") & (dunn_df["group2"] == "G9")].iloc[0]
        assert bool(row_dunn["reject"]) is True

        tukey_df = F.tukey(*groups, names=names)
        assert len(tukey_df) == 45


class TestPillar4FormulaAndRegression:
    """Verify Wilkinson formula interface, OLS with HC3 SEs, and IRLS Logistic Regression."""

    def test_ols_hc3_and_formula_parsing(self) -> None:
        rng = np.random.default_rng(321)
        n = 150
        x1 = rng.normal(0.0, 1.0, n)
        x2 = rng.uniform(0.0, 10.0, n)
        grp = np.where(rng.random(n) > 0.5, "Treatment", "Control")
        # True model: y = 2.0 + 1.5*x1 - 0.5*x2 + 3.0*(grp=='Treatment') + 0.8*(x1*x2) + noise
        y = (
            2.0
            + 1.5 * x1
            - 0.5 * x2
            + 3.0 * (grp == "Treatment").astype(float)
            + 0.8 * (x1 * x2)
            + rng.normal(0.0, 0.5, n)
        )
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "grp": grp})

        reg = sc.ols(df, "y ~ x1 * x2 + C(grp)", robust_se="HC3")
        assert isinstance(reg, RegressionResult)
        assert reg.r_squared > 0.95
        assert reg.p < 1e-10
        assert len(reg.coef_df) == 5  # Intercept, x1, x2, x1:x2, grp[T.Treatment]

        # Verify formatting methods
        assert "OLS Regression" in reg.summary()
        assert "Significant Fit" in reg._repr_html_()
        assert isinstance(reg.to_df(), pd.DataFrame)
        assert "r_squared" in reg.to_dict()

        # Check accessor
        reg2 = df.scio.ols("y ~ x1 + x2")
        assert isinstance(reg2, RegressionResult)

    def test_logistic_irls(self) -> None:
        rng = np.random.default_rng(555)
        n = 250
        x1 = rng.normal(0.0, 1.0, n)
        x2 = rng.normal(0.0, 1.0, n)
        logits = -0.5 + 1.8 * x1 - 1.2 * x2
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_bin = (rng.random(n) < probs).astype(int)
        df = pd.DataFrame({"outcome": y_bin, "x1": x1, "x2": x2})

        log_res = sc.logistic(df, "outcome ~ x1 + x2")
        assert isinstance(log_res, RegressionResult)
        assert "or_val" in log_res.coef_df.columns
        assert log_res.p < 1e-6
        assert 0.1 < log_res.r_squared < 0.9

        # x1 has positive log-odds -> Odds Ratio > 1
        or_x1 = float(log_res.coef_df.loc[log_res.coef_df["term"] == "x1", "or_val"].iloc[0])
        assert or_x1 > 2.0

    def test_compare_with_formula(self) -> None:
        rng = np.random.default_rng(777)
        df = pd.DataFrame(
            {
                "score": np.concatenate([rng.normal(10, 1, 30), rng.normal(13, 1, 30)]),
                "arm": ["A"] * 30 + ["B"] * 30,
                "covar": rng.normal(0, 1, 60),
            }
        )
        # Single categorical RHS routes to two-sample test
        res_2samp = sc.compare(df, "score ~ arm")
        assert isinstance(res_2samp, TestResult)
        assert "t-Test" in res_2samp.test_name

        # Multivariate RHS routes to OLS and attaches coef_df in posthoc
        res_multi = df.scio.compare("score ~ arm + covar")
        assert isinstance(res_multi, TestResult)
        assert "OLS" in res_multi.test_name
        assert res_multi.posthoc is not None
        assert "coef" in res_multi.posthoc.columns

    def test_compare_numeric_vs_numeric_correlation(self) -> None:
        rng = np.random.default_rng(808)
        x = rng.normal(0.0, 1.0, 60)
        y = 2.5 * x + rng.normal(0.0, 0.5, 60)
        df = pd.DataFrame({"x": x, "y": y})

        res = sc.compare(df, x="x", y="y")
        assert res.test_name == "Pearson Correlation Test"
        assert res.effect is not None and res.effect > 0.85
        assert res.effect_ci is not None and res.effect_ci[0] > 0.70


class TestResamplingAndRobust:
    """Verify permutation tests, BCa bootstrap CIs, TOST equivalence, and robust statistics."""

    def test_permutation_test_exact_and_mc(self) -> None:
        a = [10.2, 11.5, 12.1, 10.8, 11.9, 12.5]
        b = [7.1, 8.0, 7.5, 8.3, 7.9, 8.4]

        # Small N (6 + 6 = 12, C(12, 6) = 924 <= 10000) -> exact enumeration
        res_exact = F.permutation_test(a, b)
        assert res_exact.p < 0.01

        # Monte Carlo mode
        rng = np.random.default_rng(11)
        g1 = rng.normal(5.0, 1.0, 40)
        g2 = rng.normal(4.0, 1.0, 40)
        res_mc = F.permutation_test(g1, g2, n_resamples=2000, seed=11)
        assert res_mc.p < 0.01

        # Paired permutation test
        res_paired = F.permutation_test(a, b, paired=True)
        assert res_paired.p < 0.05

    def test_bootstrap_ci_bca_and_percentile(self) -> None:
        rng = np.random.default_rng(88)
        sample = rng.lognormal(mean=1.0, sigma=0.6, size=60)

        ci_bca = F.bootstrap_ci(sample, method="bca", n_boot=2000, seed=88)
        ci_pct = F.bootstrap_ci(sample, method="percentile", n_boot=2000, seed=88)
        sample_mean = float(np.mean(sample))
        assert ci_bca[0] < sample_mean < ci_bca[1]
        assert ci_pct[0] < sample_mean < ci_pct[1]

    def test_tost_equivalence(self) -> None:
        rng = np.random.default_rng(404)
        # Two groups with virtually identical means (diff ~ 0.02), equivalence bound = 0.5
        a = rng.normal(10.0, 0.4, 50)
        b = rng.normal(10.02, 0.4, 50)

        res_eq = sc.tost(a, b, bound=0.5, alpha=0.05)
        assert isinstance(res_eq, TestResult)
        assert res_eq.sig is True
        assert res_eq.p < 0.05
        assert res_eq.ci is not None
        assert res_eq.ci[0] > -0.5 and res_eq.ci[1] < 0.5

        with pytest.raises(StatError):
            sc.tost(a, b, bound=-0.5)

    def test_robust_estimators_and_yuen_t(self) -> None:
        rng = np.random.default_rng(909)
        clean_a = rng.normal(10.0, 1.0, 30)
        clean_b = rng.normal(7.0, 1.0, 30)

        # Inject extreme outliers into group A that inflate classical variance
        contam_a = clean_a.copy()
        contam_a[0] = 500.0
        contam_a[1] = -450.0

        # Trimmed mean, Winsorized mean, and Huber M-estimator remain anchored near 10.0
        assert abs(F.trimmed_mean(contam_a, proportiontocut=0.2) - 10.0) < 0.6
        assert abs(F.winsorized_mean(contam_a, proportiontocut=0.2) - 10.0) < 1.5
        huber_mu, huber_mad = F.huber_location(contam_a)
        assert abs(huber_mu - 10.0) < 0.6
        assert huber_mad > 0.0

        # Classical Welch's t is destroyed by the +/- 500 outliers (p > 0.3)
        welch_res = F.welch_t(contam_a, clean_b)
        assert welch_res.p > 0.3

        # Yuen-Welch trimmed t-test recovers the true signal (p < 0.0001)
        yuen_res = F.yuen_t(contam_a, clean_b, trim=0.2)
        assert yuen_res.p < 1e-4
        assert yuen_res.mean_diff > 2.0
