"""Tests for autonomous statistical router, compare(), and multi-feature discovery screening."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyscio as sc
from pyscio.core.errors import MissingDataError
from pyscio.core.results import ScreenReport, TestResult
from pyscio.stat.compare import compare
from pyscio.stat.router import (
    evaluate_normality_with_clt,
    route_contingency,
    route_independent_two_sample,
    route_k_sample,
    route_one_sample,
    route_paired_two_sample,
)
from pyscio.stat.screen import screen


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_evaluate_normality_with_clt(rng: np.random.Generator) -> None:
    # 1. Truly normal data (N=100) -> Passed via Shapiro-Wilk
    normal_data = rng.normal(10, 2, 100)
    passed, check = evaluate_normality_with_clt(normal_data, name="test_normal")
    assert passed is True
    assert check.passed is True
    assert check.clt_note is None

    # 2. Large-N sample with slight non-normality (N=600, t-distribution with df=15)
    # Shapiro or D'Agostino will reject normality due to extreme power on N=600,
    # but |skewness| < 1.0 and |excess kurtosis| < 2.0, so CLT guard activates!
    large_n_mild = rng.standard_t(df=15, size=600)
    passed_clt, check_clt = evaluate_normality_with_clt(large_n_mild, name="large_clt")
    assert passed_clt is True
    assert check_clt.passed is True
    assert check_clt.clt_note is not None
    assert "Central Limit Theorem holds" in check_clt.clt_note

    # 3. Severe non-normality (exponential, high skew) -> Rejected, CLT does not apply
    exp_data = rng.exponential(scale=2.0, size=100)
    passed_exp, check_exp = evaluate_normality_with_clt(exp_data, name="severe_nonnormal")
    assert passed_exp is False
    assert check_exp.passed is False
    assert check_exp.clt_note is None


def test_route_independent_two_sample_normal_equal_var(rng: np.random.Generator) -> None:
    # Equal variance, normal -> Student's t-test
    g1 = rng.normal(10, 2, 40)
    g2 = rng.normal(12, 2, 40)

    res = route_independent_two_sample(g1, g2, x_name="treatment", y_name="response")
    assert isinstance(res, TestResult)
    assert res.test_name == "Student's Two-Sample t-Test"
    assert res.sig is True
    assert res.effect is not None
    assert res.metric in ("Cohen's d", "Hedges' g")
    assert res.ci is not None
    assert len(res.checks) == 3  # 2 normality + 1 homoscedasticity


def test_route_independent_two_sample_normal_unequal_var(rng: np.random.Generator) -> None:
    # Unequal variance (std 1 vs std 5), normal -> Welch's t-test
    g1 = rng.normal(10, 1, 50)
    g2 = rng.normal(14, 6, 50)

    res = route_independent_two_sample(g1, g2, x_name="dose", y_name="effect")
    assert res.test_name == "Welch's Two-Sample t-Test"
    assert res.sig is True
    assert res.ci is not None
    # Check that homoscedasticity check failed
    homo_check = next(c for c in res.checks if "Homoscedasticity" in c.name)
    assert homo_check.passed is False


def test_route_independent_two_sample_non_normal(rng: np.random.Generator) -> None:
    # Exponential distributions -> Mann-Whitney U test
    g1 = rng.exponential(1.0, 40)
    g2 = rng.exponential(3.0, 40)

    res = route_independent_two_sample(g1, g2, x_name="platform", y_name="latency")
    assert res.test_name == "Mann-Whitney U Test"
    assert res.metric == "Cliff's delta"
    assert res.effect is not None
    assert -1.0 <= res.effect <= 1.0


def test_route_paired_two_sample(rng: np.random.Generator) -> None:
    # 1. Normal differences -> Paired t-test
    before = rng.normal(100, 15, 30)
    after = before + rng.normal(5, 2, 30)
    res_t = route_paired_two_sample(before, after, x_name="before", y_name="after")
    assert res_t.test_name == "Paired Samples t-Test"
    assert res_t.sig is True
    assert res_t.ci is not None
    assert res_t.metric == "Cohen's d (paired)"

    # 2. Non-normal differences -> Wilcoxon Signed-Rank Test
    before_skew = rng.normal(100, 15, 35)
    exp_diff = rng.exponential(5.0, 35)
    after_skew = before_skew + exp_diff
    res_w = route_paired_two_sample(before_skew, after_skew)
    assert res_w.test_name == "Wilcoxon Signed-Rank Test"
    assert res_w.metric == "Rank-biserial r"


def test_route_one_sample(rng: np.random.Generator) -> None:
    # 1. Normal sample vs popmean
    g = rng.normal(5.0, 1.0, 40)
    res = route_one_sample(g, popmean=0.0)
    assert res.test_name == "One-Sample t-Test"
    assert res.sig is True
    assert res.ci is not None
    assert res.metric == "Cohen's d (one-sample)"

    # 2. Non-normal sample vs popmean
    g_exp = rng.exponential(scale=2.0, size=40)
    res_w = route_one_sample(g_exp, popmean=1.0)
    assert res_w.test_name == "Wilcoxon Signed-Rank Test (One-Sample)"


def test_route_k_sample_anova_and_kruskal(rng: np.random.Generator) -> None:
    # 1. Normal and homoscedastic -> One-Way ANOVA + Tukey HSD posthoc
    g1 = rng.normal(10, 2, 30)
    g2 = rng.normal(15, 2, 30)
    g3 = rng.normal(20, 2, 30)

    res_anova = route_k_sample([g1, g2, g3], group_names=["A", "B", "C"])
    assert res_anova.test_name == "One-Way ANOVA"
    assert res_anova.sig is True
    assert res_anova.metric == "Eta-squared (η²)"
    assert res_anova.posthoc is not None
    assert len(res_anova.posthoc) == 3  # (A-B, A-C, B-C)

    # 2. Normal and heteroscedastic -> Welch's ANOVA + Games-Howell posthoc
    g4 = rng.normal(10, 1, 30)
    g5 = rng.normal(15, 4, 30)
    g6 = rng.normal(20, 8, 30)
    res_welch_anova = route_k_sample([g4, g5, g6], group_names=["Low", "Med", "High"])
    assert res_welch_anova.test_name == "Welch's ANOVA"
    assert res_welch_anova.posthoc is not None

    # 3. Non-normal -> Kruskal-Wallis + Dunn posthoc
    k1 = rng.exponential(1.0, 30)
    k2 = rng.exponential(2.0, 30)
    k3 = rng.exponential(4.0, 30)
    res_kw = route_k_sample([k1, k2, k3], group_names=["X", "Y", "Z"])
    assert res_kw.test_name == "Kruskal-Wallis H Test"
    assert res_kw.metric == "Epsilon-squared (ε²)"
    assert res_kw.posthoc is not None


def test_route_contingency() -> None:
    # 1. Large counts -> Pearson's Chi-Squared
    table_large = np.array([[50, 30], [20, 60]])
    res_chi = route_contingency(table_large)
    assert res_chi.test_name == "Pearson's Chi-Squared Test"
    assert res_chi.metric == "Cramér's V"

    # 2. Low counts (< 5 in 2x2) -> Fisher's Exact Test
    table_small = np.array([[3, 1], [1, 4]])
    res_fisher = route_contingency(table_small)
    assert res_fisher.test_name == "Fisher's Exact Test"
    assert res_fisher.metric == "Odds Ratio"
    assert res_fisher.ci is not None


def test_compare_dataframe_api(rng: np.random.Generator) -> None:
    df = pd.DataFrame(
        {
            "group": ["Control"] * 30 + ["Treatment"] * 30,
            "score": np.concatenate([rng.normal(50, 5, 30), rng.normal(60, 5, 30)]),
            "score_post": np.concatenate([rng.normal(55, 5, 30), rng.normal(70, 5, 30)]),
            "variant": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "metric": np.concatenate(
                [rng.normal(10, 2, 20), rng.normal(15, 2, 20), rng.normal(20, 2, 20)]
            ),
            "category": ["X", "Y"] * 30,
            "preference": ["Yes", "No", "Yes"] * 20,
        }
    )

    # 1. DataFrame 2-group comparison
    res_2g = compare(df, x="group", y="score")
    assert res_2g.sig is True
    assert "Student" in res_2g.test_name or "Welch" in res_2g.test_name

    # 2. DataFrame K-group comparison
    res_kg = compare(df, x="variant", y="metric")
    assert res_kg.sig is True
    assert res_kg.posthoc is not None

    # 3. DataFrame paired comparison
    res_pair = compare(df, x="score", y="score_post", paired=True)
    assert res_pair.test_name == "Paired Samples t-Test"

    # 4. DataFrame categorical contingency
    res_cat = compare(df, x="category", y="preference")
    assert res_cat.test_name == "Pearson's Chi-Squared Test"

    # 5. DataFrame 1-sample test
    res_1s = compare(df, y="score", mu=50.0)
    assert "One-Sample" in res_1s.test_name

    # 6. Pandas accessor: df.scio.compare
    res_accessor = df.scio.compare(x="group", y="score")
    assert res_accessor.test_name == res_2g.test_name


def test_compare_direct_arrays(rng: np.random.Generator) -> None:
    a = rng.normal(10, 2, 35)
    b = rng.normal(15, 2, 35)

    res = compare(a, b)
    assert isinstance(res, TestResult)
    assert res.sig is True

    # One sample direct array
    res_one = compare(a, mu=10.0)
    assert isinstance(res_one, TestResult)


def test_compare_error_handling(rng: np.random.Generator) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Missing column
    with pytest.raises(KeyError):
        compare(df, x="nonexistent", y="a")

    # Missing data / empty slice
    df_empty = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    with pytest.raises(MissingDataError):
        compare(df_empty, x="a", y="b", paired=True)


def test_multi_feature_screening(rng: np.random.Generator) -> None:
    n = 100
    df = pd.DataFrame(
        {
            "user_id": [f"ID_{i}" for i in range(n)],  # ID (should be excluded)
            "constant_col": [42] * n,  # Constant (should be excluded)
            "sig_numeric": rng.normal(10, 2, n),
            "sig_strong": rng.normal(25, 3, n),
            "noise_feature": rng.normal(0, 1, n),
            "binary_group": rng.choice(["A", "B"], n),
            "target": rng.choice(["Healthy", "Disease"], n),
        }
    )
    # Ensure sig_strong has massive effect vs target
    df.loc[df["target"] == "Disease", "sig_strong"] += 15.0

    # Run multi-feature screen
    report = screen(df, y="target")
    assert isinstance(report, ScreenReport)
    assert report.target == "target"
    assert report.n_features > 0

    lb = report.leaderboard()
    assert isinstance(lb, pd.DataFrame)
    assert "user_id" not in lb["feature"].values
    assert "constant_col" not in lb["feature"].values
    assert "p_adj" in lb.columns
    assert "significant_fdr" in lb.columns

    # Verify strong signal is at the top of the leaderboard
    top_feature = lb.iloc[0]["feature"]
    assert top_feature == "sig_strong"
    assert bool(lb.iloc[0]["significant_fdr"]) is True

    # Test top-level and accessor aliases
    report_alias = sc.screen(df, y="target")
    assert report_alias.target == "target"

    report_accessor = df.scio.screen(y="target")
    assert report_accessor.target == "target"
