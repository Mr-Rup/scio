"""Public API surface snapshot tests to guarantee zero regressions across refactoring phases."""

from __future__ import annotations

import pandas as pd

import omnistats as sc
import omnistats.core as core
import omnistats.eda as eda
import omnistats.functional as F
import omnistats.stat as stat


def test_top_level_public_api() -> None:
    expected = {
        "__version__",
        "F",
        "functional",
        "eda",
        "stat",
        "read",
        "compare",
        "screen",
        "ols",
        "logistic",
        "poisson",
        "anova_table",
        "tost",
        "solve_power",
    }
    assert expected.issubset(set(sc.__all__))
    for name in expected:
        assert hasattr(sc, name), f"Missing top-level export: sc.{name}"


def test_core_public_api() -> None:
    expected = {
        "BaseResult",
        "ScioError",
        "StatError",
        "SampleSizeError",
        "ZeroVarError",
        "DtypeError",
        "AssumptionError",
        "MissingDataError",
        "EPSILON",
        "clamp_p",
        "clean_1d",
        "check_sample_size",
        "check_variance",
        "check_groups",
        "check_paired",
        "check_low_power",
        "CheckResult",
        "TestResult",
        "RegressionResult",
        "ColProfile",
        "EDAReport",
        "ScreenReport",
        "RMAnovaResult",
        "BayesFactorResult",
    }
    assert expected.issubset(set(core.__all__))
    for name in expected:
        assert hasattr(core, name), f"Missing core export: core.{name}"


def test_functional_public_api() -> None:
    expected = {
        "student_t",
        "welch_t",
        "paired_t",
        "one_sample_t",
        "anova",
        "welch_anova",
        "rm_anova",
        "TTestResult",
        "ANOVAResult",
        "RMAnovaResult",
        "mann_whitney",
        "wilcoxon",
        "kruskal",
        "friedman",
        "mcnemar",
        "NonParametricResult",
        "chisq",
        "fisher",
        "ChiSqResult",
        "FisherResult",
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
        "cohen_d",
        "cohen_d_ci",
        "hedges_g",
        "hedges_g_ci",
        "cliffs_d",
        "cliffs_d_ci",
        "rank_biserial",
        "eta_sq",
        "eps_sq",
        "kendalls_w",
        "cramers_v",
        "odds_ratio",
        "interpret_effect",
        "OddsRatioResult",
        "tukey",
        "games_howell",
        "dunn",
        "pearson_r",
        "spearman_r",
        "kendall_tau",
        "partial_corr",
        "bayes_factor_t",
        "interpret_bf",
        "BayesFactorResult",
        "permutation_test",
        "bootstrap_ci",
        "tost",
        "trimmed_mean",
        "winsorized_mean",
        "winsorized_var",
        "huber_location",
        "yuen_t",
        "fdr_bh",
        "bonferroni",
        "holm",
        "power_t",
        "power_anova",
        "solve_power_t",
    }
    assert expected.issubset(set(F.__all__))
    for name in expected:
        assert hasattr(F, name), f"Missing functional export: F.{name}"


def test_stat_and_eda_public_api() -> None:
    expected_stat = {
        "compare",
        "screen",
        "ols",
        "logistic",
        "poisson",
        "anova_table",
        "parse_formula",
        "power_t",
        "power_anova",
        "solve_power",
        "solve_power_t",
        "route_independent_two_sample",
        "route_paired_two_sample",
        "route_one_sample",
        "route_k_sample",
        "route_paired_k_sample",
        "route_contingency",
        "evaluate_normality_with_clt",
    }
    assert expected_stat.issubset(set(stat.__all__))
    for name in expected_stat:
        assert hasattr(stat, name)

    expected_eda = {
        "read",
        "Role",
        "classify",
        "classify_series",
        "profile",
        "profile_column",
        "audit_quality",
        "correlations",
        "cramers_matrix",
        "vif",
        "missingness_correlation",
    }
    assert expected_eda.issubset(set(eda.__all__))
    for name in expected_eda:
        assert hasattr(eda, name)


def test_dataframe_accessor_methods() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
    for method in ("eda", "compare", "screen", "ols", "logistic", "poisson", "anova_table"):
        assert hasattr(df.scio, method), f"Missing df.scio.{method}"
