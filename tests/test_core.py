"""Unit tests for pyscio.core exceptions, guards, result containers, and formatters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyscio.core.errors import (
    AssumptionError,
    DtypeError,
    MissingDataError,
    SampleSizeError,
    ScioError,
    StatError,
    ZeroVarError,
)
from pyscio.core.fmt import (
    format_html_card,
    format_latex_table,
    format_markdown_table,
)
from pyscio.core.guards import (
    EPSILON,
    check_groups,
    check_low_power,
    check_paired,
    check_sample_size,
    check_variance,
    clamp_p,
    clean_1d,
)
from pyscio.core.results import (
    CheckResult,
    ColProfile,
    EDAReport,
    ScreenReport,
    TestResult,
)

# =============================================================================
# 1. Exception Hierarchy Tests
# =============================================================================


def test_exception_hierarchy() -> None:
    """Verify ScioError is the base for all domain exceptions."""
    assert issubclass(StatError, ScioError)
    assert issubclass(SampleSizeError, StatError)
    assert issubclass(ZeroVarError, StatError)
    assert issubclass(AssumptionError, StatError)
    assert issubclass(DtypeError, ScioError)
    assert issubclass(MissingDataError, ScioError)


def test_exception_attributes() -> None:
    """Verify custom exception attributes."""
    err = SampleSizeError("Insufficient size", n=2, min_n=3)
    assert err.n == 2
    assert err.min_n == 3

    zerr = ZeroVarError("No variance", sample_name="group1")
    assert zerr.sample_name == "group1"

    aerr = AssumptionError("Normality violated", assumption_name="normality", p_value=0.001)
    assert aerr.assumption_name == "normality"
    assert aerr.p_value == 0.001


# =============================================================================
# 2. Input Guards & Numerical Clamps Tests
# =============================================================================


def test_clamp_p_underflow() -> None:
    """Verify p-value clamping at machine epsilon."""
    # Underflow clamp
    p_clamped, display = clamp_p(1e-25)
    assert p_clamped == EPSILON
    assert "p < 2.22e-16" in display

    # Normal p-value
    p_norm, display_norm = clamp_p(0.042)
    assert p_norm == 0.042
    assert "0.042" in display_norm

    # Upper clamp
    p_upper, _ = clamp_p(1.5)
    assert p_upper == 1.0


def test_clean_1d_valid_inputs() -> None:
    """Verify clean_1d converts various inputs into a contiguous 1D float64 array."""
    data_list = [1, 2, 3, 4]
    arr = clean_1d(data_list)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.shape == (4,)

    # Pandas Series
    s = pd.Series([10.5, 20.2, np.nan, 30.1])
    arr_s = clean_1d(s, dropna=True)
    assert len(arr_s) == 3
    assert not np.isnan(arr_s).any()


def test_clean_1d_invalid_inputs() -> None:
    """Verify clean_1d raises errors on invalid shapes or infinite values."""
    # 2D matrix
    with pytest.raises(ValueError, match="1-dimensional"):
        clean_1d(np.zeros((3, 3)))

    # Infs
    with pytest.raises(ValueError, match="infinite"):
        clean_1d([1.0, 2.0, np.inf])

    # All NaNs
    with pytest.raises(MissingDataError):
        clean_1d([np.nan, np.nan], dropna=True)


def test_check_sample_size() -> None:
    """Verify check_sample_size raises SampleSizeError when len < min_n."""
    valid = np.array([1.0, 2.0, 3.0])
    check_sample_size(valid, min_n=3)  # Should not raise

    invalid = np.array([1.0, 2.0])
    with pytest.raises(SampleSizeError) as exc:
        check_sample_size(invalid, min_n=3)
    assert exc.value.n == 2
    assert exc.value.min_n == 3


def test_check_variance() -> None:
    """Verify check_variance raises ZeroVarError for constant arrays."""
    const_arr = np.array([5.0, 5.0, 5.0, 5.0])
    with pytest.raises(ZeroVarError) as exc:
        check_variance(const_arr, name="test_col")
    assert exc.value.sample_name == "test_col"

    varying = np.array([5.0, 5.1, 5.0])
    check_variance(varying)  # Should not raise


def test_check_groups_and_paired() -> None:
    """Verify check_groups and check_paired operations."""
    g1 = np.array([1.0, 2.0, 3.0, 4.0])
    g2 = np.array([2.0, 4.0, 6.0, 8.0])
    check_groups(g1, g2)
    check_paired(g1, g2)

    # Paired length mismatch
    with pytest.raises(ValueError, match="identical lengths"):
        check_paired(g1, g2[:3])


def test_check_low_power() -> None:
    """Verify low sample size warning generation."""
    assert check_low_power(10, threshold=15) is not None
    assert check_low_power(20, threshold=15) is None


# =============================================================================
# 3. Formatters Tests
# =============================================================================


def test_format_markdown_and_latex_table() -> None:
    """Verify Markdown and LaTeX table renderers."""
    headers = ["Metric", "Value"]
    rows = [["Statistic", "3.48"], ["p-value", "0.001"]]

    md = format_markdown_table(headers, rows)
    assert "| Metric" in md
    assert "| Statistic" in md

    latex = format_latex_table(headers, rows, caption="Test Results")
    assert "\\begin{table}" in latex
    assert "\\caption{Test Results}" in latex


def test_format_html_card() -> None:
    """Verify HTML card renderer contains title and metrics."""
    card = format_html_card(
        title="Welch's t-test",
        subtitle="Group A vs Group B",
        metrics=[("t-stat", "2.45"), ("p-value", "0.015")],
        badge=("Significant", "#10b981"),
        details=[("df", "48.2")],
        warnings=["Low sample power"],
    )
    assert "Welch&#x27;s t-test" in card or "Welch's t-test" in card
    assert "Significant" in card
    assert "Low sample power" in card


# =============================================================================
# 4. Result Containers Tests
# =============================================================================


def test_check_result() -> None:
    """Verify CheckResult container and serialization."""
    check = CheckResult(name="Normality", stat=0.98, p=0.45, passed=True, details="Shapiro-Wilk")
    d = check.to_dict()
    assert d["passed"] is True
    assert d["stat"] == 0.98


def test_test_result_exports_and_representations() -> None:
    """Verify TestResult exports to dict, DataFrame, summary, and HTML card."""
    res = TestResult(
        test_name="Welch's Two-Sample t-test",
        x_name="treatment",
        y_name="outcome",
        stat=3.1415,
        p=0.002,
        p_display="p = 0.002",
        dof=42.0,
        ci=(1.2, 3.8),
        sig=True,
        effect=0.65,
        metric="Hedges' g",
        magnitude="medium",
        checks=[CheckResult("Homoscedasticity", stat=1.2, p=0.25, passed=True)],
        warns=["Low power warning"],
    )

    # Dictionary export
    d = res.to_dict()
    assert d["test_name"] == "Welch's Two-Sample t-test"
    assert d["sig"] is True
    assert d["effect"] == 0.65

    # DataFrame export
    df = res.to_df()
    assert isinstance(df, pd.DataFrame)
    assert df["test"].iloc[0] == "Welch's Two-Sample t-test"
    assert bool(df["significant"].iloc[0]) is True
    assert df["ci_low"].iloc[0] == 1.2

    # String & Markdown summaries
    text_summary = res.summary(format="text")
    assert "Welch's Two-Sample t-test" in text_summary
    assert "Significant" in text_summary

    md_summary = res.summary(format="markdown")
    assert "| Test" in md_summary

    # Notebook HTML representation
    html_repr = res._repr_html_()
    assert "Welch&#x27;s Two-Sample t-test" in html_repr or "Welch's Two-Sample t-test" in html_repr
    assert "Hedges&#x27; g" in html_repr or "Hedges' g" in html_repr


def test_col_profile_and_eda_report(tmp_path: pytest.TempPathFactory) -> None:
    """Verify ColProfile and EDAReport containers, summary, and HTML generation."""
    cp = ColProfile(
        name="age",
        role="numeric",
        dtype="float64",
        n_total=100,
        n_missing=2,
        missing_pct=2.0,
        n_unique=45,
        mean=35.4,
        std=10.2,
        median=34.0,
        iqr=12.0,
    )
    assert cp.name == "age"
    assert cp.to_dict()["mean"] == 35.4
    assert isinstance(cp.to_df(), pd.DataFrame)

    report = EDAReport(
        n_rows=100,
        n_cols=1,
        profiles={"age": cp},
        health_score=94.5,
        quality_alerts=["2% missing values in age"],
    )
    assert report.health_score == 94.5
    assert "age" in report.to_dict()["profiles"]
    assert "94.5/100" in report.summary()

    # Standalone HTML export
    out_file = str(tmp_path / "eda_report.html")
    report.to_html(out_file)
    with open(out_file, encoding="utf-8") as f:
        html_content = f.read()
    assert "pyscio Exploratory Data Analysis" in html_content
    assert "94.5/100" in html_content


def test_screen_report() -> None:
    """Verify ScreenReport container, leaderboard, and summaries."""
    leaderboard = pd.DataFrame(
        {
            "feature": ["gene_A", "gene_B"],
            "test": ["Welch's t", "Welch's t"],
            "p_value": [0.0001, 0.045],
            "p_adjusted_bh": [0.0002, 0.045],
            "significant_fdr": [True, True],
            "effect_size": [1.45, 0.52],
            "effect_metric": ["Cohen's d", "Cohen's d"],
            "magnitude": ["large", "medium"],
        }
    )
    screen = ScreenReport(
        target="disease_status",
        n_features=2,
        alpha=0.05,
        results=[],
        leaderboard_df=leaderboard,
    )

    assert screen.target == "disease_status"
    assert len(screen.leaderboard()) == 2
    assert "gene_A" in screen.summary()
    assert "2 Significant" in screen._repr_html_()
