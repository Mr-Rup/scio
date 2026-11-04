"""Comprehensive test suite for omnistats.eda (ingestion, 8-role classification, quality, correlations, profiling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import omnistats as sc
from omnistats.eda import (
    Role,
    audit_quality,
    classify,
    classify_series,
    correlations,
    cramers_matrix,
    missingness_correlation,
    profile,
    read,
    vif,
)


@pytest.fixture
def complex_dataframe() -> pd.DataFrame:
    """Multi-type DataFrame with all 8 semantic roles, missing values, and duplicates."""
    n = 100
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "user_id": [f"USR_{i:04d}" for i in range(n)],  # ID
            "age": rng.normal(35, 10, n),  # NUMERIC
            "satisfaction_rating": rng.integers(1, 6, n),  # DISCRETE
            "is_active": rng.choice([True, False], n),  # BINARY
            "signup_date": pd.date_range("2024-01-01", periods=n, freq="D"),  # TEMPORAL
            "department": rng.choice(["HR", "Engineering", "Sales"], n),  # CATEGORICAL
            "customer_feedback": [
                f"This is an extended detailed customer review {i} with plenty of descriptive feedback explaining experience."
                for i in range(n)
            ],  # TEXT
            "country_code": ["US"] * n,  # CONSTANT
        }
    )


# =============================================================================
# 1. Ingestion Tests
# =============================================================================


def test_read_dataframe_and_csv(tmp_path: pytest.TempPathFactory) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    # DataFrame direct pass
    df_read = read(df)
    assert df_read.equals(df)

    # CSV with comma
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)
    df_from_csv = read(csv_file)
    assert len(df_from_csv) == 3
    assert "a" in df_from_csv.columns

    # TSV with tab delimiter
    tsv_file = tmp_path / "data.tsv"
    df.to_csv(tsv_file, sep="\t", index=False)
    df_from_tsv = read(tsv_file)
    assert len(df_from_tsv) == 3

    # Top-level alias sc.read
    assert sc.read(df).equals(df)


# =============================================================================
# 2. 8-Role Semantic Classification Tests
# =============================================================================


def test_semantic_role_classification(complex_dataframe: pd.DataFrame) -> None:
    roles = classify(complex_dataframe)

    assert roles["user_id"] == Role.ID
    assert roles["age"] == Role.NUMERIC
    assert roles["satisfaction_rating"] == Role.DISCRETE
    assert roles["is_active"] == Role.BINARY
    assert roles["signup_date"] == Role.TEMPORAL
    assert roles["department"] == Role.CATEGORICAL
    assert roles["customer_feedback"] == Role.TEXT
    assert roles["country_code"] == Role.CONSTANT


def test_classify_series_edge_cases() -> None:
    # Single value series
    s_const = pd.Series([42, 42, 42, 42])
    assert classify_series(s_const) == Role.CONSTANT

    # Binary boolean
    s_bin = pd.Series([1, 0, 1, 1, 0])
    assert classify_series(s_bin) == Role.BINARY


# =============================================================================
# 3. Data Quality Engine Tests
# =============================================================================


def test_audit_quality(complex_dataframe: pd.DataFrame) -> None:
    # Inject missing values and duplicate rows
    df_dirty = complex_dataframe.copy()
    df_dirty.loc[:10, "age"] = np.nan
    df_dirty = pd.concat([df_dirty, df_dirty.iloc[:5]], ignore_index=True)

    score, alerts = audit_quality(df_dirty)
    assert 0.0 <= score < 100.0
    assert any("Missing Values" in a for a in alerts)
    assert any("Duplicate Rows" in a for a in alerts)
    assert any("Constant Feature" in a for a in alerts)


# =============================================================================
# 4. Relationships and Multicollinearity Tests
# =============================================================================


def test_correlations_and_cramers(complex_dataframe: pd.DataFrame) -> None:
    # Numeric correlations
    corr = correlations(complex_dataframe)
    assert isinstance(corr, pd.DataFrame)
    assert "age" in corr.columns

    # Cramér's V matrix
    cr_matrix = cramers_matrix(complex_dataframe)
    assert isinstance(cr_matrix, pd.DataFrame)

    # Missingness correlation
    miss_corr = missingness_correlation(complex_dataframe)
    assert isinstance(miss_corr, pd.DataFrame)


def test_vif_multicollinearity() -> None:
    # Collinear data: x2 = 2*x1 + noise
    x1 = np.linspace(1, 100, 100)
    x2 = 2 * x1 + np.random.normal(0, 0.01, 100)
    x3 = np.random.normal(0, 1, 100)

    df_collinear = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    vif_df = vif(df_collinear)

    assert len(vif_df) == 3
    # x1 and x2 should have high VIF (> 10)
    assert vif_df.loc[vif_df["feature"] == "x1", "vif"].iloc[0] > 10.0


# =============================================================================
# 5. Full Dataset Profiling and One-Tap Accessor Tests
# =============================================================================


def test_profile_and_pandas_accessor(
    complex_dataframe: pd.DataFrame, tmp_path: pytest.TempPathFactory
) -> None:
    # Direct profile call
    report = profile(complex_dataframe)
    assert report.n_rows == 100
    assert report.n_cols == 8
    assert len(report.profiles) == 8
    assert 0.0 <= report.health_score <= 100.0

    # Test one-tap Pandas accessor: df.scio.eda()
    accessor_report = complex_dataframe.scio.eda()
    assert accessor_report.n_rows == 100
    assert accessor_report.health_score == report.health_score

    # Export standalone HTML
    out_path = str(tmp_path / "report.html")
    report.to_html(out_path)
    with open(out_path, encoding="utf-8") as f:
        html_text = f.read()
    assert "omnistats Exploratory Data Analysis" in html_text
