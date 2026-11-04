"""omnistats.eda: Universal Pandas-native Exploratory Data Analysis & Profiling suite."""

from __future__ import annotations

from omnistats.eda.io import read
from omnistats.eda.profile import profile, profile_column
from omnistats.eda.quality import audit_quality
from omnistats.eda.rel import (
    correlations,
    cramers_matrix,
    missingness_correlation,
    vif,
)
from omnistats.eda.roles import Role, classify, classify_series

__all__ = [
    # Semantic Classification
    "Role",
    # Quality Scoring & Rules
    "audit_quality",
    "classify",
    "classify_series",
    # Relationships & Associations
    "correlations",
    "cramers_matrix",
    "missingness_correlation",
    # Profiling & Report
    "profile",
    "profile_column",
    # Ingestion
    "read",
    "vif",
]
