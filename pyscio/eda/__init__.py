"""pyscio.eda: Universal Pandas-native Exploratory Data Analysis & Profiling suite."""

from __future__ import annotations

from pyscio.eda.io import read
from pyscio.eda.profile import profile, profile_column
from pyscio.eda.quality import audit_quality
from pyscio.eda.rel import (
    correlations,
    cramers_matrix,
    missingness_correlation,
    vif,
)
from pyscio.eda.roles import Role, classify, classify_series

__all__ = [
    # Ingestion
    "read",
    # Profiling & Report
    "profile",
    "profile_column",
    # Semantic Classification
    "Role",
    "classify",
    "classify_series",
    # Quality Scoring & Rules
    "audit_quality",
    # Relationships & Associations
    "correlations",
    "cramers_matrix",
    "vif",
    "missingness_correlation",
]
