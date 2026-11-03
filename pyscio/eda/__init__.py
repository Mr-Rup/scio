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
