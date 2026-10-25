"""Shared test fixtures, synthetic datasets, and R 4.3 golden reference vectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible test runs."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_normal_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two independent Gaussian distributed groups with known effect size."""
    g1 = rng.normal(loc=10.0, scale=2.0, size=50)
    g2 = rng.normal(loc=12.0, scale=2.0, size=50)
    return g1, g2


@pytest.fixture
def sample_dataframe(sample_normal_data: tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
    """Pandas DataFrame suitable for two-group comparison tests."""
    g1, g2 = sample_normal_data
    return pd.DataFrame(
        {
            "group": ["A"] * len(g1) + ["B"] * len(g2),
            "value": np.concatenate([g1, g2]),
        }
    )
