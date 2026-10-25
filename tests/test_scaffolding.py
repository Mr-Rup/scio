"""Verify package scaffolding, imports, and accessor registration."""

from __future__ import annotations

import pandas as pd

import pyscio as sc
import pyscio.functional as F


def test_package_metadata() -> None:
    """Verify package version and top-level exports."""
    assert hasattr(sc, "__version__")
    assert isinstance(sc.__version__, str)
    assert sc.F is sc.functional
    assert F is sc.functional


def test_subpackages_present() -> None:
    """Verify core, functional, eda, and stat modules exist."""
    assert hasattr(sc, "functional")
    assert hasattr(sc, "eda")
    assert hasattr(sc, "stat")


def test_pandas_accessor_registration(sample_dataframe: pd.DataFrame) -> None:
    """Verify `df.scio` accessor is registered and callable."""
    assert hasattr(sample_dataframe, "scio")
    assert hasattr(sample_dataframe.scio, "eda")
    assert hasattr(sample_dataframe.scio, "compare")
    assert hasattr(sample_dataframe.scio, "screen")
