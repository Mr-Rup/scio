"""Immutable, typed result containers with dual notebook/terminal rendering."""

from __future__ import annotations

from pyscio.core.results.eda import ColProfile, EDAReport
from pyscio.core.results.inference import (
    CheckResult,
    RegressionResult,
    ScreenReport,
    TestResult,
)

__all__ = [
    "CheckResult",
    "ColProfile",
    "EDAReport",
    "RegressionResult",
    "ScreenReport",
    "TestResult",
]
