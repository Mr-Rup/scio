"""Immutable result containers for pyscio."""

from __future__ import annotations

from pyscio.core.results.eda import ColProfile, EDAReport
from pyscio.core.results.inference import CheckResult, ScreenReport, TestResult
from pyscio.core.results.regression import RegressionResult

__all__ = [
    "CheckResult",
    "ColProfile",
    "EDAReport",
    "RegressionResult",
    "ScreenReport",
    "TestResult",
]
