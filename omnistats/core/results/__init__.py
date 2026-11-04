"""Immutable result containers for omnistats."""

from __future__ import annotations

from omnistats.core.results.eda import ColProfile, EDAReport
from omnistats.core.results.inference import CheckResult, ScreenReport, TestResult
from omnistats.core.results.regression import RegressionResult

__all__ = [
    "CheckResult",
    "ColProfile",
    "EDAReport",
    "RegressionResult",
    "ScreenReport",
    "TestResult",
]
