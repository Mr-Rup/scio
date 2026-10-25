"""Typed domain exceptions for pyscio."""

from __future__ import annotations


class ScioError(Exception):
    """Base exception for all errors raised within pyscio."""


class StatError(ScioError):
    """Base exception for statistical calculation, inference, and routing errors."""


class SampleSizeError(StatError):
    """Raised when the sample size is insufficient to compute a valid test or assumption check.

    Standard rule: $N < 3$ for two-sample continuous tests; $N < 5$ for contingency tables.
    """

    def __init__(self, message: str, n: int | None = None, min_n: int | None = None) -> None:
        super().__init__(message)
        self.n: int | None = n
        self.min_n: int | None = min_n


class ZeroVarError(StatError):
    """Raised when sample variance is exactly zero (degenerate constant distribution)."""

    def __init__(self, message: str, sample_name: str | None = None) -> None:
        super().__init__(message)
        self.sample_name: str | None = sample_name


class DtypeError(ScioError):
    """Raised when an array or column has a data type incompatible with the requested operation."""

    def __init__(self, message: str, actual_dtype: str | None = None, expected: str | None = None) -> None:
        super().__init__(message)
        self.actual_dtype: str | None = actual_dtype
        self.expected: str | None = expected


class AssumptionError(StatError):
    """Raised when a mandatory statistical assumption fails and no valid fallback is permitted."""

    def __init__(self, message: str, assumption_name: str | None = None, p_value: float | None = None) -> None:
        super().__init__(message)
        self.assumption_name: str | None = assumption_name
        self.p_value: float | None = p_value


class MissingDataError(ScioError):
    """Raised when missing values violate constraints or drop operations leave zero observations."""
