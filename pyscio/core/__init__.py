"""Core data structures, exceptions, validation guards, and formatters for pyscio."""

from __future__ import annotations

from pyscio.core.base import BaseResult
from pyscio.core.errors import (
    AssumptionError,
    DtypeError,
    MissingDataError,
    SampleSizeError,
    ScioError,
    StatError,
    ZeroVarError,
)
from pyscio.core.guards import (
    EPSILON,
    check_groups,
    check_low_power,
    check_paired,
    check_sample_size,
    check_variance,
    clamp_p,
    clean_1d,
)
from pyscio.core.results import (
    CheckResult,
    ColProfile,
    EDAReport,
    ScreenReport,
    TestResult,
)

__all__ = [
    # Base and exceptions
    "BaseResult",
    "ScioError",
    "StatError",
    "SampleSizeError",
    "ZeroVarError",
    "DtypeError",
    "AssumptionError",
    "MissingDataError",
    # Guards and clamps
    "EPSILON",
    "clamp_p",
    "clean_1d",
    "check_sample_size",
    "check_variance",
    "check_groups",
    "check_paired",
    "check_low_power",
    # Results
    "CheckResult",
    "TestResult",
    "ColProfile",
    "EDAReport",
    "ScreenReport",
]
