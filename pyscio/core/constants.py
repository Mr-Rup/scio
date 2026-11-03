"""Global mathematical and statistical threshold constants for pyscio."""

from __future__ import annotations

import numpy as np

# Machine epsilon for float64 p-value underflow clamping (~2.220446049250313e-16)
EPSILON: float = float(np.finfo(np.float64).eps)

# Large-N Central Limit Theorem (CLT) thresholds
CLT_MIN_N: int = 500
CLT_MAX_SKEW: float = 1.0
CLT_MAX_KURT: float = 2.0

# Distributional pathology guard thresholds
PATHOLOGY_TIE_THRESHOLD: float = 0.30
PATHOLOGY_TAIL_RATIO_MAX: float = 2.2
PATHOLOGY_EXTREME_OUTLIER_MAX: float = 0.025
PATHOLOGY_BIMODALITY_MAX: float = 0.555
NORMAL_IQR_SCALE: float = 1.3489795003921634

# Sample size and power thresholds
LOW_POWER_THRESHOLD: int = 15
SMALL_SAMPLE_HEDGES_N: int = 50
