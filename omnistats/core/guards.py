"""Input validation, array sanitization, degenerate variance guards, and numerical clamps."""

from __future__ import annotations

from typing import Any

import numpy as np

from omnistats.core.errors import MissingDataError, SampleSizeError, ZeroVarError

# IEEE 754 float64 machine epsilon
EPSILON: float = float(np.finfo(np.float64).eps)  # 2.220446049250313e-16


def clamp_p(p: float) -> tuple[float, str]:
    """Clamp p-value to avoid floating-point underflow hazards and produce clean display text.

    Parameters
    ----------
    p : float
        Raw calculated p-value.

    Returns
    -------
    tuple[float, str]
        (clamped_p, display_str). Clamped to [EPSILON, 1.0].
        If p < EPSILON, display_str is "p < 2.22e-16".
    """
    if np.isnan(p):
        return float("nan"), "p = nan"

    if p < EPSILON:
        return EPSILON, "p < 2.22e-16"
    if p > 1.0:
        return 1.0, "p = 1.0"
    return float(p), f"p = {p:.5g}"


def clean_1d(
    arr: Any,
    name: str = "sample",
    dropna: bool = True,
) -> np.ndarray:
    """Convert input into a clean 1D float64 NumPy array, handling Series, lists, and missing values.

    Parameters
    ----------
    arr : Any
        Input array-like data (list, tuple, np.ndarray, pd.Series).
    name : str
        Variable name for error messages.
    dropna : bool
        Whether to drop NaN values automatically.

    Returns
    -------
    np.ndarray
        Cleaned 1D numpy array of float64.

    Raises
    ------
    MissingDataError
        If dropna is True and all values are missing.
    ValueError
        If the array cannot be flattened or contains infinite values.
    """
    if hasattr(arr, "to_numpy"):
        # Pandas Series or DataFrame column
        arr = arr.to_numpy()

    try:
        a = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Cannot convert '{name}' to a continuous float64 array: {err}") from err

    if a.ndim == 0:
        raise ValueError(f"'{name}' must be 1D, but got a 0-dimensional scalar.")
    if a.ndim > 1:
        if a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1):
            a = a.ravel()
        else:
            raise ValueError(f"'{name}' must be 1-dimensional, but got shape {a.shape}.")

    if np.isinf(a).any():
        raise ValueError(
            f"'{name}' contains infinite values (+/- inf), which are invalid for statistical testing."
        )

    if dropna:
        valid_mask = ~np.isnan(a)
        a = a[valid_mask]
        if len(a) == 0:
            raise MissingDataError(f"'{name}' contains no valid observations after dropping NaNs.")

    return np.ascontiguousarray(a)


def check_sample_size(arr: np.ndarray, min_n: int = 3, name: str = "sample") -> None:
    """Ensure array meets the minimum required sample size for inference.

    Parameters
    ----------
    arr : np.ndarray
        Cleaned 1D data array.
    min_n : int
        Minimum allowed observations (default 3).
    name : str
        Variable or group name.

    Raises
    ------
    SampleSizeError
        If len(arr) < min_n.
    """
    n = len(arr)
    if n < min_n:
        raise SampleSizeError(
            f"Sample '{name}' has N={n}, which is below the minimum required N={min_n}.",
            n=n,
            min_n=min_n,
        )


def check_variance(arr: np.ndarray, name: str = "sample") -> None:
    """Ensure sample has non-zero variance.

    Parameters
    ----------
    arr : np.ndarray
        Cleaned 1D data array.
    name : str
        Variable or group name.

    Raises
    ------
    ZeroVarError
        If all values are identical or sample variance is exactly zero.
    """
    if len(arr) <= 1:
        return
    var = float(np.var(arr, ddof=1))
    if var == 0.0 or np.isclose(var, 0.0, atol=1e-15):
        raise ZeroVarError(
            f"Sample '{name}' has zero variance (all {len(arr)} values are identical: {arr[0]}).",
            sample_name=name,
        )


def check_groups(group_a: np.ndarray, group_b: np.ndarray, min_n: int = 3) -> None:
    """Run sample size and variance guards across two independent groups."""
    check_sample_size(group_a, min_n=min_n, name="group_a")
    check_sample_size(group_b, min_n=min_n, name="group_b")
    check_variance(group_a, name="group_a")
    check_variance(group_b, name="group_b")


def check_paired(
    group_a: np.ndarray,
    group_b: np.ndarray,
    min_n: int = 3,
    check_diff_var: bool = False,
) -> None:
    """Ensure paired samples have equal lengths and optional non-zero diff variance."""
    if len(group_a) != len(group_b):
        raise ValueError(
            f"Paired samples must have identical lengths, but got N1={len(group_a)} and N2={len(group_b)}."
        )
    check_sample_size(group_a, min_n=min_n, name="paired_samples")
    if check_diff_var:
        diffs = group_a - group_b
        check_variance(diffs, name="paired_differences")


def check_low_power(n: int, threshold: int = 15) -> str | None:
    """Return a warning string when sample size is in the low-power zone [3, threshold)."""
    if 3 <= n < threshold:
        return (
            f"Low sample size warning: N={n} (< {threshold}). Statistical power is reduced; "
            f"subtle or medium effects may fail to reach significance."
        )
    return None
