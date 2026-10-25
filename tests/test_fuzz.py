"""Property-based fuzz and invariant testing using Hypothesis."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from pyscio.core.errors import MissingDataError, ZeroVarError
from pyscio.core.guards import EPSILON, clamp_p, clean_1d
from pyscio.functional.adjust import fdr_bh
from pyscio.functional.effects import cliffs_d
from pyscio.stat.compare import compare


@given(st.floats(allow_nan=True, allow_infinity=True))
def test_clamp_p_invariants(val: float) -> None:
    clamped, display = clamp_p(val)

    if math.isnan(val):
        assert math.isnan(clamped)
        assert display == "p = nan"
    else:
        assert clamped >= EPSILON
        assert clamped <= 1.0
        assert isinstance(display, str)
        assert display.startswith("p")


@given(
    st.lists(
        st.one_of(
            st.floats(min_value=-1e6, max_value=1e6),
            st.just(float("nan")),
            st.just(float("inf")),
            st.just(float("-inf")),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_clean_1d_invariants(vals: list[float]) -> None:
    try:
        cleaned = clean_1d(vals, dropna=True)
        assert isinstance(cleaned, np.ndarray)
        assert cleaned.ndim == 1
        assert not np.any(np.isnan(cleaned))
        assert not np.any(np.isinf(cleaned))
    except (ValueError, MissingDataError) as e:
        msg = str(e).lower()
        assert any(k in msg for k in ("no valid", "empty", "infinite", "missing", "nan"))


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=80,
    )
)
@settings(max_examples=50)
def test_fdr_bh_invariants(p_vals: list[float]) -> None:
    reject, q_values = fdr_bh(p_vals, alpha=0.05)

    assert len(reject) == len(p_vals)
    assert len(q_values) == len(p_vals)
    assert np.all(q_values >= 0.0)
    assert np.all(q_values <= 1.0)

    # Invariant: q-values must be >= raw p-values (except for rounding epsilon)
    p_arr = np.array(p_vals)
    assert np.all(q_values >= p_arr - 1e-12)

    # Invariant: Monotonicity on sorted p-values
    sort_idx = np.argsort(p_arr)
    sorted_q = q_values[sort_idx]
    assert np.all(np.diff(sorted_q) >= -1e-12)


@given(
    st.lists(st.floats(min_value=-1000.0, max_value=1000.0), min_size=5, max_size=40),
    st.lists(st.floats(min_value=-1000.0, max_value=1000.0), min_size=5, max_size=40),
)
@settings(max_examples=40)
def test_cliffs_d_invariants(g1: list[float], g2: list[float]) -> None:
    try:
        delta = cliffs_d(g1, g2)
        assert -1.0 <= delta <= 1.0
        assert not math.isnan(delta)
    except ZeroVarError:
        pass


@given(
    st.lists(st.floats(min_value=-100.0, max_value=100.0), min_size=6, max_size=30),
    st.lists(st.floats(min_value=-100.0, max_value=100.0), min_size=6, max_size=30),
)
@settings(max_examples=30)
def test_compare_fuzz_two_sample(g1: list[float], g2: list[float]) -> None:
    try:
        res = compare(g1, g2)
        assert res.stat is not None
        assert 0.0 <= res.p <= 1.0
        assert res.sig == (res.p < 0.05)
    except ZeroVarError:
        pass
