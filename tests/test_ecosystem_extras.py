"""Tests for F7 (Plotting), F8 (Polars/Universal DataFrame Adapter), and F10 (Mixed-Effects Models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import pyscio as sc
from pyscio.core import SVGFigure, ensure_pandas_df, is_dataframe_like
from pyscio.integrations import PolarsScioNamespace


class _MockPolarsDF:
    """Mock Polars DataFrame implementing `.to_pandas()`."""

    def __init__(self, pdf: pd.DataFrame) -> None:
        self._pdf = pdf
        self.scio = PolarsScioNamespace(self)

    def to_pandas(self) -> pd.DataFrame:
        return self._pdf.copy()


class _MockPolarsLazyFrame:
    """Mock Polars LazyFrame implementing `.collect().to_pandas()`."""

    def __init__(self, pdf: pd.DataFrame) -> None:
        self._pdf = pdf

    def collect(self) -> _MockPolarsDF:
        return _MockPolarsDF(self._pdf)


class _MockAxes:
    """Minimal mock Matplotlib Axes for testing `backend='matplotlib'` without requiring GUI."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def axvline(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("axvline")

    def errorbar(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("errorbar")

    def barh(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("barh")

    def fill_between(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("fill_between")

    def plot(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("plot")

    def scatter(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("scatter")

    def set_yticks(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("set_yticks")

    def set_yticklabels(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("set_yticklabels")

    def invert_yaxis(self) -> None:
        self.calls.append("invert_yaxis")

    def set_xlim(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("set_xlim")

    def set_xlabel(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("set_xlabel")

    def set_ylabel(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("set_ylabel")

    def set_title(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append("set_title")


def test_f7_plotting_svg_and_matplotlib_axes(tmp_path: Path) -> None:
    """Verify SVGFigure and matplotlib Axes rendering across TestResult, RegressionResult, ScreenReport, EDAReport, and Raincloud."""
    rng = np.random.default_rng(101)
    df = pd.DataFrame(
        {
            "group": ["A"] * 30 + ["B"] * 30,
            "x1": rng.normal(0.0, 1.0, 60),
            "y": np.concatenate([rng.normal(10.0, 1.5, 30), rng.normal(13.0, 1.5, 30)]),
        }
    )

    # 1. TestResult.plot()
    t_res = sc.compare(df, x="group", y="y")
    svg_t = t_res.plot(backend="svg")
    assert isinstance(svg_t, SVGFigure)
    assert "<svg" in svg_t._repr_svg_()
    out_svg = tmp_path / "test_plot.svg"
    svg_t.save(out_svg)
    assert out_svg.exists()

    ax_t = _MockAxes()
    t_res.plot(ax=ax_t)
    assert "errorbar" in ax_t.calls

    # 2. RegressionResult.plot()
    reg_res = sc.ols(df, "y ~ x1 + C(group)")
    svg_r = reg_res.plot(backend="svg")
    assert isinstance(svg_r, SVGFigure)
    assert "Intercept" not in svg_r.svg or "x1" in svg_r.svg

    ax_r = _MockAxes()
    reg_res.plot(ax=ax_r)
    assert "errorbar" in ax_r.calls

    # 3. ScreenReport.plot()
    scr_res = sc.screen(df, y="y")
    svg_s = scr_res.plot(backend="svg")
    assert isinstance(svg_s, SVGFigure)
    ax_s = _MockAxes()
    scr_res.plot(ax=ax_s)
    assert "barh" in ax_s.calls

    # 4. EDAReport.plot()
    eda_res = df.scio.eda()
    svg_e = eda_res.plot(backend="svg")
    assert isinstance(svg_e, SVGFigure)
    ax_e = _MockAxes()
    eda_res.plot(ax=ax_e)
    assert "barh" in ax_e.calls

    # 5. Raincloud plot (standalone & accessor)
    svg_rc = df.scio.plot_raincloud(x="group", y="y", backend="svg")
    assert isinstance(svg_rc, SVGFigure)
    assert "<polygon" in svg_rc.svg
    ax_rc = _MockAxes()
    sc.plot_raincloud(df, x="group", y="y", ax=ax_rc)
    assert "fill_between" in ax_rc.calls
    assert "scatter" in ax_rc.calls


def test_f8_polars_lazyframe_and_dict_coercion() -> None:
    """Verify Polars DataFrame, LazyFrame, and dict coercion across pyscio entry points."""
    raw_dict = {
        "treatment": ["ctrl"] * 25 + ["drug"] * 25,
        "dose": list(np.linspace(1.0, 10.0, 50)),
        "outcome": [5.0 + 0.4 * i for i in range(25)] + [12.0 + 0.4 * i for i in range(25)],
    }
    assert is_dataframe_like(raw_dict)
    pdf = ensure_pandas_df(raw_dict)
    assert isinstance(pdf, pd.DataFrame)

    # Dict directly into compare, screen, ols
    res_dict = sc.compare(raw_dict, x="treatment", y="outcome")
    assert res_dict.sig

    # Mock Polars DataFrame and LazyFrame
    pl_df = _MockPolarsDF(pdf)
    pl_lazy = _MockPolarsLazyFrame(pdf)

    assert is_dataframe_like(pl_df)
    assert is_dataframe_like(pl_lazy)

    res_pl = pl_df.scio.compare(x="treatment", y="outcome")
    assert abs(res_pl.stat - res_dict.stat) < 1e-10

    ols_lazy = sc.ols(pl_lazy, "outcome ~ dose + C(treatment)")
    assert ols_lazy.p < 0.001

    scr_pl = pl_df.scio.screen(y="outcome")
    assert scr_pl.n_features == 2

    eda_pl = pl_df.scio.eda()
    assert eda_pl.n_rows == 50


def test_f10_linear_mixed_effects_model() -> None:
    """Verify Linear Mixed-Effects Model (REML & ML) recovery of fixed effects, ICC, and R²."""
    rng = np.random.default_rng(2026)
    n_subjects = 20
    n_per_subj = 6
    subj_ids = np.repeat([f"S{i:02d}" for i in range(n_subjects)], n_per_subj)
    random_intercepts = np.repeat(rng.normal(0.0, 3.0, size=n_subjects), n_per_subj)
    time = np.tile(np.arange(n_per_subj, dtype=float), n_subjects)
    y = 10.0 + 1.8 * time + random_intercepts + rng.normal(0.0, 1.0, size=n_subjects * n_per_subj)

    df = pd.DataFrame({"subject": subj_ids, "time": time, "score": y})

    # Fit via Wilkinson random-intercept syntax
    lmm = sc.mixedlm(df, "score ~ time + (1 | subject)", reml=True)
    assert "Linear Mixed Model (REML)" in lmm.model_type
    time_row = lmm.coef_df[lmm.coef_df["term"] == "time"].iloc[0]
    assert 1.5 < float(time_row["coef"]) < 2.1
    assert float(time_row["p_value"]) < 1e-10
    # Conditional R^2 (fixed + random) should exceed Marginal R^2 (fixed only)
    assert lmm.adj_r_squared > lmm.r_squared > 0.15
    assert any("ICC =" in w for w in lmm.warns)

    # Fit via compare() automatic routing for `(1 | subject)`
    cmp_lmm = sc.compare(df, "score ~ time + (1 | subject)")
    assert "Linear Mixed Model" in cmp_lmm.test_name
    assert cmp_lmm.sig
