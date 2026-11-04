"""Universal DataFrame coercion supporting Pandas, Polars, PyArrow, Narwhals, and Interchange objects."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from omnistats.core.errors import DtypeError


def is_dataframe_like(obj: Any) -> bool:
    """Return True if `obj` is a DataFrame, LazyFrame, PyArrow Table, Interchange object, or column dict."""
    if isinstance(obj, (pd.DataFrame, dict)):
        return True
    if hasattr(obj, "to_pandas") and callable(obj.to_pandas):
        return True
    if hasattr(obj, "__dataframe__") and callable(obj.__dataframe__):
        return True
    return bool(hasattr(obj, "collect") and callable(obj.collect))


def ensure_pandas_df(data: Any) -> pd.DataFrame:
    """Coerce a tabular dataset (Pandas, Polars, Narwhals, PyArrow, or dict) into a Pandas DataFrame.

    Fast-paths existing `pd.DataFrame` instances without copying.
    """
    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, dict):
        try:
            return pd.DataFrame(data)
        except Exception as exc:
            raise DtypeError(f"Could not convert dictionary to DataFrame: {exc}") from exc

    # Optional Narwhals universal DataFrame bridge if installed
    try:
        import narwhals as nw

        nw_obj = nw.from_native(data, pass_through=True)
        if isinstance(nw_obj, nw.LazyFrame):
            nw_obj = nw_obj.collect()
        if isinstance(nw_obj, nw.DataFrame):
            return cast(pd.DataFrame, nw_obj.to_pandas())
    except ImportError:
        pass
    except Exception:
        pass

    # Handle lazy frames (e.g., polars.LazyFrame) before calling .to_pandas()
    target = data
    if hasattr(target, "collect") and callable(target.collect):
        target = target.collect()

    if hasattr(target, "to_pandas") and callable(target.to_pandas):
        res = target.to_pandas()
        if isinstance(res, pd.DataFrame):
            return res

    # Python DataFrame Interchange Protocol (__dataframe__)
    if hasattr(target, "__dataframe__") and callable(target.__dataframe__):
        return pd.api.interchange.from_dataframe(target)

    raise DtypeError(
        f"Expected a DataFrame-compatible object (pandas.DataFrame, polars.DataFrame, "
        f"pyarrow.Table, or dict), got {type(data).__name__}."
    )
