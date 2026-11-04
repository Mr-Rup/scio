"""Multi-format dataset ingestion with automatic delimiter sniffing and encoding detection."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyscio.core.dataframe import ensure_pandas_df, is_dataframe_like


def read(
    source: str | Path | pd.DataFrame | Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Ingest tabular data from multiple formats into a clean Pandas DataFrame.

    Supported formats:
    - Pandas DataFrame (returns clean copy)
    - CSV / TSV / Delimited text (automatic delimiter and encoding sniffing)
    - Apache Parquet (.parquet)
    - Microsoft Excel (.xlsx, .xls)
    - JSON / JSON Lines (.json, .jsonl)

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        Filepath, path object, or existing DataFrame.
    **kwargs : Any
        Additional keyword arguments forwarded to the underlying pandas reader.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if not isinstance(source, (str, Path)) and is_dataframe_like(source):
        return ensure_pandas_df(source).copy()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    suffix = path.suffix.lower()

    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path, **kwargs)

    if suffix in (".xlsx", ".xls"):
        return cast(pd.DataFrame, pd.read_excel(path, **kwargs))

    if suffix in (".json", ".jsonl"):
        lines = kwargs.pop("lines", suffix == ".jsonl")
        return cast(pd.DataFrame, pd.read_json(path, lines=lines, **kwargs))

    # Delimited text (CSV, TSV, TXT, etc.): Auto-sniff delimiter
    encoding = kwargs.pop("encoding", "utf-8")
    sep = kwargs.pop("sep", None)

    if sep is None:
        try:
            with open(path, encoding=encoding, errors="replace") as f:
                sample = f.read(8192)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",\t;| ")
            sep = dialect.delimiter
        except Exception:
            sep = "," if suffix != ".tsv" else "\t"

    return cast(pd.DataFrame, pd.read_csv(path, sep=sep, encoding=encoding, **kwargs))
