"""8-role semantic column classification for automatic inference routing."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


class Role(StrEnum):
    """Semantic data role for a column in a dataset."""

    NUMERIC = "numeric"
    DISCRETE = "discrete"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    ID = "id"
    CONSTANT = "constant"


_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^uuid$|^guid$|^ssn$|^key$|^token$|^code$|^hash$)",
    re.IGNORECASE,
)


def classify_series(s: pd.Series) -> Role:
    """Classify a Pandas Series into one of 8 semantic roles.

    Evaluation hierarchy:
    1. Constant: <= 1 distinct value.
    2. Temporal: datetime / timestamp / period.
    3. Binary: exactly 2 distinct non-null values.
    4. Identifier: column name matches ID patterns AND high uniqueness (> 90%), or 100% unique strings.
    5. Numeric / Discrete: numeric dtypes partitioned by cardinality.
    6. Text: string objects with long average character length (> 50 chars).
    7. Categorical: all other discrete string/categorical variables.
    """
    valid = s.dropna()
    n_valid = len(valid)

    # 1. Constant
    if n_valid <= 1 or valid.nunique() <= 1:
        return Role.CONSTANT

    n_unique = valid.nunique()
    uniqueness_ratio = n_unique / n_valid

    # 2. Temporal
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
        return Role.TEMPORAL

    # 3. Binary
    if n_unique == 2:
        return Role.BINARY

    # 4. Text vs Identifier vs Categorical for String / Object
    name_str = str(s.name) if s.name is not None else ""
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        sample = valid.iloc[:100].astype(str)
        avg_len = float(sample.str.len().mean()) if len(sample) > 0 else 0.0
        if avg_len > 50.0:
            return Role.TEXT
        if _ID_PATTERNS.search(name_str) and uniqueness_ratio > 0.85:
            return Role.ID
        if uniqueness_ratio > 0.98:
            return Role.ID
        return Role.CATEGORICAL

    # Check ID patterns for numeric identifiers (e.g., employee_id = 1001, 1002...)
    if _ID_PATTERNS.search(name_str) and uniqueness_ratio > 0.90:
        return Role.ID

    # 5. Numeric / Discrete
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and n_unique <= 20:
            return Role.DISCRETE
        if pd.api.types.is_float_dtype(s) and n_unique <= 10 and np.all(valid == valid.round()):
            return Role.DISCRETE
        return Role.NUMERIC

    if isinstance(s.dtype, pd.CategoricalDtype):
        return Role.CATEGORICAL

    return Role.CATEGORICAL


def classify(df: pd.DataFrame) -> dict[str, Role]:
    """Classify all columns in a DataFrame into their semantic roles."""
    return {str(col): classify_series(df[col]) for col in df.columns}
