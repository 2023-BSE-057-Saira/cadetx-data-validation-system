"""Metadata Extraction Engine — Module 1, Week 1."""
import re, json
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,15}$")
ID_NAME_HINTS = ("id", "code", "no", "number", "sku", "key")
EMAIL_NAME_HINTS = ("email", "mail")
PHONE_NAME_HINTS = ("phone", "tel", "mobile", "contact")
DATE_NAME_HINTS = ("date", "time", "dob", "created", "updated")
NAME_NAME_HINTS = ("name", "customer", "client", "firstname", "lastname")


@dataclass
class ColumnMetadata:
    column: str
    pandas_dtype: str
    semantic_type: str
    is_mixed_type: bool
    null_count: int
    null_percent: float
    unique_count: int
    sample_values: list


def _looks_like_email(series):
    vals = series.dropna().astype(str)
    return (vals.str.match(EMAIL_RE)).mean() if len(vals) else 0.0


def _looks_like_phone(series):
    vals = series.dropna().astype(str)
    return (vals.str.match(PHONE_RE)).mean() if len(vals) else 0.0


def _looks_like_date(series):
    """Fraction of values that look like dates.

    Bare digit strings ('07123', '2021') are NOT counted: pandas happily
    reads them as a year, which would make any numeric code / postcode
    column look like a date column. Real date strings contain a separator
    or month name ('2021-01-04', '4 Jan 2021'). A column whose NAME says
    it is a date is still caught by the name-hint check.
    """
    vals = series.dropna().astype(str).str.strip()
    if len(vals) == 0:
        return 0.0
    pure_digits = vals.str.fullmatch(r"\d+")
    parsed = pd.to_datetime(vals, errors="coerce", format="mixed")
    return float((parsed.notna() & ~pure_digits).mean())


def _name_hint(col_name, hints):
    lname = col_name.lower()
    return any(h in lname for h in hints)


def _detect_mixed_type(series):
    vals = series.dropna()
    if len(vals) == 0:
        return False
    types_seen = {type(v) for v in vals}
    non_numeric_types = {t for t in types_seen if t not in (int, float, np.int64, np.float64)}
    is_text_dtype = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    if is_text_dtype:
        if non_numeric_types and any(t in (int, float, np.int64, np.float64) for t in types_seen):
            return True
        numeric_coerced = pd.to_numeric(vals, errors="coerce")
        frac_numeric = numeric_coerced.notna().mean()
        if 0 < frac_numeric < 1:
            return True
    return False


def infer_semantic_type(series, col_name):
    if pd.api.types.is_numeric_dtype(series) and not _detect_mixed_type(series):
        if _name_hint(col_name, ID_NAME_HINTS):
            return "id"
        return "numeric"

    # Name-hint checks take priority over content heuristics: heuristics
    # like the phone regex can ambiguously match other formats (e.g. a
    # date string "2021-01-04" also matches a loose phone-number pattern),
    # so a column's own name is a more reliable signal when available.
    if _name_hint(col_name, EMAIL_NAME_HINTS):
        return "email"
    if _name_hint(col_name, PHONE_NAME_HINTS):
        return "phone"
    if _name_hint(col_name, DATE_NAME_HINTS):
        return "date"
    if _name_hint(col_name, NAME_NAME_HINTS):
        return "name"
    if _name_hint(col_name, ID_NAME_HINTS):
        return "id"

    # No name hint — fall back to content-based heuristics, checking the
    # more specific/unambiguous format (email) before the looser one (phone).
    if _looks_like_email(series) > 0.6:
        return "email"
    if _looks_like_date(series) > 0.6:
        return "date"
    if _looks_like_phone(series) > 0.6:
        return "phone"

    n_unique = series.nunique(dropna=True)
    n_total = len(series.dropna())
    if n_total > 0 and n_unique / n_total < 0.05 and n_unique < 50:
        return "categorical"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        avg_len = series.dropna().astype(str).str.len().mean() if n_total else 0
        if avg_len and avg_len > 30:
            return "free_text"
        return "categorical" if n_unique < 50 else "unknown"
    return "unknown"


def extract_column_metadata(df, col):
    series = df[col]
    null_count = int(series.isna().sum())
    total = len(series)
    return ColumnMetadata(
        column=col, pandas_dtype=str(series.dtype),
        semantic_type=infer_semantic_type(series, col),
        is_mixed_type=_detect_mixed_type(series), null_count=null_count,
        null_percent=round(100 * null_count / total, 2) if total else 0.0,
        unique_count=int(series.nunique(dropna=True)),
        sample_values=[str(v) for v in series.dropna().unique()[:5].tolist()],
    )


def extract_metadata(df):
    return {"n_rows": len(df), "n_columns": len(df.columns),
            "columns": [asdict(extract_column_metadata(df, c)) for c in df.columns]}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    print(json.dumps(extract_metadata(pd.read_csv(path)), indent=2))
