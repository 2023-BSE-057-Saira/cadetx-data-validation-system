"""
Data-Quality Scoring
Module 2 — Week 6 deliverable

Scores a dataset from 0-100 across four standard data-quality dimensions,
so cleaning can be measured (score before, score after, delta) instead of
just asserted:

  - completeness : share of cells that are not missing
  - uniqueness   : share of rows that are not exact duplicates
  - validity     : share of values that conform to their column's expected
                   type/format (numeric parses, dates parse, emails/phones
                   match a valid pattern)
  - consistency  : share of text values that are written the same way as
                   the dominant form of that value (casing / whitespace)

Overall score = weighted average of the four dimensions.

Both the "before" and "after" scores must be computed against the SAME
schema (inferred from the raw data), otherwise the comparison is unfair —
see compare_quality().
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module1_profiling"))
from metadata_extractor import _looks_like_email, _looks_like_phone
from schema_inference import build_schema

WEIGHTS = {"completeness": 0.30, "uniqueness": 0.20, "validity": 0.30, "consistency": 0.20}


def _is_text(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def completeness_score(df):
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 100.0, {}
    per_col = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    missing = sum(per_col.values())
    return round(100 * (1 - missing / total_cells), 2), {"missing_cells_by_column": per_col}


def uniqueness_score(df):
    n = len(df)
    if n == 0:
        return 100.0, {}
    dups = int(df.duplicated().sum())
    return round(100 * (1 - dups / n), 2), {"duplicate_rows": dups}


def _column_validity(series, semantic):
    """Fraction (0-1) of non-null values valid for the column's semantic
    type, or None if validity isn't checkable for this column."""
    vals = series.dropna()
    if len(vals) == 0:
        return None
    if semantic == "numeric":
        if pd.api.types.is_numeric_dtype(series):
            return 1.0
        return float(pd.to_numeric(vals, errors="coerce").notna().mean())
    if semantic == "date":
        if pd.api.types.is_datetime64_any_dtype(series):
            return 1.0
        return float(pd.to_datetime(vals.astype(str), errors="coerce", format="mixed").notna().mean())
    if semantic == "email":
        return float(_looks_like_email(series))
    if semantic == "phone":
        return float(_looks_like_phone(series))
    return None


def validity_score(df, schema):
    per_col = {}
    for col in df.columns:
        semantic = schema.get(col, {}).get("semantic_type")
        rate = _column_validity(df[col], semantic)
        if rate is not None:
            per_col[col] = round(100 * rate, 2)
    if not per_col:
        return 100.0, {"validity_by_column": {}}
    return round(sum(per_col.values()) / len(per_col), 2), {"validity_by_column": per_col}


def _column_consistency(series):
    """Share of values equal to the dominant stripped spelling of their
    case-insensitive group. '  Spain', 'spain' and 'Spain' are one value
    written three ways; only the dominant form counts as consistent."""
    vals = series.dropna().astype(str)
    if len(vals) == 0:
        return None
    stripped = vals.str.strip()
    dominant = stripped.groupby(stripped.str.lower()).agg(lambda x: x.value_counts().idxmax())
    canonical = stripped.str.lower().map(dominant)
    return float((vals == canonical).mean())


def consistency_score(df, schema):
    per_col = {}
    for col in df.columns:
        if schema.get(col, {}).get("semantic_type") != "categorical" or not _is_text(df[col]):
            continue
        rate = _column_consistency(df[col])
        if rate is not None:
            per_col[col] = round(100 * rate, 2)
    if not per_col:
        return 100.0, {"consistency_by_column": {}}
    return round(sum(per_col.values()) / len(per_col), 2), {"consistency_by_column": per_col}


def score_dataset(df, schema=None):
    """Returns {'overall': float, 'dimensions': {...}, 'details': {...}}."""
    if schema is None:
        schema = build_schema(df)
    comp, comp_d = completeness_score(df)
    uniq, uniq_d = uniqueness_score(df)
    val, val_d = validity_score(df, schema)
    cons, cons_d = consistency_score(df, schema)
    dimensions = {"completeness": comp, "uniqueness": uniq, "validity": val, "consistency": cons}
    overall = round(sum(dimensions[k] * WEIGHTS[k] for k in WEIGHTS), 2)
    return {"overall": overall, "dimensions": dimensions,
            "details": {**comp_d, **uniq_d, **val_d, **cons_d}}


def compare_quality(before_df, after_df, schema=None):
    """Scores both DataFrames against one shared schema (inferred from the
    raw data) and reports the improvement per dimension."""
    if schema is None:
        schema = build_schema(before_df)
    before = score_dataset(before_df, schema)
    after = score_dataset(after_df, schema)
    delta = {"overall": round(after["overall"] - before["overall"], 2),
             **{k: round(after["dimensions"][k] - before["dimensions"][k], 2) for k in WEIGHTS}}
    return {"before": before, "after": after, "delta": delta}


if __name__ == "__main__":
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    print(json.dumps(score_dataset(pd.read_csv(path)), indent=2, default=str))
