"""Schema Inference Engine — Module 2, Week 4."""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module1_profiling"))
from metadata_extractor import infer_semantic_type


def infer_expected_type(series, col_name):
    semantic = infer_semantic_type(series, col_name)
    if semantic in ("numeric", "id"):
        expected_dtype = "integer" if pd.api.types.is_integer_dtype(series.dropna()) else "float"
    elif semantic == "date":
        expected_dtype = "datetime"
    else:
        expected_dtype = "string"
    return {"semantic_type": semantic, "expected_dtype": expected_dtype}


def infer_expected_range(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 4:
        return None
    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {"observed_min": float(numeric.min()), "observed_max": float(numeric.max()),
            "plausible_lower_bound": float(q1 - 1.5 * iqr), "plausible_upper_bound": float(q3 + 1.5 * iqr)}


def infer_expected_format(series, col_name, semantic):
    if semantic == "email":
        return "user@domain.tld"
    if semantic == "phone":
        return "+<country code> <digits>, 7-15 characters"
    if semantic == "date":
        return "parseable date/datetime"
    if semantic == "categorical":
        allowed = sorted(series.dropna().astype(str).str.strip().unique().tolist())
        return f"one of: {allowed}" if len(allowed) <= 20 else "categorical (>20 distinct values)"
    return None


def build_schema(df):
    schema = {}
    for col in df.columns:
        semantic = infer_semantic_type(df[col], col)
        entry = infer_expected_type(df[col], col)
        entry["expected_range"] = infer_expected_range(df[col])
        entry["expected_format"] = infer_expected_format(df[col], col, semantic)
        schema[col] = entry
    return schema


if __name__ == "__main__":
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    print(json.dumps(build_schema(pd.read_csv(path)), indent=2, default=str))
