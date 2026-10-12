"""Schema Inference Engine — Module 2, Week 4."""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module1_profiling"))
from metadata_extractor import infer_semantic_type


def _mostly_numeric_text(series, threshold=0.8):
    """True if a text column is a numeric column with a few stray
    non-numeric values (e.g. '38', '12', 'unknown').

    Deliberately strict, so real text columns are never coerced:
      - must be MIXED: at least `threshold` of values numeric, but not all
        (a fully-numeric text column is left alone)
      - values with leading zeros ('07123', '00123') are codes/postcodes,
        not quantities; coercing them would silently destroy the zeros
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    vals = series.dropna().astype(str).str.strip()
    if len(vals) == 0:
        return False
    if vals.str.match(r"^0\d+").any():
        return False
    frac_numeric = pd.to_numeric(vals, errors="coerce").notna().mean()
    return threshold <= frac_numeric < 1.0


def infer_expected_type(series, col_name):
    semantic = infer_semantic_type(series, col_name)

    # Module 1 labels a mixed-type numeric column "unknown" or "categorical"
    # (depending only on how many distinct values it has); recover it as
    # numeric so cleaning can coerce the stray text and impute.
    if semantic in ("unknown", "categorical") and _mostly_numeric_text(series):
        semantic = "numeric"
        parsed = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        expected_dtype = "integer" if (parsed % 1 == 0).all() else "float"
        return {"semantic_type": semantic, "expected_dtype": expected_dtype}

    if semantic in ("numeric", "id"):
        if not pd.api.types.is_numeric_dtype(series):
            # A TEXT identifier ('C536379', '85123A') is a label, not a number:
            # coercing it would blank every value containing a letter.
            expected_dtype = "string"
        else:
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
        entry = infer_expected_type(df[col], col)
        semantic = entry["semantic_type"]
        entry["expected_range"] = infer_expected_range(df[col])
        entry["expected_format"] = infer_expected_format(df[col], col, semantic)
        schema[col] = entry
    return schema


if __name__ == "__main__":
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    print(json.dumps(build_schema(pd.read_csv(path)), indent=2, default=str))
