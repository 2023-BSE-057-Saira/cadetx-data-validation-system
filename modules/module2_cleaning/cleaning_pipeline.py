"""Cleaning Pipeline (basics) — Module 2, Week 4."""
import sys, os
import pandas as pd
from thefuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from schema_inference import build_schema


# Columns whose values identify a specific record or person. Filling a
# missing CustomerID with a mean, or a missing email with the most common
# email, invents data that looks real but is wrong — so these are left
# missing (and reported) rather than imputed.
NON_IMPUTABLE_SEMANTICS = {"id", "email", "phone", "name", "date", "free_text", "unknown"}


def impute_missing_statistical(df, schema):
    df = df.copy()
    log = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        spec = schema.get(col, {})
        semantic = spec.get("semantic_type")
        if semantic in NON_IMPUTABLE_SEMANTICS:
            log.append({"column": col, "strategy": "skipped_not_imputable",
                        "reason": f"{semantic} values cannot be meaningfully inferred",
                        "rows_still_missing": n_missing})
            continue
        expected_dtype = spec.get("expected_dtype", "string")
        if expected_dtype in ("integer", "float"):
            numeric = pd.to_numeric(df[col], errors="coerce")
            fill_value = numeric.mean()
            if pd.isna(fill_value):
                log.append({"column": col, "strategy": "skipped_no_observed_values",
                            "rows_still_missing": n_missing})
                continue
            df[col] = numeric.fillna(fill_value)
            log.append({"column": col, "strategy": "mean_imputation",
                        "fill_value": round(float(fill_value), 4), "rows_affected": n_missing})
        else:
            mode_vals = df[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if len(mode_vals) else "unknown"
            df[col] = df[col].fillna(fill_value)
            log.append({"column": col, "strategy": "mode_imputation",
                        "fill_value": str(fill_value), "rows_affected": n_missing})
    return df, log


def detect_exact_duplicates(df):
    return df.duplicated(keep="first")


def detect_fuzzy_duplicates(df, column, threshold=90):
    values = df[column].dropna().astype(str).unique().tolist()
    if len(values) > 500:
        values = values[:500]
    matches = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            score = fuzz.token_sort_ratio(values[i], values[j])
            if score >= threshold and values[i] != values[j]:
                matches.append({"value_a": values[i], "value_b": values[j], "similarity": score})
    return matches


def normalise_types_and_formats(df, schema):
    df = df.copy()
    log = []
    for col, spec in schema.items():
        if col not in df.columns:
            continue
        semantic = spec["semantic_type"]

        if semantic == "date":
            before_nulls = df[col].isna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
            after_nulls = df[col].isna().sum()
            log.append({"column": col, "action": "parsed_to_datetime",
                        "newly_unparseable": int(after_nulls - before_nulls)})

        elif semantic in ("numeric", "id") and spec["expected_dtype"] in ("integer", "float"):
            before = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            newly_null = int((df[col].isna() & before.notna()).sum())
            if newly_null:
                log.append({"column": col, "action": "coerced_to_numeric", "newly_unparseable": newly_null})

        elif semantic == "categorical" and (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            stripped = df[col].astype(str).str.strip()
            lower_to_common = (stripped[stripped.notna()].groupby(stripped.str.lower())
                                .agg(lambda x: x.value_counts().idxmax()))
            n_changed = 0
            def _normalise(v):
                nonlocal n_changed
                if pd.isna(v):
                    return v
                key = str(v).strip().lower()
                canonical = lower_to_common.get(key, v)
                if canonical != v:
                    n_changed += 1
                return canonical
            df[col] = df[col].apply(_normalise)
            if n_changed:
                log.append({"column": col, "action": "normalised_casing_whitespace", "rows_changed": n_changed})

    return df, log


VALID_NUMERIC_IMPUTATION = ("statistical", "knn", "iterative")


def run_cleaning_pipeline(df, schema=None, numeric_imputation="statistical"):
    """
    Order matters:
      1. drop exact duplicates
      2. normalise types/casing/whitespace
      3. drop duplicates again (rows that differed only by whitespace/casing
         are identical now)
      4. impute numeric columns (statistical, KNN or iterative)
      5. statistical imputation for whatever remains imputable
    Duplicates are removed BEFORE imputation so imputed values can never
    create or hide duplicates.
    """
    if numeric_imputation not in VALID_NUMERIC_IMPUTATION:
        raise ValueError(f"numeric_imputation must be one of {VALID_NUMERIC_IMPUTATION}")
    if schema is None:
        schema = build_schema(df)

    cleaning_log = {"steps": []}

    exact_dup_mask = detect_exact_duplicates(df)
    df_deduped = df[~exact_dup_mask].reset_index(drop=True)
    cleaning_log["steps"].append({"step": "exact_duplicate_removal", "rows_removed": int(exact_dup_mask.sum())})

    text_cols = [c for c, s in schema.items() if s["semantic_type"] in ("categorical", "free_text")]
    fuzzy_findings = {}
    for col in text_cols:
        if col in df_deduped.columns:
            matches = detect_fuzzy_duplicates(df_deduped, col)
            if matches:
                fuzzy_findings[col] = matches
    cleaning_log["fuzzy_duplicate_findings"] = fuzzy_findings

    df_normalised, normalise_log = normalise_types_and_formats(df_deduped, schema)
    cleaning_log["steps"].append({"step": "type_format_normalisation", "details": normalise_log})

    post_mask = detect_exact_duplicates(df_normalised)
    df_normalised = df_normalised[~post_mask].reset_index(drop=True)
    cleaning_log["steps"].append({"step": "post_normalisation_duplicate_removal",
                                  "rows_removed": int(post_mask.sum())})

    if numeric_imputation != "statistical":
        from ml_imputation import knn_impute, iterative_impute
        ml_fn = knn_impute if numeric_imputation == "knn" else iterative_impute
        df_normalised, ml_log = ml_fn(df_normalised, schema)
        cleaning_log["steps"].append({"step": "ml_numeric_imputation", "details": ml_log})

    df_imputed, impute_log = impute_missing_statistical(df_normalised, schema)
    cleaning_log["steps"].append({"step": "missing_value_imputation", "details": impute_log})

    cleaning_log["numeric_imputation"] = numeric_imputation
    cleaning_log["rows_before"] = len(df)
    cleaning_log["rows_after"] = len(df_imputed)

    return {"cleaned_df": df_imputed, "cleaning_log": cleaning_log}


if __name__ == "__main__":
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    df = pd.read_csv(path)
    result = run_cleaning_pipeline(df)
    os.makedirs("data/processed", exist_ok=True)
    result["cleaned_df"].to_csv("data/processed/cleaned_data_partial.csv", index=False)
    with open("data/processed/cleaning_log_partial.json", "w") as f:
        json.dump(result["cleaning_log"], f, indent=2, default=str)
    print("done")
