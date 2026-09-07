"""
Cleaning Pipeline (basics)
Module 2 — Week 4 deliverable

First pass of the cleaning engine:
  - missing-value imputation (statistical: mean/median for numeric,
    mode for categorical)
  - duplicate detection: exact-match duplicates, plus fuzzy-match
    near-duplicates on text columns (e.g. "United Kingdom" vs "UK",
    or minor typos in a description field)
  - type / string / date / categorical normalisation, driven by the
    schema built in schema_inference.py

ML-based imputation (KNN/regression/iterative) and the transformation
engine (scaling/encoding/feature extraction) are Week 5 work — this
week is the statistical baseline + duplicate/format cleanup.
"""

import sys
import os

import numpy as np
import pandas as pd
from thefuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from schema_inference import build_schema


def impute_missing_statistical(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list]:
    """Fills missing values: mean for numeric columns, mode for
    categorical/text columns. Returns the filled DataFrame and a log of
    what was done, per column."""
    df = df.copy()
    log = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        expected_dtype = schema.get(col, {}).get("expected_dtype", "string")
        if expected_dtype in ("integer", "float"):
            numeric = pd.to_numeric(df[col], errors="coerce")
            fill_value = numeric.mean()
            df[col] = numeric.fillna(fill_value)
            log.append({
                "column": col, "strategy": "mean_imputation",
                "fill_value": round(float(fill_value), 4), "rows_affected": n_missing,
            })
        else:
            mode_vals = df[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if len(mode_vals) else "unknown"
            df[col] = df[col].fillna(fill_value)
            log.append({
                "column": col, "strategy": "mode_imputation",
                "fill_value": str(fill_value), "rows_affected": n_missing,
            })
    return df, log


def detect_exact_duplicates(df: pd.DataFrame) -> pd.Series:
    """Returns a boolean mask of exact-match duplicate rows (keeping the first)."""
    return df.duplicated(keep="first")


def detect_fuzzy_duplicates(df: pd.DataFrame, column: str, threshold: int = 90) -> list:
    """
    Finds near-duplicate values within a single text column using fuzzy
    string matching (e.g. catches typos or near-identical descriptions
    that exact matching would miss). Returns pairs of (index_a, index_b,
    value_a, value_b, similarity_score) for values above the threshold.

    Runs on unique values only (not row-by-row) to keep it fast on larger
    datasets — still O(n^2) on unique values, so intended for columns with
    a moderate number of distinct values (e.g. product descriptions,
    country names), not free-text columns with thousands of uniques.
    """
    values = df[column].dropna().astype(str).unique().tolist()
    if len(values) > 500:
        values = values[:500]  # cap for performance on this basic pass

    matches = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            score = fuzz.token_sort_ratio(values[i], values[j])
            if score >= threshold and values[i] != values[j]:
                matches.append({
                    "value_a": values[i], "value_b": values[j], "similarity": score,
                })
    return matches


def normalise_types_and_formats(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list]:
    """Applies normalisation based on the inferred schema:
    - strips/lowercases-consistency for categorical text (preserving
      original casing of the most common variant)
    - parses date columns into actual datetime dtype
    - coerces numeric-typed columns to numeric dtype
    """
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
            log.append({
                "column": col, "action": "parsed_to_datetime",
                "newly_unparseable": int(after_nulls - before_nulls),
            })

        elif semantic in ("numeric", "id") and spec["expected_dtype"] in ("integer", "float"):
            before = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            newly_null = int((df[col].isna() & before.notna()).sum())
            if newly_null:
                log.append({"column": col, "action": "coerced_to_numeric", "newly_unparseable": newly_null})

        elif semantic == "categorical" and (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            stripped = df[col].astype(str).str.strip()
            lower_to_common = (
                stripped[stripped.notna()]
                .groupby(stripped.str.lower())
                .agg(lambda x: x.value_counts().idxmax())
            )
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


def run_cleaning_pipeline(df: pd.DataFrame, schema: dict | None = None) -> dict:
    """Runs the Week 4 cleaning steps end-to-end. Returns the cleaned
    DataFrame plus a structured cleaning log (this becomes cleaning_log.json
    once the API wrapper is added in Week 6)."""
    if schema is None:
        schema = build_schema(df)

    cleaning_log = {"steps": []}

    exact_dup_mask = detect_exact_duplicates(df)
    n_exact_dups = int(exact_dup_mask.sum())
    df_deduped = df[~exact_dup_mask].reset_index(drop=True)
    cleaning_log["steps"].append({
        "step": "exact_duplicate_removal", "rows_removed": n_exact_dups,
    })

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

    df_imputed, impute_log = impute_missing_statistical(df_normalised, schema)
    cleaning_log["steps"].append({"step": "missing_value_imputation", "details": impute_log})

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
    print("Wrote data/processed/cleaned_data_partial.csv and cleaning_log_partial.json")
    print(f"Rows before: {result['cleaning_log']['rows_before']}, after: {result['cleaning_log']['rows_after']}")
