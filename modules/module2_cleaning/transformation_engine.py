"""
Transformation Engine
Module 2 — Week 5 deliverable

Prepares cleaned data for downstream use (modeling, analytics) via:
  - scaling: numeric columns to a common scale (standardisation)
  - encoding: categorical columns to numeric form (one-hot for low
    cardinality, label/ordinal-style for higher cardinality)
  - feature extraction: a few generic, dataset-agnostic derived features
    (e.g. extracting year/month/day-of-week from date columns, and a
    total-value feature when quantity + unit-price-like columns exist)
"""

import sys
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from schema_inference import build_schema


def scale_numeric_columns(df: pd.DataFrame, schema: dict) -> tuple:
    """Standardises numeric (non-ID) columns to mean 0, std 1. ID-like
    columns are left untouched — scaling an identifier is meaningless."""
    df = df.copy()
    log = []
    scaler = StandardScaler()

    cols_to_scale = [
        c for c, s in schema.items()
        if c in df.columns and s["expected_dtype"] in ("integer", "float") and s["semantic_type"] == "numeric"
    ]
    if not cols_to_scale:
        return df, {"scaled_columns": [], "note": "no eligible numeric columns"}

    numeric_block = df[cols_to_scale].apply(pd.to_numeric, errors="coerce")
    valid_rows = numeric_block.dropna()
    if len(valid_rows) == 0:
        return df, {"scaled_columns": [], "note": "no complete rows to fit scaler"}

    scaler.fit(valid_rows)
    scaled = scaler.transform(numeric_block.fillna(valid_rows.mean()))
    for i, col in enumerate(cols_to_scale):
        df[f"{col}_scaled"] = scaled[:, i]

    return df, {"scaled_columns": cols_to_scale,
                "means": {c: round(float(m), 4) for c, m in zip(cols_to_scale, scaler.mean_)},
                "stds": {c: round(float(s), 4) for c, s in zip(cols_to_scale, scaler.scale_)}}


def encode_categorical_columns(df: pd.DataFrame, schema: dict, onehot_max_categories: int = 10) -> tuple:
    """
    One-hot encodes categorical columns with few distinct values;
    label-encodes ones with more (to avoid an explosion of columns).
    """
    df = df.copy()
    log = {"onehot_encoded": [], "label_encoded": []}

    cat_cols = [c for c, s in schema.items() if c in df.columns and s["semantic_type"] == "categorical"]

    for col in cat_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique == 0:
            continue
        if n_unique <= onehot_max_categories:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df, dummies], axis=1)
            log["onehot_encoded"].append({"column": col, "n_categories": int(n_unique),
                                           "new_columns": dummies.columns.tolist()})
        else:
            le = LabelEncoder()
            non_null_mask = df[col].notna()
            df.loc[non_null_mask, f"{col}_encoded"] = le.fit_transform(df.loc[non_null_mask, col].astype(str))
            log["label_encoded"].append({"column": col, "n_categories": int(n_unique)})

    return df, log


def extract_date_features(df: pd.DataFrame, schema: dict) -> tuple:
    """Extracts year/month/day-of-week/is_weekend from date columns."""
    df = df.copy()
    log = []
    date_cols = [c for c, s in schema.items() if c in df.columns and s["semantic_type"] == "date"]

    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        df[f"{col}_year"] = parsed.dt.year
        df[f"{col}_month"] = parsed.dt.month
        df[f"{col}_day_of_week"] = parsed.dt.dayofweek
        df[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6])
        log.append({"column": col, "features_added":
                     [f"{col}_year", f"{col}_month", f"{col}_day_of_week", f"{col}_is_weekend"]})

    return df, log


def extract_derived_numeric_features(df: pd.DataFrame, schema: dict) -> tuple:
    """Generic derived feature: if the dataset has quantity-like and
    price-like numeric columns, add a total-value feature. Dataset-agnostic
    by matching on column-name hints rather than hardcoding retail-specific
    names."""
    df = df.copy()
    log = []
    qty_col = next((c for c in df.columns if "quantity" in c.lower() or "qty" in c.lower()), None)
    price_col = next((c for c in df.columns if "price" in c.lower() or "amount" in c.lower()), None)

    if qty_col and price_col:
        qty_numeric = pd.to_numeric(df[qty_col], errors="coerce")
        price_numeric = pd.to_numeric(df[price_col], errors="coerce")
        df["total_value"] = qty_numeric * price_numeric
        log.append({"feature": "total_value", "derived_from": [qty_col, price_col]})

    return df, log


def run_transformation_engine(df: pd.DataFrame, schema: dict = None) -> dict:
    if schema is None:
        schema = build_schema(df)

    transform_log = {}
    df, transform_log["scaling"] = scale_numeric_columns(df, schema)
    df, transform_log["encoding"] = encode_categorical_columns(df, schema)
    df, transform_log["date_features"] = extract_date_features(df, schema)
    df, transform_log["derived_features"] = extract_derived_numeric_features(df, schema)

    return {"transformed_df": df, "transform_log": transform_log}


if __name__ == "__main__":
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    df = pd.read_csv(path)
    result = run_transformation_engine(df)
    os.makedirs("data/processed", exist_ok=True)
    result["transformed_df"].to_csv("data/processed/transformed_data_partial.csv", index=False)
    with open("data/processed/transform_log_partial.json", "w") as f:
        json.dump(result["transform_log"], f, indent=2, default=str)
    print(f"Columns before: {len(df.columns)}, after: {len(result['transformed_df'].columns)}")
