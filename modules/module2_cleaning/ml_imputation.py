"""
ML-Based Imputation
Module 2 — Week 5 deliverable

Week 4 gave us a statistical baseline (mean/mode). This week adds smarter,
model-based imputation for numeric columns, which tends to preserve
relationships between columns better than filling every missing value
with the same constant:

  - KNN imputer: fills a missing value using the average of the
    k most similar rows (by other numeric columns)
  - Regression imputer: trains a linear regression on rows that HAVE a
    value for the target column, using other numeric columns as
    predictors, then predicts the missing ones
  - Iterative imputer (MICE-style): models each column with missing
    values as a function of all other columns, round-robin, repeating
    until estimates stabilise — generally the strongest of the three,
    at the cost of speed

All three only operate on numeric columns (categorical/text columns stay
on the Week 4 mode-imputation strategy, which is standard practice —
imputing text meaningfully needs NLP techniques, out of scope here).
"""

import sys
import os

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 — required to unlock IterativeImputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from schema_inference import build_schema


def _numeric_columns(df: pd.DataFrame, schema: dict) -> list:
    # Only true numeric measures: identifiers (CustomerID etc.) are neither
    # meaningful predictors nor something that should ever be imputed.
    return [c for c, s in schema.items()
            if c in df.columns and s["expected_dtype"] in ("integer", "float")
            and s["semantic_type"] == "numeric"]


def _usable(numeric_block):
    """sklearn imputers silently drop all-NaN columns, which breaks shape
    alignment — exclude them up front."""
    return [c for c in numeric_block.columns if numeric_block[c].notna().any()]


def knn_impute(df: pd.DataFrame, schema: dict, n_neighbors: int = 5) -> tuple:
    """Fills missing numeric values using K-Nearest-Neighbours imputation."""
    df = df.copy()
    num_cols = _numeric_columns(df, schema)
    if not num_cols:
        return df, {"strategy": "knn", "columns_imputed": [], "note": "no numeric columns"}

    numeric_block = df[num_cols].apply(pd.to_numeric, errors="coerce")
    num_cols = _usable(numeric_block)
    numeric_block = numeric_block[num_cols]
    missing_before = numeric_block.isna().sum()

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_values = imputer.fit_transform(numeric_block)
    imputed_df = pd.DataFrame(imputed_values, columns=num_cols, index=df.index)

    for col in num_cols:
        df[col] = imputed_df[col]

    return df, {
        "strategy": "knn", "n_neighbors": n_neighbors,
        "columns_imputed": [c for c in num_cols if missing_before[c] > 0],
        "values_filled": {c: int(missing_before[c]) for c in num_cols if missing_before[c] > 0},
    }


def regression_impute(df: pd.DataFrame, schema: dict, target_col: str) -> tuple:
    """
    Fills missing values in a single numeric column using linear
    regression against the other numeric columns.
    """
    df = df.copy()
    num_cols = _numeric_columns(df, schema)
    predictor_cols = [c for c in num_cols if c != target_col]

    if target_col not in df.columns or not predictor_cols:
        return df, {"strategy": "regression", "column": target_col, "note": "insufficient predictors"}

    numeric_block = df[num_cols].apply(pd.to_numeric, errors="coerce")
    missing_mask = numeric_block[target_col].isna()
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        return df, {"strategy": "regression", "column": target_col, "rows_filled": 0}

    train = numeric_block.dropna(subset=[target_col] + predictor_cols)
    predict_rows = numeric_block[missing_mask].dropna(subset=predictor_cols)

    if len(train) < 5 or len(predict_rows) == 0:
        return df, {"strategy": "regression", "column": target_col,
                     "note": "not enough complete rows to train a reliable model", "rows_filled": 0}

    model = LinearRegression()
    model.fit(train[predictor_cols], train[target_col])
    predictions = model.predict(predict_rows[predictor_cols])

    df.loc[predict_rows.index, target_col] = predictions

    return df, {
        "strategy": "regression", "column": target_col, "predictors": predictor_cols,
        "rows_filled": len(predict_rows), "r2_on_training_data": round(float(model.score(
            train[predictor_cols], train[target_col])), 4),
    }


def iterative_impute(df: pd.DataFrame, schema: dict, max_iter: int = 10) -> tuple:
    """Fills missing numeric values using round-robin iterative imputation
    (each column with missing values modeled as a function of the others,
    repeated until estimates stabilise)."""
    df = df.copy()
    num_cols = _numeric_columns(df, schema)
    if not num_cols:
        return df, {"strategy": "iterative", "columns_imputed": [], "note": "no numeric columns"}

    numeric_block = df[num_cols].apply(pd.to_numeric, errors="coerce")
    num_cols = _usable(numeric_block)
    numeric_block = numeric_block[num_cols]
    missing_before = numeric_block.isna().sum()

    imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    imputed_values = imputer.fit_transform(numeric_block)
    imputed_df = pd.DataFrame(imputed_values, columns=num_cols, index=df.index)

    for col in num_cols:
        df[col] = imputed_df[col]

    return df, {
        "strategy": "iterative", "max_iter": max_iter,
        "columns_imputed": [c for c in num_cols if missing_before[c] > 0],
        "values_filled": {c: int(missing_before[c]) for c in num_cols if missing_before[c] > 0},
    }


def compare_imputation_strategies(df: pd.DataFrame, schema: dict = None) -> dict:
    """Runs all three ML imputation strategies and reports how many values
    each filled, so a strategy can be chosen deliberately rather than
    arbitrarily. Does not mutate the input; returns diagnostics only."""
    if schema is None:
        schema = build_schema(df)
    num_cols = _numeric_columns(df, schema)

    _, knn_log = knn_impute(df, schema)
    _, iter_log = iterative_impute(df, schema)

    regression_logs = []
    for col in num_cols:
        if df[col].isna().sum() > 0:
            _, reg_log = regression_impute(df, schema, col)
            regression_logs.append(reg_log)

    return {"knn": knn_log, "iterative": iter_log, "regression": regression_logs}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    df = pd.read_csv(path)
    schema = build_schema(df)
    comparison = compare_imputation_strategies(df, schema)
    import json
    print(json.dumps(comparison, indent=2, default=str))
