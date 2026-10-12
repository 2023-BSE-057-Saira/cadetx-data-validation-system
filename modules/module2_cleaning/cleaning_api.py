"""
Cleaning API
Module 2 — Week 6 deliverable

Public entry point for Module 2, mirroring Module 1's api.py so Module 4's
orchestrator can chain them the same way:

    Input : a pandas DataFrame, or a path to a CSV
    Output: cleaned_data.csv + cleaning_log.json
            (the log embeds quality scores before/after cleaning)

See docs/module2_api.md for the cleaning_log.json schema.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from schema_inference import build_schema
from cleaning_pipeline import run_cleaning_pipeline, VALID_NUMERIC_IMPUTATION
from quality_scoring import compare_quality


def clean_dataset(input_data, numeric_imputation="statistical"):
    """
    Clean a dataset in memory.

    Args:
        input_data: DataFrame, or str path to a CSV.
        numeric_imputation: 'statistical' (mean), 'knn', or 'iterative'.
            Applies to numeric measures only; identifiers and contact
            details are never imputed.

    Returns:
        {'cleaned_df': DataFrame, 'cleaning_log': dict}
    """
    if numeric_imputation not in VALID_NUMERIC_IMPUTATION:
        raise ValueError(f"numeric_imputation must be one of {VALID_NUMERIC_IMPUTATION}")

    df = pd.read_csv(input_data) if isinstance(input_data, str) else input_data

    # One schema, inferred from the RAW data, drives both cleaning and scoring
    schema = build_schema(df)
    result = run_cleaning_pipeline(df, schema=schema, numeric_imputation=numeric_imputation)

    log = result["cleaning_log"]
    log["schema"] = schema
    log["quality"] = compare_quality(df, result["cleaned_df"], schema)
    return {"cleaned_df": result["cleaned_df"], "cleaning_log": log}


def clean_dataset_to_files(input_path, output_dir="data/processed", numeric_imputation="statistical"):
    """Reads a CSV, cleans it, writes cleaned_data.csv and cleaning_log.json.
    Returns (cleaned_csv_path, log_path) for chaining into Module 3."""
    result = clean_dataset(input_path, numeric_imputation=numeric_imputation)
    os.makedirs(output_dir, exist_ok=True)
    cleaned_path = os.path.join(output_dir, "cleaned_data.csv")
    log_path = os.path.join(output_dir, "cleaning_log.json")
    result["cleaned_df"].to_csv(cleaned_path, index=False)
    with open(log_path, "w") as f:
        json.dump(result["cleaning_log"], f, indent=2, default=str)
    return cleaned_path, log_path


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    method = sys.argv[2] if len(sys.argv) > 2 else "statistical"
    cleaned, log = clean_dataset_to_files(input_path, numeric_imputation=method)
    print(f"Wrote {cleaned} and {log}")
