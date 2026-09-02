"""
Profiling API
Module 1 — Week 3 deliverable

Public entry point for Module 1. Wraps metadata_extractor, profiler, and
rule_engine into a single function so downstream modules (and Module 4's
orchestrator) only need to call one thing.

    Input:  a pandas DataFrame (or a path to a CSV)
    Output: profiling_report.json — see docs/module1_api.md for the schema
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from profiler import build_profiling_report
from rule_engine import run_rule_engine


def profile_dataset(input_data, viz_out_dir: str = "data/processed/visualisations") -> dict:
    """
    Run the full Module 1 profiling pipeline on a dataset.

    Args:
        input_data: a pandas DataFrame, or a string/path to a CSV file.
        viz_out_dir: directory to write generated visualisation PNGs to.

    Returns:
        dict matching the profiling_report.json schema (see docs/module1_api.md).
    """
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    else:
        df = input_data

    report = build_profiling_report(df, viz_out_dir=viz_out_dir)
    rule_findings = run_rule_engine(df)
    report["suspicious_format_findings"] = rule_findings["suspicious_format_findings"]
    report["potential_pii_columns"] = rule_findings["potential_pii_columns"]
    return report


def profile_dataset_to_file(
    input_path: str,
    output_path: str = "data/processed/profiling_report.json",
    viz_out_dir: str = "data/processed/visualisations",
) -> str:
    """Convenience wrapper: reads a CSV, profiles it, writes profiling_report.json."""
    report = profile_dataset(input_path, viz_out_dir=viz_out_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return output_path


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    out = profile_dataset_to_file(input_path)
    print(f"Profiling report written to {out}")
