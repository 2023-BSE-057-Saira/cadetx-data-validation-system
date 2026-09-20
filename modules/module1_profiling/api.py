"""Profiling API — Module 1, Week 3."""
import json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from profiler import build_profiling_report
from rule_engine import run_rule_engine


def profile_dataset(input_data, viz_out_dir="data/processed/visualisations"):
    df = pd.read_csv(input_data) if isinstance(input_data, str) else input_data
    report = build_profiling_report(df, viz_out_dir=viz_out_dir)
    rule_findings = run_rule_engine(df)
    report["suspicious_format_findings"] = rule_findings["suspicious_format_findings"]
    report["potential_pii_columns"] = rule_findings["potential_pii_columns"]
    return report


def profile_dataset_to_file(input_path, output_path="data/processed/profiling_report.json",
                             viz_out_dir="data/processed/visualisations"):
    report = profile_dataset(input_path, viz_out_dir=viz_out_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return output_path


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    print(f"Profiling report written to {profile_dataset_to_file(input_path)}")
