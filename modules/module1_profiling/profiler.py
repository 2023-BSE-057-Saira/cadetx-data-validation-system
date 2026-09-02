"""
Profiling Engine
Module 1 — Week 2 deliverable
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from metadata_extractor import extract_metadata, _detect_mixed_type


def missing_value_matrix(df: pd.DataFrame) -> dict:
    total = len(df)
    return {
        col: {
            "missing_count": int(df[col].isna().sum()),
            "missing_percent": round(100 * df[col].isna().sum() / total, 2) if total else 0.0,
        }
        for col in df.columns
    }


def dtype_consistency_check(df: pd.DataFrame) -> dict:
    issues = {}
    for col in df.columns:
        if _detect_mixed_type(df[col]):
            issues[col] = "declared as text/numeric but contains inconsistent value types"
    return issues


def cardinality_analysis(df: pd.DataFrame) -> dict:
    total = len(df)
    result = {}
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        result[col] = {
            "unique_count": int(n_unique),
            "cardinality_ratio": round(n_unique / total, 4) if total else 0.0,
            "is_likely_constant": n_unique <= 1,
            "is_likely_id_like": total > 0 and n_unique / total > 0.95,
        }
    return result


def correlation_matrix(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {}
    return numeric_df.corr(numeric_only=True).round(3).to_dict()


def generate_visualisations(df: pd.DataFrame, out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    generated = []

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isna(), cbar=False, cmap="rocket_r", yticklabels=False, ax=ax)
    ax.set_title("Missing Value Heatmap")
    path = os.path.join(out_dir, "missing_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    generated.append(path)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 0:
        fig, axes = plt.subplots(1, numeric_df.shape[1], figsize=(4 * numeric_df.shape[1], 5))
        if numeric_df.shape[1] == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_df.columns):
            sns.boxplot(y=numeric_df[col].dropna(), ax=ax, color="skyblue")
            ax.set_title(col)
        fig.suptitle("Outlier Distribution by Column")
        fig.tight_layout()
        path = os.path.join(out_dir, "outlier_distribution.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        generated.append(path)

    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        path = os.path.join(out_dir, "correlation_heatmap.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        generated.append(path)

    return generated


def build_profiling_report(df: pd.DataFrame, viz_out_dir: str) -> dict:
    return {
        "metadata": extract_metadata(df),
        "missing_value_matrix": missing_value_matrix(df),
        "dtype_consistency_issues": dtype_consistency_check(df),
        "cardinality_analysis": cardinality_analysis(df),
        "correlation_matrix": correlation_matrix(df),
        "visualisations_generated": generate_visualisations(df, viz_out_dir),
    }


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_retail.csv"
    df = pd.read_csv(input_path)
    report = build_profiling_report(df, viz_out_dir="data/processed/visualisations")
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/profiling_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("Profiling report written to data/processed/profiling_report.json")
