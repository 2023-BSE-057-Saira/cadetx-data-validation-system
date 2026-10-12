import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "modules")
sys.path.insert(0, os.path.join(ROOT, "module1_profiling"))
sys.path.insert(0, os.path.join(ROOT, "module2_cleaning"))

from schema_inference import build_schema
from cleaning_pipeline import impute_missing_statistical, run_cleaning_pipeline
from quality_scoring import (
    completeness_score, uniqueness_score, validity_score, consistency_score,
    score_dataset, compare_quality,
)
from cleaning_api import clean_dataset, clean_dataset_to_files
from api import profile_dataset


def messy_df():
    return pd.DataFrame({
        "InvoiceNo": [1001, 1002, 1003, 1004, 1005, 1005, 1006, 1007],
        "Quantity": ["5", "3", "unknown", "8", "2", "2", "7", "4"],
        "UnitPrice": [2.5, 3.0, 4.0, None, 1.5, 1.5, 2.0, 3.5],
        "CustomerID": [1.0, None, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
        "Country": ["UK", "UK", "  France", "France", "UK", "UK", "UK", "France"],
        "ContactEmail": ["a@b.com", None, "c@d.com", "bad", "e@f.com", "e@f.com", "g@h.com", "i@j.com"],
    })


# ---------- quality scoring: unit tests ----------

def test_completeness_score_counts_missing_cells():
    df = pd.DataFrame({"a": [1, None], "b": [None, None]})
    score, details = completeness_score(df)
    assert score == 25.0
    assert details["missing_cells_by_column"] == {"a": 1, "b": 2}


def test_uniqueness_score_penalises_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 3]})
    assert uniqueness_score(df)[0] == 75.0


def test_validity_score_flags_unparseable_numeric():
    df = pd.DataFrame({"Quantity": ["1", "2", "3", "oops"]})
    schema = {"Quantity": {"semantic_type": "numeric"}}
    assert validity_score(df, schema)[0] == 75.0


def test_validity_score_flags_bad_emails():
    df = pd.DataFrame({"ContactEmail": ["a@b.com", "nope", "c@d.com", "x"]})
    schema = {"ContactEmail": {"semantic_type": "email"}}
    assert validity_score(df, schema)[0] == 50.0


def test_consistency_score_penalises_whitespace_variants():
    df = pd.DataFrame({"Country": ["UK"] * 8 + ["  UK", "uk"]})
    schema = {"Country": {"semantic_type": "categorical"}}
    assert consistency_score(df, schema)[0] == 80.0


def test_perfect_dataset_scores_100():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 45.0]})
    assert score_dataset(df)["overall"] == 100.0


def test_overall_score_is_bounded_and_weighted():
    result = score_dataset(messy_df())
    assert 0 <= result["overall"] <= 100
    assert set(result["dimensions"]) == {"completeness", "uniqueness", "validity", "consistency"}


def test_compare_quality_uses_shared_schema_and_reports_delta():
    df = messy_df()
    cleaned = run_cleaning_pipeline(df)["cleaned_df"]
    cmp = compare_quality(df, cleaned, build_schema(df))
    assert cmp["after"]["overall"] > cmp["before"]["overall"]
    assert cmp["delta"]["overall"] == pytest.approx(cmp["after"]["overall"] - cmp["before"]["overall"], abs=0.01)


# ---------- regression tests for fixes made this week ----------

def test_identifiers_and_contacts_are_never_imputed():
    df = pd.DataFrame({"CustomerID": [1.0, None, 3.0, 4.0], "ContactEmail": ["a@b.com", None, "a@b.com", "c@d.com"]})
    filled, log = impute_missing_statistical(df, build_schema(df))
    assert filled["CustomerID"].isna().sum() == 1
    assert filled["ContactEmail"].isna().sum() == 1
    assert all(entry["strategy"] == "skipped_not_imputable" for entry in log)


def test_mixed_type_numeric_column_is_recovered_as_numeric():
    df = pd.DataFrame({"Quantity": ["5", "3", "unknown", "8", "2", "7", "4", "9", "1", "6"]})
    spec = build_schema(df)["Quantity"]
    assert spec["semantic_type"] == "numeric"
    assert spec["expected_dtype"] == "integer"


def test_mixed_type_column_is_coerced_and_imputed_by_pipeline():
    cleaned = run_cleaning_pipeline(messy_df())["cleaned_df"]
    assert pd.api.types.is_numeric_dtype(cleaned["Quantity"])
    assert cleaned["Quantity"].isna().sum() == 0


def test_duplicates_that_only_match_after_normalisation_are_removed():
    df = pd.DataFrame({"Country": ["France", "  France", "UK", "UK", "UK"], "n": [1, 1, 2, 3, 4]})
    result = run_cleaning_pipeline(df)
    steps = {s["step"]: s for s in result["cleaning_log"]["steps"]}
    assert steps["post_normalisation_duplicate_removal"]["rows_removed"] == 1
    assert result["cleaning_log"]["rows_after"] == 4


# ---------- cleaning API ----------

def test_clean_dataset_returns_df_and_log_with_quality():
    out = clean_dataset(messy_df())
    log = out["cleaning_log"]
    assert {"steps", "schema", "quality", "rows_before", "rows_after"} <= set(log)
    assert log["rows_before"] == 8 and log["rows_after"] == 7  # one exact duplicate
    assert log["quality"]["delta"]["overall"] > 0


def test_clean_dataset_does_not_mutate_input():
    df = messy_df()
    snapshot = df.copy(deep=True)
    clean_dataset(df)
    pd.testing.assert_frame_equal(df, snapshot)


@pytest.mark.parametrize("method", ["statistical", "knn", "iterative"])
def test_clean_dataset_all_numeric_imputation_methods(method):
    out = clean_dataset(messy_df(), numeric_imputation=method)
    assert out["cleaned_df"]["Quantity"].isna().sum() == 0
    assert out["cleaning_log"]["numeric_imputation"] == method


def test_clean_dataset_rejects_unknown_imputation_method():
    with pytest.raises(ValueError):
        clean_dataset(messy_df(), numeric_imputation="magic")


def test_clean_dataset_to_files_writes_both_outputs(tmp_path):
    src = tmp_path / "raw.csv"
    messy_df().to_csv(src, index=False)
    cleaned_path, log_path = clean_dataset_to_files(str(src), output_dir=str(tmp_path / "out"))
    assert os.path.basename(cleaned_path) == "cleaned_data.csv"
    assert os.path.basename(log_path) == "cleaning_log.json"
    assert len(pd.read_csv(cleaned_path)) == 7
    with open(log_path) as f:
        assert "quality" in json.load(f)


# ---------- integration ----------

def test_integration_profile_then_clean_roundtrip(tmp_path):
    """Module 1 output and Module 2 output describe the same dataset consistently."""
    src = tmp_path / "raw.csv"
    messy_df().to_csv(src, index=False)

    report = profile_dataset(str(src), viz_out_dir=str(tmp_path / "viz"))
    assert report["metadata"]["n_rows"] == 8

    cleaned_path, _ = clean_dataset_to_files(str(src), output_dir=str(tmp_path / "out"))
    cleaned = pd.read_csv(cleaned_path)
    assert len(cleaned) < report["metadata"]["n_rows"]          # duplicate removed
    assert cleaned["Country"].str.strip().eq(cleaned["Country"]).all()  # whitespace gone
    assert cleaned["CustomerID"].isna().sum() == 1              # id left missing, not invented


def test_integration_cleaning_is_idempotent():
    """Cleaning already-clean data must change nothing and never lower quality."""
    once = clean_dataset(messy_df())["cleaned_df"]
    twice = clean_dataset(once)
    assert len(twice["cleaned_df"]) == len(once)
    assert twice["cleaning_log"]["quality"]["delta"]["overall"] >= 0


def test_integration_pipeline_never_introduces_nulls():
    df = messy_df()
    cleaned = run_cleaning_pipeline(df)["cleaned_df"]
    for col in ["InvoiceNo", "Country", "Quantity", "UnitPrice"]:
        assert cleaned[col].isna().sum() <= df[col].isna().sum()


def test_code_like_text_columns_are_not_coerced_to_numeric():
    """Leading-zero codes (postcodes etc.) must survive; coercing would strip the zeros."""
    df = pd.DataFrame({"Region": ["07123", "07124", "07125", "07126", "N/A", "07128", "07129", "07130", "07131", "07132"]})
    assert build_schema(df)["Region"]["semantic_type"] != "numeric"
    cleaned = run_cleaning_pipeline(df)["cleaned_df"]
    assert cleaned["Region"].dropna().astype(str).str.startswith("0").any()


def test_fully_numeric_text_column_is_left_alone():
    df = pd.DataFrame({"Level": ["1", "2", "3", "1", "2", "3", "1", "2"]})
    assert build_schema(df)["Level"]["semantic_type"] != "numeric"


def test_text_identifiers_with_letters_are_never_blanked():
    """Regression (found with UCI-shaped data): 'C536379' cancellation invoices and
    '85123A' stock codes were coerced to numbers and silently turned into NaN."""
    df = pd.DataFrame({
        "InvoiceNo": ["536365", "536366", "C536379", "536367", "C536380", "536368"],
        "StockCode": ["85123A", "71053", "84406B", "22752", "85123A", "22633"],
        "Quantity": [6, 8, -1, 2, -3, 4],
    })
    spec = build_schema(df)
    assert spec["InvoiceNo"]["expected_dtype"] == "string"
    assert spec["StockCode"]["expected_dtype"] == "string"
    cleaned = run_cleaning_pipeline(df)["cleaned_df"]
    assert cleaned["InvoiceNo"].tolist() == df["InvoiceNo"].tolist()
    assert cleaned["StockCode"].tolist() == df["StockCode"].tolist()
