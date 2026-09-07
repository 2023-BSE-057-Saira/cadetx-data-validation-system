import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules", "module2_cleaning"))
from schema_inference import build_schema, infer_expected_range
from cleaning_pipeline import (
    impute_missing_statistical, detect_exact_duplicates, detect_fuzzy_duplicates,
    normalise_types_and_formats, run_cleaning_pipeline,
)


def test_build_schema_has_entry_per_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    schema = build_schema(df)
    assert set(schema.keys()) == {"a", "b"}


def test_expected_range_returns_none_for_short_series():
    assert infer_expected_range(pd.Series([1, 2])) is None


def test_expected_range_computed_for_numeric_series():
    r = infer_expected_range(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100]))
    assert r["observed_max"] == 100
    assert r["plausible_upper_bound"] < 100


def test_impute_missing_statistical_fills_numeric_with_mean():
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
    schema = {"a": {"expected_dtype": "float"}}
    filled, log = impute_missing_statistical(df, schema)
    assert filled["a"].isna().sum() == 0
    assert log[0]["strategy"] == "mean_imputation"


def test_impute_missing_statistical_fills_categorical_with_mode():
    df = pd.DataFrame({"a": ["x", "x", None, "y"]})
    schema = {"a": {"expected_dtype": "string"}}
    filled, log = impute_missing_statistical(df, schema)
    assert filled["a"].isna().sum() == 0
    assert log[0]["strategy"] == "mode_imputation"
    assert log[0]["fill_value"] == "x"


def test_detect_exact_duplicates():
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    mask = detect_exact_duplicates(df)
    assert mask.tolist() == [False, False, True]


def test_detect_fuzzy_duplicates_finds_near_matches():
    df = pd.DataFrame({"desc": ["United Kingdom", "United Kingdon", "France"]})
    matches = detect_fuzzy_duplicates(df, "desc", threshold=85)
    assert len(matches) >= 1


def test_normalise_casing_and_whitespace():
    df = pd.DataFrame({"Country": ["  Spain", "Spain", "France"]})
    schema = build_schema(df)
    normalised, log = normalise_types_and_formats(df, schema)
    assert normalised["Country"].nunique() == 2  # Spain + France
    assert any(l["action"] == "normalised_casing_whitespace" for l in log)


def test_run_cleaning_pipeline_removes_exact_duplicates():
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    result = run_cleaning_pipeline(df)
    assert result["cleaning_log"]["rows_after"] == 2
    assert result["cleaning_log"]["steps"][0]["rows_removed"] == 1


def test_run_cleaning_pipeline_no_nulls_remain():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
    result = run_cleaning_pipeline(df)
    assert result["cleaned_df"].isna().sum().sum() == 0
