import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules", "module2_cleaning"))
from schema_inference import build_schema
from ml_imputation import knn_impute, regression_impute, iterative_impute, compare_imputation_strategies
from transformation_engine import (
    scale_numeric_columns, encode_categorical_columns, extract_date_features,
    extract_derived_numeric_features, run_transformation_engine,
)


def _correlated_df():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, 50)
    b = a * 2 + rng.normal(0, 1, 50)  # strongly correlated with a
    df = pd.DataFrame({"a": a, "b": b})
    df.loc[5:9, "b"] = np.nan  # inject missing values in b
    return df


def test_knn_impute_fills_all_missing():
    df = _correlated_df()
    schema = build_schema(df)
    filled, log = knn_impute(df, schema)
    assert filled["b"].isna().sum() == 0
    assert log["values_filled"]["b"] == 5


def test_iterative_impute_fills_all_missing():
    df = _correlated_df()
    schema = build_schema(df)
    filled, log = iterative_impute(df, schema)
    assert filled["b"].isna().sum() == 0


def test_regression_impute_uses_correlation():
    df = _correlated_df()
    schema = build_schema(df)
    filled, log = regression_impute(df, schema, "b")
    assert filled["b"].isna().sum() == 0
    # b = 2a + noise, so R^2 should be high given the strong correlation
    assert log["r2_on_training_data"] > 0.9


def test_regression_impute_handles_no_predictors_gracefully():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    schema = build_schema(df)
    filled, log = regression_impute(df, schema, "a")
    assert "note" in log  # no predictor columns available


def test_compare_imputation_strategies_returns_all_three():
    df = _correlated_df()
    result = compare_imputation_strategies(df)
    assert "knn" in result and "iterative" in result and "regression" in result


def test_scale_numeric_columns_produces_mean_zero():
    df = pd.DataFrame({"UnitPrice": [10.0, 20.0, 30.0, 40.0, 50.0]})
    schema = build_schema(df)
    scaled, log = scale_numeric_columns(df, schema)
    assert abs(scaled["UnitPrice_scaled"].mean()) < 1e-6


def test_encode_categorical_onehot_for_low_cardinality():
    df = pd.DataFrame({"Country": ["UK", "France", "UK", "Germany"]})
    schema = build_schema(df)
    encoded, log = encode_categorical_columns(df, schema, onehot_max_categories=10)
    assert len(log["onehot_encoded"]) == 1
    assert "Country_UK" in encoded.columns


def test_extract_date_features_adds_columns():
    df = pd.DataFrame({"InvoiceDate": ["2021-01-04", "2021-06-15"]})  # Mon, Tue
    schema = build_schema(df)
    result, log = extract_date_features(df, schema)
    assert "InvoiceDate_year" in result.columns
    assert result["InvoiceDate_year"].iloc[0] == 2021


def test_derived_total_value_feature():
    df = pd.DataFrame({"Quantity": [2, 3], "UnitPrice": [10.0, 5.0]})
    schema = build_schema(df)
    result, log = extract_derived_numeric_features(df, schema)
    assert result["total_value"].tolist() == [20.0, 15.0]


def test_run_transformation_engine_end_to_end():
    df = pd.DataFrame({
        "Quantity": [1, 2, 3], "UnitPrice": [10.0, 20.0, 30.0],
        "Country": ["UK", "France", "UK"], "InvoiceDate": ["2021-01-01", "2021-01-02", "2021-01-03"],
    })
    result = run_transformation_engine(df)
    out_cols = result["transformed_df"].columns
    assert "total_value" in out_cols
    assert "InvoiceDate_year" in out_cols
    assert any(c.startswith("Country_") for c in out_cols)
