import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules", "module1_profiling"))
from profiler import (
    missing_value_matrix,
    dtype_consistency_check,
    cardinality_analysis,
    correlation_matrix,
    build_profiling_report,
)


def test_missing_value_matrix():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    result = missing_value_matrix(df)
    assert result["a"]["missing_count"] == 1
    assert result["b"]["missing_count"] == 0


def test_dtype_consistency_flags_mixed_column():
    df = pd.DataFrame({"a": ["1", "2", "unknown", "4"]})
    issues = dtype_consistency_check(df)
    assert "a" in issues


def test_cardinality_analysis_flags_constant_column():
    df = pd.DataFrame({"a": [1, 1, 1, 1]})
    result = cardinality_analysis(df)
    assert result["a"]["is_likely_constant"] is True


def test_cardinality_analysis_flags_id_like_column():
    df = pd.DataFrame({"id": [1, 2, 3, 4]})
    result = cardinality_analysis(df)
    assert result["id"]["is_likely_id_like"] is True


def test_correlation_matrix_returns_empty_for_single_numeric_col():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert correlation_matrix(df) == {}


def test_correlation_matrix_computed_for_two_numeric_cols():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    corr = correlation_matrix(df)
    assert "a" in corr and "b" in corr


def test_build_profiling_report_has_all_sections(tmp_path):
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    report = build_profiling_report(df, viz_out_dir=str(tmp_path))
    for key in [
        "metadata", "missing_value_matrix", "dtype_consistency_issues",
        "cardinality_analysis", "correlation_matrix", "visualisations_generated",
    ]:
        assert key in report
