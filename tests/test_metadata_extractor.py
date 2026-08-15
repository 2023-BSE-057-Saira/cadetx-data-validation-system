import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules", "module1_profiling"))
from metadata_extractor import extract_metadata, infer_semantic_type, _detect_mixed_type


def test_extract_metadata_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = extract_metadata(df)
    assert result["n_rows"] == 3
    assert result["n_columns"] == 2
    assert len(result["columns"]) == 2


def test_mixed_type_detected():
    s = pd.Series(["1", "2", "unknown", "4"])
    assert _detect_mixed_type(s) is True


def test_clean_numeric_not_mixed():
    s = pd.Series([1, 2, 3, 4])
    assert _detect_mixed_type(s) is False


def test_email_semantic_type():
    s = pd.Series(["a@b.com", "c@d.com", None])
    assert infer_semantic_type(s, "ContactEmail") == "email"


def test_id_semantic_type_from_name():
    s = pd.Series([1001, 1002, 1003])
    assert infer_semantic_type(s, "CustomerID") == "id"


def test_null_count_and_percent():
    df = pd.DataFrame({"a": [1, None, 3, None]})
    result = extract_metadata(df)
    col = result["columns"][0]
    assert col["null_count"] == 2
    assert col["null_percent"] == 50.0
