import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules", "module1_profiling"))
from metadata_extractor import extract_metadata, infer_semantic_type, _detect_mixed_type


def test_extract_metadata_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = extract_metadata(df)
    assert result["n_rows"] == 3
    assert result["n_columns"] == 2


def test_mixed_type_detected():
    assert _detect_mixed_type(pd.Series(["1", "2", "unknown", "4"])) is True


def test_clean_numeric_not_mixed():
    assert _detect_mixed_type(pd.Series([1, 2, 3, 4])) is False


def test_email_semantic_type():
    s = pd.Series(["a@b.com", "c@d.com", None])
    assert infer_semantic_type(s, "ContactEmail") == "email"


def test_id_semantic_type_from_name():
    s = pd.Series([1001, 1002, 1003])
    assert infer_semantic_type(s, "CustomerID") == "id"


def test_null_count_and_percent():
    df = pd.DataFrame({"a": [1, None, 3, None]})
    col = extract_metadata(df)["columns"][0]
    assert col["null_count"] == 2
    assert col["null_percent"] == 50.0


def test_digit_codes_are_not_mistaken_for_dates():
    """Regression: '07123' parses as year 7123, which made postcode columns look like dates."""
    from metadata_extractor import _looks_like_date
    assert _looks_like_date(pd.Series(["07123", "07124", "07125"])) == 0.0
    assert infer_semantic_type(pd.Series(["07123", "07124", "07125", "N/A"]), "Region") != "date"


def test_real_date_strings_are_still_detected():
    from metadata_extractor import _looks_like_date
    assert _looks_like_date(pd.Series(["2021-01-04", "2021-02-10", "2021-03-15"])) == 1.0
