import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules", "module1_profiling"))
from rule_engine import (
    rule_inconsistent_casing_or_whitespace, rule_unparseable_dates,
    rule_malformed_emails_or_phones, rule_suspicious_numeric_ranges,
    detect_pii_columns, run_rule_engine,
)
from api import profile_dataset


def test_inconsistent_casing_flagged():
    df = pd.DataFrame({"Country": ["UK", "uk", " UK", "France"]})
    findings = rule_inconsistent_casing_or_whitespace(df)
    assert any(f["column"] == "Country" for f in findings)


def test_unparseable_dates_flagged():
    df = pd.DataFrame({"InvoiceDate": ["2021-01-01", "not-a-date", "2021-01-03", "??", "bad"]})
    findings = rule_unparseable_dates(df)
    assert any(f["column"] == "InvoiceDate" for f in findings)


def test_malformed_email_flagged():
    df = pd.DataFrame({"ContactEmail": ["a@b.com", "not-an-email", "c@d.com", "bad", "x"]})
    findings = rule_malformed_emails_or_phones(df)
    assert any(f["column"] == "ContactEmail" for f in findings)


def test_negative_price_flagged():
    df = pd.DataFrame({"UnitPrice": [10.0, -5.0, 20.0]})
    findings = rule_suspicious_numeric_ranges(df)
    assert any(f["column"] == "UnitPrice" for f in findings)


def test_generic_id_not_flagged_as_pii():
    df = pd.DataFrame({"InvoiceNo": [1001, 1002, 1003]})
    findings = detect_pii_columns(df)
    assert not any(f["column"] == "InvoiceNo" for f in findings)


def test_customer_id_flagged_as_pii():
    df = pd.DataFrame({"CustomerID": [1001, 1002, 1003]})
    findings = detect_pii_columns(df)
    assert any(f["column"] == "CustomerID" for f in findings)


def test_email_column_flagged_as_pii():
    df = pd.DataFrame({"ContactEmail": ["a@b.com", "c@d.com", None]})
    findings = detect_pii_columns(df)
    assert any(f["column"] == "ContactEmail" for f in findings)


def test_run_rule_engine_has_both_sections():
    df = pd.DataFrame({"ContactEmail": ["a@b.com", "bad", None]})
    result = run_rule_engine(df)
    assert "suspicious_format_findings" in result
    assert "potential_pii_columns" in result


def test_profile_dataset_api_includes_rule_engine_output(tmp_path):
    df = pd.DataFrame({"a": [1, 2, None], "ContactEmail": ["a@b.com", "bad", None]})
    report = profile_dataset(df, viz_out_dir=str(tmp_path))
    assert "suspicious_format_findings" in report
    assert "potential_pii_columns" in report
    assert "metadata" in report
