import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "modules")
for m in ("module1_profiling", "module2_cleaning", "module3_validation"):
    sys.path.insert(0, os.path.join(ROOT, m))

from rule_validation import validate_rules, validate_rules_to_files, DEFAULT_CONFIG
from cleaning_api import clean_dataset


def flagged(result, rule, column=None):
    v = result["violations_df"]
    v = v[v["rule"] == rule]
    if column:
        v = v[v["column"] == column]
    return sorted(v["row_index"].tolist())


def rule_entry(result, rule):
    return next(r for r in result["report"]["rules"] if r["rule"] == rule)


# ---------------- email / phone / postcode ----------------

def test_email_format_valid_and_invalid():
    emails = ["a@b.com", "first.last+tag@sub.example.co.uk",          # valid  (rows 0-1)
              "not-an-email", "user@@bad.com", "a@b", "@x.com", ".a@b.com",
              "a..b@c.com", "a@b..com", "a b@c.com", "a@b.c"]          # invalid (rows 2-10)
    res = validate_rules(pd.DataFrame({"ContactEmail": emails}))
    assert flagged(res, "email_format") == list(range(2, 11))


def test_phone_format_valid_and_invalid():
    phones = ["+44 20 7946 0958", "07123456789", "(555) 123-4567", "555.123.4567",   # valid (0-3)
              "12345", "not-a-phone", "+1234567890123456", "12-34", "++4412345678"]  # invalid (4-8)
    res = validate_rules(pd.DataFrame({"ContactPhone": phones}))
    assert flagged(res, "phone_format") == [4, 5, 6, 7, 8]


def test_postcode_default_accepts_uk_and_us():
    codes = ["SW1A 1AA", "M1 1AE", "EC1A1BB", "90210", "12345-6789",      # valid (0-4)
             "ABC", "1234", "123456", "SW1A 1A"]                          # invalid (5-8)
    res = validate_rules(pd.DataFrame({"Postcode": codes}))
    assert flagged(res, "postcode_format") == [5, 6, 7, 8]


def test_postcode_country_is_configurable():
    df = pd.DataFrame({"PostalCode": ["44000", "SW1A 1AA"]})
    res = validate_rules(df, config={"postcode_countries": ["PK"]})
    assert flagged(res, "postcode_format") == [1]


def test_postcode_numeric_dtype_column():
    res = validate_rules(pd.DataFrame({"zip": [90210, 1234, 10001]}))
    assert flagged(res, "postcode_format") == [1]


def test_missing_values_are_not_format_violations():
    df = pd.DataFrame({"ContactEmail": ["a@b.com", None, np.nan], "ContactPhone": [None, "07123456789", None]})
    res = validate_rules(df)
    assert flagged(res, "email_format") == [] and flagged(res, "phone_format") == []


# ---------------- numeric ----------------

def test_numeric_validity_flags_text_and_infinity():
    df = pd.DataFrame({"Quantity": ["5", "x", "7"], "Weight": [1.0, np.inf, 2.0]})
    schema = {"Quantity": {"semantic_type": "numeric"}, "Weight": {"semantic_type": "numeric"}}
    res = validate_rules(df, schema=schema)
    assert flagged(res, "numeric_validity", "Quantity") == [1]
    assert flagged(res, "numeric_validity", "Weight") == [1]


def test_numeric_range_configured_bounds_and_severity():
    df = pd.DataFrame({"Age": [5, 30, 150, -2, None]})
    res = validate_rules(df, config={"ranges": {"Age": {"min": 0, "max": 120, "severity": "warning"}}})
    v = res["violations_df"]
    assert sorted(v["row_index"]) == [2, 3]
    assert set(v["severity"]) == {"warning"}


def test_negative_price_flagged_by_default_but_negative_quantity_is_not():
    df = pd.DataFrame({"UnitPrice": [2.5, -1.0, 3.0], "Quantity": [-5, 2, 3]})   # negative qty = a return
    res = validate_rules(df)
    assert flagged(res, "numeric_range", "UnitPrice") == [1]
    assert flagged(res, "numeric_range", "Quantity") == []


def test_explicit_range_overrides_default_price_rule():
    df = pd.DataFrame({"UnitPrice": [-5.0, 2.0]})
    res = validate_rules(df, config={"ranges": {"UnitPrice": {"min": -10}}})
    assert flagged(res, "numeric_range") == []


# ---------------- dates ----------------

def test_date_plausibility_future_past_and_unparseable():
    df = pd.DataFrame({"InvoiceDate": ["2020-05-01", "2030-01-01", "1850-01-01", "not a date", "2020-12-31"]})
    res = validate_rules(df, config={"as_of": "2021-01-01"})
    assert flagged(res, "date_plausibility") == [1, 2, 3]


def test_date_plausibility_datetime_dtype():
    df = pd.DataFrame({"created_date": pd.to_datetime(["2020-01-01", "2099-01-01"])})
    res = validate_rules(df, config={"as_of": "2021-01-01"})
    assert flagged(res, "date_plausibility") == [1]


# ---------------- categorical ----------------

def test_case_whitespace_variants_flagged_as_warnings():
    df = pd.DataFrame({"Country": ["UK"] * 8 + ["  UK", "uk"]})
    schema = {"Country": {"semantic_type": "categorical"}}
    res = validate_rules(df, schema=schema)
    assert flagged(res, "categorical_case_whitespace") == [8, 9]
    assert set(res["violations_df"]["severity"]) == {"warning"}


def test_near_duplicate_flags_rare_typo_only():
    df = pd.DataFrame({"Country": ["United Kingdom"] * 20 + ["United Kingdon"] + ["France"] * 15})
    schema = {"Country": {"semantic_type": "categorical"}}
    res = validate_rules(df, schema=schema)
    assert flagged(res, "categorical_near_duplicate") == [20]
    assert "United Kingdom" in res["violations_df"].iloc[0]["message"]


def test_near_duplicate_ignores_two_equally_common_values():
    df = pd.DataFrame({"Grade": ["Grade A"] * 10 + ["Grade B"] * 10})
    res = validate_rules(df, schema={"Grade": {"semantic_type": "categorical"}})
    assert flagged(res, "categorical_near_duplicate") == []


def test_placeholder_values_flagged():
    df = pd.DataFrame({"Notes": ["fine", "unknown", "?", "  -  ", "TBD", "ok"]})
    res = validate_rules(df)
    assert flagged(res, "placeholder_value") == [1, 2, 3, 4]


def test_allowed_values_and_patterns_and_required():
    df = pd.DataFrame({"Country": ["UK", "FR", "XX", None],
                       "InvoiceNo": ["536365", "C536379", "ABC", "12"],
                       "CustomerID": [1.0, None, 3.0, 4.0]})
    cfg = {"allowed_values": {"Country": ["UK", "FR"]},
           "patterns": {"InvoiceNo": r"^C?\d{6}$"},
           "required": ["CustomerID"]}
    res = validate_rules(df, config=cfg)
    assert flagged(res, "allowed_values") == [2]
    assert flagged(res, "pattern_match") == [2, 3]
    assert flagged(res, "required_fields") == [1]


def test_whole_number_floats_match_patterns_without_dot_zero():
    df = pd.DataFrame({"CustomerID": [12346.0, 17850.0, 1.5]})
    res = validate_rules(df, config={"patterns": {"CustomerID": r"^\d{5}$"}})
    assert flagged(res, "pattern_match") == [2]


# ---------------- configuration ----------------

def test_unknown_config_key_raises():
    with pytest.raises(ValueError, match="Unknown config keys"):
        validate_rules(pd.DataFrame({"a": [1]}), config={"range": {}})        # typo of 'ranges'


def test_invalid_regex_raises():
    with pytest.raises(ValueError, match="valid regex"):
        validate_rules(pd.DataFrame({"a": ["x"]}), config={"patterns": {"a": "([unclosed"}})


def test_unsupported_postcode_country_raises():
    with pytest.raises(ValueError, match="postcode_countries"):
        validate_rules(pd.DataFrame({"a": [1]}), config={"postcode_countries": ["ZZ"]})


def test_bad_range_keys_and_severity_raise():
    with pytest.raises(ValueError):
        validate_rules(pd.DataFrame({"a": [1]}), config={"ranges": {"a": {"minimum": 0}}})
    with pytest.raises(ValueError):
        validate_rules(pd.DataFrame({"a": [1]}), config={"ranges": {"a": {"min": 0, "severity": "fatal"}}})


def test_config_mentioning_missing_column_is_a_warning_not_a_crash():
    res = validate_rules(pd.DataFrame({"a": [1]}), config={"required": ["Nope"], "patterns": {"Ghost": "x"}})
    assert len(res["report"]["config_warnings"]) == 2


def test_default_config_is_not_mutated_between_calls():
    validate_rules(pd.DataFrame({"a": [1]}), config={"required": ["a"]})
    assert DEFAULT_CONFIG["required"] == []


# ---------------- report ----------------

def test_rules_without_applicable_columns_report_not_applicable():
    res = validate_rules(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    e = rule_entry(res, "email_format")
    assert e["status"] == "not_applicable" and "no email column" in e["reason"]


def test_report_structure_totals_and_strict_json():
    df = pd.DataFrame({"ContactEmail": ["a@b.com", "bad", "worse"], "UnitPrice": [1.0, -2.0, 3.0]})
    res = validate_rules(df)
    r = res["report"]
    assert {"summary", "rules", "violations_by_column", "config_warnings", "config_used"} <= set(r)
    s = r["summary"]
    assert s["total_violations"] == s["errors"] + s["warnings"] == len(res["violations_df"])
    assert s["total_violations"] == 3                   # 2 bad emails + 1 negative price
    assert s["rows_with_violations"] == 2               # ...but only rows 1 and 2 are affected
    assert s["rows_checked"] == 3
    assert r["violations_by_column"]["ContactEmail"] == {"email_format": 2}
    json.dumps(r)                                       # strict: raises on numpy types


def test_examples_are_capped():
    df = pd.DataFrame({"ContactEmail": ["bad"] * 50})
    res = validate_rules(df, config={"max_examples_per_rule": 5})
    e = rule_entry(res, "email_format")
    assert e["violations"] == 50 and len(e["examples"]) == 5


def test_clean_dataset_has_zero_violations():
    df = pd.DataFrame({"ContactEmail": ["a@b.com", "c@d.org"], "UnitPrice": [1.0, 2.0], "Quantity": [3, 4]})
    assert validate_rules(df)["report"]["summary"]["total_violations"] == 0


def test_input_not_mutated_and_row_index_is_positional():
    df = pd.DataFrame({"ContactEmail": ["a@b.com", "bad"]}, index=[100, 200])
    snapshot = df.copy(deep=True)
    res = validate_rules(df)
    pd.testing.assert_frame_equal(df, snapshot)
    assert flagged(res, "email_format") == [1]


def test_validate_rules_to_files(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"ContactEmail": ["a@b.com", "bad"]}).to_csv(src, index=False)
    report_path, violations_path = validate_rules_to_files(str(src), output_dir=str(tmp_path / "out"))
    assert json.load(open(report_path))["summary"]["total_violations"] == 1
    assert len(pd.read_csv(violations_path)) == 1


# ---------------- integration with Module 2 ----------------

def _raw():
    return pd.DataFrame({
        "InvoiceNo": ["536365", "C536379", "536367", "536368", "536369", "536370", "536371", "536372"],
        "Quantity": ["5", "-1", "unknown", "8", "2", "7", "4", "9"],
        "UnitPrice": [2.5, 3.0, 4.0, -1.0, 1.5, 2.0, 3.5, 2.2],
        "Country": ["UK", "UK", "UK", "  UK", "UK", "UK", "UK", "UK"],
        "ContactEmail": ["a@b.com", "bad", "c@d.com", None, "e@f.com", "g@h.com", "i@j.com", "k@l.com"],
    })


def test_integration_cleaning_fixes_what_it_can_and_validation_reports_the_rest():
    raw = _raw()
    out = clean_dataset(raw)
    schema = out["cleaning_log"]["schema"]

    before = validate_rules(raw, schema=schema)
    after = validate_rules(out["cleaned_df"], schema=schema)

    # fixed by cleaning
    assert flagged(before, "numeric_validity", "Quantity") != []
    assert flagged(after, "numeric_validity", "Quantity") == []
    assert flagged(before, "categorical_case_whitespace") != []
    assert flagged(after, "categorical_case_whitespace") == []
    # NOT fixable by cleaning -> still reported by validation
    assert flagged(after, "email_format") != []
    assert flagged(after, "numeric_range", "UnitPrice") != []
    assert after["report"]["summary"]["total_violations"] < before["report"]["summary"]["total_violations"]


def test_text_identifiers_with_letters_are_not_numeric_violations():
    """Regression: 'C536379' / '85123A' must never be treated as bad numbers."""
    raw = _raw()
    out = clean_dataset(raw)
    res = validate_rules(out["cleaned_df"], schema=out["cleaning_log"]["schema"])
    assert flagged(res, "numeric_validity", "InvoiceNo") == []
    assert "C536379" in out["cleaned_df"]["InvoiceNo"].tolist()
