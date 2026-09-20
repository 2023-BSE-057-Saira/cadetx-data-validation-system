"""Profiling Rule Engine — Module 1, Week 3."""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from metadata_extractor import infer_semantic_type, _looks_like_email, _looks_like_phone

PII_SEMANTIC_TYPES = {"email", "phone", "name", "id"}
PII_NAME_HINTS = ("email", "phone", "name", "customer", "ssn", "passport", "address", "dob")


def rule_inconsistent_casing_or_whitespace(df):
    findings = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        has_whitespace_issue = (vals != vals.str.strip()).any()
        normalised = vals.str.strip().str.lower()
        collapses = normalised.nunique() < vals.nunique()
        if has_whitespace_issue or collapses:
            findings.append({"column": col, "issue": "inconsistent casing or stray whitespace",
                              "detail": "values that should likely be identical differ in case/whitespace"})
    return findings


def rule_unparseable_dates(df):
    findings = []
    for col in df.columns:
        if not any(h in col.lower() for h in ("date", "time", "dob")):
            continue
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        parsed = pd.to_datetime(vals, errors="coerce", format="mixed")
        fail_rate = 1 - parsed.notna().mean()
        if fail_rate > 0.02:
            findings.append({"column": col, "issue": "unparseable date values",
                              "detail": f"{fail_rate:.1%} of values could not be parsed"})
    return findings


def rule_malformed_emails_or_phones(df):
    findings = []
    for col in df.columns:
        lname = col.lower()
        if "email" in lname or "mail" in lname:
            rate = _looks_like_email(df[col])
            if rate < 0.9:
                findings.append({"column": col, "issue": "malformed email values",
                                  "detail": f"only {rate:.1%} match a valid email format"})
        if "phone" in lname or "tel" in lname or "mobile" in lname:
            rate = _looks_like_phone(df[col])
            if rate < 0.9:
                findings.append({"column": col, "issue": "malformed phone values",
                                  "detail": f"only {rate:.1%} match a valid phone format"})
    return findings


def rule_suspicious_numeric_ranges(df):
    findings = []
    for col in df.select_dtypes(include="number").columns:
        lname = col.lower()
        if "price" in lname or "amount" in lname or "cost" in lname:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                findings.append({"column": col, "issue": "negative values in a price/amount column",
                                  "detail": f"{int(n_negative)} row(s) negative — verify legitimacy"})
    return findings


def detect_pii_columns(df):
    findings = []
    for col in df.columns:
        semantic = infer_semantic_type(df[col], col)
        name_flag = any(h in col.lower() for h in PII_NAME_HINTS)
        if semantic == "id" and not name_flag:
            continue
        if semantic in PII_SEMANTIC_TYPES or name_flag:
            findings.append({"column": col,
                              "likely_pii_type": semantic if semantic in PII_SEMANTIC_TYPES else "unknown",
                              "reason": "semantic type matched a PII category" if semantic in PII_SEMANTIC_TYPES
                                        else "column name suggests personal data"})
    return findings


RULES = [rule_inconsistent_casing_or_whitespace, rule_unparseable_dates,
         rule_malformed_emails_or_phones, rule_suspicious_numeric_ranges]


def run_rule_engine(df):
    all_findings = []
    for rule_fn in RULES:
        all_findings.extend(rule_fn(df))
    return {"suspicious_format_findings": all_findings, "potential_pii_columns": detect_pii_columns(df)}
