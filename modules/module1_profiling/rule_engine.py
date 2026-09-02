"""
Profiling Rule Engine
Module 1 — Week 3 deliverable

Applies a set of standalone rules over a dataset (independent of the
statistical profiling in profiler.py) to flag:
  - suspicious formats: columns whose values don't consistently match
    the format their name/content implies (e.g. an "InvoiceDate" column
    with unparseable dates, a "Country" column with inconsistent casing
    or leading/trailing whitespace)
  - potential PII columns: columns likely to contain personally
    identifiable information (emails, phone numbers, names, ID numbers)
    so they can be flagged for special handling downstream

Rules are kept simple and explainable on purpose — each rule is a small
function that returns a finding dict, so new rules can be added without
touching existing ones.
"""

import re
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from metadata_extractor import infer_semantic_type, _looks_like_email, _looks_like_phone

PII_SEMANTIC_TYPES = {"email", "phone", "name", "id"}
PII_NAME_HINTS = ("email", "phone", "name", "customer", "ssn", "passport", "address", "dob")


def rule_inconsistent_casing_or_whitespace(df: pd.DataFrame) -> list:
    """Flags categorical/text columns with leading/trailing whitespace or
    inconsistent casing across otherwise-identical values (e.g. 'eire'
    vs 'Eire', '  Spain' vs 'Spain')."""
    findings = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        has_whitespace_issue = (vals != vals.str.strip()).any()
        normalised = vals.str.strip().str.lower()
        collapses = normalised.nunique() < vals.nunique()
        if has_whitespace_issue or collapses:
            findings.append({
                "column": col,
                "issue": "inconsistent casing or stray whitespace",
                "detail": "values that should likely be treated as identical differ only "
                          "in case or surrounding whitespace",
            })
    return findings


def rule_unparseable_dates(df: pd.DataFrame) -> list:
    """Flags date-like columns where a meaningful fraction of values fail to parse."""
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
            findings.append({
                "column": col,
                "issue": "unparseable date values",
                "detail": f"{fail_rate:.1%} of values could not be parsed as dates",
            })
    return findings


def rule_malformed_emails_or_phones(df: pd.DataFrame) -> list:
    """Flags email/phone columns where a meaningful fraction don't match a valid format."""
    findings = []
    for col in df.columns:
        lname = col.lower()
        if "email" in lname or "mail" in lname:
            rate = _looks_like_email(df[col])
            if rate < 0.9:
                findings.append({
                    "column": col,
                    "issue": "malformed email values",
                    "detail": f"only {rate:.1%} of non-null values match a valid email format",
                })
        if "phone" in lname or "tel" in lname or "mobile" in lname:
            rate = _looks_like_phone(df[col])
            if rate < 0.9:
                findings.append({
                    "column": col,
                    "issue": "malformed phone values",
                    "detail": f"only {rate:.1%} of non-null values match a valid phone format",
                })
    return findings


def rule_suspicious_numeric_ranges(df: pd.DataFrame) -> list:
    """Flags numeric columns with implausible values for common business fields
    (e.g. negative prices, negative quantities aren't necessarily wrong for a
    retail returns dataset, but are worth surfacing for review)."""
    findings = []
    for col in df.select_dtypes(include="number").columns:
        lname = col.lower()
        if "price" in lname or "amount" in lname or "cost" in lname:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                findings.append({
                    "column": col,
                    "issue": "negative values in a price/amount column",
                    "detail": f"{int(n_negative)} row(s) have negative values — verify whether "
                              "these are legitimate (e.g. refunds) or data errors",
                })
    return findings


def detect_pii_columns(df: pd.DataFrame) -> list:
    """Flags columns likely to contain personally identifiable information.
    Generic transactional IDs (invoice numbers, SKUs) are NOT PII on their
    own — only IDs whose name suggests they identify a person are flagged."""
    findings = []
    for col in df.columns:
        semantic = infer_semantic_type(df[col], col)
        name_flag = any(h in col.lower() for h in PII_NAME_HINTS)

        if semantic == "id" and not name_flag:
            continue  # generic transactional/product ID, not personal data

        if semantic in PII_SEMANTIC_TYPES or name_flag:
            findings.append({
                "column": col,
                "likely_pii_type": semantic if semantic in PII_SEMANTIC_TYPES else "unknown",
                "reason": "semantic type matched a PII category" if semantic in PII_SEMANTIC_TYPES
                          else "column name suggests personal data",
            })
    return findings


RULES = [
    rule_inconsistent_casing_or_whitespace,
    rule_unparseable_dates,
    rule_malformed_emails_or_phones,
    rule_suspicious_numeric_ranges,
]


def run_rule_engine(df: pd.DataFrame) -> dict:
    all_findings = []
    for rule_fn in RULES:
        all_findings.extend(rule_fn(df))
    return {
        "suspicious_format_findings": all_findings,
        "potential_pii_columns": detect_pii_columns(df),
    }
