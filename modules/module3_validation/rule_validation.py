"""
Rule-Based Validation
Module 3 — Week 7 deliverable

Checks a (cleaned) dataset against explicit, explainable rules and reports
every violation: which row, which column, which rule, how severe.

Rules (each is a small independent function; see RULES at the bottom):

  required_fields              configured columns must not be missing
  email_format                 valid email syntax
  phone_format                 7-15 digits, only digits and  + ( ) . - space
  postcode_format              UK / US / PK / CA postcode formats (configurable)
  numeric_validity             numeric columns hold finite numbers
  numeric_range                min/max bounds (configured) + no negative prices
  date_plausibility            parseable, not in the future, not before min_date
  categorical_case_whitespace  value is not a case/whitespace variant of the dominant form
  categorical_near_duplicate   rare value that looks like a typo of a common one
  placeholder_value            disguised missing values: 'unknown', '?', '-', 'tbd', ...
                               (pandas.read_csv already turns 'n/a', 'NA', 'null', 'None'
                               into real missing values on load, so for CSV input this rule
                               catches the placeholders pandas does NOT recognise)
  allowed_values               value must be in a configured list
  pattern_match                value must fully match a configured regex

Severity: 'error' = definitely invalid, 'warning' = suspicious / worth review.

Rules only fire where they apply: a dataset with no email column simply
reports email_format as "not_applicable" rather than passing vacuously.
Missing values are not format violations (that is what required_fields and
Module 2's completeness score are for).

Statistical / ML anomaly detection is deliberately NOT here — that is Week 8.
"""

import copy
import os
import re
import sys

import numpy as np
import pandas as pd
from thefuzz import fuzz

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "module1_profiling"))
sys.path.insert(0, os.path.join(_HERE, "..", "module2_cleaning"))
from schema_inference import build_schema

VIOLATION_COLUMNS = ["row_index", "column", "rule", "severity", "value", "message"]

EMAIL_PATTERN = (
    r"^[A-Za-z0-9_%+\-]+(\.[A-Za-z0-9_%+\-]+)*"       # local part: no leading/trailing/double dots
    r"@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$"  # domain with a real TLD
)

POSTCODE_PATTERNS = {
    "UK": r"[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}",
    "US": r"\d{5}(-\d{4})?",
    "PK": r"\d{5}",
    "CA": r"[A-Z]\d[A-Z]\s?\d[A-Z]\d",
}

PLACEHOLDER_TOKENS = {
    "", "n/a", "na", "n.a.", "#n/a", "none", "null", "nil", "nan", "unknown", "unk",
    "?", "??", "-", "--", "tbd", "missing", "not available", "undefined",
}

DEFAULT_CONFIG = {
    "required": [],                  # columns that must not be missing
    "ranges": {},                    # {"col": {"min": 0, "max": 100, "severity": "error"}}
    "patterns": {},                  # {"col": "^regex$"}
    "allowed_values": {},            # {"col": ["a", "b"]}
    "postcode_countries": ["UK", "US"],
    "near_duplicate_threshold": 90,  # fuzzy similarity (0-100) to call two values the same
    "near_duplicate_rare_ratio": 0.1,  # a typo must be at most this fraction as common
    "as_of": None,                   # date used for 'in the future' (default: now)
    "min_date": "1900-01-01",
    "max_examples_per_rule": 20,
}

RULE_DESCRIPTIONS = {
    "required_fields": "Configured columns must not be missing",
    "email_format": "Email addresses must be syntactically valid",
    "phone_format": "Phone numbers must have 7-15 digits and only valid characters",
    "postcode_format": "Postcodes must match a configured country format",
    "numeric_validity": "Numeric columns must contain finite numbers",
    "numeric_range": "Numeric values must be within configured bounds; prices must not be negative",
    "date_plausibility": "Dates must parse, not be in the future, and not precede min_date",
    "categorical_case_whitespace": "Categorical values must use one consistent casing/whitespace form",
    "categorical_near_duplicate": "Rare categorical values that look like typos of common ones",
    "placeholder_value": "Disguised missing values such as 'unknown', '?' or '-'",
    "allowed_values": "Values must belong to the configured allowed list",
    "pattern_match": "Values must fully match the configured regex",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _empty():
    return pd.DataFrame(columns=VIOLATION_COLUMNS)


def _na(rule, reason):
    return {"rule": rule, "columns": [], "violations": _empty(), "reason": reason}


def _is_text(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _semantic(schema, col):
    return schema.get(col, {}).get("semantic_type")


def _to_text(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _text_series(series):
    """Non-null values as text; whole-number floats rendered without '.0'."""
    return series.dropna().map(_to_text)


def _bool(x):
    return x.fillna(False).astype(bool)


def _violations(df, idx, column, rule, severity, message):
    """Build violation rows for `idx` (row labels) of `column`.
    `message` is a string, or a Series of per-row messages indexed like df."""
    idx = pd.Index(idx)
    if len(idx) == 0:
        return _empty()
    msgs = message.loc[idx].to_numpy() if isinstance(message, pd.Series) else message
    return pd.DataFrame({
        "row_index": idx.to_numpy(),
        "column": column,
        "rule": rule,
        "severity": severity,
        "value": df.loc[idx, column].map(_to_text).to_numpy(),
        "message": msgs,
    })


def _result(rule, columns, parts):
    parts = [p for p in parts if len(p)]
    return {"rule": rule, "columns": columns,
            "violations": pd.concat(parts, ignore_index=True) if parts else _empty(),
            "reason": None}


def _columns_matching(df, schema, semantics=(), name_hints=()):
    return [c for c in df.columns
            if _semantic(schema, c) in semantics or any(h in c.lower() for h in name_hints)]


def _numeric_columns(df, schema):
    cols = []
    for c in df.columns:
        sem = _semantic(schema, c)
        if sem == "numeric" or (sem is None and pd.api.types.is_numeric_dtype(df[c])):
            cols.append(c)
    return cols


def _coerce_numeric(series):
    return series if pd.api.types.is_numeric_dtype(series) else pd.to_numeric(series, errors="coerce")


def _parse_dates(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    else:
        parsed = pd.to_datetime(series, errors="coerce")           # fast: infers one format
        if parsed.isna().sum() > series.isna().sum():               # mixed formats: slower fallback
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def _merge_config(user_config):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    user_config = user_config or {}
    unknown = set(user_config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}. Valid keys: {sorted(DEFAULT_CONFIG)}")
    cfg.update(copy.deepcopy(user_config))

    bad = [c for c in cfg["postcode_countries"] if c not in POSTCODE_PATTERNS]
    if bad:
        raise ValueError(f"Unsupported postcode_countries {bad}; supported: {sorted(POSTCODE_PATTERNS)}")
    for col, rng in cfg["ranges"].items():
        extra = set(rng) - {"min", "max", "severity"}
        if extra:
            raise ValueError(f"ranges['{col}'] has unknown keys {sorted(extra)}; use min, max, severity")
        if rng.get("severity", "error") not in ("error", "warning"):
            raise ValueError(f"ranges['{col}'].severity must be 'error' or 'warning'")
    for col, pat in cfg["patterns"].items():
        try:
            re.compile(pat)
        except re.error as e:
            raise ValueError(f"patterns['{col}'] is not a valid regex: {e}")
    return cfg


def _config_warnings(df, cfg):
    """Config that mentions columns the dataset doesn't have is reported, not fatal."""
    warnings = []
    referenced = (
        [("required", c) for c in cfg["required"]]
        + [("ranges", c) for c in cfg["ranges"]]
        + [("patterns", c) for c in cfg["patterns"]]
        + [("allowed_values", c) for c in cfg["allowed_values"]]
    )
    for key, col in referenced:
        if col not in df.columns:
            warnings.append(f"config '{key}' refers to column '{col}', which is not in the dataset")
    return warnings


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------

def rule_required_fields(df, schema, cfg):
    cols = [c for c in cfg["required"] if c in df.columns]
    if not cols:
        return _na("required_fields", "no required columns configured")
    parts = [_violations(df, df.index[df[c].isna()], c, "required_fields", "error", "required value is missing")
             for c in cols]
    return _result("required_fields", cols, parts)


def rule_email_format(df, schema, cfg):
    cols = [c for c in _columns_matching(df, schema, {"email"}, ("email",)) if _is_text(df[c])]
    if not cols:
        return _na("email_format", "no email column found")
    parts = []
    for c in cols:
        s = _text_series(df[c])
        valid = _bool(s.str.match(EMAIL_PATTERN)) & (s.str.len() <= 254)
        parts.append(_violations(df, s.index[~valid.to_numpy()], c, "email_format", "error", "invalid email format"))
    return _result("email_format", cols, parts)


def rule_phone_format(df, schema, cfg):
    cols = [c for c in _columns_matching(df, schema, {"phone"}, ("phone", "mobile")) if _is_text(df[c])]
    if not cols:
        return _na("phone_format", "no phone column found")
    parts = []
    for c in cols:
        s = _text_series(df[c])
        chars_ok = _bool(s.str.match(r"^\+?[\d\s().\-]+$"))
        digits = s.str.replace(r"\D", "", regex=True).str.len()
        valid = chars_ok & digits.between(7, 15).to_numpy()
        parts.append(_violations(df, s.index[~valid.to_numpy()], c, "phone_format", "error",
                                 "invalid phone number (needs 7-15 digits, only digits and + ( ) . - space)"))
    return _result("phone_format", cols, parts)


def rule_postcode_format(df, schema, cfg):
    cols = _columns_matching(df, schema, (), ("postcode", "post_code", "postal", "zip"))
    if not cols:
        return _na("postcode_format", "no postcode column found")
    countries = cfg["postcode_countries"]
    pattern = "^(?:" + "|".join(f"(?:{POSTCODE_PATTERNS[c]})" for c in countries) + ")$"
    parts = []
    for c in cols:
        s = _text_series(df[c])
        valid = _bool(s.str.upper().str.match(pattern))
        parts.append(_violations(df, s.index[~valid.to_numpy()], c, "postcode_format", "error",
                                 f"does not match any configured postcode format ({', '.join(countries)})"))
    return _result("postcode_format", cols, parts)


def rule_numeric_validity(df, schema, cfg):
    cols = _numeric_columns(df, schema)
    if not cols:
        return _na("numeric_validity", "no numeric column found")
    parts = []
    for c in cols:
        values = _coerce_numeric(df[c])
        not_a_number = df[c].notna() & values.isna()
        parts.append(_violations(df, df.index[not_a_number.to_numpy()], c, "numeric_validity", "error",
                                 "value is not a number"))
        non_finite = values.notna() & ~np.isfinite(values.astype(float))
        parts.append(_violations(df, df.index[non_finite.to_numpy()], c, "numeric_validity", "error",
                                 "value is infinite"))
    return _result("numeric_validity", cols, parts)


def rule_numeric_range(df, schema, cfg):
    bounds = {}
    for c in _numeric_columns(df, schema):
        if c in cfg["ranges"]:
            bounds[c] = cfg["ranges"][c]
        elif any(h in c.lower() for h in ("price", "amount", "cost")):
            bounds[c] = {"min": 0, "severity": "error"}   # sensible default: no negative prices
    if not bounds:
        return _na("numeric_range", "no ranges configured and no price/amount/cost column found")
    parts = []
    for c, rng in bounds.items():
        values = _coerce_numeric(df[c])
        severity = rng.get("severity", "error")
        if "min" in rng:
            below = (values < rng["min"]).fillna(False)
            parts.append(_violations(df, df.index[below.to_numpy()], c, "numeric_range", severity,
                                     f"below minimum {rng['min']}"))
        if "max" in rng:
            above = (values > rng["max"]).fillna(False)
            parts.append(_violations(df, df.index[above.to_numpy()], c, "numeric_range", severity,
                                     f"above maximum {rng['max']}"))
    return _result("numeric_range", list(bounds), parts)


def rule_date_plausibility(df, schema, cfg):
    cols = [c for c in df.columns if _semantic(schema, c) == "date"]
    if not cols:
        return _na("date_plausibility", "no date column found")
    as_of = pd.Timestamp(cfg["as_of"]) if cfg["as_of"] else pd.Timestamp.now()
    min_date = pd.Timestamp(cfg["min_date"])
    parts = []
    for c in cols:
        parsed = _parse_dates(df[c])
        unparseable = df[c].notna() & parsed.isna()
        parts.append(_violations(df, df.index[unparseable.to_numpy()], c, "date_plausibility", "error",
                                 "value is not a parseable date"))
        future = (parsed > as_of).fillna(False)
        parts.append(_violations(df, df.index[future.to_numpy()], c, "date_plausibility", "error",
                                 f"date is after {as_of.date()}"))
        early = (parsed < min_date).fillna(False)
        parts.append(_violations(df, df.index[early.to_numpy()], c, "date_plausibility", "error",
                                 f"date is before {min_date.date()}"))
    return _result("date_plausibility", cols, parts)


def _categorical_text_columns(df, schema):
    return [c for c in df.columns if _semantic(schema, c) == "categorical" and _is_text(df[c])]


def rule_categorical_case_whitespace(df, schema, cfg):
    cols = _categorical_text_columns(df, schema)
    if not cols:
        return _na("categorical_case_whitespace", "no categorical text column found")
    parts = []
    for c in cols:
        s = _text_series(df[c])
        stripped = s.str.strip()
        key = stripped.str.lower()
        dominant = stripped.groupby(key).agg(lambda x: x.value_counts().idxmax())
        canonical = key.map(dominant)
        bad = (s != canonical).to_numpy()
        msg = "not the dominant form '" + canonical + "'"
        parts.append(_violations(df, s.index[bad], c, "categorical_case_whitespace", "warning", msg))
    return _result("categorical_case_whitespace", cols, parts)


def rule_categorical_near_duplicate(df, schema, cfg):
    cols = _categorical_text_columns(df, schema)
    if not cols:
        return _na("categorical_near_duplicate", "no categorical text column found")
    threshold, rare_ratio = cfg["near_duplicate_threshold"], cfg["near_duplicate_rare_ratio"]
    parts = []
    for c in cols:
        counts = df[c].dropna().astype(str).value_counts()
        if not 2 <= len(counts) <= 100:
            continue
        values = counts.index.tolist()
        norm = {v: v.strip().lower() for v in values}
        fixes = {}
        for rare in values:
            best = None
            for common in values:
                if common == rare or counts[common] <= counts[rare]:
                    continue
                if counts[rare] > rare_ratio * counts[common] or norm[rare] == norm[common]:
                    continue
                score = fuzz.ratio(norm[rare], norm[common])
                if score >= threshold and (best is None or score > best[1]):
                    best = (common, score)
            if best:
                fixes[rare] = best
        if fixes:
            mask = df[c].isin(list(fixes))
            msg = df.loc[mask, c].map(
                lambda v: f"'{v}' looks like a misspelling of '{fixes[v][0]}' (similarity {fixes[v][1]})")
            parts.append(_violations(df, msg.index, c, "categorical_near_duplicate", "warning", msg))
    return _result("categorical_near_duplicate", cols, parts)


def rule_placeholder_value(df, schema, cfg):
    cols = [c for c in df.columns if _is_text(df[c])]
    if not cols:
        return _na("placeholder_value", "no text column found")
    parts = []
    for c in cols:
        s = _text_series(df[c])
        hit = s.str.strip().str.lower().isin(PLACEHOLDER_TOKENS).to_numpy()
        parts.append(_violations(df, s.index[hit], c, "placeholder_value", "warning",
                                 "looks like a disguised missing value"))
    return _result("placeholder_value", cols, parts)


def rule_allowed_values(df, schema, cfg):
    cols = [c for c in cfg["allowed_values"] if c in df.columns]
    if not cols:
        return _na("allowed_values", "no allowed_values configured")
    parts = []
    for c in cols:
        allowed = {str(v) for v in cfg["allowed_values"][c]}
        s = _text_series(df[c])
        parts.append(_violations(df, s.index[~s.isin(allowed).to_numpy()], c, "allowed_values", "error",
                                 "value is not in the allowed list"))
    return _result("allowed_values", cols, parts)


def rule_pattern_match(df, schema, cfg):
    cols = [c for c in cfg["patterns"] if c in df.columns]
    if not cols:
        return _na("pattern_match", "no patterns configured")
    parts = []
    for c in cols:
        rx = re.compile(cfg["patterns"][c])
        s = _text_series(df[c])
        ok = s.map(lambda v: rx.fullmatch(v) is not None).to_numpy()
        parts.append(_violations(df, s.index[~ok], c, "pattern_match", "error",
                                 f"does not match required pattern {cfg['patterns'][c]}"))
    return _result("pattern_match", cols, parts)


RULES = [
    rule_required_fields, rule_email_format, rule_phone_format, rule_postcode_format,
    rule_numeric_validity, rule_numeric_range, rule_date_plausibility,
    rule_categorical_case_whitespace, rule_categorical_near_duplicate,
    rule_placeholder_value, rule_allowed_values, rule_pattern_match,
]


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def validate_rules(df, schema=None, config=None):
    """
    Validate a DataFrame against all rules.

    Args:
        df: the dataset (normally Module 2's cleaned output). Not modified.
        schema: {column: {"semantic_type": ...}}. Pass cleaning_log["schema"]
            (inferred from the RAW data) for best results; if omitted it is
            inferred from `df` itself.
        config: dict of overrides — see DEFAULT_CONFIG. Unknown keys raise
            ValueError so typos are not silently ignored.

    Returns:
        {"report": dict (JSON-serialisable), "violations_df": DataFrame}
        `row_index` is the 0-based row position in the dataset passed in.
    """
    cfg = _merge_config(config)
    df = df.reset_index(drop=True)
    if schema is None:
        schema = build_schema(df)

    results = [rule(df, schema, cfg) for rule in RULES]
    frames = [r["violations"] for r in results if len(r["violations"])]
    violations = pd.concat(frames, ignore_index=True) if frames else _empty()
    if len(violations):
        violations["row_index"] = violations["row_index"].astype(int)
        violations = violations.sort_values(["row_index", "column", "rule"]).reset_index(drop=True)

    n_rows = len(df)
    n_examples = cfg["max_examples_per_rule"]
    rule_summaries = []
    for r in results:
        v = r["violations"]
        entry = {
            "rule": r["rule"],
            "description": RULE_DESCRIPTIONS[r["rule"]],
            "status": "applied" if r["reason"] is None else "not_applicable",
            "columns": r["columns"],
            "violations": int(len(v)),
            "errors": int((v["severity"] == "error").sum()),
            "warnings": int((v["severity"] == "warning").sum()),
        }
        if r["reason"]:
            entry["reason"] = r["reason"]
        if len(v):
            examples = v.head(n_examples)
            entry["examples"] = [
                {"row_index": int(row.row_index), "column": row.column, "value": row.value,
                 "severity": row.severity, "message": row.message}
                for row in examples.itertuples(index=False)
            ]
        rule_summaries.append(entry)

    by_column = {}
    if len(violations):
        for (col, rule), n in violations.groupby(["column", "rule"]).size().items():
            by_column.setdefault(col, {})[rule] = int(n)

    rows_any = int(violations["row_index"].nunique()) if len(violations) else 0
    rows_err = int(violations.loc[violations["severity"] == "error", "row_index"].nunique()) if len(violations) else 0
    pct = lambda n: round(100 * n / n_rows, 2) if n_rows else 0.0

    report = {
        "summary": {
            "rows_checked": n_rows,
            "columns_checked": int(df.shape[1]),
            "rules_applied": sum(1 for r in rule_summaries if r["status"] == "applied"),
            "rules_not_applicable": sum(1 for r in rule_summaries if r["status"] == "not_applicable"),
            "total_violations": int(len(violations)),
            "errors": int((violations["severity"] == "error").sum()) if len(violations) else 0,
            "warnings": int((violations["severity"] == "warning").sum()) if len(violations) else 0,
            "rows_with_violations": rows_any,
            "rows_with_violations_percent": pct(rows_any),
            "rows_with_errors": rows_err,
            "rows_with_errors_percent": pct(rows_err),
        },
        "rules": rule_summaries,
        "violations_by_column": by_column,
        "config_warnings": _config_warnings(df, cfg),
        "config_used": cfg,
    }
    return {"report": report, "violations_df": violations}


def validate_rules_to_files(input_path, output_dir="data/processed", cleaning_log_path=None, config=None):
    """Reads a CSV, validates it, writes rule_validation_report.json and
    rule_violations.csv. If `cleaning_log_path` is given, its schema (inferred
    from the raw data) is used. Returns (report_path, violations_path)."""
    import json
    schema = None
    if cleaning_log_path:
        with open(cleaning_log_path) as f:
            schema = json.load(f).get("schema")
    result = validate_rules(pd.read_csv(input_path), schema=schema, config=config)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "rule_validation_report.json")
    violations_path = os.path.join(output_dir, "rule_violations.csv")
    with open(report_path, "w") as f:
        json.dump(result["report"], f, indent=2, default=str)
    result["violations_df"].to_csv(violations_path, index=False)
    return report_path, violations_path


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Rule-based validation (Module 3, Week 7)")
    ap.add_argument("input", nargs="?", default="data/processed/cleaned_data.csv")
    ap.add_argument("--log", default="data/processed/cleaning_log.json", help="cleaning_log.json (for the schema)")
    ap.add_argument("--config", help="JSON file of config overrides")
    ap.add_argument("--output-dir", default="data/processed")
    args = ap.parse_args()

    config = json.load(open(args.config)) if args.config else None
    log = args.log if os.path.exists(args.log) else None
    report_path, violations_path = validate_rules_to_files(
        args.input, args.output_dir, cleaning_log_path=log, config=config)
    s = json.load(open(report_path))["summary"]
    print(f"Wrote {report_path} and {violations_path}")
    print(f"{s['total_violations']} violations ({s['errors']} errors, {s['warnings']} warnings) "
          f"in {s['rows_with_violations']} of {s['rows_checked']} rows")
