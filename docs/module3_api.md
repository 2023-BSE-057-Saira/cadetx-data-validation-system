# Module 3 — Validation: API Documentation

**Status:** Week 7 (rule-based validation) is done. Anomaly detection, the NLP
column classifier, error severity scoring, the dataset health score and the
final `validation_report.json` arrive in Weeks 8-9; this document grows with them.

Entry point so far: `modules/module3_validation/rule_validation.py`

## Function signatures

### `validate_rules(df, schema=None, config=None) -> dict`
- `df`: the dataset, normally Module 2's cleaned output. Not modified.
- `schema`: `{column: {"semantic_type": ...}}`. Pass `cleaning_log["schema"]`
  (inferred from the **raw** data) for best results; if omitted it is inferred from `df`.
- `config`: dict of overrides (below). Unknown keys raise `ValueError`, so a typo
  such as `"range"` instead of `"ranges"` is never silently ignored.
- Returns `{"report": dict, "violations_df": DataFrame}`.
  `row_index` is the 0-based row position in the dataset passed in.

### `validate_rules_to_files(input_path, output_dir="data/processed", cleaning_log_path=None, config=None) -> (str, str)`
Writes `rule_validation_report.json` and `rule_violations.csv` (one row per violation).

CLI:
```
python3 modules/module3_validation/rule_validation.py data/processed/cleaned_data.csv \
    --log data/processed/cleaning_log.json --config configs/retail_validation.json
```

## Chaining with Module 2

```python
from modules.module2_cleaning.cleaning_api import clean_dataset
from modules.module3_validation.rule_validation import validate_rules

out = clean_dataset("data/raw/online_retail.csv")
res = validate_rules(out["cleaned_df"], schema=out["cleaning_log"]["schema"])
```

Division of labour: cleaning fixes what is safely fixable (types, casing,
whitespace, duplicates, numeric gaps); validation **reports** what it cannot
fix (malformed emails, negative prices, impossible dates) instead of guessing.

## Rules

| Rule | Severity | Fires on |
|---|---|---|
| `required_fields` | error | configured columns that are missing |
| `email_format` | error | invalid email syntax (double `@`, no TLD, leading/double dots, spaces, >254 chars) |
| `phone_format` | error | not 7-15 digits, or characters other than digits and `+ ( ) . - space` |
| `postcode_format` | error | no match for any configured country (UK, US, PK, CA) |
| `numeric_validity` | error | non-numeric text or infinity in a numeric column |
| `numeric_range` | error / warning | outside configured `min`/`max`; negative values in price/amount/cost columns by default |
| `date_plausibility` | error | unparseable, after `as_of` (default now), or before `min_date` (default 1900-01-01) |
| `categorical_case_whitespace` | warning | not the dominant casing/whitespace form of the value |
| `categorical_near_duplicate` | warning | rare value (<=10% as common) that is >=90% similar to a common one — likely typo |
| `placeholder_value` | warning | `unknown`, `?`, `-`, `tbd`, `missing`, ... |
| `allowed_values` | error | value not in the configured list |
| `pattern_match` | error | value does not fully match the configured regex |

Rules only apply where they can: with no email column, `email_format` is reported
as `not_applicable` (with a reason) instead of passing vacuously. Missing values
are not format violations — use `required_fields` or Module 2's completeness score.

Notes:
- Negative **quantities** are deliberately not flagged by default (returns are legitimate in retail); set a `ranges` entry if you want a bound.
- `pandas.read_csv` already converts `n/a`, `NA`, `null`, `None`, `nan` to real missing values on load, so for CSV input `placeholder_value` catches the placeholders pandas does *not* recognise.
- Statistical outliers (e.g. an unusually large but valid quantity) are Week 8's job, not a rule.

## Config

```json
{
  "required": ["CustomerID"],
  "ranges": {"UnitPrice": {"min": 0}, "Quantity": {"min": -100000, "max": 100000, "severity": "warning"}},
  "patterns": {"InvoiceNo": "^C?\\d{6}$"},
  "allowed_values": {"Country": ["United Kingdom", "France"]},
  "postcode_countries": ["UK", "US"],
  "near_duplicate_threshold": 90,
  "near_duplicate_rare_ratio": 0.1,
  "as_of": "2011-12-31",
  "min_date": "1900-01-01",
  "max_examples_per_rule": 20
}
```
All keys are optional. `configs/retail_validation.json` is a starting point for the
retail dataset: check its bounds and patterns against the real data before relying
on them. Config that names a column the dataset lacks is reported in
`config_warnings`, not treated as a crash.

## `rule_validation_report.json` schema

```
{
  "summary": {
    "rows_checked": int, "columns_checked": int,
    "rules_applied": int, "rules_not_applicable": int,
    "total_violations": int, "errors": int, "warnings": int,
    "rows_with_violations": int, "rows_with_violations_percent": float,
    "rows_with_errors": int, "rows_with_errors_percent": float
  },
  "rules": [{
    "rule": str, "description": str, "status": "applied"|"not_applicable",
    "columns": [str], "violations": int, "errors": int, "warnings": int,
    "reason": str,                                   // only when not_applicable
    "examples": [{"row_index": int, "column": str, "value": str, "severity": str, "message": str}]
  }],
  "violations_by_column": {"<col>": {"<rule>": int}},
  "config_warnings": [str],
  "config_used": {...}
}
```
`total_violations` counts rule hits; one row can break several rules, so
`rows_with_violations` is usually smaller.
