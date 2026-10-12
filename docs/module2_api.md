# Module 2 — Cleaning: API Documentation

Single public entry point: `modules/module2_cleaning/cleaning_api.py`.

## Function signatures

### `clean_dataset(input_data, numeric_imputation="statistical") -> dict`
Cleans a dataset in memory. Does not modify the input.
- `input_data`: `DataFrame` or `str` path to a CSV
- `numeric_imputation`: `"statistical"` (mean), `"knn"`, or `"iterative"`;
  anything else raises `ValueError`
- Returns `{"cleaned_df": DataFrame, "cleaning_log": dict}`

### `clean_dataset_to_files(input_path, output_dir="data/processed", numeric_imputation="statistical") -> (str, str)`
Writes `cleaned_data.csv` and `cleaning_log.json`; returns both paths
(the cleaned CSV path is what Module 3 takes as input).

```python
from modules.module2_cleaning.cleaning_api import clean_dataset_to_files
cleaned_csv, log_json = clean_dataset_to_files("data/raw/online_retail.csv", numeric_imputation="knn")
```
CLI: `python3 modules/module2_cleaning/cleaning_api.py data/raw/sample_retail.csv [statistical|knn|iterative]`

## Pipeline order (and why)

1. Drop exact duplicate rows
2. Fuzzy-duplicate detection on text columns (reported, not auto-merged)
3. Normalise: dates -> datetime, numeric-typed columns -> numeric, categorical
   casing/whitespace -> dominant form
4. Drop duplicates again (rows that differed only by whitespace/case are now identical)
5. Impute numeric columns (statistical / KNN / iterative)
6. Statistical imputation for remaining imputable columns

Duplicates are removed **before** imputation so imputed values can never
create or hide duplicates.

## Imputation policy

| Column type | Behaviour |
|---|---|
| numeric measures | imputed (mean, KNN or iterative) |
| categorical | mode imputation |
| id, email, phone, name, date, free_text, unknown | **never imputed** — left missing and logged as `skipped_not_imputable` |

A mean `CustomerID` or a most-common email would be invented data that looks
real. Missing identifiers are surfaced in the log, not hidden.

**Text identifiers stay text.** Columns such as `InvoiceNo` (`C536379` for cancellations) and
`StockCode` (`85123A`) are labels, not numbers; they are never coerced to numeric, so no value
containing a letter is lost.

A text column that is mostly numeric with a few stray words (`"38"`, `"12"`,
`"unknown"`) is recovered as numeric and coerced. Code-like values with
leading zeros (`"07123"`) and fully-numeric text columns are left alone.

## Quality scoring (`quality_scoring.py`)

Each dimension is 0-100; overall is the weighted average.

| Dimension | Weight | Measures |
|---|---|---|
| completeness | 0.30 | share of cells not missing |
| uniqueness | 0.20 | share of rows that are not exact duplicates |
| validity | 0.30 | values parse / match format for their column (numeric, date, email, phone) |
| consistency | 0.20 | text values written in the dominant casing/whitespace form |

Before and after are scored against the **same schema, inferred from the raw
data**, so the comparison is fair.

Caveats: completeness counts imputed cells as present — it measures *presence*,
not *truth*. Validity only improves where cleaning can fix the problem
(type coercion); malformed emails/phones are not repaired here and are left
for Module 3 to flag.

## `cleaning_log.json` schema

```
{
  "steps": [
    {"step": "exact_duplicate_removal", "rows_removed": int},
    {"step": "type_format_normalisation", "details": [{"column": str, "action": str, ...}]},
    {"step": "post_normalisation_duplicate_removal", "rows_removed": int},
    {"step": "ml_numeric_imputation", "details": {...}},          // knn / iterative only
    {"step": "missing_value_imputation", "details": [
        {"column": str, "strategy": "mean_imputation"|"mode_imputation", "fill_value": ..., "rows_affected": int}
      | {"column": str, "strategy": "skipped_not_imputable", "reason": str, "rows_still_missing": int}
    ]}
  ],
  "fuzzy_duplicate_findings": { "<col>": [{"value_a": str, "value_b": str, "similarity": int}] },
  "numeric_imputation": str,
  "rows_before": int, "rows_after": int,
  "schema": { "<col>": {"semantic_type": str, "expected_dtype": str,
                         "expected_range": {...}|null, "expected_format": str|null} },
  "quality": {
    "before": {"overall": float, "dimensions": {...}, "details": {...}},
    "after":  {"overall": float, "dimensions": {...}, "details": {...}},
    "delta":  {"overall": float, "completeness": float, "uniqueness": float,
               "validity": float, "consistency": float}
  }
}
```

## Transformation engine (separate, optional)

`transformation_engine.run_transformation_engine(df)` adds scaled columns,
one-hot/label encodings and date-part features. It is for modelling use and
is **not** part of `cleaned_data.csv`; run it on the cleaned output.
