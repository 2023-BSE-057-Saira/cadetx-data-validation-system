# Module 1 — Profiling: API Documentation

Module 1 exposes a single public entry point in `modules/module1_profiling/api.py`.
Downstream modules (Cleaning, Validation, Automation) should only need to
import from this file — everything else in `module1_profiling/` is an
internal implementation detail.

## Function Signatures

### `profile_dataset(input_data, viz_out_dir="data/processed/visualisations") -> dict`
Runs the full Module 1 pipeline in memory and returns the report as a dict.

- `input_data`: a `pandas.DataFrame`, or a `str` path to a CSV file
- `viz_out_dir`: directory to write the generated PNG visualisations to
- Returns: `dict` matching the `profiling_report.json` schema below

### `profile_dataset_to_file(input_path, output_path="data/processed/profiling_report.json", viz_out_dir="data/processed/visualisations") -> str`
Convenience wrapper: reads a CSV from disk, profiles it, and writes the
report to a JSON file.

- `input_path`: path to the input CSV
- `output_path`: where to write `profiling_report.json`
- Returns: the `output_path` string, for chaining into Module 4's orchestrator

## Integration Instructions

To use Module 1 from another module (e.g. Module 4's pipeline orchestrator):

```python
from modules.module1_profiling.api import profile_dataset_to_file

report_path = profile_dataset_to_file(
    input_path="data/raw/online_retail.csv",
    output_path="data/processed/profiling_report.json",
)
# report_path now points to the JSON file Module 2 (Cleaning) should read
```

Or, to work with the report in memory without touching disk:

```python
from modules.module1_profiling.api import profile_dataset
import pandas as pd

df = pd.read_csv("data/raw/online_retail.csv")
report = profile_dataset(df)
suspicious_columns = [f["column"] for f in report["suspicious_format_findings"]]
```

## `profiling_report.json` Schema

```
{
  "metadata": {
    "n_rows": int,
    "n_columns": int,
    "columns": [
      {
        "column": str,
        "pandas_dtype": str,
        "semantic_type": "id" | "numeric" | "email" | "phone" | "date" |
                          "name" | "categorical" | "free_text" | "unknown",
        "is_mixed_type": bool,
        "null_count": int,
        "null_percent": float,
        "unique_count": int,
        "sample_values": [str, ...]
      },
      ...
    ]
  },
  "missing_value_matrix": {
    "<column_name>": { "missing_count": int, "missing_percent": float },
    ...
  },
  "dtype_consistency_issues": {
    "<column_name>": "<description of the inconsistency>",
    ...
  },
  "cardinality_analysis": {
    "<column_name>": {
      "unique_count": int,
      "cardinality_ratio": float,
      "is_likely_constant": bool,
      "is_likely_id_like": bool
    },
    ...
  },
  "correlation_matrix": {
    "<column_name>": { "<other_column_name>": float, ... },
    ...
  },
  "visualisations_generated": [str, ...],   // file paths to PNGs
  "suspicious_format_findings": [
    { "column": str, "issue": str, "detail": str },
    ...
  ],
  "potential_pii_columns": [
    { "column": str, "likely_pii_type": str, "reason": str },
    ...
  ]
}
```

## Rule Engine — Rules Currently Implemented

| Rule | What it flags |
|---|---|
| `rule_inconsistent_casing_or_whitespace` | Text columns where the same value appears with different casing or stray whitespace (e.g. `"UK"` vs `" uk"`) |
| `rule_unparseable_dates` | Date-named columns where >2% of values fail to parse |
| `rule_malformed_emails_or_phones` | Email/phone-named columns where <90% of values match a valid format |
| `rule_suspicious_numeric_ranges` | Price/amount columns containing negative values |

PII detection (`detect_pii_columns`) flags columns by semantic type
(`email`, `phone`, `name`, or an `id` column whose *name* also suggests a
person — e.g. `CustomerID`) — generic transactional/product IDs (invoice
numbers, SKUs) are intentionally excluded.

Adding a new rule: write a function `(df: pd.DataFrame) -> list[dict]`
following the existing pattern in `rule_engine.py`, then add it to the
`RULES` list at the bottom of that file. No other file needs to change.
