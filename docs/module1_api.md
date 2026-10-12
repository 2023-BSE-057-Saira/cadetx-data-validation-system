# Module 1 — Profiling: API Documentation

Single public entry point: `modules/module1_profiling/api.py`. Downstream
modules should import only from this file.

## Function signatures

### `profile_dataset(input_data, viz_out_dir="data/processed/visualisations") -> dict`
Runs the full Module 1 pipeline in memory.
- `input_data`: a `pandas.DataFrame`, or a `str` path to a CSV
- `viz_out_dir`: where the PNG visualisations are written
- Returns: dict matching the `profiling_report.json` schema below

### `profile_dataset_to_file(input_path, output_path="data/processed/profiling_report.json", viz_out_dir="data/processed/visualisations") -> str`
Reads a CSV, profiles it, writes `profiling_report.json`; returns `output_path`.

## Integration

```python
from modules.module1_profiling.api import profile_dataset_to_file
report_path = profile_dataset_to_file("data/raw/online_retail.csv")
```

## `profiling_report.json` schema

```
{
  "metadata": {
    "n_rows": int, "n_columns": int,
    "columns": [{
      "column": str, "pandas_dtype": str,
      "semantic_type": "id"|"numeric"|"email"|"phone"|"date"|"name"|"categorical"|"free_text"|"unknown",
      "is_mixed_type": bool, "null_count": int, "null_percent": float,
      "unique_count": int, "sample_values": [str]
    }]
  },
  "missing_value_matrix": { "<col>": {"missing_count": int, "missing_percent": float} },
  "dtype_consistency_issues": { "<col>": str },
  "cardinality_analysis": { "<col>": {"unique_count": int, "cardinality_ratio": float,
                                       "is_likely_constant": bool, "is_likely_id_like": bool} },
  "correlation_matrix": { "<col>": { "<other_col>": float } },
  "visualisations_generated": [str],
  "suspicious_format_findings": [{"column": str, "issue": str, "detail": str}],
  "potential_pii_columns": [{"column": str, "likely_pii_type": str, "reason": str}]
}
```

## Semantic-type detection order

1. Numeric dtype -> `id` (if the name hints at an identifier) or `numeric`
2. **Column-name hints** (email, phone, date, name, id) — always first, because
   content heuristics can be ambiguous (a date like `2021-01-04` also fits a
   loose phone pattern)
3. Content heuristics, most specific first: email, then date, then phone
4. Cardinality fallback: `categorical` / `free_text` / `unknown`

Bare digit strings (`07123`) are never treated as dates by content inference.

## Rule engine

| Rule | Flags |
|---|---|
| `rule_inconsistent_casing_or_whitespace` | Same value written with different case / stray whitespace |
| `rule_unparseable_dates` | Date-named columns where >2% of values fail to parse |
| `rule_malformed_emails_or_phones` | Email/phone-named columns where <90% match a valid format |
| `rule_suspicious_numeric_ranges` | Negative values in price/amount/cost columns |

PII detection flags email, phone, name columns and ID columns whose *name*
implies a person (e.g. `CustomerID`); generic invoice/SKU IDs are excluded.
New rule: add a `(df) -> list[dict]` function and append it to `RULES`.
