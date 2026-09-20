# Progress Log

## Week 0 — Foundation
Selected domain (Retail/E-commerce) and dataset (UCI Online Retail). Scaffolded repo.

## Week 1 — Metadata Extraction Engine (Module 1)
Built `metadata_extractor.py`: dtype + semantic type inference, mixed-type detection. 6 tests.

## Week 2 — Profiling Engine + Visualisations (Module 1)
Built `profiler.py`: missing-value matrix, dtype consistency, cardinality, correlation, 3 visualisations. 7 new tests.

## Week 3 — Profiling API + Rule Engine + Docs (Module 1 complete)
Built `rule_engine.py` (suspicious formats, PII detection) and `api.py` (single entry point).
Wrote `docs/module1_api.md`. 9 new tests.

## Week 4 — Schema Inference + Cleaning Pipeline Basics (Module 2 begins)
Built `schema_inference.py` and `cleaning_pipeline.py`: statistical imputation, exact +
fuzzy duplicate detection, type/casing/whitespace normalisation. Found and fixed a bug
where un-stripped values weren't matching their own stripped canonical form. 10 new tests.

## Week 5 — ML-Based Imputation + Transformation Engine
**Built:**
- `ml_imputation.py` — three ML imputation strategies for numeric columns:
  - KNN imputer (fills using the average of similar rows)
  - Regression imputer (predicts missing values from correlated numeric columns)
  - Iterative imputer / MICE-style (models each column as a function of the others,
    repeated until estimates stabilise)
  - `compare_imputation_strategies()` runs all three so a strategy can be chosen
    deliberately based on the data, rather than picking one arbitrarily
- `transformation_engine.py`:
  - Scaling: standardises numeric columns (mean 0, std 1), skipping ID-like columns
  - Encoding: one-hot for low-cardinality categorical columns, label encoding for
    higher-cardinality ones
  - Feature extraction: year/month/day-of-week/is-weekend from date columns, and a
    dataset-agnostic `total_value` feature when quantity- and price-like columns
    are both present (matched by column-name pattern, not hardcoded to this dataset)

**Bug found & fixed during testing:** the phone-number regex from Week 1
(`^\+?[\d\s().-]{7,15}$`) is loose enough that plain date strings like
`"2021-01-04"` also match it (digits + dashes, right length). This meant
a date column with no obvious name hint could get mis-classified as a phone
column. Fixed by restructuring `infer_semantic_type()` so column-name hints
are always checked before ambiguous content-based regex heuristics — the
column's own name is a more reliable signal than a regex that happens to
match multiple formats. Added a regression test using a 2-row date-only
DataFrame (the case that originally exposed the bug) so it can't silently
come back.

**Tested:** 11 new unit tests, 41/41 passing project-wide. Verified
end-to-end: KNN/iterative/regression imputation all correctly fill
`CustomerID`'s missing values on the sample dataset; regression achieves
high R² on a synthetic strongly-correlated pair, confirming the model is
learning a real relationship rather than guessing.

**Next (Week 6):** Data-quality scoring (before/after cleaning), the
Cleaning API wrapper (raw dataset in → `cleaned_data.csv` +
`cleaning_log.json` out), and pytest unit + integration tests — closes
out Module 2.
