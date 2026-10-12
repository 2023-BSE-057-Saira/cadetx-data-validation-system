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

## Week 6 — Data-Quality Scoring + Cleaning API + Tests (Module 2 complete)
**Built:**
- `quality_scoring.py` — scores a dataset 0-100 on completeness (30%), uniqueness (20%),
  validity (30%) and consistency (20%); `compare_quality()` scores raw vs cleaned data
  against one shared schema and reports the delta, so cleaning is measured, not asserted
- `cleaning_api.py` — Module 2's public entry point: `clean_dataset()` /
  `clean_dataset_to_files()` take a raw dataset and produce `cleaned_data.csv` +
  `cleaning_log.json` (log embeds the schema and before/after quality scores); selectable
  numeric imputation (`statistical`, `knn`, `iterative`)
- `docs/module2_api.md` — signatures, pipeline order, imputation policy, log schema
- 26 new tests (unit + integration), 67/67 passing project-wide

**Result on the sample dataset:** overall quality 89.8 -> 93.1 (uniqueness 97.1 -> 100,
consistency 92.8 -> 100, completeness 92.0 -> 95.9). Validity barely moves (80.7 -> 81.1)
because malformed emails/phones cannot be repaired by cleaning; Module 3 will flag them.

**Problems found while testing, and fixed:**
1. *Fabricated data.* Weeks 4-5 filled missing `CustomerID` with a mean and missing
   emails/phones with the most common value. Identifier and contact columns are now never
   imputed; they stay missing and appear in the log as `skipped_not_imputable`.
2. *Mixed-type numeric column never coerced.* `Quantity` (numbers plus stray `"unknown"`)
   stayed text. Schema inference now recovers such columns as numeric; coercion + imputation
   then handle them. First attempt only worked on large columns (it relied on distinct-value
   count) — caught by a unit test on a small DataFrame and made size-independent.
3. *Codes read as dates.* Module 1's date detector parsed `"07123"` as the year 7123, so a
   postcode column was classed as dates and rewritten as `7123-01-01`. Bare digit strings
   are no longer treated as dates. Leading-zero codes are also protected from numeric coercion.
4. *Missed duplicates.* Rows differing only by whitespace/case (`"  France"` vs `"France"`)
   are identical only after normalisation; a second duplicate pass now runs after it.
5. `docs/module1_api.md` was missing from the Week 5 package; restored.

**Known limitations (documented, not hidden):** completeness counts imputed cells as present
(presence, not truth); mode imputation of categoricals (e.g. Country -> most common) is a
baseline assumption, logged per column; fuzzy duplicates are reported, not auto-merged.

**Module 2 is complete.** Output: `cleaned_data.csv` + `cleaning_log.json`.

**Next (Week 7):** Module 3 begins — rule-based validation (email/phone/postcode format,
numeric ranges, categorical consistency) feeding `validation_report.json`.
