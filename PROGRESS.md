# Progress Log

## Week 0 — Foundation
- Selected domain: Retail / E-commerce
- Selected dataset: UCI Online Retail dataset (real transactions, Dec 2010–Dec 2011)
- Scaffolded repo: modules/, data/, tests/, docs/

## Week 1 — Metadata Extraction Engine (Module 1)
**Built:** `metadata_extractor.py` — infers pandas dtype + semantic type
(id, email, phone, date, name, categorical, numeric, free_text, unknown)
per column, detects mixed-type columns, reports null%/unique/samples.
**Tested:** 6 unit tests, all passing.

## Week 2 — Profiling Engine + Visualisations (Module 1)
**Built:** `profiler.py` — missing-value matrix, dtype consistency check,
cardinality analysis, correlation matrix, and 3 backend visualisations
(missing heatmap, outlier boxplots, correlation heatmap).
**Tested:** 7 new unit tests, 13/13 passing project-wide.

## Week 3 — Profiling API + Rule Engine + Docs (Module 1 complete)
**Built:**
- `rule_engine.py` — standalone rules flagging suspicious formats
  (inconsistent casing/whitespace, unparseable dates, malformed
  emails/phones, negative price values) and potential PII columns
  (email, phone, name, and personally-identifying ID columns —
  generic transactional IDs like invoice numbers are excluded)
- `api.py` — single public entry point (`profile_dataset` /
  `profile_dataset_to_file`) wrapping metadata extraction + profiling +
  rule engine into one call for downstream modules to use
- `docs/module1_api.md` — full JSON schema, function signatures, and
  integration instructions for Module 2 onward

**Decisions:**
- Kept PII detection conservative: only flags an "id" column as PII if
  its name also implies a person (e.g. `CustomerID`), not generic
  product/transaction IDs — avoids false positives that would clutter
  downstream handling.
- Rules are small, independent functions in a `RULES` list, so new rules
  can be added without touching existing logic.

**Tested:** 9 new unit tests, 22/22 passing project-wide.

**Module 1 is now complete.** Output: `profiling_report.json` containing
metadata, missing-value matrix, dtype consistency issues, cardinality
analysis, correlation matrix, visualisation file paths, suspicious-format
findings, and potential PII columns.

**Next (Week 4):** Module 2 begins — schema inference engine (expected
types/ranges/formats) and the first pass of the cleaning pipeline
(missing-value imputation, duplicate detection, format normalisation).
