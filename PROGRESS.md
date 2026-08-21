# Progress Log

## Week 0 — Foundation
- Selected domain: Retail / E-commerce
- Selected dataset: UCI Online Retail dataset (real transactions, Dec 2010–Dec 2011)
- Scaffolded repo: modules/, data/, tests/, docs/
- Wrote project description (see README.md)

## Week 1 — Metadata Extraction Engine (Module 1)
**Built:**
- `modules/module1_profiling/metadata_extractor.py`
- Infers pandas dtype + semantic type (id, email, phone, date, name,
  categorical, numeric, free_text, unknown) per column using name-hints
  and content-pattern heuristics
- Detects mixed-type columns (e.g. a numeric column with stray text values)
- Outputs null count/%, unique count, sample values per column

**Decisions:**
- Used heuristic/pattern matching rather than an ML classifier for semantic
  type detection at this stage — simpler, explainable, and sufficient for
  the profiling layer. Can revisit with an NLP classifier in Module 3.
- Handled pandas 3.0's new default `str` dtype (not just legacy `object`)
  so type-mixing detection works correctly on this environment.

**Tested:**
- 6 unit tests in `tests/test_metadata_extractor.py`, all passing
- Verified against a 515-row synthetic messy retail dataset with known
  injected issues (nulls, duplicates, mixed-type column, malformed emails)

**Next (Week 2):** Profiling engine — missing-value matrix, data-type
consistency check, cardinality analysis, correlation matrix, and the
backend visualisations (missing heatmap, outlier distribution, correlation
heatmap).

## Week 2 — Profiling Engine + Visualisations (Module 1)
**Built:**
- `modules/module1_profiling/profiler.py`
- Missing-value matrix (count + % per column)
- Data-type consistency check (flags columns whose values don't match their
  declared type — reuses the mixed-type detector from Week 1)
- Cardinality analysis (unique count, cardinality ratio, flags constant
  and ID-like columns)
- Correlation matrix for numeric columns
- Backend visualisations, saved as PNGs: missing-value heatmap, per-column
  outlier boxplots, correlation heatmap
- `build_profiling_report()` ties it all together with the Week 1 metadata
  extractor into one `profiling_report.json`

**Tested:**
- 7 new unit tests in `tests/test_profiler.py`, 13/13 passing project-wide
- Verified visualisations render correctly against the sample dataset

**Next (Week 3):** Profiling API wrapper + rule engine (suspicious formats,
potential PII columns) + documentation (JSON schema, function signatures,
integration instructions) — closes out Module 1.
