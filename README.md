# Automated Data Cleaning & Validation System
CadetX Data & AI Virtual Work Experience — Project Submission

## Domain & Dataset
**Domain:** Retail / E-commerce
**Dataset:** UCI "Online Retail" — real UK-based e-commerce transactions (Dec 2010–Dec 2011).
Chosen because it contains realistic, naturally-occurring data quality issues:
missing customer IDs, negative quantities (returns), inconsistent product
descriptions, and duplicate invoice line items.

> `data/raw/sample_retail.csv` is a small synthetic stand-in used during
> development; it reproduces the same categories of messiness (nulls,
> duplicates, mixed types, bad emails/phones/dates) for fast iteration.

## Project Structure
```
modules/
  module1_profiling/    → profiling_report.json  (COMPLETE)
    metadata_extractor.py   Week 1: dtype/semantic type inference, mixed-type detection
    profiler.py              Week 2: missing-value matrix, cardinality, correlation, visualisations
    rule_engine.py            Week 3: suspicious-format rules, PII detection
    api.py                    Week 3: public entry point (profile_dataset / profile_dataset_to_file)
  module2_cleaning/     → cleaned_data.csv + cleaning_log.json (not started)
  module3_validation/   → validation_report.json (not started)
  module4_automation/   → end-to-end CLI pipeline (not started)
data/
  raw/                  → input datasets
  processed/            → pipeline outputs + visualisations
tests/                  → 22 pytest unit tests, all passing
docs/                   → module1_api.md (schema, function signatures, integration guide)
```

## Status
- [x] Week 0: Domain & dataset selected, repo scaffolded
- [x] Week 1: Metadata extraction engine
- [x] Week 2: Profiling engine + visualisations
- [x] Week 3: Profiling API + rule engine + docs — **Module 1 complete**
- [ ] Week 4: Schema inference + cleaning pipeline basics (Module 2 begins)

## Setup
```
pip install -r requirements.txt
python3 modules/module1_profiling/api.py data/raw/sample_retail.csv
python3 -m pytest tests/ -v
```

See `docs/module1_api.md` for the full API reference and JSON schema.
