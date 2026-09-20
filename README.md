# Automated Data Cleaning & Validation System
CadetX Data & AI Virtual Work Experience — Project Submission

## Domain & Dataset
**Domain:** Retail / E-commerce
**Dataset:** UCI "Online Retail" — real UK-based e-commerce transactions (Dec 2010–Dec 2011).

> `data/raw/sample_retail.csv` is a synthetic stand-in with the same
> categories of messiness (nulls, duplicates, mixed types, bad
> emails/phones/dates) used for development and testing.

## Project Structure
```
modules/
  module1_profiling/    → profiling_report.json  (COMPLETE)
    metadata_extractor.py   Week 1: dtype/semantic type inference, mixed-type detection
    profiler.py              Week 2: missing-value matrix, cardinality, correlation, visualisations
    rule_engine.py            Week 3: suspicious-format rules, PII detection
    api.py                    Week 3: public entry point
  module2_cleaning/     → cleaned_data.csv + cleaning_log.json (in progress)
    schema_inference.py      Week 4: expected type/range/format per column
    cleaning_pipeline.py     Week 4: statistical imputation, duplicate detection, normalisation
    ml_imputation.py         Week 5: KNN, regression, and iterative imputation
    transformation_engine.py Week 5: scaling, encoding, feature extraction
  module3_validation/   → validation_report.json (not started)
  module4_automation/   → end-to-end CLI pipeline (not started)
data/
  raw/                  → input datasets
  processed/            → pipeline outputs + visualisations
tests/                  → 41 pytest unit tests, all passing
docs/                   → module1_api.md (schema, function signatures, integration guide)
```

## Status
- [x] Week 0: Domain & dataset selected, repo scaffolded
- [x] Week 1: Metadata extraction engine
- [x] Week 2: Profiling engine + visualisations
- [x] Week 3: Profiling API + rule engine + docs — **Module 1 complete**
- [x] Week 4: Schema inference + cleaning pipeline basics
- [x] Week 5: ML-based imputation + transformation engine
- [ ] Week 6: Data-quality scoring + cleaning API + tests — closes Module 2

## Setup
```
pip install -r requirements.txt
python3 modules/module1_profiling/api.py data/raw/sample_retail.csv
python3 modules/module2_cleaning/cleaning_pipeline.py data/raw/sample_retail.csv
python3 modules/module2_cleaning/ml_imputation.py data/raw/sample_retail.csv
python3 modules/module2_cleaning/transformation_engine.py data/raw/sample_retail.csv
python3 -m pytest tests/ -v
```

See `docs/module1_api.md` for the Module 1 API reference and JSON schema.
