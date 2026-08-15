# Automated Data Cleaning & Validation System
CadetX Data & AI Virtual Work Experience — Project Submission

## Domain & Dataset
**Domain:** Retail / E-commerce
**Dataset:** UCI "Online Retail" — real UK-based e-commerce transactions (Dec 2010–Dec 2011).
Chosen because it contains realistic, naturally-occurring data quality issues:
missing customer IDs, negative quantities (returns), inconsistent product
descriptions, and duplicate invoice line items — the exact problems this
system is built to detect and fix.

> `data/raw/sample_retail.csv` is a small synthetic stand-in used during
> development before the full UCI dataset was in place; it deliberately
> reproduces the same categories of messiness (nulls, duplicates, mixed
> types, bad emails/formats) for fast iteration and unit testing.

## Project Structure
```
modules/
  module1_profiling/    → profiling_report.json
  module2_cleaning/     → cleaned_data.csv + cleaning_log.json
  module3_validation/   → validation_report.json
  module4_automation/   → end-to-end CLI pipeline
data/
  raw/                  → input datasets
  processed/            → pipeline outputs
tests/                  → pytest unit + integration tests
docs/                   → architecture, API docs, schemas
```

## Status
- [x] Week 0: Domain & dataset selected, repo scaffolded
- [x] Week 1: Metadata extraction engine
- [ ] Week 2: Profiling engine + visualisations
- [ ] Week 3: Profiling API + rule engine + docs

## Setup
```
pip install -r requirements.txt
python3 modules/module1_profiling/metadata_extractor.py data/raw/sample_retail.csv
python3 -m pytest tests/ -v
```
