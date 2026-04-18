# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Cyber risk detection ML pipeline using the CICIDS2017 network intrusion dataset. Classifies network traffic as benign or one of five attack types (DoS/DDoS, Reconnaissance, Brute Force, Botnet, Web Attack).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run src/ui/app.py

# Run a single test file
python -m pytest tests/test_preprocess.py -v

# Run all tests
python -m pytest
```

## Architecture

```
src/
  data/
    loader.py       # CSV → merged.parquet (caching), NaN/inf cleaning
    preprocess.py   # Label mapping, train/test split, scaling, SMOTE
  models/           # Model definitions and wrappers
  training/         # Training loops, hyperparameter tuning
  reporting/        # PDF/HTML report generation (reportlab)
  ui/
    pages/          # Streamlit multi-page components
data/raw/           # CICIDS2017 CSV files go here; merged.parquet cached here
models/             # Saved model artifacts (.pkl), scaler.pkl
outputs/
  plots/            # Generated matplotlib/plotly figures
  reports/          # Generated PDF/HTML reports
config/             # YAML/JSON configuration files
```

## Data Pipeline

1. `load_dataset(data_dir)` — scans `data/raw/*.csv`, merges, deduplicates, drops >50%-NaN columns, replaces inf with NaN, fills NaN with column medians, caches to `data/raw/merged.parquet`. Re-uses parquet if it is newer than all CSVs.
2. `preprocess(df, task, use_smote, sample_size)` — maps raw labels via `LABEL_MAP`, drops rare classes (Infiltration, Heartbleed mapped to -1), optionally does binary vs. multiclass (`task`), stratified sampling (`sample_size`), StandardScaler (saved to `models/scaler.pkl`), optional SMOTE on train set only.

## Label Schema

`LABEL_MAP` and `CLASS_NAMES` are exported from `src/data/preprocess.py` and must be the single source of truth for label encoding throughout the project. 15 raw labels collapse to 6 classes (0–5); labels mapped to -1 are dropped entirely.

## Dataset

CICIDS2017 from Kaggle (`cicdataset/cicids2017`). Place all CSV files in `data/raw/`. CSVs use `cp1252` encoding and have whitespace-padded column names — both are handled in `loader.py`.
