# Cyber Risk Detection using Machine Learning
## Yerevan State University — Diploma Project 2025

This project implements a full ML pipeline for network intrusion detection using the CICIDS2017 dataset. It trains and compares five classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM) on labeled network flow records, supporting both binary (benign vs. attack) and multiclass (6-class) tasks. A Streamlit web interface provides end-to-end interaction: dataset loading, exploratory analysis, model configuration, training with live progress, results visualization, and PDF report export.

---

## Models

| Model | Type | Key Strength | Expected F1 (multiclass) | Training Speed |
|---|---|---|---|---|
| Logistic Regression | Linear | Fast, interpretable baseline | ~0.82 | Very fast (< 30s) |
| Decision Tree | Tree | Highly interpretable, no scaling needed | ~0.92 | Fast (< 10s) |
| Random Forest | Ensemble | Robust, strong feature importances | ~0.97 | Medium (1–5 min) |
| XGBoost | Gradient Boosting | High accuracy, handles imbalance well | ~0.98 | Medium (1–3 min) |
| LightGBM | Gradient Boosting | Fastest boosting, low memory usage | ~0.98 | Fast (30s–2 min) |

---

## Dataset — CICIDS2017

- **Source:** Canadian Institute for Cybersecurity, University of New Brunswick
- **URL:** https://www.kaggle.com/datasets/cicdataset/cicids2017
- **Size:** ~2.8 million network flow records
- **Features:** 78 numerical features per flow (packet lengths, IAT, flags, etc.)
- **Encoding:** CSV files use `cp1252` (Windows-1252) encoding

### Class Distribution

| Class | Label | Approx. Records | Attack Category |
|---|---|---|---|
| Benign | 0 | 2,273,097 | Normal traffic |
| DoS/DDoS | 1 | 380,688 | DoS Hulk, GoldenEye, slowloris, Slowhttptest, DDoS |
| Reconnaissance | 2 | 158,930 | PortScan |
| Brute Force | 3 | 13,835 | FTP-Patator, SSH-Patator |
| Botnet | 4 | 1,966 | Bot |
| Web Attack | 5 | 2,180 | SQLi, XSS, Brute Force |

> **Dropped classes:** Heartbleed (11 records) and Infiltration (36 records) are excluded — insufficient samples for reliable stratified training.

---

## Download Dataset

### Option A — Kaggle CLI
```bash
pip install kaggle
# 1. Go to https://www.kaggle.com/settings → API → Create New Token
# 2. Place downloaded kaggle.json in ~/.kaggle/kaggle.json
# 3. Run:
kaggle datasets download cicdataset/cicids2017
unzip cicids2017.zip -d data/raw/
```

### Option B — Browser
1. Go to https://www.kaggle.com/datasets/cicdataset/cicids2017
2. Click Download (requires free Kaggle account)
3. Unzip the archive
4. Place all CSV files into `data/raw/`

---

## Installation & Running

```bash
# Quick start (installs deps + launches app)
bash run.sh

# Manual
pip install -r requirements.txt
streamlit run src/ui/app.py
```

Open http://localhost:8501 in your browser.

---

## Usage Guide

1. **Dataset** — Load the CICIDS2017 CSV files; view class distribution and feature list
2. **EDA** — Explore feature distributions by class, correlation heatmap, imbalance analysis
3. **Configuration** — Select which models to train; tune hyperparameters per model; save/load config
4. **Training** — Start training with live per-model progress bars; results saved automatically
5. **Results** — View confusion matrix, ROC curve, feature importances, and classification report per model
6. **Comparison** — Side-by-side metrics table and charts across all trained models
7. **Export** — Generate a PDF report and download individual plots or `results.json`

Use the **sidebar** to set global settings (binary vs. multiclass, SMOTE on/off, sample size) before training.

---

## Project Structure

```
cyber-risk-ml/
├── src/
│   ├── data/
│   │   ├── loader.py          # CSV → merged.parquet with caching
│   │   ├── preprocess.py      # Label mapping, scaling, SMOTE, train/test split
│   │   └── eda.py             # Class distribution, correlation, feature stats
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── lightgbm_model.py  # Each exports get_model(), get_default_hyperparams(), get_hyperparam_schema()
│   ├── training/
│   │   ├── trainer.py         # train_model(), load_results(), save_result()
│   │   └── evaluator.py       # evaluate(), confusion matrix, ROC, feature importance plots
│   ├── reporting/
│   │   └── pdf_report.py      # ReportLab PDF generation (5 sections)
│   └── ui/
│       ├── app.py             # Streamlit entry point + sidebar global settings
│       └── pages/
│           ├── 1_Dataset.py
│           ├── 2_EDA.py
│           ├── 3_Configuration.py
│           ├── 4_Training.py
│           ├── 5_Results.py
│           ├── 6_Comparison.py
│           └── 7_Export.py
├── data/
│   └── raw/                   # Place CICIDS2017 CSV files here
├── models/                    # Saved .pkl model files and scaler.pkl
├── outputs/
│   ├── plots/                 # Generated PNG plots
│   └── reports/               # Generated PDF reports
├── config/
│   └── last_run.json          # Saved UI configuration
├── requirements.txt
├── run.sh
└── README.md
```

---

## Output Files

| File / Folder | Contents | Format |
|---|---|---|
| `models/*.pkl` | Trained model objects (one per model+task) | joblib pickle |
| `models/scaler.pkl` | Fitted StandardScaler | joblib pickle |
| `outputs/plots/*.png` | Confusion matrices, ROC curves, feature importance, EDA, comparison charts | PNG (150 dpi) |
| `outputs/reports/*.pdf` | Full analysis report with all sections and figures | PDF (A4) |
| `outputs/results.json` | All training run metadata and metrics | JSON array |
| `config/last_run.json` | Last saved UI hyperparameter configuration | JSON |

---

## Citation

If you use the CICIDS2017 dataset, please cite:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on Information Systems Security and Privacy (ICISSP), Porto, Portugal, January 2018.
