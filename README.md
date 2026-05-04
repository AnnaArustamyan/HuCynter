# Cyber Risk Detection using Machine Learning
## Yerevan State University — Diploma Project 2025

This project implements a full ML pipeline for network intrusion detection using the CICIDS2017 dataset. It trains and compares six classifiers and two ensemble methods on labeled network flow records, supporting both binary (benign vs. attack) and multiclass (6-class) tasks. A Streamlit web interface provides end-to-end interaction: dataset loading, exploratory analysis, model configuration, training with live progress, results visualization, prediction on new data, and PDF report export.

---

## Models & Results

Results from a 200K stratified sample, multiclass task, with SMOTE oversampling:

| Model | Type | Accuracy | Precision | Recall | F1 (macro) | Training Time |
|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.9569 | 0.8542 | 0.9666 | 0.8938 | 22.6s |
| Decision Tree | Tree | 0.9921 | 0.9789 | 0.9929 | 0.9856 | 2.9s |
| Random Forest | Ensemble | 0.9964 | 0.9817 | 0.9941 | 0.9876 | 3.9s |
| XGBoost | Gradient Boosting | 0.9983 | 0.9931 | 0.9968 | 0.9949 | 3.7s |
| **LightGBM** | Gradient Boosting | **0.9985** | 0.9940 | 0.9965 | 0.9952 | 11.5s |
| Neural Network | MLP | 0.9905 | 0.9530 | 0.9870 | 0.9688 | 32.8s |
| Soft Voting Ensemble | Ensemble (voting) | 0.9976 | 0.9890 | 0.9956 | 0.9922 | 77.2s |
| **Stacking Ensemble** | Ensemble (stacking) | **0.9985** | **0.9952** | 0.9961 | **0.9956** | 2020.6s |

---

## Dataset — CICIDS2017

- **Source:** Canadian Institute for Cybersecurity, University of New Brunswick
- **URL:** https://www.kaggle.com/datasets/cicdataset/cicids2017
- **Size:** ~2.8 million network flow records
- **Features:** 78 numerical features per flow (packet lengths, IAT, flags, etc.)
- **Encoding:** CSV files use `cp1252` (Windows-1252) encoding

### Class Distribution (after cleaning)

| Class | Label | Records | % of Total | Attack Category |
|---|---|---|---|---|
| Benign | 0 | 2,096,484 | 83.12% | Normal traffic |
| DoS/DDoS | 1 | 321,764 | 12.76% | DoS Hulk, GoldenEye, slowloris, Slowhttptest, DDoS |
| Reconnaissance | 2 | 90,819 | 3.60% | PortScan |
| Brute Force | 3 | 9,152 | 0.36% | FTP-Patator, SSH-Patator |
| Web Attack | 5 | 2,143 | 0.09% | SQLi, XSS, Brute Force |
| Botnet | 4 | 1,953 | 0.08% | Bot |

**Total after cleaning:** 2,522,315 records, 78 features, 6 classes.

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

### CLI Training

Train all models from the terminal without the Streamlit UI:

```bash
# All 6 base models, multiclass, 200K sample, with SMOTE (default)
python train_all.py --task multiclass --sample 200000

# Include ensemble training (Soft Voting + Stacking)
python train_all.py --task multiclass --sample 200000 --ensemble

# Specific models only, binary task, no SMOTE
python train_all.py --models xgboost lightgbm --task binary --no-smote

# Full analysis (ensemble + feature selection + learning curves)
python train_all.py --all-analysis
```

### Thesis Data Export

Generate a comprehensive JSON snapshot with all dataset statistics, preprocessing details, and model metrics:

```bash
python export_thesis_snapshot.py
```

Output: `outputs/thesis_run_snapshot.json`

---

## Usage Guide (Streamlit UI)

1. **Dataset** — Load the CICIDS2017 CSV files; view class distribution and feature list
2. **EDA** — Explore feature distributions by class, correlation heatmap, imbalance analysis
3. **Configuration** — Select which models to train; tune hyperparameters per model; save/load config
4. **Training** — Start training with live per-model progress bars; results saved automatically
5. **Results** — View confusion matrix, ROC curve, feature importances, and classification report per model
6. **Comparison** — Side-by-side metrics table and charts across all trained models
7. **Export** — Generate a PDF report and download individual plots or `results.json`
8. **Prediction** — Run predictions on new data using trained or pretrained models (CSV upload or manual input)

Use the **sidebar** to set global settings (binary vs. multiclass, SMOTE on/off, sample size) before training.

---

## Project Structure

```
cyber-risk-ml/
├── src/
│   ├── data/
│   │   ├── loader.py              # CSV → merged.parquet with caching
│   │   ├── preprocess.py          # Label mapping, scaling, SMOTE, train/test split
│   │   └── eda.py                 # Class distribution, correlation, feature stats
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── neural_network.py      # Each exports get_model(), get_default_hyperparams(), get_hyperparam_schema()
│   │   ├── ensemble.py            # Soft Voting + Stacking ensemble training
│   │   └── pretrained_registry.py # Loader for pretrained_models/ registry
│   ├── analysis/
│   │   ├── feature_selection.py   # RF-based importance + LightGBM subset evaluation
│   │   └── learning_curves.py     # Accuracy vs sample size curves
│   ├── training/
│   │   ├── trainer.py             # train_model(), train_ensembles(), load_results()
│   │   └── evaluator.py           # evaluate(), confusion matrix, ROC, feature importance plots
│   ├── reporting/
│   │   └── pdf_report.py          # ReportLab PDF generation
│   └── ui/
│       ├── app.py                 # Streamlit entry point + sidebar global settings
│       └── pages/
│           ├── 1_Dataset.py
│           ├── 2_EDA.py
│           ├── 3_Configuration.py
│           ├── 4_Training.py
│           ├── 5_Results.py
│           ├── 6_Comparison.py
│           ├── 7_Export.py
│           └── 8_Prediction.py    # Run predictions on new data
├── pretrained_models/             # Pretrained model packages for inference
│   ├── registry.json
│   ├── feature_names.json
│   └── *.pkl
├── data/
│   └── raw/                       # Place CICIDS2017 CSV files here
├── models/                        # Saved .pkl model files and scaler.pkl
├── outputs/
│   ├── plots/                     # Generated PNG plots
│   ├── reports/                   # Generated PDF reports
│   ├── results.json               # All training run metrics (append-only)
│   └── thesis_run_snapshot.json   # Thesis-ready data export
├── tests/                         # Unit tests (pytest)
├── config/
│   └── last_run.json              # Saved UI configuration
├── train_all.py                   # CLI training entry point
├── export_thesis_snapshot.py      # Thesis data JSON exporter
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
| `outputs/thesis_run_snapshot.json` | Comprehensive thesis data export (dataset stats, preprocessing, all model metrics) | JSON |
| `config/last_run.json` | Last saved UI hyperparameter configuration | JSON |

---

## Tests

```bash
python -m pytest              # Run all tests
python -m pytest tests/ -v    # Verbose output
```

---

## Citation

If you use the CICIDS2017 dataset, please cite:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 4th International Conference on Information Systems Security and Privacy (ICISSP), Porto, Portugal, January 2018.
