# CLAUDE.md — Comprehensive Project Reference

This file provides exhaustive guidance to Claude AI when working with this repository. It covers every folder, file, function, constant, and convention in detail.

## Project Purpose

Cyber risk detection ML pipeline built as a diploma thesis (Yerevan State University, 2025). Uses the CICIDS2017 network intrusion dataset to classify network traffic as **benign** or one of **five attack types**: DoS/DDoS, Reconnaissance, Brute Force, Botnet, Web Attack.

The project provides:
- A full training/evaluation pipeline for 6 ML models + 2 ensemble methods
- A Streamlit UI with 8 pages (dataset exploration through prediction)
- A pretrained model registry for inference without retraining
- PDF report generation with comparative analysis
- Feature selection and learning curve analysis tools
- A CLI script (`train_all.py`) for headless training

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

# CLI training (all models, multiclass, 200K sample, with SMOTE)
python train_all.py --task multiclass --sample 200000

# CLI training (specific models, no SMOTE, with ensemble)
python train_all.py --models xgboost lightgbm --task binary --no-smote --ensemble

# CLI training with all analyses
python train_all.py --all-analysis
```

## Dependencies (`requirements.txt`)

| Package | Min Version | Purpose |
|---------|-------------|---------|
| pandas | >=2.2.0 | Data loading, manipulation |
| numpy | >=2.0.0 | Numerical operations |
| scikit-learn | >=1.5.0 | ML models, metrics, preprocessing, ensembles |
| xgboost | >=2.1.0 | XGBoost classifier |
| lightgbm | >=4.4.0 | LightGBM classifier |
| imbalanced-learn | >=0.13.0 | SMOTE oversampling |
| matplotlib | >=3.9.0 | Static plots (confusion matrices, ROC, feature importance) |
| seaborn | >=0.13.2 | Heatmaps and styled plots |
| plotly | >=5.22.0 | Interactive plots in Streamlit UI |
| streamlit | >=1.35.0 | Web UI framework |
| reportlab | >=4.2.0 | PDF report generation |
| joblib | >=1.4.2 | Model serialization (.pkl) |
| kaggle | >=1.6.14 | Dataset download CLI |
| tqdm | >=4.66.4 | Progress bars during CSV loading |
| pyarrow | >=16.0.0 | Parquet read/write |

---

## Complete Directory Structure

```
cyber-risk-ml/
├── CLAUDE.md                          # This file — AI guidance
├── README.md                          # Human-facing project overview
├── requirements.txt                   # Pinned minimum dependency versions
├── run.sh                             # Setup + launch script (pip install, streamlit run)
├── train_all.py                       # CLI training entry point
│
├── config/
│   └── last_run.json                  # Last UI training configuration (persisted by page 3)
│
├── data/
│   └── raw/
│       ├── *.csv                      # CICIDS2017 day CSVs (8 files, ~2.8M rows total)
│       └── merged.parquet             # Auto-generated cache (rebuilt if CSVs are newer)
│
├── models/                            # Trained model artifacts
│   ├── scaler.pkl                     # StandardScaler fitted during preprocessing
│   ├── logistic_regression_multiclass.pkl
│   ├── decision_tree_multiclass.pkl
│   ├── random_forest_multiclass.pkl
│   ├── xgboost_multiclass.pkl
│   ├── lightgbm_multiclass.pkl
│   ├── neural_network_multiclass.pkl
│   ├── soft_voting_multiclass.pkl     # Ensemble
│   └── stacking_multiclass.pkl        # Ensemble
│
├── pretrained_models/                 # Pretrained model packages for inference
│   ├── registry.json                  # Model registry (JSON array of entries)
│   ├── feature_names.json             # JSON array of 78 feature name strings
│   ├── scaler.pkl                     # Scaler for pretrained models
│   └── *.pkl                          # Pretrained model files
│
├── outputs/
│   ├── results.json                   # Append-only JSON array of all training run records
│   ├── plots/                         # Generated figures (PNG, 150 DPI)
│   │   ├── {ModelName}_{task}_confusion_matrix.png
│   │   ├── {ModelName}_{task}_roc.png
│   │   ├── {ModelName}_feature_importance.png
│   │   ├── eda_class_distribution.png
│   │   ├── eda_correlation.png
│   │   ├── eda_feature_{name}.png
│   │   ├── comparison_metrics.png
│   │   ├── comparison_time.png
│   │   ├── feature_selection_curve.png
│   │   └── learning_curves.png
│   └── reports/
│       └── report_{task}_{timestamp}.pdf
│
├── src/
│   ├── __init__.py                    # Empty package marker
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # CSV loading → merged.parquet caching
│   │   ├── preprocess.py              # Label mapping, split, scaling, SMOTE
│   │   └── eda.py                     # Exploratory data analysis functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py     # LogisticRegression wrapper
│   │   ├── decision_tree.py           # DecisionTreeClassifier wrapper
│   │   ├── random_forest.py           # RandomForestClassifier wrapper
│   │   ├── xgboost_model.py           # XGBClassifier wrapper
│   │   ├── lightgbm_model.py          # LGBMClassifier wrapper
│   │   ├── neural_network.py          # MLPClassifier wrapper
│   │   ├── ensemble.py                # Soft Voting + Stacking ensemble training
│   │   └── pretrained_registry.py     # Loader for pretrained_models/ registry
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── feature_selection.py       # RF-based importance + LightGBM subset eval
│   │   └── learning_curves.py         # Accuracy vs sample size curves
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Orchestrates load → preprocess → train → eval → save
│   │   └── evaluator.py               # Metrics computation + plot generation
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── pdf_report.py              # ReportLab-based PDF report generation
│   └── ui/
│       ├── __init__.py
│       ├── app.py                     # Streamlit entry point + global sidebar settings
│       └── pages/
│           ├── __init__.py
│           ├── 1_Dataset.py           # Load & explore CICIDS2017
│           ├── 2_EDA.py               # Exploratory data analysis
│           ├── 3_Configuration.py     # Model selection & hyperparameters
│           ├── 4_Training.py          # Train selected models
│           ├── 5_Results.py           # Per-model metrics and plots
│           ├── 6_Comparison.py        # Side-by-side model comparison
│           ├── 7_Export.py            # PDF report + plot downloads
│           └── 8_Prediction.py        # Run predictions on new data
│
└── tests/
    ├── __init__.py
    ├── test_preprocess.py             # Preprocessing edge cases
    ├── test_evaluator.py              # Evaluator metric/CM shape tests
    ├── test_prediction_mapping.py     # Column auto-mapping logic
    ├── test_pretrained_registry.py    # Registry loading + validation
    └── test_ensemble_training.py      # Ensemble minimum-model constraints
```

---

## Data Pipeline (Detailed)

### Step 1: Loading — `src/data/loader.py`

**`load_dataset(data_dir: Path) -> pd.DataFrame`**

1. Scans `data_dir` for `*.csv` files. Raises `FileNotFoundError` if none found.
2. Checks for `merged.parquet` — reuses it if it exists and is newer than ALL CSV files.
3. Otherwise rebuilds: reads each CSV with `encoding="latin-1"` (handles cp1252), strips column whitespace, fixes encoding glitches in `Label` column (replaces `ï¿½` with `-`).
4. Concatenates all DataFrames, deduplicates rows.
5. Drops columns where >50% of values are NaN.
6. Replaces `inf`/`-inf` with NaN in numeric columns, then fills NaN with column medians.
7. Saves to `merged.parquet` for future cache hits.
8. Returns the cleaned DataFrame (~2.8M rows, ~79 columns including `Label`).

**Key constant:** `PROJECT_ROOT` — resolved from file path, 2 levels up.

### Step 2: Preprocessing — `src/data/preprocess.py`

**`preprocess(df, task, use_smote, sample_size) -> (X_train, X_test, y_train, y_test, feature_names)`**

1. Finds the `Label` column case-insensitively (handles whitespace-padded names from CSVs).
2. Normalizes labels via `_normalize_label()` — strips whitespace, converts all dash variants (en-dash, em-dash, cp1252 dashes) to plain hyphens.
3. Maps labels via `LABEL_MAP` (see below). Unmapped labels are warned and dropped. Labels mapped to `-1` (Infiltration, Heartbleed) are dropped.
4. Separates features (`X`) and target (`y`). `feature_names` = all columns except `Label`.
5. **Binary mode:** if `task == "binary"`, collapses `y` to `(y > 0).astype(int)` — 0=Benign, 1=Attack.
6. **Sampling:** if `sample_size` is set, does stratified per-class sampling with a minimum of 6 rows per class. Raises `ValueError` if `sample_size < n_classes * 6`.
7. **Train/test split:** 80/20, stratified by `y`, `random_state=42`.
8. **Scaling:** fits `StandardScaler` on `X_train`, transforms both splits. Saves scaler to `models/scaler.pkl`.
9. **SMOTE:** if enabled and smallest class has ≥6 samples in training set, applies SMOTE with `random_state=42`. Skips with warning if class too small.

### Label Schema — Single Source of Truth

**`LABEL_MAP`** maps 15 raw CICIDS2017 labels to 6 integer classes (plus `-1` for drop):

| Raw Label | Class ID | Class Name |
|-----------|----------|------------|
| BENIGN | 0 | Benign |
| DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, DDoS | 1 | DoS/DDoS |
| PortScan | 2 | Reconnaissance |
| FTP-Patator, SSH-Patator | 3 | Brute Force |
| Bot | 4 | Botnet |
| Web Attack - Brute Force, Web Attack - XSS, Web Attack - Sql Injection | 5 | Web Attack |
| Infiltration, Heartbleed | -1 | (dropped — too few samples) |

**`CLASS_NAMES`** dict: `{0: "Benign", 1: "DoS/DDoS", 2: "Reconnaissance", 3: "Brute Force", 4: "Botnet", 5: "Web Attack"}`

Both `LABEL_MAP` and `CLASS_NAMES` are exported from `src/data/preprocess.py` and **must be the single source of truth** for label encoding throughout the project. All other modules import from here.

---

## Model Modules (Detailed)

Each model module in `src/models/` follows a consistent API pattern with three functions:

### Common API Pattern

```python
def get_model(hyperparams: dict) -> sklearn_estimator:
    """Instantiate and return a configured model."""

def get_default_hyperparams() -> dict:
    """Return sensible defaults for this model."""

def get_hyperparam_schema() -> list[dict]:
    """Return schema for Streamlit UI widgets (sliders, selects, etc.)."""
```

The schema list entries have keys: `name`, `type` (`float`/`int`/`select`/`text`), `default`, `min`, `max`, `step`, `options`.

### `src/models/logistic_regression.py`
- **Model:** `LogisticRegression(multi_class="multinomial")`
- **Hyperparams:** `C` (regularization), `max_iter`, `solver` (lbfgs/liblinear/saga)

### `src/models/decision_tree.py`
- **Model:** `DecisionTreeClassifier`
- **Hyperparams:** `max_depth`, `min_samples_split`, `criterion` (gini/entropy)

### `src/models/random_forest.py`
- **Model:** `RandomForestClassifier(n_jobs=-1)`
- **Hyperparams:** `n_estimators`, `max_depth`, `min_samples_split`

### `src/models/xgboost_model.py`
- **Model:** `XGBClassifier(tree_method="hist", eval_metric="mlogloss")`
- **Task-aware:** reads `hyperparams.get("task")` to set `objective` and `num_class=6` for multiclass
- **Hyperparams:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`

### `src/models/lightgbm_model.py`
- **Model:** `LGBMClassifier(verbose=-1)`
- **Task-aware:** sets multiclass objective + `num_class=6` when `task == "multiclass"`
- **Hyperparams:** `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`

### `src/models/neural_network.py`
- **Model:** `MLPClassifier(early_stopping=True, validation_fraction=0.1)`
- **Hyperparams:** `hidden_layer_sizes` (string like `"(128,64)"`, parsed to tuple), `max_iter`, `learning_rate_init`, `activation` (relu/tanh/logistic)

### `src/models/ensemble.py`

Provides two ensemble methods built on top of already-trained base models:

**`get_soft_voting_model(trained_models, model_names)`**
- Creates `VotingClassifier(voting="soft")` from estimators that have `predict_proba`

**`get_stacking_model(trained_models, model_names)`**
- Creates `StackingClassifier` with `LogisticRegression(max_iter=1000)` as meta-estimator, `cv=3`

**`train_and_evaluate_ensembles(X_train, X_test, y_train, y_test, trained_models, feature_names, model_names, task) -> dict`**
- Fits both soft voting and stacking, evaluates each (accuracy, precision, recall, F1, confusion matrix)
- Generates dual confusion matrix plots (raw counts + normalized) for each
- Returns dict with keys `"soft_voting"` and `"stacking"`, each containing metrics, plot paths, and `_model` (the fitted object)
- Plot files: `Soft_Voting_{task}_confusion_matrix.png`, `Stacking_{task}_confusion_matrix.png`

### `src/models/pretrained_registry.py`

Manages the `pretrained_models/` directory for inference without retraining:

**`load_registry() -> list[dict]`** — reads `registry.json`, attaches `_model_exists`, `_scaler_exists`, `_ready` boolean flags.

**`get_ready_entries(task=None) -> list[dict]`** — filters to entries where model + scaler files exist on disk. Optional task filter.

**`load_feature_names(entry) -> list[str]`** — loads from JSON file path in entry, or falls back to scaler's `feature_names_in_`.

**`load_pretrained_model(entry) -> (model, scaler, feature_names, class_names)`** — loads all artifacts. `class_names` dict has int keys. Raises if files missing.

---

## Training Pipeline (Detailed)

### `src/training/trainer.py`

**Constants:**
- `MODEL_MODULES` — maps model keys to importable module paths (e.g., `"xgboost"` → `"src.models.xgboost_model"`)
- `MODEL_DISPLAY_NAMES` — maps keys to human-readable names (e.g., `"xgboost"` → `"XGBoost"`)
- `ENSEMBLE_DISPLAY_NAMES` — `"soft_voting"` → `"Soft Voting Ensemble"`, `"stacking"` → `"Stacking Ensemble"`
- `MIN_ENSEMBLE_BASE_MODELS = 3` — minimum base models required for ensemble training
- `RESULTS_FILE` = `outputs/results.json`
- `MODELS_DIR` = `models/`

**`load_results() -> list`** — reads `outputs/results.json`, returns `[]` on error or missing file.

**`save_result(result: dict)`** — appends a result dict to `results.json`.

**`train_model(model_name, hyperparams, task, use_smote, sample_size, progress_callback=None) -> dict`**
1. Loads dataset via `loader.load_dataset()`
2. Merges user hyperparams with defaults, adds `task` key
3. Preprocesses data (preprocess returns scaled numpy arrays)
4. Instantiates model via `module.get_model(params)`
5. Fits model, times training
6. Evaluates via `evaluator.evaluate()` — generates metrics + plots
7. Saves model to `models/{model_name}_{task}.pkl`
8. Appends full result dict to `results.json`
9. Calls `progress_callback(step_name, percent)` at each stage if provided
10. Returns result dict with: `model_name`, `model_key`, `task`, `timestamp`, `hyperparams`, `use_smote`, `sample_size`, `training_time_seconds`, `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `roc_auc`, `confusion_matrix`, `classification_report`, `feature_importances`, `plot_paths`

**`train_ensembles(base_model_keys, task, use_smote, sample_size, progress_callback=None) -> list[dict]`**
1. Loads base model `.pkl` files from `models/` directory
2. Raises `ValueError` if fewer than `MIN_ENSEMBLE_BASE_MODELS` (3) are found
3. Reloads + preprocesses the dataset fresh
4. Calls `train_and_evaluate_ensembles()` from `ensemble.py`
5. Saves ensemble models to `models/soft_voting_{task}.pkl` and `models/stacking_{task}.pkl`
6. Appends results to `results.json`

**Important note:** `train_all.py` (CLI) checks for ≥2 models before calling ensemble, but `trainer.train_ensembles` enforces ≥3. This mismatch means the CLI may attempt ensemble with 2 models and get a `ValueError`.

### `src/training/evaluator.py`

**`evaluate(model, X_test, y_test, feature_names, model_name, task) -> dict`**
- Computes: accuracy, macro precision, macro recall, macro F1, classification report
- Confusion matrix always uses canonical label order (`[0,1]` for binary, `[0,1,2,3,4,5]` for multiclass) so shape is always fixed (2x2 or 6x6 even if some classes are absent)
- ROC AUC: binary uses `predict_proba[:, 1]`; multiclass uses one-vs-rest with column alignment (handles models whose `classes_` may differ from canonical order)
- Feature importances: top 20 from `model.feature_importances_` (if available)
- Generates 3 plot types per model:
  - **Confusion matrix** — dual heatmap (raw counts + row-normalized percentages)
  - **ROC curve** — binary AUC curve or multiclass one-vs-rest with per-class AUC
  - **Feature importance** — horizontal bar chart of top 20 features (only for tree-based models)

**`generate_comparison_plots(results) -> dict`**
- Grouped bar chart of accuracy/precision/recall/F1 across all models (y-axis 0.8–1.0)
- Horizontal bar chart of training times
- Saves to `comparison_metrics.png` and `comparison_time.png`

**Plot naming convention:** `{model_name}_{task}_confusion_matrix.png`, `{model_name}_{task}_roc.png`, `{model_name}_feature_importance.png`

---

## Analysis Modules (Detailed)

### `src/analysis/feature_selection.py`

**`run_feature_selection(X_train, X_test, y_train, y_test, feature_names, task) -> dict`**
1. Fits `RandomForestClassifier(n_estimators=100)` to get feature importances
2. Ranks all 78 features by importance
3. Tests subsets of top-N features (N ∈ `[5, 10, 15, 20, 30, 40, 50, 78]`) by training LightGBM on each subset
4. Finds `optimal_n`: smallest N within 0.1% of maximum accuracy
5. Plots accuracy + F1 vs N features with vertical line at optimal
6. Returns: `feature_importances` (all features), `top_20_features`, `results_by_n`, `optimal_n`, `plot_path`

### `src/analysis/learning_curves.py`

**`run_learning_curves(df, task, use_smote) -> dict`**
1. Tests sample sizes: `[10K, 25K, 50K, 100K, 200K, 500K]` (filtered to max 80% of dataset)
2. For each size: preprocesses fresh, trains Decision Tree + LightGBM with default hyperparams
3. Records accuracy, F1, and training time per size per model
4. Plots accuracy vs sample size with 200K vertical marker
5. Returns nested dict `{model_key: {size: {accuracy, f1, time}}, "plot_path": ...}`

---

## Streamlit UI (Detailed)

### Entry Point: `src/ui/app.py`

- Sets page config: title "Cyber Risk Detection", wide layout
- **Sidebar** (persists across all pages):
  - Dataset status (CSV file count)
  - **`task`**: radio — `"binary"` or `"multiclass"` (stored in `st.session_state['task']`)
  - **`use_smote`**: toggle (stored in `st.session_state['use_smote']`)
  - **`sample_size`**: select slider — `100K` / `200K` / `500K` / `Full` (stored as both `sample_label` string and `sample_size` int/None)
- Welcome page lists all 8 sub-pages

### Page 1 — Dataset (`1_Dataset.py`)
- Loads dataset via `load_dataset()`, stores in `st.session_state['df']`
- Shows: total records, feature count, benign/attack split
- Class distribution bar chart (Plotly), class counts table
- Data preview (first rows), full feature list
- If no CSVs found: shows Kaggle download instructions

### Page 2 — EDA (`2_EDA.py`)
- Requires `df` in session state (from page 1)
- Class distribution pie/bar charts
- Per-feature histograms colored by class (binary: Benign vs Attack; multiclass: all 6 classes)
- Feature correlation heatmap (top 20 highest-variance numeric columns)
- Dataset imbalance explanation + SMOTE info

### Page 3 — Configuration (`3_Configuration.py`)
- Per-model expanders with Streamlit widgets generated from `get_hyperparam_schema()`
- Checkboxes to include/exclude each of the 6 models
- **Save/Load** buttons: persists to `config/last_run.json`
- JSON format: `{task, use_smote, sample_label, models: {key: {include, hyperparams}}}`

### Page 4 — Training (`4_Training.py`)
- Requires `df` in session state and model selections from page 3
- Trains each selected model via `train_model()` in a thread with progress callbacks
- Progress bar + status text per model via `st.session_state['_ts']`
- Optional ensemble training after base models complete (if sufficient base models exist)
- Shows completion summary with accuracy/F1/time per model

### Page 5 — Results (`5_Results.py`)
- Reads `outputs/results.json` via `load_results()`
- Filters by current task and selected model
- Per-model display: metrics table, confusion matrix image, ROC curve image, feature importance image
- Expandable sections: hyperparameters, classification report text, training details
- History of all runs for each model (not just latest)

### Page 6 — Comparison (`6_Comparison.py`)
- Shows latest result per model name for the current task
- Styled metrics table (highlights best model per metric)
- Plotly grouped bar charts for metrics and training time
- Button to regenerate comparison plots via `generate_comparison_plots()`
- Ensemble section: shows existing confusion matrix PNGs or triggers `train_ensembles()` from saved base model pickles

### Page 7 — Export (`7_Export.py`)
- Section toggles: include EDA, confusion matrices, ROC curves, feature importances, comparison plots
- Model source radio: local vs pretrained
- Generates PDF via `generate_pdf()`
- Download buttons: last generated PDF, all `outputs/plots/*.png`, `results.json`

### Page 8 — Prediction (`8_Prediction.py`)
- **Model source selection:** local (from `models/`) or pretrained (from `pretrained_models/`)
  - Local: picks from `*.pkl` files in `models/` (excluding `scaler.pkl`), loads matching scaler
  - Pretrained: picks from ready entries in `registry.json`, loads model + scaler + feature names
- **Input methods:**
  - **CSV upload:** auto-maps columns case-insensitively, shows interactive mapping UI for unmatched columns, fills unmapped with 0.0
  - **Manual input:** grouped number inputs organized by `FEATURE_GROUPS` (Flow, Packet Length, Flag, IAT, Active/Idle), "Fill Benign" (zeros) and "Fill Attack" (random) buttons
- **Prediction pipeline:** `_run_prediction()` fills missing features with 0, coerces to numeric, clips to [-1e12, 1e12], fills NaN with 0, scales, predicts
- **Single-row output:** class label, confidence %, probability bar chart per class (green for Benign, red for attacks)
- **Batch output (CSV):** risk summary (total/benign/threats/top threat), class distribution bar chart, threat breakdown table, downloadable predictions CSV with confidence scores

---

## PDF Report Generation (Detailed)

### `src/reporting/pdf_report.py`

**`generate_pdf(task, include_eda, include_confusion, include_roc, include_importance, include_comparison, model_source) -> Path`**

Generates a multi-section A4 PDF using ReportLab:

1. **Title page** — project title, metadata table (date, task, model count, source, dataset, institution)
2. **Dataset Overview** — CICIDS2017 description, class info table (6 classes with approx record counts), optional EDA distribution plot
3. **Results Summary** — table of all models with metrics; best F1 row highlighted in green
4. **Per-Model Analysis** — for each model: hyperparameters table, metrics table, confusion matrix image, ROC curve image, feature importance image (each optional)
5. **Model Comparison** — comparison metrics plot + training time plot
6. **Conclusions** — auto-generated text citing best accuracy, best F1, best recall, fastest model; recommendation table; discussion of class imbalance and SMOTE; future work suggestions

Custom ReportLab styles: `STYLE_TITLE`, `STYLE_SUBTITLE`, `STYLE_H1`, `STYLE_H2`, `STYLE_BODY`, `STYLE_CAPTION`.

Output: `outputs/reports/report_{task}_{timestamp}.pdf`

---

## EDA Module (Detailed)

### `src/data/eda.py`

**`_apply_label_map(df)`** — applies same label mapping as training pipeline (for stats-only use, does not split/scale).

**`get_class_distribution(df) -> dict`** — returns `{class_name: count}`.

**`get_dataset_stats(df) -> dict`** — returns `{total_records, feature_count, benign_count, attack_count, benign_pct, attack_pct, class_counts}`.

**`plot_class_distribution(df) -> Path`** — log-scale bar chart → `eda_class_distribution.png`.

**`plot_feature_correlation(df, top_n=20) -> Path`** — selects top-20 highest-variance numeric columns, heatmap → `eda_correlation.png`.

**`plot_feature_distribution(df, feature_name) -> Path`** — histograms per class (Plotly if available, else matplotlib) → `eda_feature_{name}.png`.

**`get_feature_stats(df, feature_name) -> pd.DataFrame`** — mean/std/min/max per class.

---

## CLI Training Script (Detailed)

### `train_all.py`

Argparse-based CLI with these options:

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | all 6 | Space-separated model keys |
| `--task` | multiclass | `binary` or `multiclass` |
| `--sample` | None (full) | Total rows to sample |
| `--no-smote` | False | Disable SMOTE |
| `--ensemble` | False | Run ensemble after base models |
| `--feature-selection` | False | Run feature selection analysis |
| `--learning-curves` | False | Run learning curve analysis |
| `--all-analysis` | False | Run all analyses |

Prints a formatted table of results. Progress callback shows `[idx/total] ModelName [█████░░░░░] 55.0%  step_name`.

---

## Pretrained Models Registry (Detailed)

### `pretrained_models/registry.json`

JSON array of model entries. Currently contains 5 multiclass models:

| model_id | display_name | accuracy | f1 |
|----------|-------------|----------|-----|
| decision_tree_multiclass | Decision Tree (Multiclass, 200K) | 0.9921 | 0.9856 |
| random_forest_multiclass | Random Forest (Multiclass, 200K) | 0.9964 | 0.9876 |
| xgboost_multiclass | XGBoost (Multiclass, 200K) | 0.9983 | 0.9949 |
| lightgbm_multiclass | LightGBM (Multiclass, 200K) | 0.9985 | 0.9952 |
| neural_network_multiclass | Neural Network (Multiclass, 200K) | 0.9905 | 0.9688 |

### Registry Entry Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | yes | Unique identifier |
| `display_name` | string | yes | Human-readable name shown in UI |
| `task` | string | yes | `binary` or `multiclass` |
| `model_path` | string | yes | Filename of the `.pkl` model in this directory |
| `scaler_path` | string | yes | Filename of the `.pkl` scaler in this directory |
| `feature_names_path` | string | yes | Filename of a JSON list of feature names |
| `class_names` | object | yes | `{"0": "Benign", "1": "DoS/DDoS", ...}` |
| `version` | string | no | Version tag |
| `notes` | string | no | Shown in UI as info banner |

### Adding a New Pretrained Model

1. Train the model (via UI or `train_all.py`).
2. Copy the `.pkl` model file and `scaler.pkl` into `pretrained_models/`.
3. Export feature names: `json.dump(feature_names, open("pretrained_models/feature_names.json", "w"))`.
4. Add an entry to `registry.json` with the correct filenames and class names.
5. The Prediction page (page 8) will auto-detect the new entry.

### `pretrained_models/feature_names.json`

JSON array of 78 string feature names matching CICIDS2017 flow columns, in training order. Shared across all pretrained models.

---

## Test Suite (Detailed)

### `tests/test_preprocess.py`
- Creates synthetic CIC-like DataFrames with known labels
- Tests: `sample_size` too small raises `ValueError`; minimum sample works; full data returns correct size; binary mode yields only classes {0, 1}; SMOTE path with tiny classes (should skip gracefully); `feature_names` correctly excludes the `Label` column

### `tests/test_evaluator.py`
- Tests `_get_labels_for_task()`: binary returns `[0, 1]`, multiclass returns `[0, 1, 2, 3, 4, 5]`
- Tests `evaluate()` with mock models:
  - Multiclass: even if model only predicts 2 classes, CM is always 6×6 (canonical shape)
  - Binary: CM is 2×2
  - Mock model uses `predict()` and `predict_proba()`

### `tests/test_pretrained_registry.py`
- `load_registry()` returns list with correct structure and boolean flags
- Missing `registry.json` returns `[]`
- `get_ready_entries()` filters by task and `_ready` flag
- `load_feature_names()` reads from JSON file; empty fallback works
- `load_pretrained_model()` raises if files are missing (uses tmp paths)

### `tests/test_prediction_mapping.py`
- Duplicates auto-mapping logic from Prediction page
- Tests: exact match, case-insensitive match, whitespace-stripped match
- Tests `_apply_column_mapping()`: fills missing features with 0.0, renames mapped columns

### `tests/test_ensemble_training.py`
- `train_ensembles()` raises `ValueError` if fewer than 3 models found on disk
- `train_and_evaluate_ensembles()` returns dict with `"soft_voting"` and `"stacking"` keys
- Each result has `_model` (fitted estimator) and metrics (accuracy, f1, etc.)
- Validates `MIN_ENSEMBLE_BASE_MODELS == 3`

---

## Configuration (Detailed)

### `config/last_run.json`

Persisted by Configuration page (page 3). Structure:

```json
{
  "task": "multiclass",
  "use_smote": true,
  "sample_label": "200K",
  "models": {
    "logistic_regression": {"include": true, "hyperparams": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}},
    "decision_tree": {"include": true, "hyperparams": {"max_depth": 10, "min_samples_split": 2, "criterion": "gini"}},
    "random_forest": {"include": true, "hyperparams": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 2}},
    "xgboost": {"include": true, "hyperparams": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8}},
    "lightgbm": {"include": true, "hyperparams": {"n_estimators": 100, "max_depth": 7, "learning_rate": 0.05, "num_leaves": 63}},
    "neural_network": {"include": true, "hyperparams": {"hidden_layer_sizes": "(128,64)", "max_iter": 200, "learning_rate_init": 0.001, "activation": "relu"}}
  }
}
```

### `outputs/results.json`

Append-only JSON array. Each entry is a full training result dict:

```json
{
  "model_name": "XGBoost",
  "model_key": "xgboost",
  "task": "multiclass",
  "timestamp": "2026-04-20T16:25:30.123456",
  "hyperparams": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8},
  "use_smote": true,
  "sample_size": 200000,
  "training_time_seconds": 12.34,
  "accuracy": 0.9983,
  "precision_macro": 0.9951,
  "recall_macro": 0.9948,
  "f1_macro": 0.9949,
  "roc_auc": null,
  "confusion_matrix": [[...], [...]],
  "classification_report": "...",
  "feature_importances": {"Destination Port": 0.15, ...},
  "plot_paths": {
    "confusion_matrix": "/.../outputs/plots/XGBoost_multiclass_confusion_matrix.png",
    "roc_curve": "/.../outputs/plots/XGBoost_multiclass_roc.png",
    "feature_importance": "/.../outputs/plots/XGBoost_feature_importance.png"
  }
}
```

Ensemble entries have the same shape but with `roc_auc: null`, `classification_report: ""`, `feature_importances: null`, and only `confusion_matrix` in `plot_paths`.

---

## Dataset

**CICIDS2017** from Kaggle (`cicdataset/cicids2017`). Place all 8 CSV files in `data/raw/`.

CSVs use `cp1252`/`latin-1` encoding and have whitespace-padded column names — both are handled in `loader.py`. The merged dataset has ~2.83 million rows and 79 columns (78 numeric features + 1 Label).

**Approximate class distribution (after label mapping):**

| Class | Records | % of Total |
|-------|---------|------------|
| Benign | 2,273,097 | ~80.3% |
| DoS/DDoS | 380,688 | ~13.5% |
| Reconnaissance | 158,930 | ~5.6% |
| Brute Force | 13,835 | ~0.5% |
| Web Attack | 2,180 | ~0.08% |
| Botnet | 1,966 | ~0.07% |

Dropped: Infiltration (36 records), Heartbleed (11 records).

---

## Key Architectural Patterns

1. **`PROJECT_ROOT`** — every module resolves this from `Path(__file__).resolve().parents[N]`. This is the repo root and all paths are built relative to it.

2. **Model key vs display name** — internally, models use snake_case keys (`"xgboost"`, `"random_forest"`). Display names are Title Case (`"XGBoost"`, `"Random Forest"`). Both mappings live in `trainer.py` as `MODEL_MODULES` and `MODEL_DISPLAY_NAMES`.

3. **Results are append-only** — `results.json` grows with every training run. The UI shows all history and picks the latest per model name for comparison. The PDF report deduplicates by model name (keeps last).

4. **Scaler coupling** — `models/scaler.pkl` is overwritten on every `preprocess()` call. Pretrained models have their own separate scaler in `pretrained_models/scaler.pkl`.

5. **Session state flow** — Streamlit pages share data through `st.session_state`. The `df` DataFrame from page 1 is required by pages 2 and 4. Global settings (`task`, `use_smote`, `sample_size`) are set in the sidebar (`app.py`) and read by all pages.

6. **Feature count** — the pipeline consistently works with 78 numeric features (after dropping the Label column and any >50%-NaN columns during loading).
