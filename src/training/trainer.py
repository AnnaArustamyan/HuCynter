import importlib
import json
import time
from datetime import datetime
from pathlib import Path

import joblib

from src.data import loader
from src.data.preprocess import preprocess
from src.training import evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_FILE = PROJECT_ROOT / "outputs" / "results.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MODULES = {
    "logistic_regression": "src.models.logistic_regression",
    "decision_tree": "src.models.decision_tree",
    "random_forest": "src.models.random_forest",
    "xgboost": "src.models.xgboost_model",
    "lightgbm": "src.models.lightgbm_model",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def load_results() -> list:
    if not RESULTS_FILE.exists():
        return []
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_result(result: dict):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = load_results()
    results.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def train_model(
    model_name: str,
    hyperparams: dict,
    task: str,
    use_smote: bool,
    sample_size,
    progress_callback=None,
) -> dict:
    def _cb(step, pct):
        if progress_callback is not None:
            progress_callback(step, pct)

    _cb("Loading dataset", 0.05)
    df = loader.load_dataset(PROJECT_ROOT / "data" / "raw")

    _cb("Preprocessing", 0.15)
    params_for_model = hyperparams.copy()
    params_for_model["task"] = task
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, task, use_smote, sample_size
    )

    _cb("Instantiating model", 0.25)
    module = importlib.import_module(MODEL_MODULES[model_name])
    model = module.get_model(params_for_model)

    _cb("Training", 0.30)
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    _cb("Evaluating", 0.80)
    eval_results = evaluator.evaluate(
        model, X_test, y_test, feature_names,
        MODEL_DISPLAY_NAMES[model_name], task,
    )

    _cb("Saving model", 0.95)
    model_path = MODELS_DIR / f"{model_name}_{task}.pkl"
    joblib.dump(model, model_path)

    result = {
        "model_name": MODEL_DISPLAY_NAMES[model_name],
        "model_key": model_name,
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "hyperparams": hyperparams,
        "use_smote": use_smote,
        "sample_size": sample_size,
        "training_time_seconds": round(training_time, 2),
        "accuracy": eval_results["accuracy"],
        "precision_macro": eval_results["precision_macro"],
        "recall_macro": eval_results["recall_macro"],
        "f1_macro": eval_results["f1_macro"],
        "roc_auc": eval_results["roc_auc"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "classification_report": eval_results["classification_report"],
        "feature_importances": eval_results["feature_importances"],
        "plot_paths": eval_results["plot_paths"],
    }

    save_result(result)
    _cb("Done", 1.0)
    return result
