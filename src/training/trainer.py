import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib

_PROJECT_ROOT_INIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT_INIT))

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
    "neural_network": "src.models.neural_network",
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "neural_network": "Neural Network",
}

ENSEMBLE_DISPLAY_NAMES = {
    "soft_voting": "Soft Voting Ensemble",
    "stacking": "Stacking Ensemble",
}

MIN_ENSEMBLE_BASE_MODELS = 3


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
    module = importlib.import_module(MODEL_MODULES[model_name])
    defaults = module.get_default_hyperparams() if hasattr(module, "get_default_hyperparams") else {}
    params_for_model = {**defaults, **hyperparams, "task": task}
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, task, use_smote, sample_size
    )

    _cb("Instantiating model", 0.25)
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


def train_ensembles(
    base_model_keys: list[str],
    task: str,
    use_smote: bool,
    sample_size,
    progress_callback=None,
) -> list[dict]:
    """Train Soft Voting + Stacking ensembles from already-saved base models.

    Returns a list of result dicts (one per ensemble).
    Raises ValueError if fewer than MIN_ENSEMBLE_BASE_MODELS base models
    have .pkl files on disk.
    """
    from src.models.ensemble import train_and_evaluate_ensembles

    def _cb(step, pct):
        if progress_callback is not None:
            progress_callback(step, pct)

    _cb("Loading base models", 0.05)
    trained_models = []
    loaded_keys = []
    for key in base_model_keys:
        pkl = MODELS_DIR / f"{key}_{task}.pkl"
        if pkl.is_file():
            trained_models.append(joblib.load(pkl))
            loaded_keys.append(key)

    if len(trained_models) < MIN_ENSEMBLE_BASE_MODELS:
        raise ValueError(
            f"Ensemble requires at least {MIN_ENSEMBLE_BASE_MODELS} trained base "
            f"models on disk, but only found {len(trained_models)} "
            f"({', '.join(loaded_keys) or 'none'})."
        )

    _cb("Preprocessing", 0.15)
    df = loader.load_dataset(PROJECT_ROOT / "data" / "raw")
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, task, use_smote, sample_size
    )

    _cb("Training ensembles", 0.30)
    ens_results = train_and_evaluate_ensembles(
        X_train, X_test, y_train, y_test,
        trained_models, feature_names, loaded_keys, task,
        progress_callback=_cb,
    )

    saved = []
    for key, er in ens_results.items():
        _cb(f"Saving {ENSEMBLE_DISPLAY_NAMES.get(key, key)}", 0.85)
        model_obj = er.pop("_model", None)
        if model_obj is not None:
            joblib.dump(model_obj, MODELS_DIR / f"{key}_{task}.pkl")

        result = {
            **er,
            "model_key": key,
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "hyperparams": {},
            "use_smote": use_smote,
            "sample_size": sample_size,
            "roc_auc": None,
            "classification_report": "",
            "feature_importances": None,
        }
        save_result(result)
        saved.append(result)

    _cb("Done", 1.0)
    return saved
