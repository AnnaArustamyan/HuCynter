import time
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.preprocess import CLASS_NAMES

PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_soft_voting_model(trained_models: list, model_names: list):
    estimators = [
        (name, model)
        for name, model in zip(model_names, trained_models)
        if hasattr(model, "predict_proba")
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


def get_stacking_model(trained_models: list, model_names: list):
    estimators = list(zip(model_names, trained_models))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        passthrough=False,
    )


def _evaluate_ensemble(model, X_test, y_test, task: str) -> dict:
    y_pred = model.predict(X_test)
    if task == "binary":
        all_labels = [0, 1]
    else:
        all_labels = sorted(CLASS_NAMES.keys())
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=all_labels).tolist(),
    }


def _plot_cm(cm_list, model_name: str, task: str, accuracy: float) -> Path:
    import numpy as np
    cm = np.array(cm_list)
    labels = (
        ["Benign", "Attack"]
        if task == "binary"
        else [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    )
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_title("Raw Counts")
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_title("Normalized")
    fig.suptitle(f"{model_name} Confusion Matrix | Accuracy: {accuracy:.4f} | Task: {task}")
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_")
    out = PLOTS_DIR / f"{safe_name}_{task}_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def train_and_evaluate_ensembles(
    X_train,
    X_test,
    y_train,
    y_test,
    trained_models: list,
    feature_names: list,
    model_names: list,
    task: str,
    progress_callback=None,
) -> dict:
    results = {}

    def _cb(step: str, pct: float):
        if progress_callback is not None:
            progress_callback(step, pct)

    # Soft Voting
    _cb("Soft Voting: fitting", 0.35)
    sv_model = get_soft_voting_model(trained_models, model_names)
    t0 = time.time()
    sv_model.fit(X_train, y_train)
    sv_time = time.time() - t0
    _cb("Soft Voting: evaluating", 0.5)
    sv_metrics = _evaluate_ensemble(sv_model, X_test, y_test, task)
    _cb("Soft Voting: plotting confusion matrix", 0.6)
    sv_cm_path = _plot_cm(sv_metrics["confusion_matrix"], "Soft_Voting", task, sv_metrics["accuracy"])
    results["soft_voting"] = {
        "model_name": "Soft Voting Ensemble",
        "accuracy": sv_metrics["accuracy"],
        "precision_macro": sv_metrics["precision_macro"],
        "recall_macro": sv_metrics["recall_macro"],
        "f1_macro": sv_metrics["f1_macro"],
        "training_time_seconds": round(sv_time, 2),
        "confusion_matrix": sv_metrics["confusion_matrix"],
        "plot_paths": {"confusion_matrix": str(sv_cm_path)},
        "_model": sv_model,
    }

    # Stacking
    _cb("Stacking: fitting (cv=3)", 0.65)
    st_model = get_stacking_model(trained_models, model_names)
    t0 = time.time()
    st_model.fit(X_train, y_train)
    st_time = time.time() - t0
    _cb("Stacking: evaluating", 0.8)
    st_metrics = _evaluate_ensemble(st_model, X_test, y_test, task)
    _cb("Stacking: plotting confusion matrix", 0.83)
    st_cm_path = _plot_cm(st_metrics["confusion_matrix"], "Stacking", task, st_metrics["accuracy"])
    results["stacking"] = {
        "model_name": "Stacking Ensemble",
        "accuracy": st_metrics["accuracy"],
        "precision_macro": st_metrics["precision_macro"],
        "recall_macro": st_metrics["recall_macro"],
        "f1_macro": st_metrics["f1_macro"],
        "training_time_seconds": round(st_time, 2),
        "confusion_matrix": st_metrics["confusion_matrix"],
        "plot_paths": {"confusion_matrix": str(st_cm_path)},
        "_model": st_model,
    }

    return results
