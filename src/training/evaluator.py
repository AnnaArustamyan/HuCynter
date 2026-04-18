from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.data.preprocess import CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(
    model,
    X_test,
    y_test,
    feature_names: list,
    model_name: str,
    task: str,
) -> dict:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    roc_auc = None
    if task == "binary" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)

    feature_importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        paired = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        feature_importances = {k: float(v) for k, v in paired[:20]}

    cm_path = _plot_confusion_matrix(cm, model_name, task, accuracy)

    roc_path = None
    if task == "binary" and hasattr(model, "predict_proba"):
        roc_path = _plot_roc_binary(model, X_test, y_test, model_name)
    elif task == "multiclass":
        roc_path = _plot_roc_multiclass(model, X_test, y_test, model_name)

    fi_path = None
    if feature_importances is not None:
        fi_path = _plot_feature_importance(feature_importances, model_name)

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": feature_importances,
        "plot_paths": {
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path) if roc_path else None,
            "feature_importance": str(fi_path) if fi_path else None,
        },
    }


def _plot_confusion_matrix(cm, model_name, task, accuracy) -> Path:
    if task == "binary":
        labels = ["Benign", "Attack"]
    else:
        labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_title("Raw Counts")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_title("Normalized")

    fig.suptitle(
        f"{model_name} Confusion Matrix | Accuracy: {accuracy:.4f} | Task: {task}"
    )
    plt.tight_layout()

    out = PLOTS_DIR / f"{model_name}_{task}_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_roc_binary(model, X_test, y_test, model_name) -> Path:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc_val = auc(fpr, tpr)

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve (Binary)")
    ax.legend()

    out = PLOTS_DIR / f"{model_name}_binary_roc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    sns.set_style("white")
    return out


def _plot_roc_multiclass(model, X_test, y_test, model_name) -> Path:
    if not hasattr(model, "predict_proba"):
        return None

    classes = sorted(CLASS_NAMES.keys())
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)
    colors = plt.cm.tab10(range(len(CLASS_NAMES)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, cls in enumerate(classes):
        if y_test_bin[:, i].sum() == 0:
            continue
        if i >= y_score.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i],
                label=f"{CLASS_NAMES[cls]} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — Multiclass ROC (One-vs-Rest)")
    ax.legend(loc="lower right")

    out = PLOTS_DIR / f"{model_name}_multiclass_roc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_feature_importance(feature_importances: dict, model_name: str) -> Path:
    names = list(feature_importances.keys())
    values = list(feature_importances.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names[::-1], values[::-1], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"{model_name} — Top 20 Feature Importances")
    plt.tight_layout()

    out = PLOTS_DIR / f"{model_name}_feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_comparison_plots(results: list) -> dict:
    model_names = [r["model_name"] for r in results]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Metrics")
    ax.legend()
    plt.tight_layout()

    path1 = PROJECT_ROOT / "outputs" / "plots" / "comparison_metrics.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)

    times = [r["training_time_seconds"] for r in results]
    sorted_pairs = sorted(zip(times, model_names))
    sorted_times, sorted_names = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_names, sorted_times)
    for bar, t in zip(bars, sorted_times):
        ax.annotate(
            f"{t:.1f}s",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            ha="left", va="center", fontsize=9,
        )
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Comparison — Training Time")
    plt.tight_layout()

    path2 = PROJECT_ROOT / "outputs" / "plots" / "comparison_time.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)

    return {"metrics": str(path1), "time": str(path2)}
