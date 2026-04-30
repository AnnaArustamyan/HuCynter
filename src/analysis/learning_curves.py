import importlib
import time
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from src.data.preprocess import preprocess

PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZES = [10_000, 25_000, 50_000, 100_000, 200_000, 500_000]
MODELS_TO_TEST = ["decision_tree", "lightgbm"]
MODEL_MODULES = {
    "decision_tree": "src.models.decision_tree",
    "lightgbm": "src.models.lightgbm_model",
}


def run_learning_curves(df, task: str, use_smote: bool) -> dict:
    n_rows = len(df)
    max_sample = int(n_rows * 0.8)
    sizes = [s for s in SAMPLE_SIZES if s <= max_sample]
    if not sizes:
        sizes = [min(SAMPLE_SIZES)]

    curves: dict = {m: {} for m in MODELS_TO_TEST}

    for size in sizes:
        X_train, X_test, y_train, y_test, _ = preprocess(df, task, use_smote, size)

        for model_key in MODELS_TO_TEST:
            mod = importlib.import_module(MODEL_MODULES[model_key])
            hp = mod.get_default_hyperparams()
            hp["task"] = task
            model = mod.get_model(hp)

            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - t0

            y_pred = model.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            curves[model_key][size] = {"accuracy": acc, "f1": f1, "time": round(elapsed, 2)}

    # Plot
    colors = {"decision_tree": "#2196F3", "lightgbm": "#4CAF50"}
    labels = {"decision_tree": "Decision Tree", "lightgbm": "LightGBM"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_key in MODELS_TO_TEST:
        xs = sorted(curves[model_key].keys())
        ys = [curves[model_key][x]["accuracy"] for x in xs]
        ax.plot(xs, ys, marker="o", label=labels[model_key],
                color=colors[model_key], linewidth=2)

    if 200_000 in sizes:
        ax.axvline(x=200_000, color="gray", linestyle="--", alpha=0.6, label="200K mark")

    ax.set_xlabel("Training Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curves ({task})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS_DIR / "learning_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    result = {m: curves[m] for m in MODELS_TO_TEST}
    result["plot_path"] = str(plot_path)
    return result
