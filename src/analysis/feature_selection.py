import time
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.models.lightgbm_model import get_default_hyperparams as lgbm_defaults, get_model as lgbm_model

PLOTS_DIR = _PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES_GRID = [5, 10, 15, 20, 30, 40, 50, 78]


def run_feature_selection(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names: list,
    task: str,
) -> dict:
    # Step 1: fit RF to get importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    paired = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    feature_importances = {k: float(v) for k, v in paired}
    ranked_features = [k for k, _ in paired]

    n_available = X_train.shape[1]
    grid = [n for n in N_FEATURES_GRID if n <= n_available]
    if n_available not in grid:
        grid.append(n_available)
    grid = sorted(set(grid))

    # Build index map for fast column selection (X arrays are numpy)
    feat_idx = {name: i for i, name in enumerate(feature_names)}

    results_by_n: dict = {}
    for n in grid:
        top_feats = ranked_features[:n]
        cols = [feat_idx[f] for f in top_feats if f in feat_idx]

        X_tr_sub = X_train[:, cols] if hasattr(X_train, "__getitem__") else X_train[:, cols]
        X_te_sub = X_test[:, cols]

        hp = lgbm_defaults()
        hp["task"] = task
        model = lgbm_model(hp)

        t0 = time.time()
        model.fit(X_tr_sub, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_te_sub)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        results_by_n[n] = {"accuracy": acc, "f1": f1, "time": round(elapsed, 2)}

    # Find optimal_n: smallest n within 0.1% of max accuracy
    max_acc = max(v["accuracy"] for v in results_by_n.values())
    optimal_n = min(
        (n for n, v in results_by_n.items() if v["accuracy"] >= max_acc - 0.001),
        default=grid[-1],
    )

    # Plot
    ns = sorted(results_by_n.keys())
    accs = [results_by_n[n]["accuracy"] for n in ns]
    f1s = [results_by_n[n]["f1"] for n in ns]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, accs, marker="o", label="Accuracy", linewidth=2)
    ax.plot(ns, f1s, marker="s", label="F1 Macro", linewidth=2)
    ax.axvline(x=optimal_n, color="red", linestyle="--", alpha=0.7,
               label=f"Optimal N={optimal_n}")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Score")
    ax.set_title("Feature Selection Curve (LightGBM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = PLOTS_DIR / "feature_selection_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "feature_importances": feature_importances,
        "top_20_features": ranked_features[:20],
        "results_by_n": results_by_n,
        "optimal_n": optimal_n,
        "plot_path": str(plot_path),
    }
