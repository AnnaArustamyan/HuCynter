import sys
import argparse
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_dataset
from src.training.trainer import train_model, save_result

MODEL_MODULES = {
    "logistic_regression": "src.models.logistic_regression",
    "decision_tree":       "src.models.decision_tree",
    "random_forest":       "src.models.random_forest",
    "xgboost":             "src.models.xgboost_model",
    "lightgbm":            "src.models.lightgbm_model",
    "neural_network":      "src.models.neural_network",
}

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_ALL = [
    "logistic_regression", "decision_tree", "random_forest",
    "xgboost", "lightgbm", "neural_network",
]
MODELS_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "decision_tree":       "Decision Tree",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
    "neural_network":      "Neural Network",
}

parser = argparse.ArgumentParser(description="Train cyber-risk ML models from the terminal.")
parser.add_argument("--models", nargs="+", choices=MODELS_ALL, default=MODELS_ALL,
                    metavar="MODEL", help="Models to train (default: all)")
parser.add_argument("--task", choices=["binary", "multiclass"], default="multiclass",
                    help="Classification task (default: multiclass)")
parser.add_argument("--sample", type=int, default=None,
                    metavar="N", help="Total rows to sample (default: full dataset)")
parser.add_argument("--no-smote", dest="no_smote", action="store_true",
                    help="Disable SMOTE oversampling")
parser.add_argument("--ensemble", action="store_true",
                    help="Run ensemble training after base models")
parser.add_argument("--feature-selection", dest="feature_selection", action="store_true",
                    help="Run feature selection analysis")
parser.add_argument("--learning-curves", dest="learning_curves", action="store_true",
                    help="Run learning curve analysis")
parser.add_argument("--all-analysis", dest="all_analysis", action="store_true",
                    help="Run all analyses (ensemble + feature-selection + learning-curves)")


def _make_callback(idx: int, total: int, display_name: str):
    def callback(step: str, percent: float):
        bar_width = 20
        filled = int(percent * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{idx}/{total}] {display_name} [{bar}] {percent*100:5.1f}%  {step:<30}",
              end="", flush=True)
    return callback


def _print_table(results: list) -> None:
    if not results:
        return

    col_w = [23, 10, 11, 10, 10, 9]
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Time"]

    def row_str(cells):
        return "│ " + " │ ".join(str(c).ljust(w) for c, w in zip(cells, col_w)) + " │"

    top    = "┌" + "┬".join("─" * (w + 2) for w in col_w) + "┐"
    mid    = "├" + "┼".join("─" * (w + 2) for w in col_w) + "┤"
    bottom = "└" + "┴".join("─" * (w + 2) for w in col_w) + "┘"

    print("\n" + top)
    print(row_str(headers))
    print(mid)
    for r in results:
        raw_name = r.get("model_name", "")
        name = MODELS_DISPLAY.get(raw_name, raw_name)
        cells = [
            name[:col_w[0]],
            f"{r['accuracy']:.4f}",
            f"{r.get('precision_macro', 0):.4f}",
            f"{r.get('recall_macro', 0):.4f}",
            f"{r['f1_macro']:.4f}",
            f"{r['training_time_seconds']:.1f}s",
        ]
        print(row_str(cells))
    print(bottom)


if __name__ == "__main__":
    args = parser.parse_args()
    use_smote = not args.no_smote
    selected = args.models
    n = len(selected)
    run_ensemble = args.ensemble or args.all_analysis
    run_fs = args.feature_selection or args.all_analysis
    run_lc = args.learning_curves or args.all_analysis

    print(f"Task      : {args.task}")
    print(f"SMOTE     : {'ON' if use_smote else 'OFF'}")
    print(f"Sample    : {args.sample if args.sample else 'full dataset'}")
    print(f"Models    : {', '.join(MODELS_DISPLAY[m] for m in selected)}")
    print()

    data_dir = PROJECT_ROOT / "data" / "raw"
    print("Loading dataset...", flush=True)
    df = load_dataset(data_dir)
    print()

    results = []
    for i, model_key in enumerate(selected, start=1):
        display = MODELS_DISPLAY[model_key]
        print(f"[{i}/{n}] Training {display}...")
        callback = _make_callback(i, n, display)

        try:
            module = importlib.import_module(MODEL_MODULES[model_key])
            hyperparams = module.get_default_hyperparams()
            result = train_model(
                model_name=model_key,
                hyperparams=hyperparams,
                task=args.task,
                use_smote=use_smote,
                sample_size=args.sample,
                progress_callback=callback,
            )
            print()
            print(
                f"[{i}/{n}] {display} ✅ Done — "
                f"Accuracy: {result['accuracy']:.4f}, "
                f"F1: {result['f1_macro']:.4f}, "
                f"Time: {result['training_time_seconds']:.1f}s"
            )
            results.append(result)
        except Exception as exc:
            print()
            print(f"[{i}/{n}] {display} ❌ Error: {exc}")

    _print_table(results)

    # ── Ensemble analysis ────────────────────────────────────────────────────
    if run_ensemble and results:
        print("\n── Ensemble Analysis ──────────────────────────────────────────")
        import joblib
        from src.models.ensemble import train_and_evaluate_ensembles
        from src.data.preprocess import preprocess

        print("Preprocessing for ensemble (reusing dataset)...")
        X_train, X_test, y_train, y_test, feature_names = preprocess(
            df, args.task, use_smote, args.sample
        )

        # Load trained models from disk
        models_dir = PROJECT_ROOT / "models"
        trained_models, model_names = [], []
        for model_key in selected:
            pkl = models_dir / f"{model_key}_{args.task}.pkl"
            if pkl.exists():
                trained_models.append(joblib.load(pkl))
                model_names.append(model_key)

        if len(trained_models) < 2:
            print("Need at least 2 trained models for ensemble — skipping.")
        else:
            print(f"Building ensembles from {len(trained_models)} models...")
            try:
                ens_results = train_and_evaluate_ensembles(
                    X_train, X_test, y_train, y_test,
                    trained_models, feature_names, model_names, args.task,
                )
                ens_rows = []
                for key, er in ens_results.items():
                    model_obj = er.pop("_model", None)
                    if model_obj is not None:
                        joblib.dump(model_obj, models_dir / f"{key}_{args.task}.pkl")
                    save_result({**er, "task": args.task, "model_key": key,
                                 "hyperparams": {}, "use_smote": use_smote,
                                 "sample_size": args.sample,
                                 "roc_auc": None, "classification_report": "",
                                 "feature_importances": None,
                                 "timestamp": __import__("datetime").datetime.now().isoformat()})
                    ens_rows.append(er)
                _print_table(ens_rows)
            except Exception as exc:
                print(f"Ensemble failed: {exc}")

    # ── Feature selection ────────────────────────────────────────────────────
    if run_fs:
        print("\n── Feature Selection ──────────────────────────────────────────")
        from src.analysis.feature_selection import run_feature_selection
        from src.data.preprocess import preprocess

        print("Preprocessing for feature selection...")
        X_train, X_test, y_train, y_test, feature_names = preprocess(
            df, args.task, use_smote, args.sample
        )
        try:
            fs = run_feature_selection(X_train, X_test, y_train, y_test, feature_names, args.task)
            opt = fs["optimal_n"]
            opt_acc = fs["results_by_n"][opt]["accuracy"]
            print(f"Optimal features: {opt} (accuracy: {opt_acc*100:.2f}%)")
            print("Top 10 features:")
            for rank, feat in enumerate(fs["top_20_features"][:10], 1):
                score = fs["feature_importances"][feat]
                print(f"  {rank:2d}. {feat} ({score:.4f})")
            print(f"Plot saved to: {fs['plot_path']}")
        except Exception as exc:
            print(f"Feature selection failed: {exc}")

    # ── Learning curves ──────────────────────────────────────────────────────
    if run_lc:
        print("\n── Learning Curves ────────────────────────────────────────────")
        from src.analysis.learning_curves import run_learning_curves

        try:
            lc = run_learning_curves(df, args.task, use_smote)
            sizes = sorted(k for k in lc["decision_tree"].keys())

            col_w2 = [10, 14, 14]
            hdr = ["Sample", "DT Accuracy", "LGBM Accuracy"]

            def row2(cells):
                return "│ " + " │ ".join(str(c).ljust(w) for c, w in zip(cells, col_w2)) + " │"

            top2    = "┌" + "┬".join("─" * (w + 2) for w in col_w2) + "┐"
            mid2    = "├" + "┼".join("─" * (w + 2) for w in col_w2) + "┤"
            bottom2 = "└" + "┴".join("─" * (w + 2) for w in col_w2) + "┘"

            print(top2)
            print(row2(hdr))
            print(mid2)
            for s in sizes:
                dt_acc  = lc["decision_tree"].get(s, {}).get("accuracy", 0)
                lgbm_acc = lc["lightgbm"].get(s, {}).get("accuracy", 0)
                print(row2([f"{s:,}", f"{dt_acc:.4f}", f"{lgbm_acc:.4f}"]))
            print(bottom2)
            print(f"Plot saved to: {lc['plot_path']}")
        except Exception as exc:
            print(f"Learning curves failed: {exc}")
