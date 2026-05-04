"""Export a comprehensive thesis-ready JSON snapshot.

Reads the full CICIDS2017 dataset for ground-truth statistics, reproduces the
preprocessing split used in training, and merges per-model metrics from
outputs/results.json into a single machine-readable file at
outputs/thesis_run_snapshot.json.
"""
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.data.eda import get_dataset_stats, get_class_distribution
from src.data.preprocess import CLASS_NAMES, LABEL_MAP, preprocess

TASK = "multiclass"
SAMPLE_SIZE = 200_000
USE_SMOTE = True

DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_FILE = PROJECT_ROOT / "outputs" / "results.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "thesis_run_snapshot.json"


def _class_counts(y) -> dict:
    c = Counter(np.asarray(y).tolist())
    return {CLASS_NAMES.get(k, str(k)): v for k, v in sorted(c.items())}


def main():
    print("Loading full dataset …")
    df = load_dataset(DATA_DIR)

    dataset_stats = get_dataset_stats(df)
    class_dist = get_class_distribution(df)

    total = dataset_stats["total_records"]
    class_pct = {
        name: round(count / total * 100, 4)
        for name, count in class_dist.items()
    }

    label_map_readable = {}
    for raw_label, class_id in sorted(LABEL_MAP.items(), key=lambda x: (x[1], x[0])):
        name = CLASS_NAMES.get(class_id, "dropped")
        label_map_readable.setdefault(name, []).append(raw_label)

    print("Reproducing preprocessing split …")
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, TASK, USE_SMOTE, SAMPLE_SIZE,
    )

    pre_smote_train_size = SAMPLE_SIZE * 4 // 5
    post_smote_train_size = X_train.shape[0]

    preprocessing_info = {
        "task": TASK,
        "sample_size": SAMPLE_SIZE,
        "use_smote": USE_SMOTE,
        "test_size_fraction": 0.2,
        "random_state": 42,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_rows_before_smote": pre_smote_train_size,
        "train_rows_after_smote": post_smote_train_size,
        "test_rows": X_test.shape[0],
        "train_class_counts_after_smote": _class_counts(y_train),
        "test_class_counts": _class_counts(y_test),
    }

    with open(RESULTS_FILE) as f:
        all_results = json.load(f)

    model_results = []
    for r in all_results:
        entry = {
            "model_name": r["model_name"],
            "model_key": r.get("model_key", ""),
            "task": r.get("task", TASK),
            "accuracy": r["accuracy"],
            "precision_macro": r.get("precision_macro"),
            "recall_macro": r.get("recall_macro"),
            "f1_macro": r["f1_macro"],
            "roc_auc": r.get("roc_auc"),
            "training_time_seconds": r.get("training_time_seconds"),
            "hyperparams": r.get("hyperparams", {}),
            "confusion_matrix": r.get("confusion_matrix"),
            "classification_report": r.get("classification_report", ""),
            "feature_importances_top20": (
                dict(list(r["feature_importances"].items())[:20])
                if r.get("feature_importances") else None
            ),
            "timestamp": r.get("timestamp", ""),
        }
        model_results.append(entry)

    best_acc = max(model_results, key=lambda x: x["accuracy"])
    best_f1 = max(model_results, key=lambda x: x["f1_macro"])
    fastest = min(
        (m for m in model_results if m["training_time_seconds"] is not None),
        key=lambda x: x["training_time_seconds"],
    )

    snapshot = {
        "generated_at": datetime.now().isoformat(),
        "command": f"python train_all.py --task {TASK} --sample {SAMPLE_SIZE} --ensemble",
        "dataset": {
            "name": "CICIDS2017",
            "source": "kaggle.com/datasets/cicdataset/cicids2017",
            "total_records_after_cleaning": dataset_stats["total_records"],
            "n_features": dataset_stats["n_features"],
            "n_classes": dataset_stats["n_classes"],
            "benign_count": dataset_stats["benign_count"],
            "attack_count": dataset_stats["attack_count"],
            "benign_pct": dataset_stats["benign_pct"],
            "attack_pct": dataset_stats["attack_pct"],
            "class_distribution": class_dist,
            "class_distribution_pct": class_pct,
            "label_mapping": label_map_readable,
            "dropped_labels": {
                raw: "too few samples"
                for raw, cid in LABEL_MAP.items() if cid == -1
            },
        },
        "preprocessing": preprocessing_info,
        "models": model_results,
        "highlights": {
            "best_accuracy": {
                "model": best_acc["model_name"],
                "value": best_acc["accuracy"],
            },
            "best_f1": {
                "model": best_f1["model_name"],
                "value": best_f1["f1_macro"],
            },
            "fastest_training": {
                "model": fastest["model_name"],
                "seconds": fastest["training_time_seconds"],
            },
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"Snapshot written to {OUTPUT_FILE}")
    print(f"  Dataset: {snapshot['dataset']['total_records_after_cleaning']:,} records, "
          f"{snapshot['dataset']['n_features']} features, "
          f"{snapshot['dataset']['n_classes']} classes")
    print(f"  Models:  {len(model_results)}")
    print(f"  Best accuracy: {best_acc['model_name']} ({best_acc['accuracy']:.4f})")
    print(f"  Best F1:       {best_f1['model_name']} ({best_f1['f1_macro']:.4f})")
    print(f"  Fastest:       {fastest['model_name']} ({fastest['training_time_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
