from lightgbm import LGBMClassifier


def get_model(hyperparams: dict):
    task = hyperparams.get("task", "binary")
    if task == "multiclass":
        extra = {"objective": "multiclass", "num_class": 6, "metric": "multi_logloss"}
    else:
        extra = {"objective": "binary", "metric": "binary_logloss"}

    return LGBMClassifier(
        n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"],
        learning_rate=hyperparams["learning_rate"],
        num_leaves=hyperparams["num_leaves"],
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
        random_state=42,
        **extra,
    )


def get_default_hyperparams() -> dict:
    return {
        "n_estimators": 100,
        "max_depth": 7,
        "learning_rate": 0.05,
        "num_leaves": 63,
    }


def get_hyperparam_schema() -> list:
    return [
        {"name": "n_estimators", "type": "int", "min": 10, "max": 300, "step": 10, "default": 100},
        {"name": "max_depth", "type": "int", "min": 1, "max": 15, "step": 1, "default": 7},
        {"name": "learning_rate", "type": "float", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.05},
        {"name": "num_leaves", "type": "int", "min": 20, "max": 200, "step": 10, "default": 63},
    ]
