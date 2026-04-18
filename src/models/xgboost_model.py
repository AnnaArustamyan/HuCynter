from xgboost import XGBClassifier


def get_model(hyperparams: dict):
    task = hyperparams.get("task", "binary")
    if task == "multiclass":
        extra = {"objective": "multi:softmax", "num_class": 6, "eval_metric": "mlogloss"}
    else:
        extra = {"objective": "binary:logistic", "eval_metric": "logloss"}

    return XGBClassifier(
        n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"],
        learning_rate=hyperparams["learning_rate"],
        subsample=hyperparams["subsample"],
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        **extra,
    )


def get_default_hyperparams() -> dict:
    return {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
    }


def get_hyperparam_schema() -> list:
    return [
        {"name": "n_estimators", "type": "int", "min": 10, "max": 300, "step": 10, "default": 100},
        {"name": "max_depth", "type": "int", "min": 1, "max": 15, "step": 1, "default": 6},
        {"name": "learning_rate", "type": "float", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.1},
        {"name": "subsample", "type": "float", "min": 0.5, "max": 1.0, "step": 0.05, "default": 0.8},
    ]
