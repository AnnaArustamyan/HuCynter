from sklearn.ensemble import RandomForestClassifier


def get_model(hyperparams: dict):
    return RandomForestClassifier(
        n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"],
        min_samples_split=hyperparams["min_samples_split"],
        random_state=42,
        n_jobs=-1,
    )


def get_default_hyperparams() -> dict:
    return {"n_estimators": 100, "max_depth": 15, "min_samples_split": 2}


def get_hyperparam_schema() -> list:
    return [
        {"name": "n_estimators", "type": "int", "min": 10, "max": 300, "step": 10, "default": 100},
        {"name": "max_depth", "type": "int", "min": 1, "max": 30, "step": 1, "default": 15},
        {"name": "min_samples_split", "type": "int", "min": 2, "max": 20, "step": 1, "default": 2},
    ]
