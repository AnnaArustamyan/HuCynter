from sklearn.tree import DecisionTreeClassifier


def get_model(hyperparams: dict):
    return DecisionTreeClassifier(
        max_depth=hyperparams["max_depth"],
        min_samples_split=hyperparams["min_samples_split"],
        criterion=hyperparams["criterion"],
        random_state=42,
    )


def get_default_hyperparams() -> dict:
    return {"max_depth": 10, "min_samples_split": 2, "criterion": "gini"}


def get_hyperparam_schema() -> list:
    return [
        {"name": "max_depth", "type": "int", "min": 1, "max": 50, "step": 1, "default": 10},
        {"name": "min_samples_split", "type": "int", "min": 2, "max": 20, "step": 1, "default": 2},
        {"name": "criterion", "type": "select", "options": ["gini", "entropy"], "default": "gini"},
    ]
