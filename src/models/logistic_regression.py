from sklearn.linear_model import LogisticRegression


def get_model(hyperparams: dict):
    return LogisticRegression(
        C=hyperparams["C"],
        max_iter=hyperparams["max_iter"],
        solver=hyperparams["solver"],
        n_jobs=-1,
    )


def get_default_hyperparams() -> dict:
    return {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}


def get_hyperparam_schema() -> list:
    return [
        {"name": "C", "type": "float", "min": 0.01, "max": 10.0, "step": 0.01, "default": 1.0},
        {"name": "max_iter", "type": "int", "min": 100, "max": 3000, "step": 100, "default": 1000},
        {"name": "solver", "type": "select", "options": ["lbfgs", "liblinear", "saga"], "default": "lbfgs"},
    ]
