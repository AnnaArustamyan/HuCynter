from sklearn.neural_network import MLPClassifier


def get_model(hyperparams: dict):
    sizes_str = hyperparams["hidden_layer_sizes"]
    hidden_layer_sizes = tuple(
        int(x) for x in sizes_str.strip("()").split(",") if x.strip()
    )
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=hyperparams["max_iter"],
        learning_rate_init=hyperparams["learning_rate_init"],
        activation=hyperparams["activation"],
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )


def get_default_hyperparams() -> dict:
    return {
        "hidden_layer_sizes": "(128,64)",
        "max_iter": 200,
        "learning_rate_init": 0.001,
        "activation": "relu",
    }


def get_hyperparam_schema() -> list:
    return [
        {
            "name": "hidden_layer_sizes",
            "type": "select",
            "options": ["(64,)", "(128,)", "(64,64)", "(128,64)", "(128,128)"],
            "default": "(128,64)",
        },
        {"name": "max_iter", "type": "int", "min": 100, "max": 500, "step": 50, "default": 200},
        {
            "name": "learning_rate_init",
            "type": "float",
            "min": 0.0001,
            "max": 0.01,
            "step": 0.0001,
            "default": 0.001,
        },
        {
            "name": "activation",
            "type": "select",
            "options": ["relu", "tanh"],
            "default": "relu",
        },
    ]
