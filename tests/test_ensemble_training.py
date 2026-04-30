"""Tests for ensemble training orchestration in trainer.py and ensemble.py."""

import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier

from src.training.trainer import (
    train_ensembles,
    MIN_ENSEMBLE_BASE_MODELS,
)
from src.models.ensemble import train_and_evaluate_ensembles


def _make_fitted_models(n_models, X_train, y_train):
    """Create n pre-fitted DecisionTree classifiers."""
    models = []
    for i in range(n_models):
        m = DecisionTreeClassifier(max_depth=2, random_state=i)
        m.fit(X_train, y_train)
        models.append(m)
    return models


class TestTrainEnsemblesValidation:
    def test_rejects_fewer_than_minimum_base_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.training.trainer.MODELS_DIR", tmp_path)
        import joblib
        m = DecisionTreeClassifier(max_depth=1)
        m.fit(np.random.randn(10, 2), np.array([0]*5 + [1]*5))
        joblib.dump(m, tmp_path / "xgboost_multiclass.pkl")

        with pytest.raises(ValueError, match="at least"):
            train_ensembles(
                base_model_keys=["xgboost"],
                task="multiclass",
                use_smote=False,
                sample_size=None,
            )

    def test_rejects_when_no_pkl_files_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.training.trainer.MODELS_DIR", tmp_path)
        with pytest.raises(ValueError, match="at least"):
            train_ensembles(
                base_model_keys=["a", "b", "c"],
                task="multiclass",
                use_smote=False,
                sample_size=None,
            )


class TestTrainAndEvaluateEnsemblesReturnModels:
    """Verify that ensemble results include fitted model objects."""

    def test_results_contain_model_key(self):
        n, d = 100, 5
        X_train = np.random.randn(n, d)
        X_test = np.random.randn(30, d)
        y_train = np.random.randint(0, 3, n)
        y_test = np.random.randint(0, 3, 30)
        models = _make_fitted_models(3, X_train, y_train)
        names = ["m1", "m2", "m3"]

        results = train_and_evaluate_ensembles(
            X_train, X_test, y_train, y_test,
            models, [f"f{i}" for i in range(d)], names, "multiclass",
        )

        assert "soft_voting" in results
        assert "stacking" in results
        for key in ("soft_voting", "stacking"):
            assert "_model" in results[key], f"{key} missing _model"
            assert results[key]["_model"] is not None
            assert "accuracy" in results[key]
            assert "f1_macro" in results[key]


class TestMinEnsembleBaseModelsConstant:
    def test_minimum_is_at_least_2(self):
        assert MIN_ENSEMBLE_BASE_MODELS >= 2

    def test_minimum_is_3(self):
        assert MIN_ENSEMBLE_BASE_MODELS == 3
