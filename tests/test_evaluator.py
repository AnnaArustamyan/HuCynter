"""Tests for src/training/evaluator.py correctness fixes."""

import numpy as np
import pytest

from src.training.evaluator import _get_labels_for_task, evaluate


class _MockModel:
    """Minimal model mock with predict and predict_proba."""

    def __init__(self, classes, predictions, probas):
        self.classes_ = np.array(classes)
        self._predictions = np.array(predictions)
        self._probas = np.array(probas)

    def predict(self, X):
        return self._predictions[: len(X)]

    def predict_proba(self, X):
        return self._probas[: len(X)]


class TestLabelsForTask:
    def test_binary_labels(self):
        assert _get_labels_for_task("binary") == [0, 1]

    def test_multiclass_labels(self):
        labels = _get_labels_for_task("multiclass")
        assert labels == [0, 1, 2, 3, 4, 5]


class TestEvaluateConfusionMatrixShape:
    def test_multiclass_cm_always_6x6(self):
        n = 20
        y_test = np.array([0, 1] * 10)
        y_pred = np.array([0, 1] * 10)
        probas = np.zeros((n, 2))
        probas[np.arange(n), y_pred] = 1.0

        model = _MockModel(classes=[0, 1], predictions=y_pred, probas=probas)
        feature_names = [f"f{i}" for i in range(5)]
        X_test = np.random.randn(n, 5)

        result = evaluate(model, X_test, y_test, feature_names, "TestModel", "multiclass")
        cm = result["confusion_matrix"]
        assert len(cm) == 6
        assert all(len(row) == 6 for row in cm)

    def test_binary_cm_always_2x2(self):
        n = 10
        y_test = np.array([0] * 5 + [1] * 5)
        y_pred = np.array([0] * 5 + [1] * 5)
        probas = np.zeros((n, 2))
        probas[np.arange(n), y_pred] = 1.0

        model = _MockModel(classes=[0, 1], predictions=y_pred, probas=probas)
        result = evaluate(
            model, np.random.randn(n, 3), y_test,
            ["a", "b", "c"], "TestBinary", "binary",
        )
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)
