"""Tests for src/data/preprocess.py edge cases."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess, LABEL_MAP, CLASS_NAMES


def _make_dataset(n_per_class=100):
    """Build a tiny synthetic DataFrame that looks like CICIDS2017."""
    rows = []
    raw_labels = ["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator", "Bot",
                  "Web Attack - Brute Force"]
    for label in raw_labels:
        for _ in range(n_per_class):
            rows.append({
                "Feature1": np.random.randn(),
                "Feature2": np.random.randn(),
                "Feature3": np.random.randn(),
                "Label": label,
            })
    return pd.DataFrame(rows)


class TestPreprocessSampling:
    def test_sample_size_too_small_raises(self):
        df = _make_dataset(n_per_class=50)
        with pytest.raises(ValueError, match="too small"):
            preprocess(df, task="multiclass", use_smote=False, sample_size=5)

    def test_sample_size_minimum_works(self):
        df = _make_dataset(n_per_class=50)
        X_train, X_test, y_train, y_test, feat = preprocess(
            df, task="multiclass", use_smote=False, sample_size=36,
        )
        assert len(y_train) > 0
        assert len(y_test) > 0

    def test_full_dataset_no_sample(self):
        df = _make_dataset(n_per_class=50)
        X_train, X_test, y_train, y_test, feat = preprocess(
            df, task="multiclass", use_smote=False, sample_size=None,
        )
        total = len(y_train) + len(y_test)
        assert total == 6 * 50

    def test_binary_task_reduces_to_two_classes(self):
        df = _make_dataset(n_per_class=50)
        X_train, X_test, y_train, y_test, feat = preprocess(
            df, task="binary", use_smote=False, sample_size=None,
        )
        unique_classes = set(np.unique(y_train)) | set(np.unique(y_test))
        assert unique_classes == {0, 1}

    def test_smote_skipped_for_tiny_classes(self, capsys):
        df = _make_dataset(n_per_class=6)
        X_train, X_test, y_train, y_test, feat = preprocess(
            df, task="multiclass", use_smote=True, sample_size=None,
        )
        assert len(y_train) > 0


class TestPreprocessFeatureNames:
    def test_feature_names_exclude_label(self):
        df = _make_dataset(n_per_class=20)
        _, _, _, _, feat = preprocess(
            df, task="multiclass", use_smote=False, sample_size=None,
        )
        assert "Label" not in feat
        assert "Feature1" in feat
