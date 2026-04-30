"""Tests for src/models/pretrained_registry.py."""

import json
import pytest
from pathlib import Path

from src.models.pretrained_registry import (
    load_registry,
    get_ready_entries,
    load_feature_names,
    load_pretrained_model,
    PRETRAINED_DIR,
    REGISTRY_FILE,
)


class TestLoadRegistry:
    def test_returns_list(self):
        entries = load_registry()
        assert isinstance(entries, list)

    def test_entries_have_availability_flags(self):
        entries = load_registry()
        for e in entries:
            assert "_model_exists" in e
            assert "_scaler_exists" in e
            assert "_ready" in e

    def test_missing_registry_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.models.pretrained_registry.REGISTRY_FILE",
            tmp_path / "nonexistent.json",
        )
        assert load_registry() == []


class TestGetReadyEntries:
    def test_filters_by_task(self):
        entries = get_ready_entries(task="multiclass")
        for e in entries:
            assert e["task"] == "multiclass"

    def test_all_returned_are_ready(self):
        entries = get_ready_entries()
        for e in entries:
            assert e["_ready"] is True


class TestLoadFeatureNames:
    def test_returns_list_from_json(self, tmp_path):
        names = ["f1", "f2", "f3"]
        fn_file = tmp_path / "features.json"
        fn_file.write_text(json.dumps(names))

        entry = {"feature_names_path": "features.json", "scaler_path": "x.pkl"}
        import src.models.pretrained_registry as reg
        orig = reg.PRETRAINED_DIR
        reg.PRETRAINED_DIR = tmp_path
        try:
            result = load_feature_names(entry)
            assert result == names
        finally:
            reg.PRETRAINED_DIR = orig

    def test_returns_empty_when_nothing(self, tmp_path):
        entry = {"feature_names_path": "nope.json", "scaler_path": "nope.pkl"}
        import src.models.pretrained_registry as reg
        orig = reg.PRETRAINED_DIR
        reg.PRETRAINED_DIR = tmp_path
        try:
            result = load_feature_names(entry)
            assert result == []
        finally:
            reg.PRETRAINED_DIR = orig


class TestLoadPretrainedModel:
    def test_raises_for_missing_model(self, tmp_path):
        entry = {"model_path": "missing.pkl", "scaler_path": "missing.pkl"}
        import src.models.pretrained_registry as reg
        orig = reg.PRETRAINED_DIR
        reg.PRETRAINED_DIR = tmp_path
        try:
            with pytest.raises(FileNotFoundError):
                load_pretrained_model(entry)
        finally:
            reg.PRETRAINED_DIR = orig
