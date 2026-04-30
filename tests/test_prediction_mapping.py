"""Tests for column-mapping helpers used in the Prediction page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest


def _auto_map_columns(upload_cols, expected_cols):
    """Reproduce the mapping logic from 8_Prediction.py for testability."""
    mapping = {}
    norm_upload = {c.strip().lower(): c for c in upload_cols}
    for exp in expected_cols:
        key = exp.strip().lower()
        if key in norm_upload:
            mapping[exp] = norm_upload[key]
    return mapping


def _apply_column_mapping(df, mapping, feature_names):
    reverse = {v: k for k, v in mapping.items()}
    df = df.rename(columns=reverse)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df


class TestAutoMapColumns:
    def test_exact_match(self):
        m = _auto_map_columns(["Flow Duration", "Label"], ["Flow Duration"])
        assert m == {"Flow Duration": "Flow Duration"}

    def test_case_insensitive_match(self):
        m = _auto_map_columns(["flow duration"], ["Flow Duration"])
        assert m == {"Flow Duration": "flow duration"}

    def test_whitespace_stripped(self):
        m = _auto_map_columns([" Flow Duration "], ["Flow Duration"])
        assert m == {"Flow Duration": " Flow Duration "}

    def test_no_match_returns_empty(self):
        m = _auto_map_columns(["Unrelated"], ["Flow Duration"])
        assert m == {}


class TestApplyColumnMapping:
    def test_missing_filled_with_zero(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = _apply_column_mapping(df.copy(), {}, ["A", "B", "C"])
        assert "C" in result.columns
        assert (result["C"] == 0.0).all()

    def test_mapped_columns_renamed(self):
        df = pd.DataFrame({"uploaded_col": [10, 20]})
        mapping = {"Expected": "uploaded_col"}
        result = _apply_column_mapping(df.copy(), mapping, ["Expected"])
        assert "Expected" in result.columns
        assert list(result["Expected"]) == [10, 20]
