"""Loader for pretrained model registry in pretrained_models/."""

import json
from pathlib import Path
from typing import Optional

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models"
REGISTRY_FILE = PRETRAINED_DIR / "registry.json"


def load_registry() -> list[dict]:
    """Return all entries from registry.json, with availability flags."""
    if not REGISTRY_FILE.exists():
        return []
    try:
        entries = json.loads(REGISTRY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    for entry in entries:
        model_path = PRETRAINED_DIR / entry.get("model_path", "")
        scaler_path = PRETRAINED_DIR / entry.get("scaler_path", "")
        entry["_model_exists"] = model_path.is_file()
        entry["_scaler_exists"] = scaler_path.is_file()
        entry["_ready"] = entry["_model_exists"] and entry["_scaler_exists"]
    return entries


def get_ready_entries(task: Optional[str] = None) -> list[dict]:
    """Return only entries whose model + scaler files exist on disk."""
    entries = load_registry()
    ready = [e for e in entries if e["_ready"]]
    if task is not None:
        ready = [e for e in ready if e.get("task") == task]
    return ready


def load_feature_names(entry: dict) -> list[str]:
    """Load feature names for a registry entry.

    Tries the dedicated feature_names JSON first, then falls back to
    scaler.feature_names_in_.
    """
    fn_path = PRETRAINED_DIR / entry.get("feature_names_path", "")
    if fn_path.is_file():
        try:
            return json.loads(fn_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    scaler_path = PRETRAINED_DIR / entry.get("scaler_path", "")
    if scaler_path.is_file():
        try:
            scaler = joblib.load(scaler_path)
            if hasattr(scaler, "feature_names_in_"):
                return list(scaler.feature_names_in_)
        except Exception:
            pass

    return []


def load_pretrained_model(entry: dict):
    """Load and return (model, scaler, feature_names, class_names) tuple."""
    model_path = PRETRAINED_DIR / entry["model_path"]
    scaler_path = PRETRAINED_DIR / entry["scaler_path"]

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = load_feature_names(entry)

    raw_class_names = entry.get("class_names", {})
    class_names = {int(k): v for k, v in raw_class_names.items()}

    return model, scaler, feature_names, class_names
