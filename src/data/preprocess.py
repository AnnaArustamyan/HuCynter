from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dash variants seen in CICIDS2017 CSVs across different encodings:
# \x96 (cp1252 en-dash), \u2013 (Unicode en-dash), \u2014 (em-dash), plain hyphen
_DASH_VARIANTS = ["\x96", "\u2013", "\u2014", "\u2012", "\u2015"]


def _normalize_label(label: str) -> str:
    """Strip whitespace and normalize all dash variants to a plain hyphen."""
    label = label.strip()
    for dash in _DASH_VARIANTS:
        label = label.replace(dash, "-")
    return label


LABEL_MAP = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "DoS GoldenEye": 1,
    "DoS slowloris": 1,
    "DoS Slowhttptest": 1,
    "DDoS": 1,
    "PortScan": 2,
    "FTP-Patator": 3,
    "SSH-Patator": 3,
    "Bot": 4,
    "Web Attack - Brute Force": 5,
    "Web Attack - XSS": 5,
    "Web Attack - Sql Injection": 5,
    "Infiltration": -1,
    "Heartbleed": -1,
}

CLASS_NAMES = {
    0: "Benign",
    1: "DoS/DDoS",
    2: "Reconnaissance",
    3: "Brute Force",
    4: "Botnet",
    5: "Web Attack",
}


def preprocess(df: pd.DataFrame, task: str, use_smote: bool, sample_size) -> tuple:
    df = df.copy()

    # Locate the label column case-insensitively (guards against whitespace-padded names)
    print(f"[preprocess] columns: {df.columns.tolist()}")
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"),
        None,
    )
    if label_col is None:
        raise ValueError(f"No 'Label' column found. Columns: {df.columns.tolist()}")
    if label_col != "Label":
        print(f"[preprocess] renaming '{label_col}' → 'Label'")
        df = df.rename(columns={label_col: "Label"})

    df["Label"] = df["Label"].astype(str).apply(_normalize_label)

    unmapped = set(df["Label"].unique()) - set(LABEL_MAP.keys())
    if unmapped:
        warnings.warn(f"Unmapped labels will be dropped: {unmapped}")

    df["Label"] = df["Label"].map(LABEL_MAP)
    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = df["Label"].astype(int)

    df = df[df["Label"] != -1]

    feature_names = [c for c in df.columns if c != "Label"]
    X = df[feature_names]
    y = df["Label"]

    if task == "binary":
        y = (y > 0).astype(int)

    if sample_size is not None:
        n_classes = y.nunique()
        min_per_class = 6
        min_sample = n_classes * min_per_class
        if sample_size < min_sample:
            raise ValueError(
                f"sample_size={sample_size} is too small for {n_classes} classes. "
                f"Minimum is {min_sample} ({min_per_class} per class)."
            )
        total = len(y)
        class_counts = y.value_counts()
        sampled_idx = []
        for cls in sorted(y.unique()):
            cls_idx = y[y == cls].index
            proportion = class_counts[cls] / total
            n_take = max(min_per_class, min(len(cls_idx), round(sample_size * proportion)))
            sampled_idx.extend(
                cls_idx.to_series().sample(n_take, random_state=42).index
            )
        X = X.loc[sampled_idx]
        y = y.loc[sampled_idx]

    if len(y) < 10:
        raise ValueError(
            f"Only {len(y)} samples remain after filtering/sampling — too few to split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")

    if use_smote:
        min_class_count = pd.Series(y_train).value_counts().min()
        if min_class_count < 6:
            print(
                f"Warning: smallest class has {min_class_count} samples — skipping SMOTE."
            )
        else:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, feature_names
