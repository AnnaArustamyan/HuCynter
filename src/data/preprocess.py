from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    "Web Attack \x96 Brute Force": 5,
    "Web Attack \x96 XSS": 5,
    "Web Attack \x96 Sql Injection": 5,
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
    df["Label"] = df["Label"].astype(str).str.strip()

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
        df_work = X.copy()
        df_work["__label__"] = y.values
        df_sampled = (
            df_work.groupby("__label__")
            .apply(
                lambda x: x.sample(
                    min(len(x), sample_size // n_classes), random_state=42
                )
            )
            .reset_index(drop=True)
        )
        y = df_sampled["__label__"]
        X = df_sampled.drop(columns=["__label__"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
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
