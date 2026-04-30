from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import CLASS_NAMES, LABEL_MAP, _normalize_label

PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _apply_label_map(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Label"] = df["Label"].astype(str).apply(_normalize_label).map(LABEL_MAP)
    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = df["Label"].astype(int)
    df = df[df["Label"] != -1]
    return df


def get_class_distribution(df: pd.DataFrame) -> dict:
    df = _apply_label_map(df)
    counts = df["Label"].value_counts()
    # Sorted by class number (0→5), skip classes with 0 records
    return {
        CLASS_NAMES[label]: int(counts.get(label, 0))
        for label in sorted(CLASS_NAMES.keys())
        if counts.get(label, 0) > 0
    }


def get_dataset_stats(df: pd.DataFrame) -> dict:
    # Single mapping pass — never pass already-mapped df into _apply_label_map again
    df = _apply_label_map(df)
    total = len(df)
    counts = df["Label"].value_counts()
    benign_count = int(counts.get(0, 0))
    attack_count = total - benign_count
    n_features = len([c for c in df.columns if c != "Label"])
    # Include all 6 classes; 0-record classes show as 0 rather than being absent
    records_per_class = {
        CLASS_NAMES[label]: int(counts.get(label, 0))
        for label in sorted(CLASS_NAMES.keys())
    }
    return {
        "total_records": total,
        "n_features": n_features,
        "n_classes": df["Label"].nunique(),
        "benign_count": benign_count,
        "attack_count": attack_count,
        "benign_pct": round(benign_count / total * 100, 2) if total else 0,
        "attack_pct": round(attack_count / total * 100, 2) if total else 0,
        "records_per_class": records_per_class,
    }


def plot_class_distribution(df: pd.DataFrame) -> Path:
    df = _apply_label_map(df)
    counts = df["Label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["class_name"] = counts["label"].map(CLASS_NAMES)
    counts.sort_values("label", inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sns.barplot(data=counts, x="class_name", y="count", ax=ax)
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height()):,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("Class Distribution — CICIDS2017")
    ax.set_ylabel("Number of Records (log scale)")
    ax.set_xlabel("")
    ax.set_yscale("log")
    plt.xticks(rotation=15)
    plt.tight_layout()

    out = PLOTS_DIR / "eda_class_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_feature_correlation(df: pd.DataFrame, top_n: int = 20) -> Path:
    df = df.drop(columns=["Label"], errors="ignore").select_dtypes(include="number")
    variances = df.var().nlargest(top_n)
    df = df[variances.index]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", center=0, ax=ax)
    plt.tight_layout()

    out = PLOTS_DIR / "eda_correlation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_feature_distribution(df: pd.DataFrame, feature_name: str) -> Path:
    df = _apply_label_map(df)
    out = PLOTS_DIR / f"eda_feature_{feature_name}.png"

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for label, class_name in CLASS_NAMES.items():
            subset = df[df["Label"] == label][feature_name].dropna()
            if len(subset) > 5000:
                subset = subset.sample(5000, random_state=42)
            fig.add_trace(go.Histogram(x=subset, name=class_name, opacity=0.6))
        fig.update_layout(
            title=f"Distribution of {feature_name} by Class",
            barmode="overlay",
        )
        fig.write_image(str(out))
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        for label, class_name in CLASS_NAMES.items():
            subset = df[df["Label"] == label][feature_name].dropna()
            if len(subset) > 5000:
                subset = subset.sample(5000, random_state=42)
            ax.hist(subset, bins=50, alpha=0.6, label=class_name)
        ax.set_title(f"Distribution of {feature_name} by Class")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    return out


def get_feature_stats(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    df = _apply_label_map(df)
    rows = []
    for label, class_name in CLASS_NAMES.items():
        subset = df[df["Label"] == label][feature_name].dropna()
        if len(subset) == 0:
            continue
        rows.append({
            "Class": class_name,
            "Mean": subset.mean(),
            "Std": subset.std(),
            "Min": subset.min(),
            "Max": subset.max(),
        })
    return pd.DataFrame(rows)
