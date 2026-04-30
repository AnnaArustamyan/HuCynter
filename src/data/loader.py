from pathlib import Path
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/. Please download the CICIDS2017 dataset "
            "from https://www.kaggle.com/datasets/cicdataset/cicids2017 and place all "
            "CSV files in data/raw/"
        )

    parquet_path = data_dir / "merged.parquet"
    if parquet_path.exists():
        parquet_mtime = parquet_path.stat().st_mtime
        if all(parquet_mtime > f.stat().st_mtime for f in csv_files):
            return pd.read_parquet(parquet_path)

    # Delete stale parquet so it gets rebuilt cleanly
    if parquet_path.exists():
        parquet_path.unlink()

    dfs = []
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        tqdm.write(f"  {csv_file.name}")
        df = pd.read_csv(csv_file, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()
        if "Label" in df.columns:
            df["Label"] = (
                df["Label"]
                .astype(str)
                .str.replace("ï¿½", "-", regex=False)
                .str.strip()
            )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Ensure column names are clean before saving (strip any residual whitespace)
    df.columns = df.columns.str.strip()

    df.drop_duplicates(inplace=True)

    threshold = len(df) * 0.5
    df.dropna(axis=1, thresh=int(threshold), inplace=True)

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([float("inf"), float("-inf")], pd.NA)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df.to_parquet(parquet_path, index=False)

    print(
        f"Loaded {len(df):,} records | {len(df.columns)} columns | "
        f"{len(csv_files)} CSV file(s) processed"
    )
    return df
