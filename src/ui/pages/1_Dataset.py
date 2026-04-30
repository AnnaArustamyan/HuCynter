import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.data.eda import get_class_distribution, get_dataset_stats, plot_class_distribution


st.header("Dataset Explorer")

data_dir = PROJECT_ROOT / 'data' / 'raw'
csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []

if not csv_files:
    st.error("No CSV files found in data/raw/")
    st.markdown("""
    ### Download Instructions

    **Option A — Kaggle CLI (recommended):**
```bash
    pip install kaggle
    # Place kaggle.json from https://www.kaggle.com/settings in ~/.kaggle/
    kaggle datasets download cicdataset/cicids2017
    unzip cicids2017.zip -d data/raw/
```

    **Option B — Browser:**
    1. Go to https://www.kaggle.com/datasets/cicdataset/cicids2017
    2. Click Download
    3. Unzip and place all CSV files in `data/raw/`

    After placing files, refresh this page.
    """)
    st.stop()

st.info(f"Found {len(csv_files)} CSV file(s) in data/raw/")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    load_btn = st.button("Load Dataset", type="primary",
                         disabled='df' in st.session_state)
with col2:
    reimport_btn = st.button(
        "Reimport Dataset",
        help="Delete the cached parquet and rebuild from CSVs. Use after updating CSV files.",
    )
with col3:
    if 'df' in st.session_state:
        st.success("Dataset loaded. Scroll down to explore.")

def _load_dataset(force_rebuild: bool = False):
    if force_rebuild:
        parquet = data_dir / "merged.parquet"
        if parquet.exists():
            parquet.unlink()
    df = load_dataset(data_dir)
    st.session_state['df'] = df
    return df

if load_btn or reimport_btn:
    label = "Reimporting" if reimport_btn else "Loading"
    with st.spinner(f"{label} dataset from CSV files… This may take 1-2 minutes."):
        try:
            df = _load_dataset(force_rebuild=reimport_btn)
            st.success(f"Loaded {len(df):,} records with {len(df.columns)} columns.")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

if 'df' not in st.session_state:
    st.stop()

df = st.session_state['df']

stats = get_dataset_stats(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records", f"{stats['total_records']:,}")
c2.metric("Features", stats['n_features'])
c3.metric("Attack Classes", stats['n_classes'])
c4.metric("Benign", f"{stats['benign_pct']:.1f}%")
c5.metric("Attack", f"{stats['attack_pct']:.1f}%")

st.divider()

st.subheader("Class Distribution")
dist = get_class_distribution(df)

import plotly.graph_objects as go
fig = go.Figure(go.Bar(
    x=list(dist.keys()),
    y=list(dist.values()),
    marker_color=['#2ecc71' if k == 'Benign' else '#e74c3c' for k in dist.keys()],
    text=[f"{v:,}" for v in dist.values()],
    textposition='outside',
))
log_scale = st.toggle("Log scale", value=True)
fig.update_layout(
    title="Records per Class",
    xaxis_title="Class",
    yaxis_title="Count",
    yaxis_type="log" if log_scale else "linear",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Records per Class")
total_records = stats['total_records']
class_df = pd.DataFrame([
    {
        'Class': k,
        'Count': f"{v:,}",
        'Percentage': f"{v / total_records * 100:.2f}%" if total_records else "0.00%",
    }
    for k, v in stats['records_per_class'].items()
    if v > 0
])
if class_df.empty:
    st.info("No class distribution available — ensure dataset is loaded correctly.")
else:
    st.dataframe(class_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Data Preview (first 500 rows)")
st.dataframe(df.head(500), use_container_width=True)

st.divider()

st.subheader("Feature List")
feat_df = pd.DataFrame([
    {'Feature': col, 'Type': str(df[col].dtype),
     'Non-null': df[col].notna().sum(),
     'Unique': df[col].nunique()}
    for col in df.columns
])
st.dataframe(feat_df, use_container_width=True, hide_index=True)
