import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eda import (
    get_class_distribution,
    get_dataset_stats,
    plot_feature_correlation,
    get_feature_stats,
)
from src.data.preprocess import LABEL_MAP, CLASS_NAMES

st.header("📊 Exploratory Data Analysis")

if 'df' not in st.session_state:
    st.warning("⚠️ Dataset not loaded. Please go to **Dataset** page first.")
    st.stop()

df = st.session_state['df']
stats = get_dataset_stats(df)
dist = get_class_distribution(df)

# --- Section 1: Class distribution ---
st.subheader("1. Class Distribution")
col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(
        values=list(dist.values()),
        names=list(dist.keys()),
        title="Class Distribution (Pie)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(
        x=list(dist.keys()),
        y=list(dist.values()),
        title="Class Distribution (Bar)",
        labels={'x': 'Class', 'y': 'Count'},
        color=list(dist.keys()),
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=True,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# --- Section 2: Feature distribution explorer ---
st.subheader("2. Feature Distribution Explorer")

numeric_cols = [c for c in df.columns if c != 'Label'
                and pd.api.types.is_numeric_dtype(df[c])]
selected_feature = st.selectbox("Select a feature to explore:", numeric_cols)

if selected_feature:
    # Apply label mapping for coloring
    df_plot = df.copy()
    df_plot['_label_int'] = df_plot['Label'].map(LABEL_MAP)
    df_plot = df_plot[df_plot['_label_int'] != -1]
    if st.session_state.get('task', 'multiclass') == 'binary':
        df_plot['_class'] = df_plot['_label_int'].apply(
            lambda x: 'Benign' if x == 0 else 'Attack'
        )
    else:
        df_plot['_class'] = df_plot['_label_int'].map(CLASS_NAMES)

    # Sample for speed
    sample = df_plot.groupby('_class').apply(
        lambda x: x.sample(min(len(x), 5000), random_state=42)
    ).reset_index(drop=True)

    fig_hist = px.histogram(
        sample,
        x=selected_feature,
        color='_class',
        barmode='overlay',
        opacity=0.6,
        title=f"Distribution of '{selected_feature}' by Class",
        nbins=50,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Stats table
    feat_stats = get_feature_stats(df, selected_feature)
    st.markdown("**Statistics per class:**")
    st.dataframe(feat_stats, use_container_width=True, hide_index=True)

st.divider()

# --- Section 3: Correlation heatmap ---
st.subheader("3. Feature Correlation Heatmap (Top 20 by Variance)")
with st.spinner("Computing correlation matrix..."):
    corr_path = plot_feature_correlation(df, top_n=20)
st.image(str(corr_path), use_container_width=True)

st.divider()

# --- Section 4: Records per CSV file ---
st.subheader("4. Records per Source File")
if 'file_counts' in st.session_state:
    file_counts = st.session_state['file_counts']
    fig_files = px.bar(
        x=list(file_counts.keys()),
        y=list(file_counts.values()),
        title="Records per CSV File",
        labels={'x': 'File', 'y': 'Records'},
        text_auto=True,
    )
    fig_files.update_xaxes(tickangle=20)
    st.plotly_chart(fig_files, use_container_width=True)
else:
    st.info("File-level counts not available. "
            "Re-load dataset to capture per-file stats.")

st.divider()

# --- Section 5: Class imbalance analysis ---
st.subheader("5. Class Imbalance Analysis")

col1, col2 = st.columns(2)
total = stats['total_records']

with col1:
    st.markdown("**Binary view**")
    binary_data = {
        'Benign': stats['benign_count'],
        'Attack': stats['attack_count'],
    }
    fig_bin = px.bar(
        x=list(binary_data.keys()),
        y=list(binary_data.values()),
        text_auto=True,
        color=list(binary_data.keys()),
        color_discrete_map={'Benign': '#2ecc71', 'Attack': '#e74c3c'},
        title="Binary Class Balance",
    )
    st.plotly_chart(fig_bin, use_container_width=True)

with col2:
    st.markdown("**Multiclass view**")
    fig_multi = px.bar(
        x=list(dist.keys()),
        y=list(dist.values()),
        text_auto=True,
        title="Multiclass Balance",
        color=list(dist.keys()),
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_multi, use_container_width=True)

smote_on = st.session_state.get('use_smote', True)
st.info(f"""
**Why SMOTE matters here:**
- Benign records: {stats['benign_count']:,} ({stats['benign_pct']:.1f}%)
- All attacks: {stats['attack_count']:,} ({stats['attack_pct']:.1f}%)
- Botnet class: ~1,966 records (0.07% of total)
- Web Attack class: ~2,180 records (0.08% of total)

Without balancing, models learn to ignore rare attack classes.
SMOTE generates synthetic minority samples during training.

**Current setting: SMOTE is {'ON ✅' if smote_on else 'OFF ❌'}**
(Change in sidebar)
""")
