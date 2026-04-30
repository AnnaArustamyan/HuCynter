import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import load_results, train_ensembles, MIN_ENSEMBLE_BASE_MODELS
from src.training.evaluator import generate_comparison_plots


st.header("Model Comparison")

all_results = load_results()

if not all_results:
    st.warning("No trained models found. Go to the Training page first.")
    st.stop()

available_tasks = sorted(set(r['task'] for r in all_results))
selected_task = st.selectbox(
    "Task to compare",
    options=available_tasks,
    index=available_tasks.index(
        st.session_state.get('task', 'multiclass')
    ) if st.session_state.get('task', 'multiclass') in available_tasks else 0,
)

filtered = [r for r in all_results if r['task'] == selected_task]

seen = {}
for r in filtered:
    seen[r['model_name']] = r
latest_results = list(seen.values())

if len(latest_results) < 2:
    st.info(
        f"Only {len(latest_results)} model(s) trained for task `{selected_task}`. "
        "Train at least 2 models to compare."
    )
    st.stop()

st.success(f"Comparing {len(latest_results)} models on `{selected_task}` task.")
st.divider()

st.subheader("Metrics Summary")

metrics_data = []
for r in latest_results:
    row = {
        'Model':     r['model_name'],
        'Accuracy':  round(r['accuracy'], 4),
        'Precision': round(r['precision_macro'], 4),
        'Recall':    round(r['recall_macro'], 4),
        'F1 Score':  round(r['f1_macro'], 4),
        'AUC':       round(r['roc_auc'], 4) if r.get('roc_auc') else None,
        'Time (s)':  round(r['training_time_seconds'], 2),
    }
    metrics_data.append(row)

df_metrics = pd.DataFrame(metrics_data)

def highlight_best(col):
    if col.name in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']:
        is_best = col == col.max()
    elif col.name == 'Time (s)':
        is_best = col == col.min()
    else:
        is_best = [False] * len(col)
    return ['background-color: #d4edda; font-weight: bold'
            if v else '' for v in is_best]

styled = df_metrics.style.apply(highlight_best)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Metrics Comparison")

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = px.colors.qualitative.Set2

fig_metrics = go.Figure()
for i, metric in enumerate(metrics_to_plot):
    fig_metrics.add_trace(go.Bar(
        name=metric,
        x=df_metrics['Model'],
        y=df_metrics[metric],
        marker_color=colors[i],
        text=df_metrics[metric].map('{:.4f}'.format),
        textposition='outside',
    ))

fig_metrics.update_layout(
    barmode='group',
    title=f'Model Metrics Comparison — {selected_task}',
    xaxis_title='Model',
    yaxis_title='Score',
    yaxis=dict(range=[
        max(0, float(df_metrics[metrics_to_plot].min().min()) - 0.05),
        1.05
    ]),
    legend_title='Metric',
    height=500,
)
st.plotly_chart(fig_metrics, use_container_width=True)

st.divider()

st.subheader("Training Time Comparison")

df_time = df_metrics[['Model', 'Time (s)']].sort_values('Time (s)')
fig_time = go.Figure(go.Bar(
    x=df_time['Time (s)'],
    y=df_time['Model'],
    orientation='h',
    marker_color=px.colors.qualitative.Pastel[0],
    text=df_time['Time (s)'].map('{:.1f}s'.format),
    textposition='outside',
))
fig_time.update_layout(
    title='Training Time per Model (lower is better)',
    xaxis_title='Seconds',
    height=350,
)
st.plotly_chart(fig_time, use_container_width=True)

st.divider()

st.subheader("Best Model Summary")

best_acc   = df_metrics.loc[df_metrics['Accuracy'].idxmax()]
best_f1    = df_metrics.loc[df_metrics['F1 Score'].idxmax()]
best_time  = df_metrics.loc[df_metrics['Time (s)'].idxmin()]

st.info(f"""
- **Highest Accuracy:** {best_acc['Model']} — `{best_acc['Accuracy']:.4f}`
- **Highest F1 Score:** {best_f1['Model']} — `{best_f1['F1 Score']:.4f}`
- **Fastest Training:** {best_time['Model']} — `{best_time['Time (s)']:.1f}s`
""")

if selected_task == 'binary' and df_metrics['AUC'].notna().any():
    best_auc = df_metrics.loc[df_metrics['AUC'].idxmax()]
    st.info(f"- **Highest AUC:** {best_auc['Model']} — `{best_auc['AUC']:.4f}`")

st.divider()

st.subheader("Generate Comparison Plots for Export")
st.caption("Saves PNG files to outputs/plots/ for use in PDF report.")

if st.button("Generate Comparison Plots"):
    with st.spinner("Generating plots..."):
        try:
            paths = generate_comparison_plots(latest_results)
            st.success("Comparison plots saved.")
            st.markdown(f"- Metrics chart: `{paths['metrics']}`")
            st.markdown(f"- Time chart: `{paths['time']}`")
        except Exception as e:
            st.error(f"Error generating plots: {e}")

st.divider()

st.subheader("Ensemble Analysis")

PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
existing_cm = list(PLOTS_DIR.glob(f"Soft_Voting_{selected_task}_confusion_matrix.png"))

if existing_cm:
    st.success("Ensemble results already exist for this task.")
    st.image(str(existing_cm[0]), caption="Soft Voting Confusion Matrix",
             use_container_width=True)
    stacking_cm = PLOTS_DIR / f"Stacking_{selected_task}_confusion_matrix.png"
    if stacking_cm.exists():
        st.image(str(stacking_cm), caption="Stacking Confusion Matrix",
                 use_container_width=True)

if st.button("Run Ensemble Analysis", type="primary"):
    models_dir = PROJECT_ROOT / "models"
    base_keys = [
        r.get("model_key", "")
        for r in latest_results
        if (models_dir / f"{r.get('model_key', '')}_{selected_task}.pkl").exists()
    ]

    if len(base_keys) < MIN_ENSEMBLE_BASE_MODELS:
        st.error(
            f"Need at least {MIN_ENSEMBLE_BASE_MODELS} trained models saved to disk. "
            f"Found {len(base_keys)}. Train more models first."
        )
    elif "df" not in st.session_state:
        st.error("Dataset not loaded. Go to Dataset page first.")
    else:
        use_smote = st.session_state.get("use_smote", True)
        sample_size = st.session_state.get("sample_size", None)

        with st.spinner("Running ensemble training..."):
            try:
                ens_results = train_ensembles(
                    base_model_keys=base_keys,
                    task=selected_task,
                    use_smote=use_smote,
                    sample_size=sample_size,
                )
                st.success("Ensemble training complete.")
                for er in ens_results:
                    st.markdown(
                        f"**{er['model_name']}** — "
                        f"Accuracy: `{er['accuracy']:.4f}` | "
                        f"F1: `{er['f1_macro']:.4f}` | "
                        f"Time: `{er['training_time_seconds']:.1f}s`"
                    )
                    cm_img = er.get("plot_paths", {}).get("confusion_matrix")
                    if cm_img and Path(cm_img).exists():
                        st.image(cm_img, use_container_width=True)
                st.rerun()
            except Exception as e:
                st.error(f"Ensemble failed: {e}")
                st.exception(e)
