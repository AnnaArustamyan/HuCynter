import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import load_results


st.header("Model Results")

all_results = load_results()

if not all_results:
    st.warning("No trained models found. Go to the Training page first.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    available_tasks = sorted(set(r['task'] for r in all_results))
    selected_task = st.selectbox(
        "Task",
        options=available_tasks,
        index=available_tasks.index(
            st.session_state.get('task', 'multiclass')
        ) if st.session_state.get('task', 'multiclass') in available_tasks else 0,
    )

filtered = [r for r in all_results if r['task'] == selected_task]

with col2:
    available_models = list(dict.fromkeys(r['model_name'] for r in filtered))
    selected_model = st.selectbox("Model", options=available_models)

model_results = [
    r for r in filtered if r['model_name'] == selected_model
]
result = model_results[-1]

st.divider()

st.subheader("Performance Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",  f"{result['accuracy']:.4f}")
c2.metric("Precision", f"{result['precision_macro']:.4f}")
c3.metric("Recall",    f"{result['recall_macro']:.4f}")
c4.metric("F1 Score",  f"{result['f1_macro']:.4f}")
auc_val = result.get('roc_auc')
c5.metric("ROC AUC", f"{auc_val:.4f}" if auc_val else "N/A (multiclass)")

st.divider()

plot_paths = result.get('plot_paths', {})

cm_path = plot_paths.get('confusion_matrix')
if cm_path and Path(cm_path).exists():
    st.subheader("Confusion Matrix")
    st.image(cm_path, use_container_width=True)
else:
    st.info("Confusion matrix plot not available.")

roc_path = plot_paths.get('roc_curve')
if roc_path and Path(roc_path).exists():
    st.subheader("ROC Curve")
    st.image(roc_path, use_container_width=True)
elif selected_task == 'binary':
    st.info("ROC curve not available.")

mc_roc_path = plot_paths.get('roc_curve')
if selected_task == 'multiclass' and mc_roc_path and Path(mc_roc_path).exists():
    st.subheader("Multiclass ROC Curves (One-vs-Rest)")
    st.image(mc_roc_path, use_container_width=True)

fi_path = plot_paths.get('feature_importance')
if fi_path and Path(fi_path).exists():
    st.subheader("Feature Importance (Top 20)")
    st.image(fi_path, use_container_width=True)

st.divider()

with st.expander("Hyperparameters Used"):
    hp = result.get('hyperparams', {})
    hp_display = {k: v for k, v in hp.items() if k != 'task'}
    st.json(hp_display)

with st.expander("Full Classification Report"):
    st.code(result.get('classification_report', 'Not available'))

with st.expander("Training Details"):
    details = {
        'Timestamp':     result.get('timestamp', 'N/A'),
        'Training Time': f"{result.get('training_time_seconds', 0):.2f}s",
        'Sample Size':   str(result.get('sample_size', 'Full')),
        'SMOTE':         'ON' if result.get('use_smote') else 'OFF',
        'Task':          result.get('task', 'N/A'),
    }
    for k, v in details.items():
        st.markdown(f"**{k}:** {v}")

fi = result.get('feature_importances')
if fi:
    with st.expander("Feature Importances (Top 20 — table)"):
        fi_df = pd.DataFrame(
            list(fi.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        fi_df['Importance'] = fi_df['Importance'].map('{:.6f}'.format)
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

if len(model_results) > 1:
    st.divider()
    with st.expander(f"Previous runs for {selected_model} ({len(model_results)-1} older)"):
        for r in reversed(model_results[:-1]):
            st.markdown(
                f"**{r['timestamp'][:19]}** — "
                f"Acc: {r['accuracy']:.4f} | "
                f"F1: {r['f1_macro']:.4f} | "
                f"Time: {r['training_time_seconds']:.1f}s"
            )
