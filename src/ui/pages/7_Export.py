import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import load_results
from src.reporting.pdf_report import generate_pdf


st.header("Export")

all_results = load_results()

st.subheader("PDF Report Options")

available_tasks = sorted(set(r['task'] for r in all_results)) if all_results else []

if not available_tasks:
    st.warning("No trained models found. Train models first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    export_task = st.selectbox(
        "Task to include",
        options=available_tasks,
        index=available_tasks.index(
            st.session_state.get('task', 'multiclass')
        ) if st.session_state.get('task', 'multiclass') in available_tasks else 0,
    )
with col2:
    n_models = len(set(
        r['model_name'] for r in all_results if r['task'] == export_task
    ))
    st.metric("Models available for this task", n_models)

model_source = st.radio(
    "Model source to document in report",
    options=["local", "pretrained"],
    format_func=lambda x: "Trained locally (models/)" if x == "local"
                          else "Pretrained (pretrained_models/)",
    horizontal=True,
)

st.markdown("**Report sections:**")
col1, col2 = st.columns(2)
with col1:
    inc_eda        = st.checkbox("Dataset overview & EDA chart", value=True)
    inc_confusion  = st.checkbox("Confusion matrices",           value=True)
    inc_roc        = st.checkbox("ROC curves",                   value=True)
with col2:
    inc_importance = st.checkbox("Feature importance charts",    value=True)
    inc_comparison = st.checkbox("Comparison charts",            value=True)

st.divider()

if st.button("Generate PDF Report", type="primary"):
    with st.spinner("Generating PDF... this may take 10-20 seconds."):
        try:
            pdf_path = generate_pdf(
                task=export_task,
                include_eda=inc_eda,
                include_confusion=inc_confusion,
                include_roc=inc_roc,
                include_importance=inc_importance,
                include_comparison=inc_comparison,
                model_source=model_source,
            )
            st.session_state['last_pdf_path'] = str(pdf_path)
            st.success(f"PDF generated: {pdf_path.name}")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            st.exception(e)

last_pdf = st.session_state.get('last_pdf_path')
if last_pdf and Path(last_pdf).exists():
    with open(last_pdf, 'rb') as f:
        st.download_button(
            label="Download PDF Report",
            data=f.read(),
            file_name=Path(last_pdf).name,
            mime="application/pdf",
            type="primary",
        )

st.divider()

st.subheader("Download Individual Plots")

plots_dir = PROJECT_ROOT / 'outputs' / 'plots'
plot_files = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

if not plot_files:
    st.info("No plots generated yet. Train models first.")
else:
    st.caption(f"{len(plot_files)} plot(s) available")
    cols = st.columns(3)
    for i, plot_path in enumerate(plot_files):
        with cols[i % 3]:
            with open(plot_path, 'rb') as f:
                st.download_button(
                    label=plot_path.stem,
                    data=f.read(),
                    file_name=plot_path.name,
                    mime="image/png",
                    key=f"dl_plot_{i}",
                )

st.divider()

st.subheader("Download Raw Results")
results_file = PROJECT_ROOT / 'outputs' / 'results.json'
if results_file.exists():
    with open(results_file, 'rb') as f:
        st.download_button(
            label="Download results.json",
            data=f.read(),
            file_name="results.json",
            mime="application/json",
        )
else:
    st.info("results.json not found yet.")
