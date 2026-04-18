import streamlit as st
import threading
import time
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import train_model, load_results, MODEL_DISPLAY_NAMES

st.header("🚀 Model Training")

MODELS_ORDER = [
    'logistic_regression',
    'decision_tree',
    'random_forest',
    'xgboost',
    'lightgbm',
]

MODELS_DISPLAY = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
}

# --- Config summary ---
task = st.session_state.get('task', 'multiclass')
use_smote = st.session_state.get('use_smote', True)
sample_label = st.session_state.get('sample_label', '200K')
sample_size = st.session_state.get('sample_size', 200_000)

selected_models = [
    k for k in MODELS_ORDER
    if st.session_state.get(f'include_{k}', True)
]

st.info(f"""
**Training configuration:**
- Task: `{task}`
- SMOTE: `{'ON' if use_smote else 'OFF'}`
- Sample size: `{sample_label}`
- Models selected: `{len(selected_models)}` → {', '.join(MODELS_DISPLAY[m] for m in selected_models)}
""")

if not selected_models:
    st.error("❌ No models selected. Go to Configuration page.")
    st.stop()

# Check dataset
if 'df' not in st.session_state:
    st.warning("⚠️ Dataset not loaded. Go to **Dataset** page first.")
    st.stop()

st.divider()

# --- Initialize training state ---
if 'training_active' not in st.session_state:
    st.session_state['training_active'] = False
if 'training_done' not in st.session_state:
    st.session_state['training_done'] = False
if 'training_results' not in st.session_state:
    st.session_state['training_results'] = []
if 'training_log' not in st.session_state:
    st.session_state['training_log'] = []
if 'training_progress' not in st.session_state:
    st.session_state['training_progress'] = {}

# --- Start button ---
if not st.session_state['training_active']:
    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button(
            "🚀 Start Training",
            type="primary",
            disabled=st.session_state['training_active'],
        )
    with col2:
        if st.session_state['training_done']:
            st.success("✅ Previous training run complete.")

    if start_btn:
        # Reset state for new run
        st.session_state['training_active'] = True
        st.session_state['training_done'] = False
        st.session_state['training_results'] = []
        st.session_state['training_log'] = []
        st.session_state['training_progress'] = {
            m: {'step': 'Queued', 'percent': 0.0, 'done': False, 'result': None}
            for m in selected_models
        }

        def run_training():
            for model_key in selected_models:
                hyperparams = st.session_state.get(
                    f'hyperparams_{model_key}', {}
                )

                def make_callback(key):
                    def callback(step, percent):
                        st.session_state['training_progress'][key]['step'] = step
                        st.session_state['training_progress'][key]['percent'] = percent
                        st.session_state['training_log'].append(
                            f"[{key}] {step} ({percent*100:.0f}%)"
                        )
                    return callback

                try:
                    result = train_model(
                        model_name=model_key,
                        hyperparams=hyperparams,
                        task=task,
                        use_smote=use_smote,
                        sample_size=sample_size,
                        progress_callback=make_callback(model_key),
                    )
                    st.session_state['training_progress'][model_key]['done'] = True
                    st.session_state['training_progress'][model_key]['result'] = result
                    st.session_state['training_results'].append(result)
                    st.session_state['training_log'].append(
                        f"[{model_key}] ✅ Done — "
                        f"Accuracy: {result['accuracy']:.4f}, "
                        f"F1: {result['f1_macro']:.4f}, "
                        f"Time: {result['training_time_seconds']:.1f}s"
                    )
                except Exception as e:
                    st.session_state['training_progress'][model_key]['step'] = f'Error: {e}'
                    st.session_state['training_progress'][model_key]['done'] = True
                    st.session_state['training_log'].append(
                        f"[{model_key}] ❌ Error: {e}"
                    )

            st.session_state['training_active'] = False
            st.session_state['training_done'] = True

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        st.rerun()

# --- Live progress display ---
if st.session_state['training_active'] or st.session_state['training_done']:
    st.subheader("Training Progress")

    progress_state = st.session_state.get('training_progress', {})
    n_total = len(selected_models)
    n_done = sum(1 for v in progress_state.values() if v.get('done', False))

    # Overall progress
    st.progress(n_done / n_total if n_total > 0 else 0,
                text=f"Overall: {n_done}/{n_total} models complete")

    st.divider()

    # Per-model cards
    for model_key in selected_models:
        pdata = progress_state.get(model_key, {})
        step = pdata.get('step', 'Queued')
        percent = pdata.get('percent', 0.0)
        done = pdata.get('done', False)
        result = pdata.get('result', None)

        icon = "✅" if (done and result) else ("❌" if (done and not result) else "⏳")
        with st.container():
            st.markdown(f"**{icon} {MODELS_DISPLAY[model_key]}**")
            if not done:
                st.progress(percent, text=f"{step}")
            elif result:
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{result['accuracy']:.4f}")
                c2.metric("F1 Score", f"{result['f1_macro']:.4f}")
                c3.metric("Time", f"{result['training_time_seconds']:.1f}s")
            else:
                st.error(f"Failed: {step}")
        st.divider()

    # Training log
    with st.expander("📋 Training Log", expanded=False):
        log = st.session_state.get('training_log', [])
        st.code('\n'.join(log) if log else 'No log entries yet.')

    # Auto-refresh while active
    if st.session_state['training_active']:
        time.sleep(1)
        st.rerun()

# --- Post-training navigation ---
if st.session_state['training_done'] and st.session_state['training_results']:
    st.divider()
    st.success(f"🎉 Training complete! "
               f"{len(st.session_state['training_results'])} model(s) trained.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 View Results →", type="primary"):
            st.switch_page("pages/5_Results.py")
    with col2:
        if st.button("📈 View Comparison →"):
            st.switch_page("pages/6_Comparison.py")
