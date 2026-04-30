import streamlit as st
import threading
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import (
    train_model,
    train_ensembles,
    MODEL_DISPLAY_NAMES,
    ENSEMBLE_DISPLAY_NAMES,
    MIN_ENSEMBLE_BASE_MODELS,
)


st.header("Model Training")

MODELS_ORDER = [
    'logistic_regression',
    'decision_tree',
    'random_forest',
    'xgboost',
    'lightgbm',
    'neural_network',
]

MODELS_DISPLAY = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'neural_network': 'Neural Network',
}

if '_ts' not in st.session_state:
    st.session_state['_ts'] = {
        'active': False,
        'done': False,
        'progress': {},
        'results': [],
        'log': [],
    }

_ts: dict = st.session_state['_ts']

task = st.session_state.get('task', 'multiclass')
use_smote = st.session_state.get('use_smote', True)
sample_label = st.session_state.get('sample_label', '200K')
sample_size = st.session_state.get('sample_size', 200_000)

selected_models = [
    k for k in MODELS_ORDER
    if st.session_state.get(f'include_{k}', True)
]

train_ensemble = st.checkbox(
    "Train ensemble models (Soft Voting + Stacking)",
    value=st.session_state.get("train_ensemble", False),
    key="cb_train_ensemble",
)
st.session_state["train_ensemble"] = train_ensemble

if train_ensemble:
    if len(selected_models) < MIN_ENSEMBLE_BASE_MODELS:
        st.error(
            f"Ensemble requires at least {MIN_ENSEMBLE_BASE_MODELS} base models. "
            f"Currently selected: {len(selected_models)}. "
            "Enable more models on the Configuration page."
        )
    else:
        st.warning(
            f"Ensemble training will first train all {len(selected_models)} selected "
            f"base model(s), then build Soft Voting and Stacking ensembles on top. "
            "This may take significantly longer."
        )

st.info(f"""
**Training configuration:**
- Task: `{task}`
- SMOTE: `{'ON' if use_smote else 'OFF'}`
- Sample size: `{sample_label}`
- Models selected: `{len(selected_models)}` — {', '.join(MODELS_DISPLAY[m] for m in selected_models)}
- Ensemble: `{'ON' if train_ensemble else 'OFF'}`
""")

if not selected_models:
    st.error("No models selected. Go to Configuration page.")
    st.stop()

if train_ensemble and len(selected_models) < MIN_ENSEMBLE_BASE_MODELS:
    st.stop()

if 'df' not in st.session_state:
    st.warning("Dataset not loaded. Go to the Dataset page first.")
    st.stop()

st.divider()

if not _ts['active']:
    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("Start Training", type="primary")
    with col2:
        if _ts['done']:
            st.success("Previous training run complete.")

    if start_btn:
        snapshot_hyperparams = {
            k: st.session_state.get(f'hyperparams_{k}', {})
            for k in selected_models
        }
        snapshot_task = task
        snapshot_smote = use_smote
        snapshot_size = sample_size
        snapshot_models = list(selected_models)
        snapshot_ensemble = train_ensemble

        _ts['active'] = True
        _ts['done'] = False
        _ts['results'] = []
        _ts['log'] = []
        progress_init = {
            m: {'step': 'Queued', 'percent': 0.0, 'done': False, 'result': None}
            for m in snapshot_models
        }
        if snapshot_ensemble:
            progress_init["ensemble"] = {
                'step': 'Waiting for base models',
                'percent': 0.0,
                'done': False,
                'result': None,
            }
            for ek in ENSEMBLE_DISPLAY_NAMES:
                progress_init[ek] = {
                    'step': 'Waiting for base models',
                    'percent': 0.0,
                    'done': False,
                    'result': None,
                }
        _ts['progress'] = progress_init

        def run_training(
            models, hyperparams_map, task_, use_smote_, sample_size_,
            run_ensemble, shared,
        ):
            def make_callback(key, state):
                def callback(step, percent):
                    state['progress'][key]['step'] = step
                    state['progress'][key]['percent'] = percent
                    state['log'].append(
                        f"[{key}] {step} ({percent * 100:.0f}%)"
                    )
                return callback

            for model_key in models:
                hyperparams = hyperparams_map.get(model_key, {})
                try:
                    result = train_model(
                        model_name=model_key,
                        hyperparams=hyperparams,
                        task=task_,
                        use_smote=use_smote_,
                        sample_size=sample_size_,
                        progress_callback=make_callback(model_key, shared),
                    )
                    shared['progress'][model_key]['done'] = True
                    shared['progress'][model_key]['result'] = result
                    shared['results'].append(result)
                    shared['log'].append(
                        f"[{model_key}] Done — "
                        f"Accuracy: {result['accuracy']:.4f}, "
                        f"F1: {result['f1_macro']:.4f}, "
                        f"Time: {result['training_time_seconds']:.1f}s"
                    )
                except Exception as e:
                    shared['progress'][model_key]['step'] = f'Error: {e}'
                    shared['progress'][model_key]['done'] = True
                    shared['log'].append(f"[{model_key}] Error: {e}")

            if run_ensemble:
                successfully_trained = [
                    k for k in models
                    if shared['progress'][k].get('result') is not None
                ]
                if len(successfully_trained) < MIN_ENSEMBLE_BASE_MODELS:
                    for ek in ENSEMBLE_DISPLAY_NAMES:
                        shared['progress'][ek]['step'] = (
                            f'Skipped — only {len(successfully_trained)} base '
                            f'model(s) succeeded (need {MIN_ENSEMBLE_BASE_MODELS})'
                        )
                        shared['progress'][ek]['done'] = True
                    shared['log'].append(
                        "[ensemble] Skipped — not enough base models succeeded."
                    )
                else:
                    for ek in ENSEMBLE_DISPLAY_NAMES:
                        shared['progress'][ek]['step'] = 'Queued'
                    try:
                        ens_results = train_ensembles(
                            base_model_keys=successfully_trained,
                            task=task_,
                            use_smote=use_smote_,
                            sample_size=sample_size_,
                            progress_callback=make_callback("ensemble", shared),
                        )
                        for er in ens_results:
                            ek = er['model_key']
                            shared['progress'][ek]['done'] = True
                            shared['progress'][ek]['result'] = er
                            shared['results'].append(er)
                            shared['log'].append(
                                f"[{ek}] Done — "
                                f"Accuracy: {er['accuracy']:.4f}, "
                                f"F1: {er['f1_macro']:.4f}, "
                                f"Time: {er['training_time_seconds']:.1f}s"
                            )
                    except Exception as e:
                        for ek in ENSEMBLE_DISPLAY_NAMES:
                            if not shared['progress'][ek].get('done'):
                                shared['progress'][ek]['step'] = f'Error: {e}'
                                shared['progress'][ek]['done'] = True
                        shared['log'].append(f"[ensemble] Error: {e}")

            shared['active'] = False
            shared['done'] = True

        thread = threading.Thread(
            target=run_training,
            args=(
                snapshot_models,
                snapshot_hyperparams,
                snapshot_task,
                snapshot_smote,
                snapshot_size,
                snapshot_ensemble,
                _ts,
            ),
            daemon=True,
        )
        thread.start()
        st.rerun()

ALL_DISPLAY = {**MODELS_DISPLAY, **ENSEMBLE_DISPLAY_NAMES}

if _ts['active'] or _ts['done']:
    st.subheader("Training Progress")

    progress_state = _ts['progress']
    all_keys = [k for k in progress_state if k != "ensemble"]
    n_total = len(all_keys)
    n_done = sum(1 for k in all_keys if progress_state[k].get('done', False))

    st.progress(
        min(1.0, n_done / n_total) if n_total > 0 else 0.0,
        text=f"Overall: {n_done}/{n_total} models complete",
    )
    st.divider()

    for model_key in all_keys:
        if model_key == "ensemble":
            continue
        pdata = progress_state.get(model_key, {})
        step = pdata.get('step', 'Queued')
        percent = min(1.0, float(pdata.get('percent', 0.0)))
        done = pdata.get('done', False)
        result = pdata.get('result', None)

        display = ALL_DISPLAY.get(model_key, model_key)
        status = "Complete" if (done and result) else ("Failed" if (done and not result) else "Running")
        with st.container():
            st.markdown(f"**{display}** — {status}")
            if not done:
                st.progress(min(1.0, percent), text=step)
            elif result:
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{result['accuracy']:.4f}")
                c2.metric("F1 Score", f"{result['f1_macro']:.4f}")
                c3.metric("Time", f"{result['training_time_seconds']:.1f}s")
            else:
                st.error(f"Failed: {step}")
        st.divider()

    with st.expander("Training Log", expanded=False):
        log = _ts['log']
        st.code('\n'.join(log) if log else 'No log entries yet.')

    if _ts['active']:
        time.sleep(1)
        st.rerun()

if _ts['done'] and _ts['results']:
    st.divider()
    st.success(f"Training complete. {len(_ts['results'])} model(s) trained.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Results", type="primary"):
            st.switch_page("pages/5_Results.py")
    with col2:
        if st.button("View Comparison"):
            st.switch_page("pages/6_Comparison.py")
