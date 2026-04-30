import streamlit as st
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.logistic_regression import get_default_hyperparams as lr_defaults, get_hyperparam_schema as lr_schema
from src.models.decision_tree import get_default_hyperparams as dt_defaults, get_hyperparam_schema as dt_schema
from src.models.random_forest import get_default_hyperparams as rf_defaults, get_hyperparam_schema as rf_schema
from src.models.xgboost_model import get_default_hyperparams as xgb_defaults, get_hyperparam_schema as xgb_schema
from src.models.lightgbm_model import get_default_hyperparams as lgbm_defaults, get_hyperparam_schema as lgbm_schema
from src.models.neural_network import get_default_hyperparams as nn_defaults, get_hyperparam_schema as nn_schema

CONFIG_FILE = PROJECT_ROOT / 'config' / 'last_run.json'
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)


st.header("Model Configuration")

task = st.session_state.get('task', 'multiclass')
use_smote = st.session_state.get('use_smote', True)
sample_label = st.session_state.get('sample_label', '200K')

st.info(f"""
**Current global settings** (change in sidebar):
Task: `{task}` | SMOTE: `{'ON' if use_smote else 'OFF'}` | Sample size: `{sample_label}`
""")

st.divider()

MODELS = {
    'logistic_regression': {
        'display': 'Logistic Regression',
        'icon': 'show_chart',
        'defaults': lr_defaults,
        'schema': lr_schema,
    },
    'decision_tree': {
        'display': 'Decision Tree',
        'icon': 'account_tree',
        'defaults': dt_defaults,
        'schema': dt_schema,
    },
    'random_forest': {
        'display': 'Random Forest',
        'icon': 'forest',
        'defaults': rf_defaults,
        'schema': rf_schema,
    },
    'xgboost': {
        'display': 'XGBoost',
        'icon': 'bolt',
        'defaults': xgb_defaults,
        'schema': xgb_schema,
    },
    'lightgbm': {
        'display': 'LightGBM',
        'icon': 'rocket_launch',
        'defaults': lgbm_defaults,
        'schema': lgbm_schema,
    },
    'neural_network': {
        'display': 'Neural Network',
        'icon': 'hub',
        'defaults': nn_defaults,
        'schema': nn_schema,
    },
}

for key, meta in MODELS.items():
    if f'include_{key}' not in st.session_state:
        st.session_state[f'include_{key}'] = True
    if f'hyperparams_{key}' not in st.session_state:
        st.session_state[f'hyperparams_{key}'] = meta['defaults']()

for model_key, meta in MODELS.items():
    col1, col2 = st.columns([0.05, 0.95])
    with col1:
        included = st.checkbox(
            "",
            value=st.session_state.get(f'include_{model_key}', True),
            key=f'cb_{model_key}',
        )
        st.session_state[f'include_{model_key}'] = included

    with col2:
        label = f"{meta['display']}"
        if not included:
            label += " *(excluded)*"
        with st.expander(label, expanded=False):
            schema = meta['schema']()
            current = st.session_state.get(
                f'hyperparams_{model_key}', meta['defaults']()
            )
            new_params = {}

            for param in schema:
                pname = param['name']
                ptype = param['type']
                default_val = current.get(pname, param['default'])

                if ptype == 'int':
                    new_params[pname] = st.slider(
                        pname,
                        min_value=int(param['min']),
                        max_value=int(param['max']),
                        value=int(default_val),
                        step=int(param['step']),
                        key=f'{model_key}_{pname}',
                    )
                elif ptype == 'float':
                    new_params[pname] = st.slider(
                        pname,
                        min_value=float(param['min']),
                        max_value=float(param['max']),
                        value=float(default_val),
                        step=float(param['step']),
                        format="%.4f",
                        key=f'{model_key}_{pname}',
                    )
                elif ptype == 'select':
                    options = param['options']
                    current_val = default_val
                    idx = options.index(current_val) if current_val in options else 0
                    new_params[pname] = st.selectbox(
                        pname,
                        options=options,
                        index=idx,
                        key=f'{model_key}_{pname}',
                    )

            st.session_state[f'hyperparams_{model_key}'] = new_params

            if st.button("Reset to defaults", key=f'reset_{model_key}'):
                st.session_state[f'hyperparams_{model_key}'] = meta['defaults']()
                st.rerun()

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("Save Configuration", type="primary"):
        config = {
            'task': task,
            'use_smote': use_smote,
            'sample_label': sample_label,
            'models': {}
        }
        for model_key in MODELS:
            config['models'][model_key] = {
                'include': st.session_state.get(f'include_{model_key}', True),
                'hyperparams': st.session_state.get(
                    f'hyperparams_{model_key}',
                    MODELS[model_key]['defaults']()
                ),
            }
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
        st.success(f"Configuration saved to {CONFIG_FILE}")

with col2:
    if st.button("Load Last Configuration"):
        if CONFIG_FILE.exists():
            try:
                config = json.loads(CONFIG_FILE.read_text())
                st.session_state['task'] = config.get('task', 'multiclass')
                st.session_state['use_smote'] = config.get('use_smote', True)
                st.session_state['sample_label'] = config.get('sample_label', '200K')
                for model_key, mdata in config.get('models', {}).items():
                    st.session_state[f'include_{model_key}'] = mdata.get('include', True)
                    st.session_state[f'hyperparams_{model_key}'] = mdata.get('hyperparams', {})
                st.success("Configuration loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load config: {e}")
        else:
            st.warning("No saved configuration found.")

st.divider()

st.subheader("Training Summary")
selected = [MODELS[k]['display']
            for k in MODELS
            if st.session_state.get(f'include_{k}', True)]
if selected:
    st.success(f"**{len(selected)} model(s) selected:** {', '.join(selected)}")
    if sample_label == 'Full':
        st.warning(
            "Full dataset selected. Random Forest and LightGBM may take "
            "10-30 minutes to train. Consider using 200K or 500K for faster runs."
        )
else:
    st.error("No models selected. Please include at least one model.")
