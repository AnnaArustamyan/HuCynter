import streamlit as st
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Cyber Risk Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🛡️ Cyber Risk Detection")
    st.caption("ML-based Network Intrusion Detection")
    st.divider()

    # Dataset status check
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
    if csv_files:
        st.success(f"✅ {len(csv_files)} CSV file(s) found")
    else:
        st.error("❌ No dataset found in data/raw/")

    st.divider()
    st.subheader("⚙️ Global Settings")

    task = st.radio(
        "Classification Task",
        options=['binary', 'multiclass'],
        format_func=lambda x: 'Binary (Normal vs Attack)'
                               if x == 'binary'
                               else 'Multiclass (6 classes)',
        index=1 if st.session_state.get('task', 'multiclass') == 'multiclass' else 0,
    )
    st.session_state['task'] = task

    use_smote = st.toggle(
        "SMOTE (class balancing)",
        value=st.session_state.get('use_smote', True),
    )
    st.session_state['use_smote'] = use_smote

    sample_options = {
        '100K': 100_000,
        '200K': 200_000,
        '500K': 500_000,
        'Full': None,
    }
    sample_label = st.select_slider(
        "Training sample size",
        options=list(sample_options.keys()),
        value=st.session_state.get('sample_label', '200K'),
    )
    st.session_state['sample_label'] = sample_label
    st.session_state['sample_size'] = sample_options[sample_label]

    st.divider()
    st.caption("Yerevan State University\nDiploma Project 2025")

st.title("🛡️ Cyber Risk Detection System")
st.markdown("""
Welcome. Use the sidebar to configure settings, then navigate through the pages:

**1. Dataset** → Load and explore CICIDS2017
**2. EDA** → Exploratory data analysis
**3. Configuration** → Select models and hyperparameters
**4. Training** → Train selected models
**5. Results** → Per-model metrics and plots
**6. Comparison** → Side-by-side model comparison
**7. Export** → Download PDF report and plots
""")

if not csv_files:
    st.warning("""
    ⚠️ No dataset found. Please go to the **Dataset** page for download instructions.
    """)
