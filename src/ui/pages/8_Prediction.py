import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import CLASS_NAMES
from src.models.pretrained_registry import (
    get_ready_entries,
    load_pretrained_model,
    load_feature_names,
    load_registry,
)


st.header("Prediction")
st.caption("Score network traffic against a trained model to detect cyber risks.")

MODELS_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

BINARY_CLASS_NAMES = {0: "Benign", 1: "Attack"}

FEATURE_GROUPS = {
    "Flow Features": [
        "Destination Port", "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
        "Down/Up Ratio", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Fwd Header Length", "Bwd Header Length", "Fwd Header Length.1",
        "Fwd Packets/s", "Bwd Packets/s",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    ],
    "Packet Length Features": [
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Min Packet Length", "Max Packet Length",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    ],
    "Flag Features": [
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "CWE Flag Count", "ECE Flag Count",
    ],
    "IAT Features": [
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
        "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min",
    ],
    "Active/Idle Features": [
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ],
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_feature_names_local():
    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
            if hasattr(scaler, "feature_names_in_"):
                return list(scaler.feature_names_in_)
        except Exception:
            pass
    results_file = PROJECT_ROOT / "outputs" / "results.json"
    if results_file.exists():
        import json
        try:
            results = json.loads(results_file.read_text())
            for r in reversed(results):
                fi = r.get("feature_importances")
                if fi:
                    return list(fi.keys())
        except Exception:
            pass
    return []


def _run_prediction(model, scaler, feature_names: list, X_raw: pd.DataFrame):
    X_raw = X_raw.copy()
    for col in feature_names:
        if col not in X_raw.columns:
            X_raw[col] = 0.0
    X = X_raw[feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    # Keep values in a numerically stable range before scaling.
    X = X.clip(lower=-1e12, upper=1e12)
    X = X.fillna(0.0)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_scaled)
        except Exception:
            pass
    return y_pred, y_proba


def _auto_map_columns(upload_cols: list[str], expected_cols: list[str]) -> dict[str, str]:
    """Case-insensitive, whitespace-stripped auto-mapping of uploaded to expected columns."""
    mapping: dict[str, str] = {}
    norm_upload = {c.strip().lower(): c for c in upload_cols}
    for exp in expected_cols:
        key = exp.strip().lower()
        if key in norm_upload:
            mapping[exp] = norm_upload[key]
    return mapping


def _apply_column_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str],
    feature_names: list[str],
) -> pd.DataFrame:
    """Rename mapped columns and fill missing features with 0."""
    reverse = {v: k for k, v in mapping.items()}
    df = df.rename(columns=reverse)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ── Section 1: Model Source ──────────────────────────────────────────────────

st.subheader("1. Select Model")

local_model_files = sorted(
    f for f in MODELS_DIR.glob("*.pkl") if f.name != "scaler.pkl"
) if MODELS_DIR.exists() else []

pretrained_entries = load_registry()
pretrained_ready = [e for e in pretrained_entries if e.get("_ready")]

has_local = len(local_model_files) > 0 and SCALER_PATH.exists()
has_pretrained = len(pretrained_ready) > 0

if not has_local and not has_pretrained:
    st.warning(
        "No models available. Either train models on the Training page, "
        "or add pretrained model files to the `pretrained_models/` folder."
    )
    st.stop()

source_options = []
if has_local:
    source_options.append("Trained locally (models/)")
if has_pretrained:
    source_options.append("Pretrained (pretrained_models/)")

model_source = st.radio("Model source", options=source_options, horizontal=True)

model = None
scaler = None
feature_names: list[str] = []
class_names_for_task: dict[int, str] = {}
selected_task = "multiclass"

if model_source == "Trained locally (models/)":
    model_labels = []
    for f in local_model_files:
        stem = f.stem
        parts = stem.rsplit("_", 1)
        name = parts[0].replace("_", " ").title() if len(parts) == 2 else stem
        task_tag = parts[1] if len(parts) == 2 else "unknown"
        model_labels.append(f"{name} ({task_tag})")

    selected_label = st.selectbox("Model", options=model_labels)
    selected_idx = model_labels.index(selected_label)
    selected_model_path = local_model_files[selected_idx]
    selected_task = selected_model_path.stem.rsplit("_", 1)[-1]
    class_names_for_task = CLASS_NAMES if selected_task == "multiclass" else BINARY_CLASS_NAMES
    st.caption(f"File: `{selected_model_path.name}` | Task: `{selected_task}`")

    feature_names = _get_feature_names_local()
    if not feature_names:
        st.error("Could not load feature names from scaler.pkl or results.json.")
        st.stop()

    try:
        model = joblib.load(selected_model_path)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Failed to load model or scaler: {e}")
        st.stop()

else:
    pretrained_labels = [
        f"{e['display_name']} (v{e.get('version', '?')})"
        for e in pretrained_ready
    ]
    selected_pt_label = st.selectbox("Pretrained model", options=pretrained_labels)
    selected_pt_idx = pretrained_labels.index(selected_pt_label)
    entry = pretrained_ready[selected_pt_idx]
    selected_task = entry.get("task", "multiclass")
    raw_cn = entry.get("class_names", {})
    class_names_for_task = {int(k): v for k, v in raw_cn.items()} if raw_cn else CLASS_NAMES
    st.caption(
        f"ID: `{entry['model_id']}` | Task: `{selected_task}` | "
        f"Version: `{entry.get('version', '-')}`"
    )
    if entry.get("notes"):
        st.info(entry["notes"])

    feature_names = load_feature_names(entry)
    if not feature_names:
        st.error("No feature names found for this pretrained model.")
        st.stop()

    try:
        model, scaler, _, _ = load_pretrained_model(entry)
    except Exception as e:
        st.error(f"Failed to load pretrained model: {e}")
        st.stop()

    # Show status of all pretrained entries
    not_ready = [e for e in pretrained_entries if not e.get("_ready")]
    if not_ready:
        with st.expander(f"{len(not_ready)} pretrained model(s) not yet available"):
            for e in not_ready:
                missing = []
                if not e.get("_model_exists"):
                    missing.append(e.get("model_path", "?"))
                if not e.get("_scaler_exists"):
                    missing.append(e.get("scaler_path", "?"))
                st.markdown(
                    f"- **{e['display_name']}** — missing: {', '.join(missing)}"
                )

st.divider()

# ── Section 2: Input Data ────────────────────────────────────────────────────

st.subheader("2. Input Data")

input_method = st.radio(
    "Input method",
    options=["Upload CSV file", "Manual input"],
    horizontal=True,
)

X_input: pd.DataFrame | None = None
run_prediction = False

if input_method == "Upload CSV file":
    uploaded = st.file_uploader(
        "Upload a CSV file with network flow features", type="csv",
    )

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        st.markdown(
            f"**Preview** ({len(df_upload):,} rows, {len(df_upload.columns)} columns)"
        )
        st.dataframe(df_upload.head(5), use_container_width=True)

        # Auto-map columns (case-insensitive)
        auto_map = _auto_map_columns(list(df_upload.columns), feature_names)
        matched = set(auto_map.keys())
        unmatched = [f for f in feature_names if f not in matched]

        if matched:
            st.success(f"{len(matched)}/{len(feature_names)} feature columns auto-matched.")

        # Interactive mapping for unmatched columns
        manual_map: dict[str, str] = {}
        if unmatched:
            with st.expander(
                f"Map {len(unmatched)} unmatched feature(s)", expanded=len(unmatched) <= 10
            ):
                st.caption(
                    "Select the uploaded column that corresponds to each expected feature, "
                    "or leave as '-- fill with 0 --' to use a default value."
                )
                upload_col_options = ["-- fill with 0 --"] + sorted(df_upload.columns.tolist())
                for feat in unmatched:
                    chosen = st.selectbox(
                        feat,
                        options=upload_col_options,
                        index=0,
                        key=f"map_{feat}",
                    )
                    if chosen != "-- fill with 0 --":
                        manual_map[feat] = chosen

        fill_count = len(unmatched) - len(manual_map)
        if fill_count > 0:
            st.warning(f"{fill_count} feature(s) will be filled with 0.")

        full_mapping = {**auto_map, **manual_map}
        X_input = _apply_column_mapping(df_upload.copy(), full_mapping, feature_names)
        run_prediction = st.button("Run Prediction", type="primary")

else:
    col_fill1, col_fill2 = st.columns(2)
    with col_fill1:
        fill_benign = st.button("Fill with Benign values (zeros)")
    with col_fill2:
        fill_attack = st.button("Fill with Attack values (random)")

    if fill_benign:
        for feat in feature_names:
            st.session_state[f"feat_{feat}"] = 0.0
    if fill_attack:
        rng = np.random.default_rng(42)
        for feat in feature_names:
            st.session_state[f"feat_{feat}"] = float(rng.uniform(0, 1000))

    grouped = {f for feats in FEATURE_GROUPS.values() for f in feats}
    other_features = [f for f in feature_names if f not in grouped]

    manual_values: dict[str, float] = {}

    all_groups = dict(FEATURE_GROUPS)
    if other_features:
        all_groups["Other Features"] = other_features

    for group_name, group_feats in all_groups.items():
        present = [f for f in group_feats if f in feature_names]
        if not present:
            continue
        with st.expander(f"{group_name} ({len(present)} features)", expanded=False):
            cols = st.columns(3)
            for i, feat in enumerate(present):
                default_val = st.session_state.get(f"feat_{feat}", 0.0)
                val = cols[i % 3].number_input(
                    feat,
                    value=float(default_val),
                    format="%.4f",
                    key=f"ni_{feat}",
                )
                manual_values[feat] = val

    run_prediction = st.button("Run Prediction", type="primary")
    if run_prediction:
        row = {feat: manual_values.get(feat, 0.0) for feat in feature_names}
        X_input = pd.DataFrame([row])

st.divider()

# ── Section 3: Results ───────────────────────────────────────────────────────

if run_prediction and X_input is not None:
    st.subheader("3. Prediction Results")

    try:
        y_pred, y_proba = _run_prediction(model, scaler, feature_names, X_input)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
        st.stop()

    is_single = len(y_pred) == 1

    if is_single:
        label_int = int(y_pred[0])
        class_label = class_names_for_task.get(label_int, f"Class {label_int}")
        is_benign = label_int == 0

        if is_benign:
            st.success("BENIGN -- Normal Traffic")
        elif selected_task == "multiclass":
            st.error(f"THREAT DETECTED -- {class_label}")
        else:
            st.error("ATTACK DETECTED")

        if selected_task == "multiclass":
            st.markdown(f"**Predicted class:** `{class_label}` (label `{label_int}`)")
        else:
            st.markdown(f"**Predicted:** `{class_label}`")

        if y_proba is not None:
            conf = float(y_proba[0][label_int]) * 100
            st.metric("Confidence", f"{conf:.1f}%")

            n_classes_model = y_proba.shape[1]
            prob_names = [
                class_names_for_task.get(i, f"Class {i}")
                for i in range(n_classes_model)
            ]
            prob_vals = [float(y_proba[0][i]) * 100 for i in range(n_classes_model)]

            fig = go.Figure(go.Bar(
                x=prob_names,
                y=prob_vals,
                marker_color=["#2ecc71" if i == 0 else "#e74c3c"
                               for i in range(n_classes_model)],
                text=[f"{v:.1f}%" for v in prob_vals],
                textposition="outside",
            ))
            fig.update_layout(
                title="Class Probabilities",
                yaxis_title="Probability (%)",
                yaxis=dict(range=[0, 110]),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Input feature values"):
            feat_df = pd.DataFrame(
                {"Feature": feature_names,
                 "Value": [X_input.iloc[0].get(f, 0.0) for f in feature_names]}
            )
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    else:
        result_df = X_input.copy()
        result_df["Predicted Label"] = y_pred
        result_df["Predicted Class"] = [
            class_names_for_task.get(int(p), f"Class {p}") for p in y_pred
        ]
        if y_proba is not None:
            result_df["Confidence %"] = [
                round(float(y_proba[i][int(y_pred[i])]) * 100, 1)
                for i in range(len(y_pred))
            ]

        # Risk summary
        n_benign = int((np.array(y_pred) == 0).sum())
        n_attack = len(y_pred) - n_benign
        risk_pct = n_attack / len(y_pred) * 100 if len(y_pred) > 0 else 0

        st.markdown("### Risk Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{len(y_pred):,}")
        c2.metric("Benign", f"{n_benign:,}", f"{n_benign/len(y_pred)*100:.1f}%")
        c3.metric("Threats", f"{n_attack:,}", f"{risk_pct:.1f}%",
                  delta_color="inverse")

        if selected_task == "multiclass" and n_attack > 0:
            attack_preds = [int(p) for p in y_pred if int(p) != 0]
            from collections import Counter
            attack_counts = Counter(attack_preds)
            top_class_id = attack_counts.most_common(1)[0][0]
            top_class_name = class_names_for_task.get(top_class_id, f"Class {top_class_id}")
            top_class_pct = attack_counts[top_class_id] / n_attack * 100
            c4.metric("Top Threat", f"{top_class_name}", f"{top_class_pct:.0f}% of threats")
        elif n_attack > 0:
            c4.metric("Attack Rate", f"{risk_pct:.1f}%")
        else:
            c4.metric("Status", "All Clear")

        # Per-class breakdown
        class_counts = pd.Series(result_df["Predicted Class"]).value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        fig = px.bar(
            class_counts, x="Class", y="Count",
            color="Class",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=True,
            title="Predicted Class Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        if selected_task == "multiclass" and n_attack > 0:
            st.markdown("**Threat Breakdown:**")
            threat_rows = result_df[result_df["Predicted Label"] != 0]
            threat_summary = (
                threat_rows["Predicted Class"]
                .value_counts()
                .reset_index()
            )
            threat_summary.columns = ["Attack Type", "Count"]
            threat_summary["% of Threats"] = (
                threat_summary["Count"] / n_attack * 100
            ).round(1)
            st.dataframe(threat_summary, use_container_width=True, hide_index=True)

        st.markdown("**Predictions (first 500 rows):**")
        display_cols = ["Predicted Class", "Predicted Label"]
        if "Confidence %" in result_df.columns:
            display_cols.append("Confidence %")
        st.dataframe(
            result_df[display_cols].head(500),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )
