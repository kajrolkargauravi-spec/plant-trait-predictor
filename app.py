"""
Plant Functional Trait Predictor
A multimodal (image + environmental) research application.
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Plant Functional Trait Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
IMG_SIZE = 224
TARGET_COLS = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
TARGET_NAMES = {
    'X4_mean': 'Stem Specific Density',
    'X11_mean': 'Specific Leaf Area',
    'X18_mean': 'Plant Height',
    'X26_mean': 'Seed Dry Mass',
    'X50_mean': 'Leaf Nitrogen per Area',
    'X3112_mean': 'Leaf Area'
}
# Default units — overridden if a units file is found (see load_units())
DEFAULT_UNITS = {
    'X4_mean': 'g/cm3',
    'X11_mean': 'mm2/mg',
    'X18_mean': 'cm',
    'X26_mean': 'g',
    'X50_mean': 'g/m2',
    'X3112_mean': 'mm2'
}

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))


def artifact_path(name):
    return os.path.join(ARTIFACT_DIR, name)


# ============================================================
# MINIMAL, CLEAN STYLING (no emojis, quiet color palette)
# ============================================================
st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .metric-card {
        background-color: #f7f8f7;
        border: 1px solid #e3e6e3;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #5c6b5c;
        margin-bottom: 0.25rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1f2b1f;
    }
    .metric-card .unit {
        font-size: 0.9rem;
        color: #7a877a;
        margin-left: 0.35rem;
    }
    .section-note {
        color: #6b6b6b;
        font-size: 0.9rem;
    }
    h1, h2, h3 { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ARTIFACT LOADING
# ============================================================
@st.cache_resource
def load_model_artifacts():
    model = tf.keras.models.load_model(artifact_path('multimodal_model.keras'))
    env_scaler = joblib.load(artifact_path('env_scaler.pkl'))
    target_scaler = joblib.load(artifact_path('target_scaler.pkl'))
    env_cols = joblib.load(artifact_path('env_cols.pkl'))
    selected_env_cols = joblib.load(artifact_path('selected_env_cols.pkl'))
    env_medians = joblib.load(artifact_path('env_medians.pkl'))
    return model, env_scaler, target_scaler, env_cols, selected_env_cols, env_medians


@st.cache_resource
def load_optional(name, _loader):
    """Load an optional artifact; return None if missing instead of crashing the app.
    The leading underscore on _loader tells Streamlit not to try to hash a function object."""
    path = artifact_path(name)
    if os.path.exists(path):
        return _loader(path)
    return None


def load_units():
    units = load_optional('target_units.json', lambda p: json.load(open(p)))
    return units if units else DEFAULT_UNITS


def load_results_table():
    return load_optional('results_table.csv', pd.read_csv)


def load_ablation_results():
    return load_optional('ablation_results.csv', pd.read_csv)


def load_shap_background():
    return load_optional('shap_background.pkl', joblib.load)


def load_shap_env_cols():
    return load_optional('shap_env_cols.pkl', joblib.load)


def load_training_distribution():
    return load_optional('target_distributions.csv', pd.read_csv)


def load_env_groups():
    """Optional mapping of environmental column name -> group (Climate/Soil/MODIS/VOD)."""
    return load_optional('env_groups.json', lambda p: json.load(open(p)))


try:
    model, env_scaler, target_scaler, ENV_COLS, SELECTED_ENV_COLS, ENV_MEDIANS = load_model_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

UNITS = load_units()
RESULTS_TABLE = load_results_table()
ABLATION_RESULTS = load_ablation_results()
SHAP_BACKGROUND = load_shap_background()
SHAP_ENV_COLS = load_shap_env_cols()
TARGET_DIST = load_training_distribution()
ENV_GROUPS = load_env_groups()


# ============================================================
# HELPER: infer environmental group from column name if no mapping file exists
# ============================================================
def infer_group(col_name):
    name = col_name.upper()
    if any(k in name for k in ['BIO', 'TEMP', 'PREC', 'CLIM']):
        return 'Climate'
    if 'SOIL' in name:
        return 'Soil'
    if 'MODIS' in name:
        return 'MODIS'
    if 'VOD' in name:
        return 'VOD'
    return 'Other'


def get_env_group(col_name):
    if ENV_GROUPS and col_name in ENV_GROUPS:
        return ENV_GROUPS[col_name]
    return infer_group(col_name)


# ============================================================
# PREDICTION LOGIC
# ============================================================
def preprocess_image(pil_image):
    img = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def build_env_vector(user_values):
    row = ENV_MEDIANS.copy()
    for feat, val in user_values.items():
        row[feat] = val
    vector = np.array([[row[c] for c in ENV_COLS]])
    return env_scaler.transform(vector), row


def predict(pil_image, user_env_values):
    img_array = preprocess_image(pil_image)
    env_scaled, full_env_row = build_env_vector(user_env_values)
    pred_scaled = model.predict([img_array, env_scaled], verbose=0)
    pred = target_scaler.inverse_transform(pred_scaled)
    pred = np.expm1(pred)  # reverse log1p applied during training
    return pred[0], full_env_row, env_scaled


def compute_shap_for_prediction(env_scaled_row, trait_idx):
    """Local SHAP explanation for one prediction, one trait, using a small precomputed background."""
    import shap
    if SHAP_BACKGROUND is None:
        return None

    def predict_fn(x):
        dummy_img = np.zeros((x.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        preds = model.predict([dummy_img, x], verbose=0)
        return preds[:, trait_idx]

    explainer = shap.KernelExplainer(predict_fn, SHAP_BACKGROUND)
    shap_values = explainer.shap_values(env_scaled_row, nsamples=100, silent=True)
    return shap_values


def compute_gradcam(pil_image, trait_idx):
    import cv2
    img_array = preprocess_image(pil_image)

    # Locate the nested EfficientNet base and the model's head layers
    base_layer = None
    for layer in model.layers:
        if 'efficientnet' in layer.name:
            base_layer = layer
            break
    if base_layer is None:
        return None

    last_conv_name = None
    for l in base_layer.layers:
        if 'top_conv' in l.name:
            last_conv_name = l.name
    if last_conv_name is None:
        return None

    feature_model = tf.keras.Model(
        inputs=base_layer.input,
        outputs=[base_layer.get_layer(last_conv_name).output, base_layer.output]
    )

    # Identify the dense/dropout/fusion path after the image branch — model-specific.
    # This assumes the multimodal architecture built in the training notebook.
    img_dense = None
    for l in model.layers:
        if l.name == 'img_features':
            img_dense = l
    if img_dense is None:
        return None

    with tf.GradientTape() as tape:
        conv_output, pooled_output = feature_model(img_array)
        tape.watch(conv_output)
        img_feat = img_dense(pooled_output)
        # NOTE: full forward pass through fusion + head is architecture-specific;
        # this focuses the attribution on the image branch's own contribution.
        loss = tf.reduce_mean(img_feat)

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    orig = np.array(pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


# ============================================================
# SESSION STATE
# ============================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
if 'last_env_row' not in st.session_state:
    st.session_state.last_env_row = None
if 'last_env_scaled' not in st.session_state:
    st.session_state.last_env_scaled = None


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("Plant Trait Predictor")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Predict",
        "Prediction Analysis",
        "Explainability",
        "Model Performance",
        "Model Comparison",
        "Environmental Analysis",
        "Ablation Study",
        "About the Research",
    ],
    label_visibility="collapsed"
)

if not MODEL_LOADED:
    st.sidebar.error("Model artifacts failed to load. Check that all files are present in the app folder.")


# ============================================================
# PAGE: HOME
# ============================================================
if page == "Home":
    st.title("Plant Functional Trait Predictor")
    st.write(
        "This application estimates six continuous plant functional traits from a plant "
        "photograph combined with environmental and geographic information. It is the "
        "deployment component of a research project comparing unimodal and multimodal "
        "approaches to plant trait prediction."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="label">Predicted traits</div>'
            '<div class="value">6</div></div>', unsafe_allow_html=True
        )
    with col2:
        n_env = len(ENV_COLS) if MODEL_LOADED else "—"
        st.markdown(
            f'<div class="metric-card"><div class="label">Environmental variables used by the model</div>'
            f'<div class="value">{n_env}</div></div>', unsafe_allow_html=True
        )
    with col3:
        n_sel = len(SELECTED_ENV_COLS) if MODEL_LOADED else "—"
        st.markdown(
            f'<div class="metric-card"><div class="label">Variables shown in the interface</div>'
            f'<div class="value">{n_sel}</div></div>', unsafe_allow_html=True
        )

    st.subheader("How the model works")
    st.write(
        "- **Image branch**: an EfficientNetB0 backbone learns visual representations "
        "from the plant photograph.\n"
        "- **Environmental branch**: a multilayer perceptron learns relationships between "
        "environmental and geographic variables and plant traits.\n"
        "- **Fusion**: the two representations are combined before producing the six "
        "numerical predictions."
    )
    st.markdown(
        '<p class="section-note">Use the navigation panel on the left to make a prediction, '
        'inspect explainability results, or review the underlying research comparisons.</p>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE: PREDICT
# ============================================================
elif page == "Predict":
    st.title("Predict Plant Traits")

    if not MODEL_LOADED:
        st.error(f"Could not load model artifacts: {LOAD_ERROR}")
    else:
        left, right = st.columns([1, 1.2])

        with left:
            st.subheader("Plant Image")
            uploaded_file = st.file_uploader(
                "Upload a plant image", type=['jpg', 'jpeg', 'png']
            )
            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file)
                st.image(pil_image, use_container_width=True)
            else:
                pil_image = None

        with right:
            st.subheader("Environmental Information")
            st.markdown(
                f'<p class="section-note">Showing the {len(SELECTED_ENV_COLS)} variables '
                f'identified as most influential during model analysis. Remaining variables '
                f'are filled in using training-data medians.</p>',
                unsafe_allow_html=True
            )

            user_values = {}
            n_cols = 2
            cols = st.columns(n_cols)
            for i, feat in enumerate(SELECTED_ENV_COLS):
                default_val = float(ENV_MEDIANS.get(feat, 0.0))
                with cols[i % n_cols]:
                    user_values[feat] = st.number_input(
                        feat, value=default_val, format="%.4f", key=f"env_{feat}"
                    )

        st.divider()
        predict_clicked = st.button("Predict Plant Traits", type="primary", use_container_width=False)

        if predict_clicked:
            if pil_image is None:
                st.error("Please upload a plant image before predicting.")
            else:
                with st.spinner("Running prediction..."):
                    pred_values, full_env_row, env_scaled = predict(pil_image, user_values)

                st.session_state.last_prediction = pred_values
                st.session_state.last_image = pil_image
                st.session_state.last_env_row = full_env_row
                st.session_state.last_env_scaled = env_scaled
                st.session_state.prediction_history.append({
                    'index': len(st.session_state.prediction_history) + 1,
                    **{TARGET_NAMES[c]: v for c, v in zip(TARGET_COLS, pred_values)}
                })

        if st.session_state.last_prediction is not None:
            st.subheader("Predicted Functional Traits")
            pred_values = st.session_state.last_prediction

            metric_cols = st.columns(3)
            for i, col_name in enumerate(TARGET_COLS):
                unit = UNITS.get(col_name, "")
                with metric_cols[i % 3]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="label">{TARGET_NAMES[col_name]}</div>'
                        f'<div class="value">{pred_values[i]:.2f}<span class="unit">{unit}</span></div>'
                        f'</div>', unsafe_allow_html=True
                    )

            st.markdown(
                '<p class="section-note">Exact units are taken from the dataset metadata where available.</p>',
                unsafe_allow_html=True
            )

            # Download report
            report_df = pd.DataFrame({
                'Trait': [TARGET_NAMES[c] for c in TARGET_COLS],
                'Predicted Value': pred_values,
                'Unit': [UNITS.get(c, "") for c in TARGET_COLS]
            })
            csv_bytes = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Prediction Report",
                data=csv_bytes,
                file_name="plant_trait_prediction.csv",
                mime="text/csv"
            )


# ============================================================
# PAGE: PREDICTION ANALYSIS
# ============================================================
elif page == "Prediction Analysis":
    st.title("Prediction Analysis")

    if st.session_state.last_prediction is None:
        st.info("Make a prediction on the Predict page first.")
    else:
        pred_values = st.session_state.last_prediction

        st.subheader("Prediction Profile")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[TARGET_NAMES[c] for c in TARGET_COLS],
            y=pred_values,
            marker_color="#4a7a4a"
        ))
        fig.update_layout(
            yaxis_title="Predicted value",
            height=420,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        if TARGET_DIST is not None:
            st.subheader("Prediction vs. Training Data Distribution")
            st.markdown(
                '<p class="section-note">Shows where this prediction falls relative to the '
                'range of values seen during training, for context rather than confidence.</p>',
                unsafe_allow_html=True
            )
            trait_choice = st.selectbox(
                "Select trait", [TARGET_NAMES[c] for c in TARGET_COLS], key="dist_trait"
            )
            col_key = [c for c, n in TARGET_NAMES.items() if n == trait_choice][0]
            if col_key in TARGET_DIST.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=TARGET_DIST[col_key], nbinsx=40, marker_color="#c9d6c9"))
                idx = TARGET_COLS.index(col_key)
                fig2.add_vline(x=pred_values[idx], line_color="#2f4f2f", line_width=2,
                                annotation_text="Prediction")
                fig2.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown(
                '<p class="section-note">Training-data distribution file not found '
                '(target_distributions.csv). This comparison is unavailable until it is added.</p>',
                unsafe_allow_html=True
            )

        if len(st.session_state.prediction_history) > 1:
            st.subheader("Prediction History (this session)")
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)


# ============================================================
# PAGE: EXPLAINABILITY
# ============================================================
elif page == "Explainability":
    st.title("Explainability")
    st.write("Why did the model make this prediction?")

    if st.session_state.last_prediction is None:
        st.info("Make a prediction on the Predict page first.")
    else:
        trait_choice = st.selectbox("Select trait", [TARGET_NAMES[c] for c in TARGET_COLS])
        trait_col = [c for c, n in TARGET_NAMES.items() if n == trait_choice][0]
        trait_idx = TARGET_COLS.index(trait_col)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Environmental Contribution")
            if SHAP_BACKGROUND is None or SHAP_ENV_COLS is None:
                st.markdown(
                    '<p class="section-note">SHAP background sample not found '
                    '(shap_background.pkl / shap_env_cols.pkl). Save a small background sample '
                    'from the training notebook to enable this panel.</p>',
                    unsafe_allow_html=True
                )
            else:
                with st.spinner("Computing local explanation..."):
                    try:
                        shap_vals = compute_shap_for_prediction(
                            st.session_state.last_env_scaled, trait_idx
                        )
                        shap_vals = np.array(shap_vals).flatten()
                        shap_df = pd.DataFrame({
                            'feature': SHAP_ENV_COLS,
                            'contribution': shap_vals
                        }).sort_values('contribution', key=abs, ascending=True).tail(10)

                        fig3 = go.Figure(go.Bar(
                            x=shap_df['contribution'],
                            y=shap_df['feature'],
                            orientation='h',
                            marker_color=["#4a7a4a" if v > 0 else "#a85c5c" for v in shap_df['contribution']]
                        ))
                        fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig3, use_container_width=True)

                        top_feat = shap_df.iloc[-1]['feature']
                        second_feat = shap_df.iloc[-2]['feature'] if len(shap_df) > 1 else None
                        note = f"The environmental variables contributing most to this prediction were **{top_feat}**"
                        if second_feat:
                            note += f" and **{second_feat}**"
                        note += "."
                        st.write(note)
                        st.markdown(
                            '<p class="section-note">These values indicate how much each variable '
                            'contributed to the model\'s prediction, not a biological cause of the trait.</p>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.warning(f"SHAP explanation could not be computed: {e}")

        with col_b:
            st.subheader("Visual Explanation")
            try:
                overlay = compute_gradcam(st.session_state.last_image, trait_idx)
                if overlay is not None:
                    st.image(overlay, use_container_width=True)
                    st.markdown(
                        '<p class="section-note">The highlighted regions indicate parts of the '
                        'image the model weighted most heavily for this prediction.</p>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<p class="section-note">Grad-CAM could not be generated for this '
                        'model architecture.</p>', unsafe_allow_html=True
                    )
            except Exception as e:
                st.warning(f"Grad-CAM could not be computed: {e}")


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.title("Model Performance")

    if RESULTS_TABLE is None:
        st.markdown(
            '<p class="section-note">results_table.csv not found in the app folder. '
            'This file is produced by the evaluation step in the training notebook.</p>',
            unsafe_allow_html=True
        )
    else:
        multimodal_results = RESULTS_TABLE[RESULTS_TABLE['Model'] == 'Multimodal CNN+MLP']

        st.subheader("Multimodal Model — Metrics by Trait")
        st.dataframe(
            multimodal_results[['Trait', 'MAE', 'RMSE', 'R2']].reset_index(drop=True),
            use_container_width=True
        )

        st.subheader("R-squared by Trait")
        fig4 = go.Figure(go.Bar(
            x=multimodal_results['Trait'], y=multimodal_results['R2'], marker_color="#4a7a4a"
        ))
        fig4.update_layout(yaxis_title="R2", height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown(
            '<p class="section-note">Actual-vs-predicted scatter plots require the saved test-set '
            'predictions (test_predictions.csv). Add this file to enable that view.</p>',
            unsafe_allow_html=True
        )
        scatter_data = load_optional('test_predictions.csv', pd.read_csv)
        if scatter_data is not None:
            trait_choice = st.selectbox(
                "Select trait for actual vs. predicted plot",
                [TARGET_NAMES[c] for c in TARGET_COLS], key="perf_trait"
            )
            trait_col = [c for c, n in TARGET_NAMES.items() if n == trait_choice][0]
            actual_col = f"{trait_col}_actual"
            pred_col = f"{trait_col}_pred"
            if actual_col in scatter_data.columns and pred_col in scatter_data.columns:
                fig5 = px.scatter(
                    scatter_data, x=actual_col, y=pred_col, opacity=0.5,
                    labels={actual_col: "Actual", pred_col: "Predicted"}
                )
                min_v = min(scatter_data[actual_col].min(), scatter_data[pred_col].min())
                max_v = max(scatter_data[actual_col].max(), scatter_data[pred_col].max())
                fig5.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                           mode='lines', line=dict(color="#a85c5c", dash="dash"),
                                           name="Ideal"))
                st.plotly_chart(fig5, use_container_width=True)


# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write(
        "This connects the deployed model to the underlying research experiment: does combining "
        "image and environmental information improve prediction compared with either source alone?"
    )

    if RESULTS_TABLE is None:
        st.markdown(
            '<p class="section-note">results_table.csv not found in the app folder.</p>',
            unsafe_allow_html=True
        )
    else:
        st.subheader("Overall Metrics by Model")
        summary = RESULTS_TABLE.groupby('Model')[['MAE', 'RMSE', 'R2']].mean().reset_index()
        st.dataframe(summary, use_container_width=True)

        st.subheader("R-squared by Trait and Model")
        pivot = RESULTS_TABLE.pivot_table(index='Trait', columns='Model', values='R2')
        fig6 = go.Figure()
        for model_name in pivot.columns:
            fig6.add_trace(go.Bar(name=model_name, x=pivot.index, y=pivot[model_name]))
        fig6.update_layout(barmode='group', height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig6, use_container_width=True)

        best_model = summary.sort_values('R2', ascending=False).iloc[0]['Model']
        multimodal_row = summary[summary['Model'] == 'Multimodal CNN+MLP']
        if not multimodal_row.empty and best_model == 'Multimodal CNN+MLP':
            st.success(f"Best-performing approach overall: {best_model}")
        else:
            wins = (pivot.idxmax(axis=1) == 'Multimodal CNN+MLP').sum()
            total = len(pivot)
            st.warning(
                f"Finding: the multimodal model did not achieve the highest R2 on average. "
                f"It performed best on {wins} of {total} traits."
            )


# ============================================================
# PAGE: ENVIRONMENTAL ANALYSIS
# ============================================================
elif page == "Environmental Analysis":
    st.title("Environmental Analysis")

    if st.session_state.last_env_row is None:
        st.info("Make a prediction on the Predict page first to view its environmental profile.")
    else:
        env_row = st.session_state.last_env_row
        groups = {}
        for col in ENV_COLS:
            g = get_env_group(col)
            groups.setdefault(g, []).append(col)

        st.subheader("Variable Groups")
        group_counts = pd.DataFrame({
            'Group': list(groups.keys()),
            'Number of variables': [len(v) for v in groups.values()]
        })
        fig7 = go.Figure(go.Bar(x=group_counts['Group'], y=group_counts['Number of variables'],
                                 marker_color="#4a7a4a"))
        fig7.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("Current Input Profile")
        selected_group = st.selectbox("Select variable group", list(groups.keys()))
        group_vals = pd.DataFrame({
            'Variable': groups[selected_group],
            'Value': [env_row.get(c, np.nan) for c in groups[selected_group]]
        })
        st.dataframe(group_vals, use_container_width=True, height=350)

        st.markdown(
            '<p class="section-note">Group contribution to prediction (e.g. climate vs. soil vs. '
            'MODIS vs. VOD) requires an attribution method run at the group level and is not yet '
            'computed. This section currently shows the input profile only.</p>',
            unsafe_allow_html=True
        )


# ============================================================
# PAGE: ABLATION STUDY
# ============================================================
elif page == "Ablation Study":
    st.title("Ablation Study")
    st.write(
        "This section examines which information sources are necessary for accurate prediction "
        "by systematically removing them and observing the effect on performance."
    )

    if ABLATION_RESULTS is None:
        st.markdown(
            '<p class="section-note">ablation_results.csv not found. This experiment '
            '(full multimodal model vs. model without image, without climate, without soil, '
            'without MODIS, without VOD) needs to be run separately in the training notebook '
            'and its results saved to this file before this page can display them.</p>',
            unsafe_allow_html=True
        )
    else:
        st.dataframe(ABLATION_RESULTS, use_container_width=True)
        fig8 = px.bar(ABLATION_RESULTS, x='Configuration', y='R2', color='Trait', barmode='group')
        fig8.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig8, use_container_width=True)


# ============================================================
# PAGE: ABOUT THE RESEARCH
# ============================================================
elif page == "About the Research":
    st.title("About the Research")

    st.subheader("Research Question")
    st.write(
        "Can plant functional traits be predicted more accurately by combining visual "
        "information from plant images with environmental and geographic information than "
        "by using either source of information independently?"
    )

    st.subheader("What the Model Predicts")
    st.write(
        "The model estimates six continuous plant functional traits: " +
        ", ".join(TARGET_NAMES.values()) + "."
    )

    st.subheader("Architecture")
    st.write(
        "- **Image branch**: EfficientNetB0, pretrained on ImageNet, learns visual representations "
        "from the plant photograph.\n"
        "- **Environmental branch**: a multilayer perceptron learns nonlinear relationships between "
        "environmental and geographic variables and plant traits.\n"
        "- **Fusion**: the two learned representations are concatenated and passed through a "
        "fusion network to produce all six predictions simultaneously."
    )

    st.subheader("Models Compared")
    st.write(
        "Linear regression (baseline), an environmental-only MLP, an image-only CNN, and the "
        "multimodal CNN+MLP model were trained and evaluated under the same train/validation/test "
        "split, using MAE, RMSE and R2 for each trait."
    )

    st.subheader("Interpretability")
    st.write(
        "SHAP is used to attribute predictions of the environmental branch to individual "
        "environmental variables. Grad-CAM is used to visualize which regions of the plant image "
        "were most influential for the CNN branch. Both methods describe what the model used to "
        "make a prediction, not the biological cause of the trait."
    )

    if MODEL_LOADED:
        st.subheader("Model Information")
        info_df = pd.DataFrame({
            'Property': [
                'Image resolution', 'Number of environmental variables (full model)',
                'Number of variables shown in interface', 'Number of predicted traits'
            ],
            'Value': [
                f"{IMG_SIZE} x {IMG_SIZE}", len(ENV_COLS), len(SELECTED_ENV_COLS), len(TARGET_COLS)
            ]
        })
        st.dataframe(info_df, use_container_width=True)