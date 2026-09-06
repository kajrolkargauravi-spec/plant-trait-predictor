```python
"""
Plant Functional Trait Predictor
A multimodal (image + environmental) research application.
"""

import os
import json
import warnings

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")


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

TARGET_COLS = [
    "X4_mean",
    "X11_mean",
    "X18_mean",
    "X26_mean",
    "X50_mean",
    "X3112_mean"
]

TARGET_NAMES = {
    "X4_mean": "Stem Specific Density",
    "X11_mean": "Specific Leaf Area",
    "X18_mean": "Plant Height",
    "X26_mean": "Seed Dry Mass",
    "X50_mean": "Leaf Nitrogen per Area",
    "X3112_mean": "Leaf Area"
}

DEFAULT_UNITS = {
    "X4_mean": "g/cm3",
    "X11_mean": "mm2/mg",
    "X18_mean": "cm",
    "X26_mean": "g",
    "X50_mean": "g/m2",
    "X3112_mean": "mm2"
}

ARTIFACT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def artifact_path(name):
    return os.path.join(
        ARTIFACT_DIR,
        name
    )


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main > div {
        padding-top: 1.5rem;
    }

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

    h1, h2, h3 {
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL + ARTIFACT LOADING
# ============================================================

@st.cache_resource
def load_model_artifacts():

    required_files = [
        "multimodal_model.keras",
        "env_scaler.pkl",
        "target_scaler.pkl",
        "env_cols.pkl",
        "selected_env_cols.pkl",
        "env_medians.pkl"
    ]

    missing_files = []

    for filename in required_files:

        if not os.path.exists(
            artifact_path(filename)
        ):
            missing_files.append(filename)

    if missing_files:

        raise FileNotFoundError(
            "The following required files are missing: "
            + ", ".join(missing_files)
        )

    # --------------------------------------------------------
    # Load Keras model
    # --------------------------------------------------------
    #
    # compile=False is important for deployment because
    # the training loss/metrics do not need to be restored.
    #
    # safe_mode=False helps with models saved using Keras
    # serialization.
    #

    model = tf.keras.models.load_model(
        artifact_path(
            "multimodal_model.keras"
        ),
        compile=False,
        safe_mode=False
    )

    # --------------------------------------------------------
    # Load preprocessing artifacts
    # --------------------------------------------------------

    env_scaler = joblib.load(
        artifact_path("env_scaler.pkl")
    )

    target_scaler = joblib.load(
        artifact_path("target_scaler.pkl")
    )

    env_cols = joblib.load(
        artifact_path("env_cols.pkl")
    )

    selected_env_cols = joblib.load(
        artifact_path("selected_env_cols.pkl")
    )

    env_medians = joblib.load(
        artifact_path("env_medians.pkl")
    )

    return (
        model,
        env_scaler,
        target_scaler,
        env_cols,
        selected_env_cols,
        env_medians
    )


# ============================================================
# OPTIONAL ARTIFACT LOADERS
# ============================================================

def load_optional_pickle(filename):

    path = artifact_path(filename)

    if not os.path.exists(path):
        return None

    try:

        return joblib.load(path)

    except Exception:

        return None


def load_optional_csv(filename):

    path = artifact_path(filename)

    if not os.path.exists(path):
        return None

    try:

        return pd.read_csv(path)

    except Exception:

        return None


def load_optional_json(filename):

    path = artifact_path(filename)

    if not os.path.exists(path):
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return None


# ============================================================
# LOAD ALL ARTIFACTS
# ============================================================

try:

    (
        model,
        env_scaler,
        target_scaler,
        ENV_COLS,
        SELECTED_ENV_COLS,
        ENV_MEDIANS

    ) = load_model_artifacts()

    MODEL_LOADED = True
    LOAD_ERROR = None

except Exception as e:

    MODEL_LOADED = False
    LOAD_ERROR = str(e)

    model = None
    env_scaler = None
    target_scaler = None
    ENV_COLS = []
    SELECTED_ENV_COLS = []
    ENV_MEDIANS = {}


# Optional files

UNITS = load_optional_json(
    "target_units.json"
)

if not isinstance(
    UNITS,
    dict
):
    UNITS = DEFAULT_UNITS


RESULTS_TABLE = load_optional_csv(
    "results_table.csv"
)

ABLATION_RESULTS = load_optional_csv(
    "ablation_results.csv"
)

SHAP_BACKGROUND = load_optional_pickle(
    "shap_background.pkl"
)

SHAP_ENV_COLS = load_optional_pickle(
    "shap_env_cols.pkl"
)

TARGET_DIST = load_optional_csv(
    "target_distributions.csv"
)

ENV_GROUPS = load_optional_json(
    "env_groups.json"
)


# ============================================================
# ENVIRONMENTAL GROUP FUNCTIONS
# ============================================================

def infer_group(
    column_name
):

    name = str(
        column_name
    ).upper()

    if any(
        keyword in name
        for keyword in [
            "BIO",
            "TEMP",
            "PREC",
            "CLIM"
        ]
    ):

        return "Climate"

    if "SOIL" in name:
        return "Soil"

    if "MODIS" in name:
        return "MODIS"

    if "VOD" in name:
        return "VOD"

    return "Other"


def get_env_group(
    column_name
):

    if isinstance(
        ENV_GROUPS,
        dict
    ):

        if column_name in ENV_GROUPS:

            return ENV_GROUPS[
                column_name
            ]

    return infer_group(
        column_name
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    pil_image
):

    image = pil_image.convert(
        "RGB"
    )

    image = image.resize(
        (
            IMG_SIZE,
            IMG_SIZE
        )
    )

    array = np.array(
        image
    ).astype(
        np.float32
    )

    array = (
        tf.keras
        .applications
        .efficientnet
        .preprocess_input(
            array
        )
    )

    return np.expand_dims(
        array,
        axis=0
    )


# ============================================================
# ENVIRONMENT PREPROCESSING
# ============================================================

def build_env_vector(
    user_values
):

    # Copy training medians
    if hasattr(
        ENV_MEDIANS,
        "copy"
    ):

        row = ENV_MEDIANS.copy()

    else:

        row = dict(
            ENV_MEDIANS
        )

    # Replace selected variables
    # with user-provided values.

    for feature, value in user_values.items():

        if feature in ENV_COLS:

            row[feature] = float(
                value
            )

    values = []

    for column in ENV_COLS:

        try:

            value = row[column]

        except Exception:

            value = 0.0

        if pd.isna(value):

            value = 0.0

        values.append(
            float(value)
        )

    vector = np.array(
        [values],
        dtype=np.float32
    )

    scaled_vector = (
        env_scaler.transform(
            vector
        )
    )

    return (
        scaled_vector,
        row
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    pil_image,
    user_env_values
):

    if model is None:

        raise RuntimeError(
            "The model is not loaded."
        )

    image_array = preprocess_image(
        pil_image
    )

    env_scaled, full_env_row = (
        build_env_vector(
            user_env_values
        )
    )

    prediction_scaled = (
        model.predict(
            [
                image_array,
                env_scaled
            ],
            verbose=0
        )
    )

    prediction_scaled = np.asarray(
        prediction_scaled
    )

    # Reverse target scaling
    prediction = (
        target_scaler
        .inverse_transform(
            prediction_scaled
        )
    )

    # Reverse log1p transformation
    prediction = np.expm1(
        prediction
    )

    # Remove extremely small
    # numerical negative values.
    prediction = np.maximum(
        prediction,
        0
    )

    return (
        prediction[0],
        full_env_row,
        env_scaled
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def compute_shap_for_prediction(
    env_scaled_row,
    trait_idx
):

    if SHAP_BACKGROUND is None:

        return None

    if model is None:

        return None

    try:

        import shap

    except ImportError:

        return None

    try:

        background = np.asarray(
            SHAP_BACKGROUND,
            dtype=np.float32
        )

        current_row = np.asarray(
            env_scaled_row,
            dtype=np.float32
        )

        if background.ndim == 1:

            background = background.reshape(
                1,
                -1
            )

        if current_row.ndim == 1:

            current_row = current_row.reshape(
                1,
                -1
            )

        if (
            background.shape[1]
            != current_row.shape[1]
        ):

            return None

        def predict_fn(x):

            dummy_images = np.zeros(
                (
                    x.shape[0],
                    IMG_SIZE,
                    IMG_SIZE,
                    3
                ),
                dtype=np.float32
            )

            predictions = model.predict(
                [
                    dummy_images,
                    x
                ],
                verbose=0
            )

            return predictions[
                :,
                trait_idx
            ]

        explainer = (
            shap.KernelExplainer(
                predict_fn,
                background
            )
        )

        shap_values = (
            explainer.shap_values(
                current_row,
                nsamples=50,
                silent=True
            )
        )

        return shap_values

    except Exception:

        return None


# ============================================================
# GRAD-CAM
# ============================================================

def compute_gradcam(
    pil_image,
    trait_idx
):

    if model is None:

        return None

    try:

        import cv2

    except ImportError:

        return None

    try:

        image_array = preprocess_image(
            pil_image
        )

        # ----------------------------------------------------
        # Find EfficientNet branch
        # ----------------------------------------------------

        base_layer = None

        for layer in model.layers:

            if (
                "efficientnet"
                in layer.name.lower()
            ):

                base_layer = layer
                break

        if base_layer is None:

            return None

        # ----------------------------------------------------
        # Find last convolutional layer
        # ----------------------------------------------------

        convolutional_layer = None

        for layer in reversed(
            base_layer.layers
        ):

            try:

                output_shape = (
                    layer.output.shape
                )

                if len(
                    output_shape
                ) == 4:

                    convolutional_layer = (
                        layer
                    )

                    break

            except Exception:

                continue

        if convolutional_layer is None:

            return None

        feature_model = tf.keras.Model(
            inputs=base_layer.input,
            outputs=convolutional_layer.output
        )

        # ----------------------------------------------------
        # Generate activation map
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            convolution_output = (
                feature_model(
                    image_array,
                    training=False
                )
            )

            tape.watch(
                convolution_output
            )

            score = tf.reduce_mean(
                convolution_output
            )

        gradients = tape.gradient(
            score,
            convolution_output
        )

        if gradients is None:

            return None

        pooled_gradients = (
            tf.reduce_mean(
                gradients,
                axis=(0, 1, 2)
            )
        )

        convolution_output = (
            convolution_output[0]
        )

        heatmap = (
            convolution_output
            @ pooled_gradients[
                ...,
                tf.newaxis
            ]
        )

        heatmap = tf.squeeze(
            heatmap
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        maximum = tf.reduce_max(
            heatmap
        )

        if float(
            maximum
        ) == 0:

            return None

        heatmap = (
            heatmap / maximum
        ).numpy()

        # ----------------------------------------------------
        # Resize heatmap
        # ----------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap_color = (
            cv2.applyColorMap(
                heatmap,
                cv2.COLORMAP_JET
            )
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        original = np.array(
            pil_image
            .convert("RGB")
            .resize(
                (
                    IMG_SIZE,
                    IMG_SIZE
                )
            )
        )

        original_bgr = (
            cv2.cvtColor(
                original,
                cv2.COLOR_RGB2BGR
            )
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        overlay_rgb = (
            cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            )
        )

        return overlay_rgb

    except Exception:

        return None


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


if "last_prediction" not in st.session_state:

    st.session_state.last_prediction = None


if "last_image" not in st.session_state:

    st.session_state.last_image = None


if "last_env_row" not in st.session_state:

    st.session_state.last_env_row = None


if "last_env_scaled" not in st.session_state:

    st.session_state.last_env_scaled = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Plant Trait Predictor"
)

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
        "About the Research"
    ],
    label_visibility="collapsed"
)

if not MODEL_LOADED:

    st.sidebar.error(
        "Model artifacts could not be loaded."
    )


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title(
        "Plant Functional Trait Predictor"
    )

    st.write(
        """
        This application estimates six continuous plant
        functional traits from a plant photograph combined
        with environmental and geographic information.

        It is the deployment component of a research project
        comparing unimodal and multimodal approaches to
        plant trait prediction.
        """
    )

    if not MODEL_LOADED:

        st.error(
            "Could not load the model."
        )

        st.code(
            LOAD_ERROR
        )

        st.info(
            """
            Check that the following files are present
            in the same folder as app.py:

            multimodal_model.keras
            env_scaler.pkl
            target_scaler.pkl
            env_cols.pkl
            selected_env_cols.pkl
            env_medians.pkl
            """
        )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="label">
                    Predicted traits
                </div>

                <div class="value">
                    6
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        n_env = (
            len(ENV_COLS)
            if MODEL_LOADED
            else "—"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">
                    Environmental variables
                </div>

                <div class="value">
                    {n_env}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        n_selected = (
            len(
                SELECTED_ENV_COLS
            )
            if MODEL_LOADED
            else "—"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">
                    Variables shown
                </div>

                <div class="value">
                    {n_selected}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader(
        "How the model works"
    )

    st.write(
        """
        Image branch: EfficientNetB0 learns visual
        representations from the plant photograph.

        Environmental branch: a multilayer perceptron
        learns nonlinear relationships between
        environmental and geographic variables.

        Fusion: the two representations are combined
        before producing the six numerical predictions.
        """
    )


# ============================================================
# PREDICT
# ============================================================

elif page == "Predict":

    st.title(
        "Predict Plant Traits"
    )

    if not MODEL_LOADED:

        st.error(
            f"Could not load model artifacts: "
            f"{LOAD_ERROR}"
        )

    else:

        left, right = st.columns(
            [1, 1.2]
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with left:

            st.subheader(
                "Plant Image"
            )

            uploaded_file = (
                st.file_uploader(
                    "Upload a plant image",
                    type=[
                        "jpg",
                        "jpeg",
                        "png"
                    ]
                )
            )

            if uploaded_file:

                try:

                    pil_image = (
                        Image.open(
                            uploaded_file
                        ).convert("RGB")
                    )

                    st.image(
                        pil_image,
                        use_container_width=True
                    )

                except Exception as e:

                    pil_image = None

                    st.error(
                        f"Could not read image: {e}"
                    )

            else:

                pil_image = None

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        with right:

            st.subheader(
                "Environmental Information"
            )

            st.markdown(
                f"""
                <p class="section-note">
                Enter values for the
                {len(SELECTED_ENV_COLS)}
                environmental variables shown below.
                All remaining variables are filled using
                training-data medians.
                </p>
                """,
                unsafe_allow_html=True
            )

            user_values = {}

            input_columns = st.columns(2)

            for i, feature in enumerate(
                SELECTED_ENV_COLS
            ):

                try:

                    default_value = float(
                        ENV_MEDIANS.get(
                            feature,
                            0.0
                        )
                    )

                except Exception:

                    default_value = 0.0

                with input_columns[
                    i % 2
                ]:

                    user_values[
                        feature
                    ] = st.number_input(
                        feature,
                        value=default_value,
                        format="%.4f",
                        key=f"environment_{feature}"
                    )

        st.divider()

        predict_clicked = (
            st.button(
                "Predict Plant Traits",
                type="primary"
            )
        )

        if predict_clicked:

            if pil_image is None:

                st.error(
                    "Please upload a plant image first."
                )

            else:

                try:

                    with st.spinner(
                        "Running prediction..."
                    ):

                        (
                            pred_values,
                            full_env_row,
                            env_scaled
                        ) = predict(
                            pil_image,
                            user_values
                        )

                    st.session_state.last_prediction = (
                        pred_values
                    )

                    st.session_state.last_image = (
                        pil_image
                    )

                    st.session_state.last_env_row = (
                        full_env_row
                    )

                    st.session_state.last_env_scaled = (
                        env_scaled
                    )

                    st.session_state.prediction_history.append(
                        {
                            "Prediction":
                                len(
                                    st.session_state
                                    .prediction_history
                                ) + 1,

                            **{
                                TARGET_NAMES[col]:
                                    float(value)

                                for col, value
                                in zip(
                                    TARGET_COLS,
                                    pred_values
                                )
                            }
                        }
                    )

                    st.success(
                        "Prediction completed successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if (
            st.session_state.last_prediction
            is not None
        ):

            st.subheader(
                "Predicted Functional Traits"
            )

            pred_values = (
                st.session_state
                .last_prediction
            )

            metric_cols = st.columns(
                3
            )

            for i, col_name in enumerate(
                TARGET_COLS
            ):

                unit = UNITS.get(
                    col_name,
                    ""
                )

                with metric_cols[
                    i % 3
                ]:

                    st.markdown(
                        f"""
                        <div class="metric-card">

                            <div class="label">
                                {TARGET_NAMES[col_name]}
                            </div>

                            <div class="value">
                                {pred_values[i]:.2f}

                                <span class="unit">
                                    {unit}
                                </span>
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ------------------------------------------------
            # DOWNLOAD REPORT
            # ------------------------------------------------

            report_df = pd.DataFrame(
                {
                    "Trait": [
                        TARGET_NAMES[c]
                        for c in TARGET_COLS
                    ],

                    "Predicted Value": [
                        float(v)
                        for v in pred_values
                    ],

                    "Unit": [
                        UNITS.get(
                            c,
                            ""
                        )

                        for c in TARGET_COLS
                    ]
                }
            )

            csv_bytes = (
                report_df
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )

            st.download_button(
                "Download Prediction Report",
                data=csv_bytes,
                file_name="plant_trait_prediction.csv",
                mime="text/csv"
            )


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

elif page == "Prediction Analysis":

    st.title(
        "Prediction Analysis"
    )

    if (
        st.session_state.last_prediction
        is None
    ):

        st.info(
            "Make a prediction on the Predict page first."
        )

    else:

        pred_values = (
            st.session_state
            .last_prediction
        )

        st.subheader(
            "Prediction Profile"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[
                    TARGET_NAMES[c]
                    for c in TARGET_COLS
                ],

                y=pred_values
            )
        )

        fig.update_layout(
            yaxis_title="Predicted value",
            height=420,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        if TARGET_DIST is not None:

            st.subheader(
                "Prediction vs. Training Data Distribution"
            )

            st.markdown(
                """
                <p class="section-note">
                This comparison provides context about
                where the prediction lies relative to
                values observed in the training data.
                It is not a confidence interval.
                </p>
                """,
                unsafe_allow_html=True
            )

            trait_choice = (
                st.selectbox(
                    "Select trait",
                    [
                        TARGET_NAMES[c]
                        for c in TARGET_COLS
                    ],
                    key="distribution_trait"
                )
            )

            col_key = next(
                (
                    c

                    for c, name
                    in TARGET_NAMES.items()

                    if name == trait_choice
                ),
                None
            )

            if (
                col_key is not None
                and col_key
                in TARGET_DIST.columns
            ):

                idx = (
                    TARGET_COLS.index(
                        col_key
                    )
                )

                fig2 = go.Figure()

                fig2.add_trace(
                    go.Histogram(
                        x=TARGET_DIST[
                            col_key
                        ],
                        nbinsx=40
                    )
                )

                fig2.add_vline(
                    x=pred_values[idx],
                    line_width=2,
                    annotation_text="Prediction"
                )

                fig2.update_layout(
                    height=350
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        if len(
            st.session_state
            .prediction_history
        ) > 1:

            st.subheader(
                "Prediction History"
            )

            history_df = pd.DataFrame(
                st.session_state
                .prediction_history
            )

            st.dataframe(
                history_df,
                use_container_width=True
            )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.title(
        "Explainability"
    )

    st.write(
        "Why did the model make this prediction?"
    )

    if (
        st.session_state.last_prediction
        is None
    ):

        st.info(
            "Make a prediction on the Predict page first."
        )

    else:

        trait_choice = (
            st.selectbox(
                "Select trait",
                [
                    TARGET_NAMES[c]
                    for c in TARGET_COLS
                ]
            )
        )

        trait_col = next(
            (
                c

                for c, name
                in TARGET_NAMES.items()

                if name == trait_choice
            ),
            None
        )

        trait_idx = (
            TARGET_COLS.index(
                trait_col
            )
        )

        col_a, col_b = st.columns(
            2
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        with col_a:

            st.subheader(
                "Environmental Contribution"
            )

            if (
                SHAP_BACKGROUND is None
                or SHAP_ENV_COLS is None
            ):

                st.info(
                    """
                    SHAP artifacts are not available.

                    Add:
                    shap_background.pkl
                    shap_env_cols.pkl

                    to enable this panel.
                    """
                )

            else:

                with st.spinner(
                    "Computing environmental explanation..."
                ):

                    shap_values = (
                        compute_shap_for_prediction(
                            st.session_state
                            .last_env_scaled,
                            trait_idx
                        )
                    )

                if shap_values is None:

                    st.warning(
                        "SHAP could not be computed. "
                        "Check that the SHAP background "
                        "matches the model input dimensions."
                    )

                else:

                    shap_values = np.asarray(
                        shap_values
                    ).flatten()

                    features = list(
                        SHAP_ENV_COLS
                    )

                    if len(features) == len(
                        shap_values
                    ):

                        shap_df = pd.DataFrame(
                            {
                                "Feature":
                                    features,

                                "Contribution":
                                    shap_values
                            }
                        )

                        shap_df[
                            "Absolute Contribution"
                        ] = (
                            shap_df[
                                "Contribution"
                            ].abs()
                        )

                        shap_df = (
                            shap_df
                            .sort_values(
                                "Absolute Contribution"
                            )
                            .tail(10)
                        )

                        fig3 = go.Figure()

                        fig3.add_trace(
                            go.Bar(
                                x=shap_df[
                                    "Contribution"
                                ],

                                y=shap_df[
                                    "Feature"
                                ],

                                orientation="h"
                            )
                        )

                        fig3.update_layout(
                            height=420
                        )

                        st.plotly_chart(
                            fig3,
                            use_container_width=True
                        )

                        st.markdown(
                            """
                            <p class="section-note">
                            SHAP values describe how environmental
                            variables influence the model output.
                            They represent model behaviour and should
                            not be interpreted as proof of biological
                            causation.
                            </p>
                            """,
                            unsafe_allow_html=True
                        )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        with col_b:

            st.subheader(
                "Visual Explanation"
            )

            if (
                st.session_state.last_image
                is None
            ):

                st.info(
                    "No prediction image is available."
                )

            else:

                with st.spinner(
                    "Generating visual explanation..."
                ):

                    overlay = compute_gradcam(
                        st.session_state.last_image,
                        trait_idx
                    )

                if overlay is not None:

                    st.image(
                        overlay,
                        use_container_width=True
                    )

                    st.markdown(
                        """
                        <p class="section-note">
                        Highlighted regions indicate areas of
                        stronger activation in the visual feature
                        representation.
                        </p>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.info(
                        "Grad-CAM could not be generated "
                        "for this model architecture."
                    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "Model Performance"
    )

    if RESULTS_TABLE is None:

        st.warning(
            """
            results_table.csv was not found.

            Run the model evaluation section of your
            training notebook and add the resulting file
            to the repository.
            """
        )

    else:

        if "Model" not in (
            RESULTS_TABLE.columns
        ):

            st.error(
                "results_table.csv must contain a Model column."
            )

        else:

            multimodal_results = (
                RESULTS_TABLE[
                    RESULTS_TABLE["Model"]
                    == "Multimodal CNN+MLP"
                ]
            )

            st.subheader(
                "Multimodal Model — Metrics by Trait"
            )

            columns_to_show = [
                c

                for c in [
                    "Trait",
                    "MAE",
                    "RMSE",
                    "R2"
                ]

                if c in
                multimodal_results.columns
            ]

            st.dataframe(
                multimodal_results[
                    columns_to_show
                ].reset_index(
                    drop=True
                ),
                use_container_width=True
            )

            if (
                "Trait"
                in multimodal_results.columns

                and

                "R2"
                in multimodal_results.columns
            ):

                st.subheader(
                    "R-squared by Trait"
                )

                fig4 = go.Figure()

                fig4.add_trace(
                    go.Bar(
                        x=multimodal_results[
                            "Trait"
                        ],

                        y=multimodal_results[
                            "R2"
                        ]
                    )
                )

                fig4.update_layout(
                    yaxis_title="R²",
                    height=380
                )

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title(
        "Model Comparison"
    )

    st.write(
        """
        This section compares the multimodal model with
        single-source approaches to determine whether
        combining image and environmental information
        improves prediction.
        """
    )

    if RESULTS_TABLE is None:

        st.warning(
            "results_table.csv was not found."
        )

    elif not all(
        column in RESULTS_TABLE.columns

        for column in [
            "Model",
            "MAE",
            "RMSE",
            "R2"
        ]
    ):

        st.error(
            """
            results_table.csv is missing one or more
            required columns: Model, MAE, RMSE, R2.
            """
        )

    else:

        st.subheader(
            "Overall Metrics by Model"
        )

        summary = (
            RESULTS_TABLE
            .groupby("Model")[
                [
                    "MAE",
                    "RMSE",
                    "R2"
                ]
            ]
            .mean()
            .reset_index()
        )

        st.dataframe(
            summary,
            use_container_width=True
        )

        if all(
            column in RESULTS_TABLE.columns

            for column in [
                "Trait",
                "R2"
            ]
        ):

            st.subheader(
                "R-squared by Trait and Model"
            )

            pivot = (
                RESULTS_TABLE
                .pivot_table(
                    index="Trait",
                    columns="Model",
                    values="R2"
                )
            )

            fig6 = go.Figure()

            for model_name in (
                pivot.columns
            ):

                fig6.add_trace(
                    go.Bar(
                        name=model_name,
                        x=pivot.index,
                        y=pivot[
                            model_name
                        ]
                    )
                )

            fig6.update_layout(
                barmode="group",
                height=450
            )

            st.plotly_chart(
                fig6,
                use_container_width=True
            )

            if not summary.empty:

                best_model = (
                    summary
                    .sort_values(
                        "R2",
                        ascending=False
                    )
                    .iloc[0]["Model"]
                )

                st.success(
                    "Best-performing model by "
                    f"average R²: {best_model}"
                )


# ============================================================
# ENVIRONMENTAL ANALYSIS
# ============================================================

elif page == "Environmental Analysis":

    st.title(
        "Environmental Analysis"
    )

    if (
        st.session_state.last_env_row
        is None
    ):

        st.info(
            "Make a prediction on the Predict page first."
        )

    else:

        env_row = (
            st.session_state
            .last_env_row
        )

        groups = {}

        for column in ENV_COLS:

            group = get_env_group(
                column
            )

            if group not in groups:

                groups[group] = []

            groups[group].append(
                column
            )

        st.subheader(
            "Variable Groups"
        )

        group_counts = pd.DataFrame(
            {
                "Group":
                    list(
                        groups.keys()
                    ),

                "Number of variables":
                    [
                        len(values)

                        for values
                        in groups.values()
                    ]
            }
        )

        fig7 = go.Figure()

        fig7.add_trace(
            go.Bar(
                x=group_counts[
                    "Group"
                ],

                y=group_counts[
                    "Number of variables"
                ]
            )
        )

        fig7.update_layout(
            height=350
        )

        st.plotly_chart(
            fig7,
            use_container_width=True
        )

        st.subheader(
            "Current Input Profile"
        )

        selected_group = (
            st.selectbox(
                "Select variable group",
                list(
                    groups.keys()
                )
            )
        )

        group_values = pd.DataFrame(
            {
                "Variable":
                    groups[
                        selected_group
                    ],

                "Value":
                    [
                        env_row.get(
                            column,
                            np.nan
                        )

                        for column
                        in groups[
                            selected_group
                        ]
                    ]
            }
        )

        st.dataframe(
            group_values,
            use_container_width=True,
            height=350
        )

        st.info(
            """
            This panel displays the environmental
            input profile. Group-level attribution requires
            a separate attribution experiment.
            """
        )


# ============================================================
# ABLATION STUDY
# ============================================================

elif page == "Ablation Study":

    st.title(
        "Ablation Study"
    )

    st.write(
        """
        This section examines which information sources
        are necessary for accurate prediction by
        systematically removing them and observing
        the effect on model performance.
        """
    )

    if ABLATION_RESULTS is None:

        st.info(
            """
            ablation_results.csv was not found.

            Run the ablation experiment in the training
            notebook and add the resulting file to the
            repository.
            """
        )

    else:

        st.dataframe(
            ABLATION_RESULTS,
            use_container_width=True
        )

        required_columns = [
            "Configuration",
            "R2",
            "Trait"
        ]

        if all(
            column in
            ABLATION_RESULTS.columns

            for column
            in required_columns
        ):

            fig8 = px.bar(
                ABLATION_RESULTS,
                x="Configuration",
                y="R2",
                color="Trait",
                barmode="group"
            )

            fig8.update_layout(
                height=450
            )

            st.plotly_chart(
                fig8,
                use_container_width=True
            )


# ============================================================
# ABOUT THE RESEARCH
# ============================================================

elif page == "About the Research":

    st.title(
        "About the Research"
    )

    st.subheader(
        "Research Question"
    )

    st.write(
        """
        Can plant functional traits be predicted more
        accurately by combining visual information from
        plant images with environmental and geographic
        information than by using either source independently?
        """
    )

    st.subheader(
        "What the Model Predicts"
    )

    st.write(
        """
        The model estimates six continuous plant
        functional traits:
        """
    )

    for trait in TARGET_NAMES.values():

        st.write(
            f"• {trait}"
        )

    st.subheader(
        "Architecture"
    )

    st.write(
        """
        Image branch: EfficientNetB0 pretrained on
        ImageNet learns visual representations from
        the plant photograph.

        Environmental branch: a multilayer perceptron
        learns nonlinear relationships between
        environmental and geographic variables.

        Fusion: the learned image and environmental
        representations are combined before producing
        the six predictions.
        """
    )

    st.subheader(
        "Models Compared"
    )

    st.write(
        """
        Linear regression is used as a baseline.
        An environmental-only MLP, an image-only CNN,
        and the multimodal CNN+MLP are compared using
        MAE, RMSE and R².
        """
    )

    st.subheader(
        "Interpretability"
    )

    st.write(
        """
        SHAP is used to investigate the contribution
        of environmental variables.

        Grad-CAM is used to visualize regions of the
        plant image associated with the visual feature
        representation.

        These techniques explain model behaviour and
        should not be interpreted as proof of biological
        causation.
        """
    )

    if MODEL_LOADED:

        st.subheader(
            "Model Information"
        )

        info_df = pd.DataFrame(
            {
                "Property": [
                    "Image resolution",
                    "Environmental variables",
                    "Variables shown",
                    "Predicted traits"
                ],

                "Value": [
                    f"{IMG_SIZE} × {IMG_SIZE}",
                    len(ENV_COLS),
                    len(SELECTED_ENV_COLS),
                    len(TARGET_COLS)
                ]
            }
        )

        st.dataframe(
            info_df,
            use_container_width=True
        )
```
