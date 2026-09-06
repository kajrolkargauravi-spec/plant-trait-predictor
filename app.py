"""Plant Functional Trait Predictor
Multimodal image + environmental plant trait prediction application.
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
    "X4_mean": "g/cm³",
    "X11_mean": "mm²/mg",
    "X18_mean": "cm",
    "X26_mean": "g",
    "X50_mean": "g/m²",
    "X3112_mean": "mm²"
}


# ============================================================
# FILE PATH
# ============================================================

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))


def artifact_path(filename):
    return os.path.join(ARTIFACT_DIR, filename)


# ============================================================
# STYLING
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
            border-radius: 10px;
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
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OPTIONAL FILE LOADER
# ============================================================

def optional_load(filename, loader):

    path = artifact_path(filename)

    if not os.path.exists(path):
        return None

    try:
        return loader(path)
    except Exception:
        return None


# ============================================================
# LOAD OPTIONAL FILES
# ============================================================

def load_units():

    path = artifact_path("target_units.json")

    if not os.path.exists(path):
        return DEFAULT_UNITS

    try:

        with open(path, "r", encoding="utf-8") as file:
            units = json.load(file)

        if isinstance(units, dict):
            return units

    except Exception:
        pass

    return DEFAULT_UNITS


def load_results_table():
    return optional_load(
        "results_table.csv",
        pd.read_csv
    )


def load_ablation_results():
    return optional_load(
        "ablation_results.csv",
        pd.read_csv
    )


def load_shap_background():
    return optional_load(
        "shap_background.pkl",
        joblib.load
    )


def load_shap_env_cols():
    return optional_load(
        "shap_env_cols.pkl",
        joblib.load
    )


def load_training_distribution():
    return optional_load(
        "target_distributions.csv",
        pd.read_csv
    )


def load_env_groups():

    path = artifact_path("env_groups.json")

    if not os.path.exists(path):
        return None

    try:

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return None


UNITS = load_units()
RESULTS_TABLE = load_results_table()
ABLATION_RESULTS = load_ablation_results()
SHAP_BACKGROUND = load_shap_background()
SHAP_ENV_COLS = load_shap_env_cols()
TARGET_DIST = load_training_distribution()
ENV_GROUPS = load_env_groups()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model_artifacts():

    model_path = artifact_path(
        "multimodal_model.keras"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            "multimodal_model.keras was not found."
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

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


MODEL_LOADED = False
LOAD_ERROR = ""

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

except Exception as error:

    MODEL_LOADED = False
    LOAD_ERROR = str(error)


# ============================================================
# ENVIRONMENTAL GROUPS
# ============================================================

def infer_group(column_name):

    name = str(column_name).upper()

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


def get_env_group(column_name):

    if ENV_GROUPS is not None:

        if column_name in ENV_GROUPS:
            return ENV_GROUPS[column_name]

    return infer_group(column_name)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        image
    ).astype(np.float32)

    image_array = (
        tf.keras.applications.efficientnet
        .preprocess_input(image_array)
    )

    return np.expand_dims(
        image_array,
        axis=0
    )


# ============================================================
# ENVIRONMENTAL VECTOR
# ============================================================

def build_env_vector(user_values):

    row = ENV_MEDIANS.copy()

    for feature, value in user_values.items():

        if feature in row:
            row[feature] = value

    vector = np.array(
        [
            [
                row[column]
                for column in ENV_COLS
            ]
        ],
        dtype=np.float32
    )

    scaled_vector = env_scaler.transform(
        vector
    )

    return scaled_vector, row


# ============================================================
# PREDICTION
# ============================================================

def predict(image, user_environment):

    image_array = preprocess_image(
        image
    )

    env_scaled, full_env_row = (
        build_env_vector(
            user_environment
        )
    )

    predictions_scaled = model.predict(
        [image_array, env_scaled],
        verbose=0
    )

    predictions = target_scaler.inverse_transform(
        predictions_scaled
    )

    predictions = np.expm1(
        predictions
    )

    return (
        predictions[0],
        full_env_row,
        env_scaled
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def compute_shap_for_prediction(
    env_scaled_row,
    trait_index
):

    if SHAP_BACKGROUND is None:
        return None

    if SHAP_ENV_COLS is None:
        return None

    try:

        import shap

        def prediction_function(x):

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
                [dummy_images, x],
                verbose=0
            )

            return predictions[
                :,
                trait_index
            ]

        explainer = shap.KernelExplainer(
            prediction_function,
            SHAP_BACKGROUND
        )

        shap_values = explainer.shap_values(
            env_scaled_row,
            nsamples=100,
            silent=True
        )

        return shap_values

    except Exception:

        return None


# ============================================================
# GRAD-CAM
# ============================================================

def compute_gradcam(image):

    try:

        import cv2

        image_array = preprocess_image(
            image
        )

        base_model = None

        for layer in model.layers:

            if (
                "efficientnet"
                in layer.name.lower()
            ):

                base_model = layer
                break

        if base_model is None:
            return None

        convolution_layer = None

        for layer in reversed(
            base_model.layers
        ):

            try:

                output_shape = layer.output.shape

                if len(output_shape) == 4:

                    convolution_layer = layer
                    break

            except Exception:

                continue

        if convolution_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[
                convolution_layer.output,
                base_model.output
            ]
        )

        with tf.GradientTape() as tape:

            convolution_output, features = (
                grad_model(
                    image_array
                )
            )

            loss = tf.reduce_mean(
                features
            )

        gradients = tape.gradient(
            loss,
            convolution_output
        )

        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
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

        heatmap = heatmap / (
            maximum + 1e-8
        )

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (IMG_SIZE, IMG_SIZE)
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        original = np.array(
            image.convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE)
            )
        )

        original_bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        overlay = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )

        return overlay

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
        "Model could not be loaded."
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
        functional traits using a plant photograph together
        with environmental and geographic information.
        """
    )

    col1, col2, col3 = st.columns(3)

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

        number_environmental = (
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
                    {number_environmental}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        number_selected = (
            len(SELECTED_ENV_COLS)
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
                    {number_selected}
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
        learns relationships between environmental variables
        and plant traits.

        Fusion: the two representations are combined before
        producing the six numerical predictions.
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
            "Could not load model artifacts."
        )

        st.code(
            LOAD_ERROR
        )

    else:

        left, right = st.columns(
            [1, 1.2]
        )

        with left:

            st.subheader(
                "Plant Image"
            )

            uploaded_file = st.file_uploader(
                "Upload a plant image",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

            if uploaded_file is not None:

                try:

                    pil_image = Image.open(
                        uploaded_file
                    )

                    st.image(
                        pil_image,
                        use_container_width=True
                    )

                except Exception:

                    pil_image = None

                    st.error(
                        "The uploaded image could not "
                        "be read."
                    )

            else:

                pil_image = None

        with right:

            st.subheader(
                "Environmental Information"
            )

            st.write(
                f"""
                Enter values for the
                {len(SELECTED_ENV_COLS)}
                selected environmental variables.

                Remaining variables are automatically
                filled using training-data median values.
                """
            )

            user_values = {}

            input_columns = st.columns(2)

            for index, feature in enumerate(
                SELECTED_ENV_COLS
            ):

                default_value = float(
                    ENV_MEDIANS.get(
                        feature,
                        0.0
                    )
                )

                with input_columns[
                    index % 2
                ]:

                    user_values[feature] = (
                        st.number_input(
                            feature,
                            value=default_value,
                            format="%.4f",
                            key=(
                                f"environment_{feature}"
                            )
                        )
                    )

        st.divider()

        predict_clicked = st.button(
            "Predict Plant Traits",
            type="primary"
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
                            predictions,
                            environment_row,
                            environment_scaled
                        ) = predict(
                            pil_image,
                            user_values
                        )

                    st.session_state.last_prediction = (
                        predictions
                    )

                    st.session_state.last_image = (
                        pil_image
                    )

                    st.session_state.last_env_row = (
                        environment_row
                    )

                    st.session_state.last_env_scaled = (
                        environment_scaled
                    )

                    history_record = {
                        "Prediction": (
                            len(
                                st.session_state
                                .prediction_history
                            ) + 1
                        )
                    }

                    for column, value in zip(
                        TARGET_COLS,
                        predictions
                    ):

                        history_record[
                            TARGET_NAMES[column]
                        ] = float(value)

                    st.session_state.prediction_history.append(
                        history_record
                    )

                    st.success(
                        "Prediction completed successfully."
                    )

                except Exception as error:

                    st.error(
                        "Prediction failed."
                    )

                    st.exception(
                        error
                    )

        if (
            st.session_state.last_prediction
            is not None
        ):

            st.subheader(
                "Predicted Functional Traits"
            )

            predictions = (
                st.session_state.last_prediction
            )

            metric_columns = st.columns(3)

            for index, column in enumerate(
                TARGET_COLS
            ):

                unit = UNITS.get(
                    column,
                    ""
                )

                with metric_columns[
                    index % 3
                ]:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="label">
                                {TARGET_NAMES[column]}
                            </div>
                            <div class="value">
                                {predictions[index]:.2f}
                                <span class="unit">
                                    {unit}
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            report = pd.DataFrame(
                {
                    "Trait": [
                        TARGET_NAMES[column]
                        for column in TARGET_COLS
                    ],
                    "Predicted Value": predictions,
                    "Unit": [
                        UNITS.get(
                            column,
                            ""
                        )
                        for column in TARGET_COLS
                    ]
                }
            )

            csv_data = report.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(
                "Download Prediction Report",
                data=csv_data,
                file_name=(
                    "plant_trait_prediction.csv"
                ),
                mime="text/csv"
            )


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

elif page == "Prediction Analysis":

    st.title(
        "Prediction Analysis"
    )

    if st.session_state.last_prediction is None:

        st.info(
            "Make a prediction on the Predict page first."
        )

    else:

        predictions = (
            st.session_state.last_prediction
        )

        st.subheader(
            "Prediction Profile"
        )

        figure = go.Figure()

        figure.add_trace(
            go.Bar(
                x=[
                    TARGET_NAMES[column]
                    for column in TARGET_COLS
                ],
                y=predictions
            )
        )

        figure.update_layout(
            yaxis_title="Predicted value",
            height=450
        )

        st.plotly_chart(
            figure,
            use_container_width=True
        )

        if TARGET_DIST is not None:

            st.subheader(
                "Prediction vs Training Distribution"
            )

            trait_choice = st.selectbox(
                "Select trait",
                [
                    TARGET_NAMES[column]
                    for column in TARGET_COLS
                ],
                key="distribution_trait"
            )

            trait_column = next(
                column
                for column, name
                in TARGET_NAMES.items()
                if name == trait_choice
            )

            if trait_column in TARGET_DIST.columns:

                figure2 = go.Figure()

                figure2.add_trace(
                    go.Histogram(
                        x=TARGET_DIST[
                            trait_column
                        ],
                        nbinsx=40
                    )
                )

                trait_index = (
                    TARGET_COLS.index(
                        trait_column
                    )
                )

                figure2.add_vline(
                    x=predictions[
                        trait_index
                    ],
                    line_width=2,
                    annotation_text="Prediction"
                )

                figure2.update_layout(
                    height=400
                )

                st.plotly_chart(
                    figure2,
                    use_container_width=True
                )

        if len(
            st.session_state.prediction_history
        ) > 1:

            st.subheader(
                "Prediction History"
            )

            st.dataframe(
                pd.DataFrame(
                    st.session_state
                    .prediction_history
                ),
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
        """
        This section shows which environmental variables
        and image regions influenced the model prediction.
        """
    )

    if st.session_state.last_prediction is None:

        st.info(
            "Make a prediction first."
        )

    else:

        trait_choice = st.selectbox(
            "Select trait",
            [
                TARGET_NAMES[column]
                for column in TARGET_COLS
            ]
        )

        trait_column = next(
            column
            for column, name
            in TARGET_NAMES.items()
            if name == trait_choice
        )

        trait_index = TARGET_COLS.index(
            trait_column
        )

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Environmental Contribution"
            )

            if (
                SHAP_BACKGROUND is None
                or SHAP_ENV_COLS is None
            ):

                st.info(
                    """
                    SHAP files are not available.

                    Add shap_background.pkl and
                    shap_env_cols.pkl to enable this section.
                    """
                )

            else:

                with st.spinner(
                    "Calculating SHAP explanation..."
                ):

                    shap_values = (
                        compute_shap_for_prediction(
                            st.session_state
                            .last_env_scaled,
                            trait_index
                        )
                    )

                if shap_values is None:

                    st.warning(
                        "SHAP explanation could not be generated."
                    )

                else:

                    shap_values = np.asarray(
                        shap_values
                    ).flatten()

                    shap_dataframe = pd.DataFrame(
                        {
                            "Feature": SHAP_ENV_COLS,
                            "Contribution": shap_values
                        }
                    )

                    shap_dataframe[
                        "Absolute"
                    ] = shap_dataframe[
                        "Contribution"
                    ].abs()

                    shap_dataframe = (
                        shap_dataframe
                        .sort_values(
                            "Absolute"
                        )
                        .tail(10)
                    )

                    figure3 = go.Figure(
                        go.Bar(
                            x=shap_dataframe[
                                "Contribution"
                            ],
                            y=shap_dataframe[
                                "Feature"
                            ],
                            orientation="h"
                        )
                    )

                    figure3.update_layout(
                        height=450
                    )

                    st.plotly_chart(
                        figure3,
                        use_container_width=True
                    )

                    st.caption(
                        """
                        SHAP values describe model attribution.
                        They should not be interpreted as proof
                        of biological causation.
                        """
                    )

        with right:

            st.subheader(
                "Visual Explanation"
            )

            overlay = compute_gradcam(
                st.session_state.last_image
            )

            if overlay is not None:

                st.image(
                    overlay,
                    use_container_width=True
                )

                st.caption(
                    """
                    Highlighted image regions represent areas
                    emphasized by the image branch.
                    """
                )

            else:

                st.info(
                    """
                    Grad-CAM could not be generated for
                    this model architecture.
                    """
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "Model Performance"
    )

    if RESULTS_TABLE is None:

        st.info(
            "results_table.csv is not available."
        )

    else:

        multimodal = RESULTS_TABLE[
            RESULTS_TABLE["Model"]
            == "Multimodal CNN+MLP"
        ]

        if multimodal.empty:

            st.warning(
                """
                The results table does not contain
                a Multimodal CNN+MLP model.
                """
            )

        else:

            st.subheader(
                "Multimodal Model Metrics"
            )

            available_columns = [
                column
                for column in [
                    "Trait",
                    "MAE",
                    "RMSE",
                    "R2"
                ]
                if column in multimodal.columns
            ]

            st.dataframe(
                multimodal[
                    available_columns
                ].reset_index(drop=True),
                use_container_width=True
            )

            if "R2" in multimodal.columns:

                figure4 = go.Figure(
                    go.Bar(
                        x=multimodal["Trait"],
                        y=multimodal["R2"]
                    )
                )

                figure4.update_layout(
                    yaxis_title="R²",
                    height=400
                )

                st.plotly_chart(
                    figure4,
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
        This page compares the multimodal model with
        image-only, environmental-only and baseline models.
        """
    )

    if RESULTS_TABLE is None:

        st.info(
            "results_table.csv is not available."
        )

    else:

        required_columns = {
            "Model",
            "MAE",
            "RMSE",
            "R2"
        }

        if not required_columns.issubset(
            RESULTS_TABLE.columns
        ):

            st.error(
                """
                results_table.csv does not contain
                the required columns.
                """
            )

        else:

            summary = (
                RESULTS_TABLE
                .groupby("Model")[
                    ["MAE", "RMSE", "R2"]
                ]
                .mean()
                .reset_index()
            )

            st.subheader(
                "Overall Metrics"
            )

            st.dataframe(
                summary,
                use_container_width=True
            )

            if {
                "Trait",
                "Model",
                "R2"
            }.issubset(
                RESULTS_TABLE.columns
            ):

                pivot = RESULTS_TABLE.pivot_table(
                    index="Trait",
                    columns="Model",
                    values="R2"
                )

                figure5 = go.Figure()

                for model_name in pivot.columns:

                    figure5.add_trace(
                        go.Bar(
                            name=model_name,
                            x=pivot.index,
                            y=pivot[
                                model_name
                            ]
                        )
                    )

                figure5.update_layout(
                    barmode="group",
                    yaxis_title="R²",
                    height=500
                )

                st.plotly_chart(
                    figure5,
                    use_container_width=True
                )


# ============================================================
# ENVIRONMENTAL ANALYSIS
# ============================================================

elif page == "Environmental Analysis":

    st.title(
        "Environmental Analysis"
    )

    if st.session_state.last_env_row is None:

        st.info(
            "Make a prediction first."
        )

    else:

        environment_row = (
            st.session_state.last_env_row
        )

        groups = {}

        for column in ENV_COLS:

            group = get_env_group(
                column
            )

            groups.setdefault(
                group,
                []
            ).append(
                column
            )

        st.subheader(
            "Environmental Variable Groups"
        )

        group_counts = pd.DataFrame(
            {
                "Group": list(
                    groups.keys()
                ),
                "Number of variables": [
                    len(values)
                    for values in groups.values()
                ]
            }
        )

        figure6 = go.Figure(
            go.Bar(
                x=group_counts["Group"],
                y=group_counts[
                    "Number of variables"
                ]
            )
        )

        figure6.update_layout(
            height=350
        )

        st.plotly_chart(
            figure6,
            use_container_width=True
        )

        selected_group = st.selectbox(
            "Select environmental group",
            list(groups.keys())
        )

        group_dataframe = pd.DataFrame(
            {
                "Variable": groups[
                    selected_group
                ],
                "Value": [
                    environment_row.get(
                        column,
                        np.nan
                    )
                    for column in groups[
                        selected_group
                    ]
                ]
            }
        )

        st.dataframe(
            group_dataframe,
            use_container_width=True,
            height=400
        )

        st.caption(
            """
            This page displays the environmental input
            profile. It does not calculate causal
            group-level effects.
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
        The ablation study evaluates how prediction
        performance changes when individual information
        sources are removed.
        """
    )

    if ABLATION_RESULTS is None:

        st.info(
            """
            ablation_results.csv is not available.

            Run the ablation experiments and add the
            resulting CSV file to the repository.
            """
        )

    else:

        st.dataframe(
            ABLATION_RESULTS,
            use_container_width=True
        )

        if {
            "Configuration",
            "R2"
        }.issubset(
            ABLATION_RESULTS.columns
        ):

            if "Trait" in ABLATION_RESULTS.columns:

                figure7 = px.bar(
                    ABLATION_RESULTS,
                    x="Configuration",
                    y="R2",
                    color="Trait",
                    barmode="group"
                )

            else:

                figure7 = px.bar(
                    ABLATION_RESULTS,
                    x="Configuration",
                    y="R2"
                )

            figure7.update_layout(
                height=500
            )

            st.plotly_chart(
                figure7,
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
        "Predicted Traits"
    )

    for column in TARGET_COLS:

        st.write(
            f"• {TARGET_NAMES[column]}"
        )

    st.subheader(
        "Model Architecture"
    )

    st.write(
        """
        The image branch uses EfficientNetB0 to extract
        visual features from plant photographs.

        The environmental branch uses a multilayer
        perceptron to process environmental and geographic
        variables.

        The outputs of both branches are fused and passed
        through a prediction head that simultaneously
        estimates six continuous plant functional traits.
        """
    )

    st.subheader(
        "Models Compared"
    )

    st.write(
        """
        The research compares a linear regression baseline,
        an environmental-only MLP, an image-only CNN, and
        the multimodal CNN+MLP architecture.
        """
    )

    st.subheader(
        "Interpretability"
    )

    st.write(
        """
        SHAP is used to investigate the contribution of
        environmental variables, while Grad-CAM is used
        to visualize influential regions of the plant image.

        These techniques describe model behavior and should
        not be interpreted as evidence of biological causation.
        """
    )

    if MODEL_LOADED:

        st.subheader(
            "Model Information"
        )

        information = pd.DataFrame(
            {
                "Property": [
                    "Image resolution",
                    "Environmental variables",
                    "Selected variables",
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
            information,
            use_container_width=True
        )
