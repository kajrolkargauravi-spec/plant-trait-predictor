# ============================================================
# PLANT FUNCTIONAL TRAIT PREDICTION
# Multimodal Deep Learning Streamlit Application
# ============================================================

import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

from PIL import Image

# SHAP is optional so that the main prediction can still work
# if SHAP encounters a deployment/environment issue.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Functional Trait Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .trait-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
        text-align: center;
    }

    .trait-name {
        font-size: 15px;
        color: #666666;
    }

    .trait-value {
        font-size: 28px;
        font-weight: 700;
    }

    .research-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7f6;
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "plant_trait_model"

MULTIMODAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "multimodal_model.keras"
)

ENVIRONMENT_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "environment_mlp.keras"
)

IMAGE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "image_cnn.keras"
)

TABULAR_IMPUTER_PATH = os.path.join(
    MODEL_DIR,
    "tabular_imputer.pkl"
)

TABULAR_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "tabular_scaler.pkl"
)

TARGET_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "target_scaler.pkl"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.json"
)

RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data
def load_config():

    if not os.path.exists(CONFIG_PATH):
        st.error(
            "config.json was not found inside plant_trait_model."
        )
        st.stop()

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return config


config = load_config()

TARGETS = config["targets"]

NUMERIC_FEATURES = config["numeric_features"]

IMAGE_SIZE = tuple(
    config.get(
        "image_size",
        [224, 224]
    )
)


# ============================================================
# HUMAN-READABLE TRAIT NAMES
# ============================================================

# These names correspond to the six intended plant traits.
# Units should ultimately be checked against target_name_meta.tsv.

TRAIT_NAMES = {
    "X4_mean":
        "Stem Specific Density",

    "X11_mean":
        "Specific Leaf Area",

    "X18_mean":
        "Plant Height",

    "X26_mean":
        "Seed Dry Mass",

    "X50_mean":
        "Leaf Nitrogen per Area",

    "X3112_mean":
        "Leaf Area"
}


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    multimodal_model = tf.keras.models.load_model(
        MULTIMODAL_MODEL_PATH,
        compile=False
    )

    environment_model = tf.keras.models.load_model(
        ENVIRONMENT_MODEL_PATH,
        compile=False
    )

    image_model = tf.keras.models.load_model(
        IMAGE_MODEL_PATH,
        compile=False
    )

    return (
        multimodal_model,
        environment_model,
        image_model
    )


# ============================================================
# LOAD PREPROCESSING
# ============================================================

@st.cache_resource
def load_preprocessors():

    imputer = joblib.load(
        TABULAR_IMPUTER_PATH
    )

    tabular_scaler = joblib.load(
        TABULAR_SCALER_PATH
    )

    target_scaler = joblib.load(
        TARGET_SCALER_PATH
    )

    return (
        imputer,
        tabular_scaler,
        target_scaler
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    uploaded_image
):

    image = Image.open(
        uploaded_image
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    # This must match the preprocessing
    # used during model training.
    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_array


# ============================================================
# TABULAR PREPROCESSING
# ============================================================

def preprocess_tabular(
    values,
    imputer,
    scaler
):

    df = pd.DataFrame(
        [values],
        columns=NUMERIC_FEATURES
    )

    df = imputer.transform(
        df
    )

    df = scaler.transform(
        df
    )

    return df.astype(
        np.float32
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_traits(
    model,
    image_array,
    tabular_array,
    target_scaler
):

    prediction_scaled = model.predict(
        {
            "image_input":
                image_array,

            "environment_input":
                tabular_array
        },
        verbose=0
    )

    prediction = target_scaler.inverse_transform(
        prediction_scaled
    )

    return prediction[0]


# ============================================================
# FORMAT TRAIT VALUE
# ============================================================

def format_value(
    value
):

    if abs(value) >= 1000:
        return f"{value:,.1f}"

    elif abs(value) >= 100:
        return f"{value:,.2f}"

    elif abs(value) >= 1:
        return f"{value:.3f}"

    else:
        return f"{value:.4f}"


# ============================================================
# SHAP EXPLANATION
# ============================================================

def calculate_shap_explanation(
    environment_model,
    tabular_array,
    tabular_scaler
):

    if not SHAP_AVAILABLE:
        return None

    try:

        # A small background set centered around
        # the standardized feature space.
        background = np.zeros(
            (
                1,
                len(NUMERIC_FEATURES)
            ),
            dtype=np.float32
        )

        def prediction_function(X):

            predictions = environment_model.predict(
                X,
                verbose=0
            )

            return predictions

        explainer = shap.Explainer(
            prediction_function,
            background
        )

        shap_values = explainer(
            tabular_array
        )

        return shap_values

    except Exception as e:

        print(
            "SHAP error:",
            e
        )

        return None


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(
    model
):

    # First search backwards for a 4D output.
    for layer in reversed(
        model.layers
    ):

        try:

            output_shape = layer.output.shape

            if len(output_shape) == 4:

                return layer.name

        except Exception:
            continue

    return None


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(
    model,
    image_array,
    target_index
):

    last_conv_layer_name = (
        find_last_conv_layer(model)
    )

    if last_conv_layer_name is None:
        return None

    try:

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(
                    last_conv_layer_name
                ).output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = (
                grad_model(
                    {
                        "image_input":
                            image_array,

                        "environment_input":
                            np.zeros(
                                (
                                    1,
                                    len(
                                        NUMERIC_FEATURES
                                    )
                                ),
                                dtype=np.float32
                            )
                    }
                )
            )

            target_output = predictions[
                :, target_index
            ]

        gradients = tape.gradient(
            target_output,
            conv_outputs
        )

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = (
            conv_outputs
            *
            pooled_gradients
        )

        heatmap = tf.reduce_sum(
            heatmap,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        maximum = tf.reduce_max(
            heatmap
        )

        if maximum > 0:

            heatmap = (
                heatmap / maximum
            )

        return heatmap.numpy()

    except Exception as e:

        print(
            "Grad-CAM error:",
            e
        )

        return None


# ============================================================
# GRAD-CAM DISPLAY
# ============================================================

def display_gradcam(
    original_image,
    heatmap
):

    if heatmap is None:

        st.warning(
            "Grad-CAM could not be generated for this model."
        )

        return

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.imshow(
        original_image
    )

    ax.imshow(
        heatmap,
        alpha=0.45,
        cmap="jet"
    )

    ax.axis("off")

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        multimodal_model,
        environment_model,
        image_model
    ) = load_models()

    (
        tabular_imputer,
        tabular_scaler,
        target_scaler
    ) = load_preprocessors()

except Exception as e:

    st.error(
        "Unable to load the trained model files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌿 Plant Trait AI"
)

st.sidebar.markdown(
    """
    **Multimodal Deep Learning**

    Predict continuous plant functional
    traits using plant imagery and
    environmental information.
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🌱 Predict Traits",
        "🔍 Explainability",
        "📊 Model Performance",
        "🧪 Validation Demo",
        "📖 Research Overview"
    ]
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "🌱 Predict Traits":

    st.markdown(
        '<div class="main-title">'
        '🌿 Plant Functional Trait Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Multimodal Deep Learning using Plant Images '
        'and Environmental Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Upload a plant image and provide environmental
        information to estimate six continuous plant
        functional traits.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📷 Plant Image'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a plant image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    image = None
    image_array = None

    if uploaded_file is not None:

        image, image_array = preprocess_image(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded plant image",
            width=400
        )

    st.divider()

    # --------------------------------------------------------
    # ENVIRONMENTAL INPUTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🌎 Environmental Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Enter the environmental/contextual values "
        "required by the trained model."
    )

    input_values = {}

    # Group features based on their prefixes where possible.
    climate_features = [
        f for f in NUMERIC_FEATURES
        if "WORLDCLIM" in f.upper()
        or "CLIMATE" in f.upper()
    ]

    soil_features = [
        f for f in NUMERIC_FEATURES
        if "SOIL" in f.upper()
    ]

    satellite_features = [
        f for f in NUMERIC_FEATURES
        if "MODIS" in f.upper()
        or "VOD" in f.upper()
        or "SAT" in f.upper()
    ]

    used_features = set()

    def create_feature_inputs(
        feature_list,
        title
    ):

        if not feature_list:
            return

        with st.expander(
            title,
            expanded=False
        ):

            columns = st.columns(2)

            for i, feature in enumerate(
                feature_list
            ):

                with columns[
                    i % 2
                ]:

                    input_values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.6f",
                        key=f"input_{feature}"
                    )

                used_features.add(
                    feature
                )

    create_feature_inputs(
        climate_features,
        "🌡️ Climate Variables"
    )

    create_feature_inputs(
        soil_features,
        "🌱 Soil Variables"
    )

    create_feature_inputs(
        satellite_features,
        "🛰️ Satellite Variables"
    )

    remaining_features = [
        f for f in NUMERIC_FEATURES
        if f not in used_features
    ]

    create_feature_inputs(
        remaining_features,
        "📊 Other Environmental Variables"
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🌱 Predict Plant Traits",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        if uploaded_file is None:

            st.warning(
                "Please upload a plant image first."
            )

        elif len(input_values) != len(
            NUMERIC_FEATURES
        ):

            st.warning(
                "Please provide all required "
                "environmental inputs."
            )

        else:

            with st.spinner(
                "Running multimodal deep-learning model..."
            ):

                tabular_array = preprocess_tabular(
                    input_values,
                    tabular_imputer,
                    tabular_scaler
                )

                predictions = predict_traits(
                    multimodal_model,
                    image_array,
                    tabular_array,
                    target_scaler
                )

            st.session_state[
                "latest_predictions"
            ] = predictions

            st.session_state[
                "latest_image"
            ] = image

            st.session_state[
                "latest_image_array"
            ] = image_array

            st.session_state[
                "latest_tabular"
            ] = tabular_array

            st.session_state[
                "latest_inputs"
            ] = input_values

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.success(
                "Prediction completed successfully."
            )

            st.markdown(
                '<div class="section-title">'
                '📈 Predicted Plant Traits'
                '</div>',
                unsafe_allow_html=True
            )

            cols = st.columns(3)

            for i, target in enumerate(
                TARGETS
            ):

                trait_name = TRAIT_NAMES.get(
                    target,
                    target
                )

                value = predictions[i]

                with cols[i % 3]:

                    st.metric(
                        label=trait_name,
                        value=format_value(value)
                    )

            st.divider()

            st.markdown(
                """
                **Model:** Multimodal CNN + MLP  
                **Image backbone:** EfficientNetB0  
                **Task:** Multi-output regression  
                **Outputs:** Six continuous plant traits
                """
            )


# ============================================================
# PAGE 2 — EXPLAINABILITY
# ============================================================

elif page == "🔍 Explainability":

    st.markdown(
        '<div class="main-title">'
        '🔍 Model Explainability'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        This section investigates which environmental
        variables and image regions contributed to the
        model's predictions.
        """
    )

    if (
        "latest_predictions"
        not in st.session_state
    ):

        st.info(
            "Make a prediction first from the "
            "'Predict Traits' page."
        )

    else:

        predictions = st.session_state[
            "latest_predictions"
        ]

        image = st.session_state[
            "latest_image"
        ]

        image_array = st.session_state[
            "latest_image_array"
        ]

        tabular_array = st.session_state[
            "latest_tabular"
        ]

        selected_trait = st.selectbox(
            "Select a plant trait",
            TARGETS,
            format_func=lambda x:
                TRAIT_NAMES.get(x, x)
        )

        target_index = TARGETS.index(
            selected_trait
        )

        st.markdown(
            '<div class="section-title">'
            '🌿 Selected Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        st.metric(
            TRAIT_NAMES.get(
                selected_trait,
                selected_trait
            ),
            format_value(
                predictions[target_index]
            )
        )

        st.divider()

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🌎 Environmental Feature Contribution'
            '</div>',
            unsafe_allow_html=True
        )

        if not SHAP_AVAILABLE:

            st.warning(
                "SHAP is not installed in this environment."
            )

        else:

            with st.spinner(
                "Calculating SHAP explanation..."
            ):

                shap_values = (
                    calculate_shap_explanation(
                        environment_model,
                        tabular_array,
                        tabular_scaler
                    )
                )

            if shap_values is None:

                st.warning(
                    "A local SHAP explanation could "
                    "not be generated."
                )

            else:

                try:

                    values = shap_values.values

                    if values.ndim == 3:

                        values = values[
                            0,
                            :,
                            target_index
                        ]

                    elif values.ndim == 2:

                        values = values[0]

                    else:

                        values = np.asarray(
                            values
                        ).flatten()

                    shap_df = pd.DataFrame({

                        "Feature":
                            NUMERIC_FEATURES,

                        "SHAP Contribution":
                            values

                    })

                    shap_df[
                        "Absolute Contribution"
                    ] = np.abs(
                        shap_df[
                            "SHAP Contribution"
                        ]
                    )

                    shap_df = shap_df.sort_values(
                        "Absolute Contribution",
                        ascending=False
                    )

                    st.dataframe(
                        shap_df[
                            [
                                "Feature",
                                "SHAP Contribution"
                            ]
                        ].head(15),
                        use_container_width=True,
                        hide_index=True
                    )

                    fig, ax = plt.subplots(
                        figsize=(9, 6)
                    )

                    plot_df = shap_df.head(
                        12
                    ).sort_values(
                        "SHAP Contribution"
                    )

                    ax.barh(
                        plot_df["Feature"],
                        plot_df[
                            "SHAP Contribution"
                        ]
                    )

                    ax.axvline(
                        0,
                        linestyle="--"
                    )

                    ax.set_xlabel(
                        "SHAP Contribution"
                    )

                    ax.set_title(
                        "Environmental Feature Contributions"
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    plt.close(fig)

                except Exception as e:

                    st.warning(
                        "SHAP visualization could "
                        "not be generated."
                    )

                    st.caption(
                        str(e)
                    )

        st.divider()

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📷 Visual Explanation — Grad-CAM'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            Grad-CAM highlights image regions that
            contributed to the selected trait prediction.
            """
        )

        heatmap = make_gradcam(
            multimodal_model,
            image_array,
            target_index
        )

        if heatmap is not None:

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Original Plant Image",
                    use_container_width=True
                )

            with col2:

                display_gradcam(
                    image,
                    heatmap
                )

        else:

            st.warning(
                "Grad-CAM could not be generated."
            )

        st.caption(
            "SHAP and Grad-CAM indicate model attribution. "
            "They should not be interpreted as evidence of "
            "biological causation."
        )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="main-title">'
        '📊 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Comparison of the regression models evaluated
        during the research experiment.
        """
    )

    if os.path.exists(
        RESULTS_PATH
    ):

        results = pd.read_csv(
            RESULTS_PATH
        )

        # Rename R2 for display if necessary
        if "R2" in results.columns:

            display_results = results.rename(
                columns={
                    "R2": "R²"
                }
            )

        else:

            display_results = results.copy()

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # R2 CHART
        # ----------------------------------------------------

        if "R2" in results.columns:

            st.subheader(
                "R² Comparison Across Traits"
            )

            pivot_r2 = results.pivot(
                index="Trait",
                columns="Model",
                values="R2"
            )

            st.bar_chart(
                pivot_r2
            )

        # ----------------------------------------------------
        # MAE CHART
        # ----------------------------------------------------

        if "MAE" in results.columns:

            st.subheader(
                "MAE Comparison Across Traits"
            )

            pivot_mae = results.pivot(
                index="Trait",
                columns="Model",
                values="MAE"
            )

            st.bar_chart(
                pivot_mae
            )

        # ----------------------------------------------------
        # RMSE CHART
        # ----------------------------------------------------

        if "RMSE" in results.columns:

            st.subheader(
                "RMSE Comparison Across Traits"
            )

            pivot_rmse = results.pivot(
                index="Trait",
                columns="Model",
                values="RMSE"
            )

            st.bar_chart(
                pivot_rmse
            )

    else:

        st.warning(
            "model_comparison.csv was not found."
        )


# ============================================================
# PAGE 4 — VALIDATION DEMO
# ============================================================

elif page == "🧪 Validation Demo":

    st.markdown(
        '<div class="main-title">'
        '🧪 Validation Sample'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Examine predictions made on validation observations
        that were not used to fit the final model.
        """
    )

    validation_path = os.path.join(
        MODEL_DIR,
        "validation_predictions.csv"
    )

    if not os.path.exists(
        validation_path
    ):

        st.warning(
            "validation_predictions.csv was not found."
        )

    else:

        validation_df = pd.read_csv(
            validation_path
        )

        row_number = st.number_input(
            "Select validation observation",
            min_value=0,
            max_value=len(
                validation_df
            ) - 1,
            value=0,
            step=1
        )

        row = validation_df.iloc[
            row_number
        ]

        st.write(
            "Observation ID:",
            row["id"]
        )

        comparison_rows = []

        for target in TARGETS:

            actual_column = (
                f"{target}_actual"
            )

            predicted_column = (
                f"{target}_predicted"
            )

            residual_column = (
                f"{target}_residual"
            )

            comparison_rows.append({

                "Trait":
                    TRAIT_NAMES.get(
                        target,
                        target
                    ),

                "Actual":
                    row[actual_column],

                "Predicted":
                    row[predicted_column],

                "Residual":
                    row[residual_column]

            })

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 5 — RESEARCH OVERVIEW
# ============================================================

elif page == "📖 Research Overview":

    st.markdown(
        '<div class="main-title">'
        '🧪 Research Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Research Question

        Can combining plant imagery with environmental
        information improve the prediction of continuous
        plant functional traits?
        """
    )

    st.divider()

    st.markdown(
        """
        ### Multimodal Architecture
        """
    )

    st.code(
        """
                    PLANT IMAGE
                         │
                         ▼
                   EfficientNetB0
                         │
                         ▼
                   Image Features
                         │
                         │
                         ├──────────────┐
                                        │
                                        ▼
                                FEATURE FUSION
                                        ▲
                                        │
                         ┌──────────────┤
                         │
                  Environmental Data
                         │
                         ▼
                 Environmental MLP
                         │
                         ▼
                Environmental Features
                                        │
                                        ▼
                              Fusion Network
                                        │
                                        ▼
                         Six Numerical Predictions
        """,
        language="text"
    )

    st.divider()

    st.markdown(
        """
        ### Models Evaluated

        **Linear Regression**

        A conventional tabular regression baseline.

        **Environmental MLP**

        A neural network using environmental/contextual
        information only.

        **Image CNN**

        An EfficientNetB0-based image regression model.

        **Multimodal CNN + MLP**

        The proposed model combines visual and environmental
        representations before producing the six continuous
        trait predictions.
        """
    )

    st.divider()

    st.markdown(
        """
        ### Explainability

        **SHAP** is used to investigate the contribution of
        environmental features to model predictions.

        **Grad-CAM** is used to visualize influential regions
        of plant images.

        These methods provide model attribution and should not
        be interpreted as proof of biological causation.
        """
    )

    st.divider()

    st.markdown(
        """
        ### Prediction Task

        This application performs **multi-output regression**.

        The model predicts six continuous plant functional
        traits rather than categorical classes.
        """
    )

    st.divider()

    st.markdown(
        """
        ### Reproducibility

        The Streamlit application uses the trained model and
        preprocessing objects generated during the research
        pipeline.

        No model training is performed inside the application.
        """
    )