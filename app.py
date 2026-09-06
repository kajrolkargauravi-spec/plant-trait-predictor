```python
import os
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Functional Trait Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "multimodal_model.keras"
ENV_COLS_PATH = "env_cols.pkl"
ENV_MEDIANS_PATH = "env_medians.pkl"
ENV_SCALER_PATH = "env_scaler.pkl"
SELECTED_ENV_COLS_PATH = "selected_env_cols.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"


# ============================================================
# TARGET INFORMATION
# ============================================================

TARGETS = [
    "X4_mean",
    "X11_mean",
    "X18_mean",
    "X26_mean",
    "X50_mean",
    "X3112_mean"
]

TRAIT_NAMES = {
    "X4_mean": "Stem Specific Density",
    "X11_mean": "Specific Leaf Area",
    "X18_mean": "Plant Height",
    "X26_mean": "Seed Dry Mass",
    "X50_mean": "Leaf Nitrogen per Area",
    "X3112_mean": "Leaf Area"
}


# ============================================================
# CSS
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
        font-size: 19px;
        color: #666666;
        margin-bottom: 25px;
    }

    .trait-card {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        text-align: center;
        min-height: 140px;
    }

    .trait-name {
        font-size: 15px;
        color: #666666;
        margin-bottom: 10px;
    }

    .trait-value {
        font-size: 28px;
        font-weight: 700;
    }

    .section-box {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        margin-bottom: 20px;
    }

    .small-text {
        color: #666666;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_and_preprocessors():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    env_cols = joblib.load(
        ENV_COLS_PATH
    )

    env_medians = joblib.load(
        ENV_MEDIANS_PATH
    )

    env_scaler = joblib.load(
        ENV_SCALER_PATH
    )

    selected_env_cols = joblib.load(
        SELECTED_ENV_COLS_PATH
    )

    target_scaler = joblib.load(
        TARGET_SCALER_PATH
    )

    return (
        model,
        env_cols,
        env_medians,
        env_scaler,
        selected_env_cols,
        target_scaler
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_PATH,
    ENV_COLS_PATH,
    ENV_MEDIANS_PATH,
    ENV_SCALER_PATH,
    SELECTED_ENV_COLS_PATH,
    TARGET_SCALER_PATH
]

missing_files = [
    f for f in required_files
    if not os.path.exists(f)
]

if missing_files:

    st.error("Some required model files are missing.")

    for f in missing_files:
        st.write(f"• `{f}`")

    st.stop()


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        model,
        env_cols,
        env_medians,
        env_scaler,
        selected_env_cols,
        target_scaler
    ) = load_model_and_preprocessors()

except Exception as e:

    st.error("The trained model could not be loaded.")

    st.exception(e)

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "image" not in st.session_state:
    st.session_state.image = None

if "image_array" not in st.session_state:
    st.session_state.image_array = None

if "environment_array" not in st.session_state:
    st.session_state.environment_array = None

if "environment_original" not in st.session_state:
    st.session_state.environment_original = None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    original_image = image.copy()

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return original_image, image_array


# ============================================================
# ENVIRONMENT PREPROCESSING
# ============================================================

def preprocess_environment(df):

    df = df.copy()

    # --------------------------------------------------------
    # Make sure every required feature exists.
    # Missing variables are filled with training medians.
    # --------------------------------------------------------

    for col in env_cols:

        if col not in df.columns:

            df[col] = env_medians.get(
                col,
                0.0
            )

    # --------------------------------------------------------
    # Keep the exact training feature order.
    # --------------------------------------------------------

    df = df[env_cols]

    # --------------------------------------------------------
    # Convert to numerical values.
    # --------------------------------------------------------

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Fill missing values.
    # --------------------------------------------------------

    for col in env_cols:

        median_value = env_medians.get(
            col,
            0.0
        )

        df[col] = df[col].fillna(
            median_value
        )

    # --------------------------------------------------------
    # Convert to NumPy.
    # --------------------------------------------------------

    values = df.values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Apply the SAME StandardScaler used during training.
    # --------------------------------------------------------

    values = env_scaler.transform(
        values
    )

    return values.astype(
        np.float32
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_traits(
    image_array,
    environment_array
):

    prediction_scaled = model.predict(
        {
            "image": image_array,
            "env": environment_array
        },
        verbose=0
    )

    prediction_scaled = np.asarray(
        prediction_scaled
    )

    prediction = target_scaler.inverse_transform(
        prediction_scaled
    )

    return prediction[0]


# ============================================================
# NUMBER FORMATTER
# ============================================================

def format_value(value):

    if not np.isfinite(value):
        return "N/A"

    return f"{value:,.4f}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌿 Plant Trait AI")

st.sidebar.caption(
    "Multimodal Deep Learning Research Demonstrator"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Research Overview",
        "🌱 Predict Traits",
        "📊 Model Performance",
        "🔍 Explainability",
        "🌍 Environmental Features",
        "🧠 Model Architecture"
    ]
)

st.sidebar.divider()

st.sidebar.write(
    f"**Environmental inputs:** {len(env_cols)}"
)

st.sidebar.write(
    f"**Selected features:** {len(selected_env_cols)}"
)

st.sidebar.write(
    "**Outputs:** 6 continuous traits"
)


# ============================================================
# PAGE 1: RESEARCH OVERVIEW
# ============================================================

if page == "🏠 Research Overview":

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
        <div class="section-box">

        <h3>Research Concept</h3>

        This project investigates whether visual information
        from plant images combined with environmental information
        can be used to predict continuous plant functional traits.

        The trained model receives two forms of information:

        <br><br>

        <b>Plant image</b><br>
        A 224 × 224 RGB image.

        <br><br>

        <b>Environmental information</b><br>
        163 climate, soil, satellite reflectance and vegetation
        variables.

        <br><br>

        The multimodal model produces six numerical plant
        functional trait predictions.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Research Pipeline")

    st.code(
        """
        PLANT IMAGE
             │
             ▼
        IMAGE BRANCH
             │
             │
             ├──────────────┐
                            │
                            ▼
                         FUSION
                            ▲
                            │
             ┌──────────────┘
             │
        ENVIRONMENTAL DATA
             │
             ▼
        163 VARIABLES
             │
             ▼
        PREPROCESSING
             │
             ▼
        ENVIRONMENT BRANCH
             │
             ▼
        MULTIMODAL MODEL
             │
             ▼
        MULTI-OUTPUT REGRESSION
             │
        ┌────┼────┬────┬────┬────┐
        ▼    ▼    ▼    ▼    ▼    ▼
       T1   T2   T3   T4   T5   T6
        """,
        language="text"
    )

    st.subheader("Model Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Image Input",
            "224 × 224 × 3"
        )

    with c2:
        st.metric(
            "Environmental Inputs",
            "163"
        )

    with c3:
        st.metric(
            "Selected Features",
            "15"
        )

    with c4:
        st.metric(
            "Predicted Traits",
            "6"
        )

    st.subheader("Predicted Traits")

    for target in TARGETS:

        st.write(
            f"• **{TRAIT_NAMES[target]}**"
        )

    st.info(
        "This is a multi-output regression problem. "
        "The model predicts continuous numerical values, "
        "not categorical classes."
    )


# ============================================================
# PAGE 2: PREDICTION
# ============================================================

elif page == "🌱 Predict Traits":

    st.markdown(
        '<div class="main-title">'
        '🌱 Predict Plant Functional Traits'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload a plant image and environmental data.'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("1. Upload Plant Image")

    uploaded_image = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    image = None
    image_array = None

    if uploaded_image is not None:

        try:

            image, image_array = preprocess_image(
                uploaded_image
            )

            st.image(
                image,
                caption="Uploaded plant image",
                width=400
            )

        except Exception as e:

            st.error(
                "Could not process the image."
            )

            st.exception(e)

    st.divider()

    st.subheader("2. Upload Environmental Data")

    st.write(
        """
        Upload a CSV containing environmental observations.
        The application automatically places the variables in
        the same order used during model training.
        """
    )

    uploaded_csv = st.file_uploader(
        "Choose environmental CSV",
        type=["csv"]
    )

    environmental_data = None

    if uploaded_csv is not None:

        try:

            environmental_data = pd.read_csv(
                uploaded_csv
            )

            st.success(
                f"Environmental dataset loaded: "
                f"{environmental_data.shape[0]} rows × "
                f"{environmental_data.shape[1]} columns."
            )

            with st.expander(
                "Preview uploaded data"
            ):

                st.dataframe(
                    environmental_data.head(),
                    use_container_width=True
                )

            present = [
                col for col in env_cols
                if col in environmental_data.columns
            ]

            missing = [
                col for col in env_cols
                if col not in environmental_data.columns
            ]

            st.write(
                f"Required environmental variables found: "
                f"**{len(present)} / {len(env_cols)}**"
            )

            if missing:

                st.warning(
                    f"{len(missing)} environmental variables "
                    "are missing. The saved training medians "
                    "will be used for those variables."
                )

        except Exception as e:

            st.error(
                "Could not read the environmental CSV."
            )

            st.exception(e)

    else:

        st.info(
            "Upload a CSV to continue."
        )

    st.divider()

    if st.button(
        "🌿 Generate Predictions",
        type="primary",
        use_container_width=True
    ):

        if image_array is None:

            st.warning(
                "Please upload a plant image first."
            )

        elif environmental_data is None:

            st.warning(
                "Please upload an environmental CSV first."
            )

        elif len(environmental_data) == 0:

            st.warning(
                "The environmental CSV contains no observations."
            )

        else:

            try:

                with st.spinner(
                    "Running multimodal deep learning model..."
                ):

                    # Currently the deployment predicts the
                    # first environmental observation.
                    first_row = environmental_data.iloc[
                        [0]
                    ]

                    environment_array = (
                        preprocess_environment(
                            first_row
                        )
                    )

                    predictions = predict_traits(
                        image_array,
                        environment_array
                    )

                st.session_state.predictions = predictions

                st.session_state.image = image

                st.session_state.image_array = image_array

                st.session_state.environment_array = (
                    environment_array
                )

                st.session_state.environment_original = (
                    first_row
                )

                st.success(
                    "Prediction completed successfully."
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if st.session_state.predictions is not None:

        st.divider()

        st.subheader(
            "📈 Predicted Plant Functional Traits"
        )

        predictions = (
            st.session_state.predictions
        )

        columns = st.columns(3)

        for i, target in enumerate(TARGETS):

            with columns[i % 3]:

                st.markdown(
                    f"""
                    <div class="trait-card">

                    <div class="trait-name">
                    {TRAIT_NAMES[target]}
                    </div>

                    <div class="trait-value">
                    {format_value(predictions[i])}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("")

        st.info(
            "The values above are continuous predictions "
            "generated by the trained multimodal regression model. "
            "Their scientific interpretation requires the trait "
            "units and target metadata from the original dataset."
        )

        # ----------------------------------------------------
        # Prediction chart
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Profile"
        )

        chart_df = pd.DataFrame(
            {
                "Trait": [
                    TRAIT_NAMES[t]
                    for t in TARGETS
                ],
                "Predicted Value": predictions
            }
        )

        st.bar_chart(
            chart_df.set_index("Trait")
        )


# ============================================================
# PAGE 3: MODEL PERFORMANCE
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
        Model performance must be calculated using predictions
        on an independent validation or test dataset.

        Because this is a regression problem, conventional
        classification accuracy is not the appropriate metric.
        Recommended metrics include R², MAE and RMSE.
        """
    )

    st.subheader(
        "Recommended Evaluation Metrics"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "R²",
            "Validation required"
        )

        st.caption(
            "Proportion of variance explained by the model."
        )

    with c2:

        st.metric(
            "MAE",
            "Validation required"
        )

        st.caption(
            "Mean absolute prediction error."
        )

    with c3:

        st.metric(
            "RMSE",
            "Validation required"
        )

        st.caption(
            "Root mean squared prediction error."
        )

    st.divider()

    st.subheader(
        "Why is there no 'Prediction Percentage'?"
    )

    st.write(
        """
        A percentage such as "87% accuracy" is generally not
        appropriate for this regression problem.

        For example, an R² value of 0.87 means that approximately
        87% of the variance in the target variable is explained
        by the model on the evaluated dataset. It does NOT mean
        that individual predictions are 87% correct.

        Therefore, the research interface should report R²,
        MAE and RMSE rather than a misleading accuracy percentage.
        """
    )

    st.divider()

    st.subheader(
        "📁 Add Validation Results"
    )

    st.write(
        """
        When you have generated your validation predictions in
        Google Colab, we can connect them here and automatically
        display:

        • R² for each trait

        • MAE for each trait

        • RMSE for each trait

        • Actual vs predicted plots

        • Residual plots

        • Overall model comparison
        """
    )

    validation_file = st.file_uploader(
        "Upload validation_predictions.csv",
        type=["csv"]
    )

    if validation_file is not None:

        try:

            validation_df = pd.read_csv(
                validation_file
            )

            st.success(
                "Validation file loaded."
            )

            st.dataframe(
                validation_df.head(),
                use_container_width=True
            )

            st.info(
                "The exact column structure of your validation "
                "file needs to be connected to the metric calculation "
                "after we generate the validation results in Colab."
            )

        except Exception as e:

            st.error(
                "Could not read validation file."
            )

            st.exception(e)

    else:

        st.warning(
            "No validation results have been uploaded yet. "
            "Do not display invented performance values."
        )


# ============================================================
# PAGE 4: EXPLAINABILITY
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
        Explainability is included to investigate which
        environmental variables and image regions contribute
        to the model's predictions.
        """
    )

    # --------------------------------------------------------
    # ENVIRONMENTAL FEATURE SELECTION
    # --------------------------------------------------------

    st.subheader(
        "🌍 Selected Environmental Features"
    )

    st.write(
        f"""
        The saved feature-selection pipeline contains
        **{len(selected_env_cols)} selected environmental variables**
        from the original set of **{len(env_cols)} variables**.
        """
    )

    selected_df = pd.DataFrame(
        {
            "Selected Environmental Feature":
                selected_env_cols
        }
    )

    st.dataframe(
        selected_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🧩 SHAP Explainability"
    )

    st.write(
        """
        SHAP can be used to estimate how environmental variables
        contribute to individual predictions or to the model's
        overall behaviour.

        The final research version should calculate SHAP values
        using the actual trained model and an appropriate
        background dataset.
        """
    )

    st.info(
        "Actual SHAP values are intentionally not fabricated "
        "here. They should be generated from the trained model "
        "and real background/validation observations in Colab."
    )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🔥 Image Explainability with Grad-CAM"
    )

    st.write(
        """
        Grad-CAM can be used to highlight image regions that
        contribute strongly to a neural-network prediction.

        This is particularly useful for demonstrating whether
        the model is focusing on biologically meaningful regions
        of the plant image.
        """
    )

    if st.session_state.image is not None:

        st.image(
            st.session_state.image,
            caption="Image used for prediction",
            width=400
        )

        st.info(
            "Grad-CAM requires identification of the appropriate "
            "convolutional layer in the saved multimodal model. "
            "Once that layer is identified, the heatmap can be "
            "added here without changing the prediction pipeline."
        )

    else:

        st.info(
            "Run a prediction first to load the plant image."
        )


# ============================================================
# PAGE 5: ENVIRONMENTAL FEATURES
# ============================================================

elif page == "🌍 Environmental Features":

    st.markdown(
        '<div class="main-title">'
        '🌍 Environmental Feature Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"""
        The model was trained with **{len(env_cols)} environmental
        variables**.
        """
    )

    climate_features = [
        x for x in env_cols
        if x.startswith("WORLDCLIM")
    ]

    soil_features = [
        x for x in env_cols
        if x.startswith("SOIL")
    ]

    modis_features = [
        x for x in env_cols
        if x.startswith("MODIS")
    ]

    vod_features = [
        x for x in env_cols
        if x.startswith("VOD")
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Climate",
            len(climate_features)
        )

    with c2:
        st.metric(
            "Soil",
            len(soil_features)
        )

    with c3:
        st.metric(
            "MODIS",
            len(modis_features)
        )

    with c4:
        st.metric(
            "VOD",
            len(vod_features)
        )

    st.divider()

    st.subheader(
        "All Environmental Variables"
    )

    feature_df = pd.DataFrame(
        {
            "Environmental Feature": env_cols,
            "Selected Feature": [
                feature in selected_env_cols
                for feature in env_cols
            ]
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Feature List",
        data=feature_df.to_csv(
            index=False
        ),
        file_name="environmental_features.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 6: MODEL ARCHITECTURE
# ============================================================

elif page == "🧠 Model Architecture":

    st.markdown(
        '<div class="main-title">'
        '🧠 Model Architecture'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The trained model is a multimodal neural network.
        It receives an image input and an environmental input,
        combines learned representations, and produces six
        continuous outputs.
        """
    )

    st.code(
        """
                 PLANT IMAGE
                     │
                     ▼
              Image Network
                     │
                     ▼
              Image Features
                     │
                     │
                     ├──────────────┐
                                    │
                                    ▼
                                  FUSION
                                    ▲
                                    │
                     ┌──────────────┘
                     │
              ENVIRONMENT
                     │
                     ▼
              163 VARIABLES
                     │
                     ▼
              Environmental
                 Network
                     │
                     ▼
             Environmental
                Features
                     │
                     ▼
                  FUSION
                     │
                     ▼
            MULTI-OUTPUT REGRESSION
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Trait 1    Trait 2     Trait 3
          │          │           │
          └──────────┼───────────┘
                     │
                  ... six
                  outputs
        """,
        language="text"
    )

    st.subheader(
        "Verified Model Input / Output Dimensions"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Image",
            "(224, 224, 3)"
        )

    with c2:
        st.metric(
            "Environment",
            "(163,)"
        )

    with c3:
        st.metric(
            "Output",
            "(6,)"
        )

    st.divider()

    st.subheader(
        "Six Numerical Outputs"
    )

    architecture_df = pd.DataFrame(
        {
            "Output": [
                f"Output {i + 1}"
                for i in range(6)
            ],
            "Trait": [
                TRAIT_NAMES[target]
                for target in TARGETS
            ],
            "Type": [
                "Continuous numerical"
                for _ in TARGETS
            ]
        }
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "⚠️ Scientific Interpretation"
    )

    st.write(
        """
        The model learns statistical relationships between
        image/environmental information and observed plant
        functional traits.

        Model explanations should therefore be interpreted as
        associations learned by the model rather than evidence
        of biological causation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Plant Functional Trait Predictor • "
    "Multimodal Deep Learning • Multi-output Regression"
)
```
