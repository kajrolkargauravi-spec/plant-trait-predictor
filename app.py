import os
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

from PIL import Image


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Plant Functional Trait Predictor",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# FILES
# ============================================================

MODEL_PATH = "multimodal_model.keras"
ENV_COLS_PATH = "env_cols.pkl"
ENV_MEDIANS_PATH = "env_medians.pkl"
ENV_SCALER_PATH = "env_scaler.pkl"
SELECTED_ENV_COLS_PATH = "selected_env_cols.pkl"
TARGET_SCALER_PATH = "target_scaler.pkl"


# ============================================================
# MODEL OUTPUTS
# ============================================================
# We know the model produces 6 numerical outputs.
# We will verify the biological names later.

TARGETS = [
    "Trait 1",
    "Trait 2",
    "Trait 3",
    "Trait 4",
    "Trait 5",
    "Trait 6"
]


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 19px;
        color: #666666;
        margin-bottom: 25px;
    }

    .trait-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        text-align: center;
        margin-bottom: 20px;
    }

    .trait-name {
        font-size: 16px;
        color: #666666;
    }

    .trait-value {
        font-size: 30px;
        font-weight: 700;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK FILES
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
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("Required model files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"• {file}")

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_everything():

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


try:

    (
        model,
        env_cols,
        env_medians,
        env_scaler,
        selected_env_cols,
        target_scaler
    ) = load_everything()

except Exception as error:

    st.error("The model could not be loaded.")

    st.exception(error)

    st.stop()


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

    # Make sure all 163 required columns exist.

    for column in env_cols:

        if column not in df.columns:

            if isinstance(env_medians, dict):

                df[column] = env_medians.get(
                    column,
                    0.0
                )

            else:

                df[column] = 0.0

    # Use exactly the same order as training.

    df = df[env_cols]

    # Convert values to numbers.

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Fill missing values.

    for column in env_cols:

        if isinstance(env_medians, dict):

            median = env_medians.get(
                column,
                0.0
            )

        else:

            median = 0.0

        df[column] = df[column].fillna(
            median
        )

    values = df.values.astype(
        np.float32
    )

    # Apply training scaler.

    values = env_scaler.transform(
        values
    )

    return values.astype(
        np.float32
    )


# ============================================================
# PREDICTION
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

    prediction = target_scaler.inverse_transform(
        prediction_scaled
    )

    return prediction[0]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌿 Plant Trait AI")

st.sidebar.write(
    "Multimodal Deep Learning"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🌱 Predict Traits",
        "📊 Model Performance",
        "🔍 Explainability",
        "🌍 Environmental Data",
        "🧠 Model Architecture"
    ]
)

st.sidebar.divider()

st.sidebar.write(
    f"Environmental variables: {len(env_cols)}"
)

st.sidebar.write(
    f"Selected variables: {len(selected_env_cols)}"
)

st.sidebar.write(
    "Outputs: 6 numerical traits"
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

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

    st.info(
        """
        This research uses a multimodal deep learning model
        to predict continuous plant functional traits by
        combining plant images with environmental information.
        """
    )

    st.subheader("How the model works")

    st.code(
        """
        PLANT IMAGE
             |
             v
        IMAGE BRANCH
             |
             v
        IMAGE FEATURES
             |
             |
             +------------+
                          |
                          v
                       FUSION
                          ^
                          |
             +------------+
             |
        ENVIRONMENTAL DATA
             |
             v
        163 VARIABLES
             |
             v
        ENVIRONMENT BRANCH
             |
             v
        ENVIRONMENT FEATURES
             |
             v
           FUSION
             |
             v
        DEEP LEARNING MODEL
             |
             v
        6 NUMERICAL OUTPUTS
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
            "Selected Variables",
            "15"
        )

    with c4:
        st.metric(
            "Outputs",
            "6"
        )

    st.subheader("Research Focus")

    st.write(
        """
        The objective is to investigate whether combining
        visual plant information with environmental information
        can provide useful predictions of continuous plant
        functional traits.
        """
    )

    st.success(
        "Problem type: Multi-output regression"
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🌱 Predict Traits":

    st.markdown(
        '<div class="main-title">'
        '🌱 Predict Plant Traits'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload a plant image and environmental data.'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("Step 1: Upload Plant Image")

    uploaded_image = st.file_uploader(
        "Upload JPG, JPEG or PNG",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    image_array = None

    if uploaded_image is not None:

        try:

            image, image_array = preprocess_image(
                uploaded_image
            )

            st.image(
                image,
                caption="Uploaded Plant",
                width=400
            )

        except Exception as error:

            st.error(
                "Could not process image."
            )

            st.exception(error)

    st.divider()

    st.subheader("Step 2: Upload Environmental Data")

    uploaded_csv = st.file_uploader(
        "Upload environmental CSV",
        type=["csv"]
    )

    environmental_data = None

    if uploaded_csv is not None:

        try:

            environmental_data = pd.read_csv(
                uploaded_csv
            )

            st.success(
                f"CSV loaded: "
                f"{environmental_data.shape[0]} rows × "
                f"{environmental_data.shape[1]} columns"
            )

            with st.expander(
                "View uploaded data"
            ):

                st.dataframe(
                    environmental_data.head(),
                    use_container_width=True
                )

            available = [
                column
                for column in env_cols
                if column in environmental_data.columns
            ]

            missing = [
                column
                for column in env_cols
                if column not in environmental_data.columns
            ]

            st.write(
                f"Environmental variables found: "
                f"**{len(available)} / {len(env_cols)}**"
            )

            if missing:

                st.warning(
                    f"{len(missing)} variables are missing. "
                    "Training medians will be used."
                )

        except Exception as error:

            st.error(
                "Could not read the CSV."
            )

            st.exception(error)

    st.divider()

    if st.button(
        "🌿 Generate Prediction",
        type="primary",
        use_container_width=True
    ):

        if image_array is None:

            st.warning(
                "Please upload a plant image."
            )

        elif environmental_data is None:

            st.warning(
                "Please upload environmental data."
            )

        elif environmental_data.empty:

            st.warning(
                "The CSV contains no data."
            )

        else:

            try:

                with st.spinner(
                    "Running deep learning model..."
                ):

                    # Use first environmental observation.

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

                st.session_state[
                    "predictions"
                ] = predictions

                st.success(
                    "Prediction completed!"
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)

    # --------------------------------------------------------
    # SHOW RESULTS
    # --------------------------------------------------------

    if "predictions" in st.session_state:

        predictions = st.session_state[
            "predictions"
        ]

        st.divider()

        st.subheader(
            "📈 Predicted Plant Functional Traits"
        )

        columns = st.columns(3)

        for i in range(6):

            with columns[i % 3]:

                st.markdown(
                    f"""
                    <div class="trait-card">

                    <div class="trait-name">
                    {TARGETS[i]}
                    </div>

                    <div class="trait-value">
                    {predictions[i]:.4f}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.subheader(
            "📊 Prediction Profile"
        )

        chart_data = pd.DataFrame(
            {
                "Trait": TARGETS,
                "Prediction": predictions
            }
        )

        st.bar_chart(
            chart_data.set_index(
                "Trait"
            )
        )

        st.info(
            """
            The model produces continuous numerical predictions.
            The biological names and units of the six outputs will
            be connected after verifying the original target data.
            """
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="main-title">'
        '📊 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Because this is a regression model, performance should
        be evaluated using R², MAE and RMSE.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "R²",
            "Pending"
        )

        st.caption(
            "Variance explained by the model."
        )

    with c2:

        st.metric(
            "MAE",
            "Pending"
        )

        st.caption(
            "Mean Absolute Error."
        )

    with c3:

        st.metric(
            "RMSE",
            "Pending"
        )

        st.caption(
            "Root Mean Squared Error."
        )

    st.divider()

    st.subheader(
        "Why not prediction accuracy %?"
    )

    st.write(
        """
        This is a numerical regression problem, not a
        classification problem.

        Therefore, a statement such as "the model is 90%
        accurate" would be misleading.

        We will instead calculate R², MAE and RMSE using
        the independent test dataset.
        """
    )

    st.divider()

    st.subheader(
        "Research Evaluation"
    )

    st.write(
        """
        The final research version of this page will include:

        • R² for each plant trait

        • MAE for each plant trait

        • RMSE for each plant trait

        • Actual vs Predicted plots

        • Residual plots

        • Comparison of performance across the six traits
        """
    )

    st.warning(
        "Do not enter or display invented performance values."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "🔍 Explainability":

    st.markdown(
        '<div class="main-title">'
        '🔍 Model Explainability'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Explainability will help us understand which
        environmental variables and image regions influence
        the model predictions.
        """
    )

    st.subheader(
        "🧩 SHAP"
    )

    st.write(
        f"""
        The model uses {len(env_cols)} environmental variables,
        with {len(selected_env_cols)} selected variables.
        """
    )

    st.write(
        "Selected environmental variables:"
    )

    shap_features = pd.DataFrame(
        {
            "Feature": selected_env_cols
        }
    )

    st.dataframe(
        shap_features,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        SHAP values will be calculated from the real trained
        model and real observations. We will add the actual
        SHAP plots after generating them in Google Colab.
        """
    )

    st.divider()

    st.subheader(
        "🔥 Grad-CAM"
    )

    st.write(
        """
        Grad-CAM will be used to visualize which parts of the
        plant image contribute to the model's prediction.
        """
    )

    st.info(
        """
        The correct convolutional layer must first be identified
        from the trained model before generating the Grad-CAM
        heatmap.
        """
    )


# ============================================================
# ENVIRONMENTAL DATA
# ============================================================

elif page == "🌍 Environmental Data":

    st.markdown(
        '<div class="main-title">'
        '🌍 Environmental Features'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"""
        The model receives {len(env_cols)} environmental
        variables.
        """
    )

    climate = [
        x for x in env_cols
        if x.startswith("WORLDCLIM")
    ]

    soil = [
        x for x in env_cols
        if x.startswith("SOIL")
    ]

    modis = [
        x for x in env_cols
        if x.startswith("MODIS")
    ]

    vod = [
        x for x in env_cols
        if x.startswith("VOD")
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Climate",
            len(climate)
        )

    with c2:
        st.metric(
            "Soil",
            len(soil)
        )

    with c3:
        st.metric(
            "MODIS",
            len(modis)
        )

    with c4:
        st.metric(
            "VOD",
            len(vod)
        )

    st.divider()

    st.subheader(
        "Environmental Variables"
    )

    feature_df = pd.DataFrame(
        {
            "Feature": env_cols,
            "Selected": [
                x in selected_env_cols
                for x in env_cols
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
# MODEL ARCHITECTURE
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
        Our model combines two information sources:

        1. Plant imagery

        2. Environmental information

        These representations are combined and used to predict
        six continuous numerical plant functional traits.
        """
    )

    st.code(
        """
             PLANT IMAGE
                  |
                  v
             IMAGE MODEL
                  |
                  v
             IMAGE FEATURES
                  |
                  |
                  +-------------+
                                |
                                v
                              FUSION
                                ^
                                |
                  +-------------+
                  |
          ENVIRONMENTAL DATA
                  |
                  v
              163 FEATURES
                  |
                  v
          ENVIRONMENT PROCESSING
                  |
                  v
          ENVIRONMENT FEATURES
                  |
                  v
                FUSION
                  |
                  v
          REGRESSION NETWORK
                  |
                  v
          SIX NUMERICAL OUTPUTS
        """,
        language="text"
    )

    st.subheader(
        "Verified Model Dimensions"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Image Input",
            "(224, 224, 3)"
        )

    with c2:

        st.metric(
            "Environmental Input",
            "(163,)"
        )

    with c3:

        st.metric(
            "Output",
            "(6,)"
        )

    st.divider()

    st.subheader(
        "Research Design"
    )

    architecture_df = pd.DataFrame(
        {
            "Component": [
                "Plant image",
                "Environmental variables",
                "Selected variables",
                "Model output",
                "Problem type"
            ],
            "Value": [
                "224 × 224 RGB",
                "163",
                "15",
                "6 numerical values",
                "Multi-output regression"
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
        "Scientific Interpretation"
    )

    st.write(
        """
        The model learns statistical relationships between
        plant images, environmental conditions and observed
        plant functional traits.

        Explainability results should therefore be interpreted
        as model associations and not automatically as evidence
        of biological causation.
        """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Plant Functional Trait Predictor | "
    "Multimodal Deep Learning | "
    "Multi-output Regression"
)
