import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Plant Functional Trait Predictor",
    page_icon="🌿",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
    margin-bottom: 15px;
}

.trait-name {
    font-size: 16px;
    color: #666;
}

.trait-value {
    font-size: 28px;
    font-weight: 700;
}

.info-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7f6;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)


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
# LOAD MODEL AND PREPROCESSORS
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

except Exception as e:

    st.error("Could not load the model files.")

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_array


# ============================================================
# ENVIRONMENTAL DATA PREPROCESSING
# ============================================================

def preprocess_environment(
    dataframe
):

    # Make a copy so original uploaded data
    # is not modified.
    df = dataframe.copy()

    # --------------------------------------------------------
    # Check for missing columns
    # --------------------------------------------------------

    missing_columns = [
        col for col in env_cols
        if col not in df.columns
    ]

    # --------------------------------------------------------
    # If columns are missing, add them using
    # saved median values.
    # --------------------------------------------------------

    for col in missing_columns:

        if col in env_medians:

            df[col] = env_medians[col]

        else:

            df[col] = 0.0

    # --------------------------------------------------------
    # Keep ONLY the 163 features used during training
    # and preserve their original order.
    # --------------------------------------------------------

    df = df[env_cols]

    # --------------------------------------------------------
    # Convert everything to numerical values
    # --------------------------------------------------------

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Fill missing values using training medians
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
    # Convert to numpy
    # --------------------------------------------------------

    values = df.values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Apply the SAME StandardScaler used during training
    # --------------------------------------------------------

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

    prediction_scaled = np.asarray(
        prediction_scaled
    )

    prediction = target_scaler.inverse_transform(
        prediction_scaled
    )

    return prediction[0]


# ============================================================
# FORMAT NUMBERS
# ============================================================

def format_number(
    value
):

    if abs(value) >= 1000:
        return f"{value:,.2f}"

    elif abs(value) >= 100:
        return f"{value:,.3f}"

    elif abs(value) >= 1:
        return f"{value:.4f}"

    else:
        return f"{value:.6f}"


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title(
    "🌿 Plant Trait AI"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Research Overview",
        "🌱 Predict Traits",
        "🔍 Explainability",
        "📊 Environmental Features"
    ]
)


# ============================================================
# HOME / RESEARCH OVERVIEW
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

    st.markdown("""
    <div class="info-box">

    <h3>Research Idea</h3>

    This application demonstrates a multimodal deep learning
    approach for predicting continuous plant functional traits.

    The model combines information from two sources:

    <br>

    <b>1. Plant imagery</b><br>
    Visual information extracted from a plant image.

    <br><br>

    <b>2. Environmental information</b><br>
    Climate, soil, satellite reflectance and vegetation
    optical-depth variables.

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.subheader(
        "Multimodal Architecture"
    )

    st.code("""
             PLANT IMAGE
                  │
                  ▼
          Image Feature Network
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
          163 Environmental
             Variables
                  │
                  ▼
        Environmental Processing
                  │
                  ▼
              Fusion
                  │
                  ▼
          Multi-output Regression
                  │
          ┌───────┼────────┐
          ▼       ▼        ▼
        Trait   Trait     Trait
          ...     ...       ...
                  │
                  ▼
              6 Traits
    """, language="text")

    st.subheader(
        "Model Inputs"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Image Size",
            "224 × 224"
        )

    with col2:
        st.metric(
            "Environmental Features",
            "163"
        )

    with col3:
        st.metric(
            "Numerical Outputs",
            "6"
        )

    st.subheader(
        "Predicted Traits"
    )

    for target in TARGETS:

        st.write(
            "•",
            TRAIT_NAMES[target]
        )

    st.info(
        "This is a regression problem. The model predicts "
        "continuous numerical trait values rather than "
        "categorical classes."
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
        'Upload a plant image and environmental data '
        'to generate six numerical predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    st.subheader(
        "📷 Plant Image"
    )

    uploaded_image = st.file_uploader(
        "Upload plant image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    image = None
    image_array = None

    if uploaded_image is not None:

        image, image_array = preprocess_image(
            uploaded_image
        )

        st.image(
            image,
            caption="Uploaded plant",
            width=400
        )

    st.divider()

    # --------------------------------------------------------
    # ENVIRONMENTAL DATA
    # --------------------------------------------------------

    st.subheader(
        "🌍 Environmental Data"
    )

    st.write(
        """
        Upload a CSV containing environmental information.
        The application automatically selects the 163 variables
        required by the trained model, fills missing values
        using the training medians, and applies the saved scaler.
        """
    )

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
                f"CSV loaded successfully: "
                f"{environmental_data.shape[0]} rows × "
                f"{environmental_data.shape[1]} columns"
            )

            with st.expander(
                "Preview environmental data"
            ):

                st.dataframe(
                    environmental_data.head(),
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Could not read the CSV file."
            )

            st.exception(e)

    else:

        st.info(
            "Please upload an environmental CSV."
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🌿 Predict Plant Traits",
        type="primary",
        use_container_width=True
    ):

        if uploaded_image is None:

            st.warning(
                "Please upload a plant image."
            )

        elif environmental_data is None:

            st.warning(
                "Please upload an environmental CSV."
            )

        elif len(environmental_data) == 0:

            st.warning(
                "The uploaded CSV contains no rows."
            )

        else:

            try:

                with st.spinner(
                    "Running multimodal deep learning model..."
                ):

                    # Use first row if CSV contains
                    # multiple observations.
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

                st.session_state[
                    "image"
                ] = image

                st.session_state[
                    "image_array"
                ] = image_array

                st.session_state[
                    "environment"
                ] = environment_array

                st.session_state[
                    "environment_original"
                ] = first_row

                st.success(
                    "Prediction completed successfully."
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    if "predictions" in st.session_state:

        st.divider()

        st.subheader(
            "📈 Predicted Plant Functional Traits"
        )

        predictions = st.session_state[
            "predictions"
        ]

        columns = st.columns(3)

        for i, target in enumerate(
            TARGETS
        ):

            with columns[i % 3]:

                st.markdown(
                    f"""
                    <div class="card">

                    <div class="trait-name">
                    {TRAIT_NAMES[target]}
                    </div>

                    <div class="trait-value">
                    {format_number(predictions[i])}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.caption(
            "The displayed values are predictions generated "
            "by the trained multimodal regression model."
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

    st.markdown("""
    This section provides an interpretable view of the model.

    The environmental feature-selection pipeline identified
    15 environmental variables as particularly relevant.

    These features are displayed here for research
    interpretation. They should not be interpreted as proof
    of biological causation.
    """)

    st.subheader(
        "🌍 Selected Environmental Features"
    )

    selected_df = pd.DataFrame({
        "Selected Feature":
            selected_env_cols
    })

    st.dataframe(
        selected_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Why Explainability?"
    )

    st.write("""
    Deep learning models can achieve strong predictive
    performance while remaining difficult to interpret.

    Feature-selection and explainability methods help us
    investigate which environmental variables are associated
    with the model's predictions.

    The selected variables include information from climate,
    soil, MODIS reflectance and vegetation optical depth
    datasets.
    """)

    st.info(
        "SHAP analysis should be performed using the exact "
        "training background data and model pipeline used "
        "during experimentation. The current deployment "
        "therefore displays the selected environmental "
        "features without fabricating SHAP values."
    )


# ============================================================
# ENVIRONMENTAL FEATURES PAGE
# ============================================================

elif page == "📊 Environmental Features":

    st.markdown(
        '<div class="main-title">'
        '📊 Environmental Variables'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"The trained model expects {len(env_cols)} "
        "environmental variables."
    )

    # --------------------------------------------------------
    # GROUP FEATURES
    # --------------------------------------------------------

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

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Climate",
            len(climate_features)
        )

    with col2:
        st.metric(
            "Soil",
            len(soil_features)
        )

    with col3:
        st.metric(
            "MODIS",
            len(modis_features)
        )

    with col4:
        st.metric(
            "VOD",
            len(vod_features)
        )

    st.divider()

    st.subheader(
        "All Environmental Features"
    )

    feature_df = pd.DataFrame({
        "Feature": env_cols,
        "Selected for Interpretation": [
            x in selected_env_cols
            for x in env_cols
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Feature List",
        data=feature_df.to_csv(
            index=False
        ),
        file_name="environmental_features.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Plant Functional Trait Prediction | "
    "Multimodal Deep Learning Research Project"
)
