import streamlit as st
import pandas as pd
import joblib
import os

# --- Model Loading ---

# Define the path to the model directory
model_dir = 'model'

# Use os.path.join to create the correct file paths for the model, scaler, and encoder
model_path = os.path.join(model_dir, 'wine_quality_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
encoder_path = os.path.join(model_dir, 'encoder.pkl')

# Load the brain (model), the measuring tool (scaler), and the encoder we saved earlier
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
except FileNotFoundError:
    st.error(
        "Could not find the model, scaler, or encoder files. "
        "Please ensure you have run the `train_model.py` script first to create the `model` folder "
        "and save the necessary files."
    )
    st.stop()  # Stop the app if the files are not found

# --- App Setup ---

# Set up the title and a nice description for the app
st.title("🍷 Wine Quality Guessing App")
st.markdown("Move the sliders to change the wine's ingredients and see what quality it might be!")

# These are the ingredients we'll have sliders for
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Set up the sliders for each ingredient
with st.form("input_form"):
    st.header("What's in Your Wine?")
    cols = st.columns(3)
    user_input = {}

    # We'll set some limits for the sliders
    data_ranges = {
        'fixed acidity': (4.6, 15.9, 7.4), 'volatile acidity': (0.12, 1.58, 0.7),
        'citric acid': (0.0, 1.0, 0.0), 'residual sugar': (0.9, 15.5, 1.9),
        'chlorides': (0.034, 0.612, 0.076), 'free sulfur dioxide': (1.0, 72.0, 11.0),
        'total sulfur dioxide': (6.0, 289.0, 34.0), 'density': (0.99, 1.004, 0.9978),
        'pH': (2.74, 4.01, 3.51), 'sulphates': (0.33, 2.0, 0.56),
        'alcohol': (8.4, 14.9, 9.4)
    }

    # Create a slider for each ingredient
    for i, feature in enumerate(features):
        min_val, max_val, default_val = data_ranges[feature]
        # Set a finer step for density
        if feature == 'density':
            step_val = 0.0001
        else:
            step_val = 0.01

        with cols[i % 3]:
            user_input[feature] = st.slider(
                label=feature.replace('_', ' ').title(),
                min_value=min_val, max_value=max_val, value=default_val, step=step_val
            )

    # Add a button to make a guess
    submitted = st.form_submit_button("Guess the Quality!")

# When the button is clicked, do the guessing!
if submitted:
    input_df = pd.DataFrame([user_input])

    # Use our measuring tool to prepare the numbers for the robot
    scaled_input = scaler.transform(input_df)

    # Ask the robot for its guess!
    # The model predicts an encoded value (0-indexed).
    prediction_encoded = model.predict(scaled_input)

    # Use the encoder to convert the prediction back to the original quality score.
    prediction_quality = encoder.inverse_transform(prediction_encoded)

    st.subheader("Robot's Guess")
    st.success(f"The robot thinks the quality is: **{prediction_quality[0]}**")
    st.info("The quality score is usually a number between 3 and 8.")