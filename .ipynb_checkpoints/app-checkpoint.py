import streamlit as st
import pickle
import numpy as np

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="NASA Hazard Predictor", page_icon="☄️")
st.title("☄️ NASA Asteroid Hazard Predictor")
st.write("Enter the asteroid's telemetry data below to predict if it poses a threat to Earth.")

# --- 2. LOAD THE SAVED MODEL ---
try:
    with open('asteroid_hazard_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file 'asteroid_hazard_model.pkl' not found. Please run the training script first!")
    st.stop()

# --- 3. SIDEBAR (CONTEXT) ---
st.sidebar.header("About This Project")
st.sidebar.info(
    "This machine learning app uses a Random Forest Classifier trained on "
    "NASA's NeoWs (Near Earth Object) API data. It analyzes asteroid size, "
    "velocity, and proximity to identify potential impact risks."
)
st.sidebar.markdown("---")
st.sidebar.write("Created by: [Your Name]")

# --- 4. INPUT FORM ---
# We use columns to make the layout look professional
col1, col2 = st.columns(2)

with col1:
    # Feature 1: Absolute Magnitude (Lower is brighter/bigger)
    magnitude = st.number_input("Absolute Magnitude (H)", value=20.0, step=0.1)
    
    # Feature 2: Velocity (km/h)
    velocity = st.number_input("Relative Velocity (km/h)", value=50000.0, step=100.0)

    # Feature 3: Miss Distance (km)
    distance = st.number_input("Miss Distance (km)", value=1000000.0, step=1000.0)

with col2:
    # Feature 4: Min Diameter (km)
    dia_min = st.number_input("Est. Diameter Min (km)", value=0.1, step=0.01)
    
    # Feature 5: Max Diameter (km)
    dia_max = st.number_input("Est. Diameter Max (km)", value=0.3, step=0.01)

# --- 5. PREDICTION LOGIC ---
if st.button("Analyze Asteroid Hazard 🚀"):
    # Create the numpy array with the EXACT same column order as training
    # Order: [absolute_magnitude, est_diameter_min, est_diameter_max, relative_velocity, miss_distance]
    input_data = np.array([[magnitude, dia_min, dia_max, velocity, distance]])
    
    # Get Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    # Display Result
    st.subheader("Analysis Result:")
    
    # prediction[0] is either True (Hazardous) or False (Safe)
    if prediction[0]:
        st.error(f"🚨 WARNING: POTENTIALLY HAZARDOUS ASTEROID DETECTED!")
        st.write(f"Confidence Level: **{probability[0][1]*100:.2f}%**")
        st.image("https://media.giphy.com/media/5wFS6a1PE62lKUWXyx/giphy.gif", caption="Impact Warning")
    else:
        st.success(f"✅ STATUS: SAFE. No impact risk detected.")
        st.write(f"Confidence Level: **{probability[0][0]*100:.2f}%**")