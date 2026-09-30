import streamlit as st
import pickle
import numpy as np
import os

# Set up the page layout
st.set_page_config(page_title="Asteroid Hazard Predictor", page_icon="☄️", layout="centered")


@st.cache_resource
def load_model():
    path = 'asteroid_hazard_model.pkl'
    if not os.path.exists(path):
        return None
    
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load up the brain
clf = load_model()

# Sidebar for context
with st.sidebar:
    st.header("Project Info")
    st.markdown(
        """
        **NASA NeoWs Data Analysis**
        
        This tool utilizes a Random Forest Classifier model and SMOTE technique to assess the potential threat of Near-Earth Objects based on trajectory and physical properties.
        """
    )
    st.write("---")
    st.caption("Made By: SHIVANSH GAUTAM")

# Main Interface
st.title("☄️ Asteroid Hazard Predictor Using NASA API")
st.markdown("Please input the telemetry data below to check for potential Earth impacts.")

# Check if model loaded correctly before showing the form
if clf is None:
    st.error("⚠️ Critical Error: Model file missing. Please ensure 'asteroid_hazard_model.pkl' is in the directory.")
    st.stop()

# Input Section - Grouped by physical vs movement properties
c1, c2 = st.columns(2)

with c1:
    st.subheader("Physical Props")
    # Absolute Magnitude: brightness acts as a proxy for size usually
    abs_mag = st.number_input("Absolute Magnitude (H)", value=20.0, step=0.1, help="Lower values mean the object is brighter/larger.")
    
    # Diameter estimates
    d_min = st.number_input("Min Diameter (km)", value=0.1, step=0.01)
    d_max = st.number_input("Max Diameter (km)", value=0.3, step=0.01)

with c2:
    st.subheader("Trajectory")
    # Speed and proximity
    v_rel = st.number_input("Velocity (km/h)", value=50000.0, step=100.0)
    miss_dist = st.number_input("Miss Distance (km)", value=1000000.0, step=1000.0)

st.write("---")

# Execution Logic
if st.button("Run Risk Assessment 🚀", use_container_width=True):
    
    # Input validation
    if d_min < 0 or d_max < 0:
        st.warning("⚠️ Diameter values cannot be negative.")
        st.stop()
    if d_min > d_max:
        st.warning("⚠️ Min Diameter cannot be greater than Max Diameter.")
        st.stop()
    if v_rel < 0 or miss_dist < 0:
        st.warning("⚠️ Velocity and Miss Distance cannot be negative.")
        st.stop()
    
    # Organizing features to match the training shape
    # [magnitude, min_dia, max_dia, velocity, distance]
    feats = np.array([[abs_mag, d_min, d_max, v_rel, miss_dist]])
    
    try:
        prediction = clf.predict(feats)[0]
        probs = clf.predict_proba(feats)[0] # Returns [prob_safe, prob_danger]
        
        st.subheader("Assessment Report")
        
        # If prediction is Hazardous (True/1)
        if prediction:
            confidence = probs[1] * 100
            st.error(f"🚨 ALERT: HAZARDOUS OBJECT DETECTED")
            st.markdown(f"**Risk Confidence:** {confidence:.2f}%")
            st.image("hazardous.png")
        else:
            confidence = probs[0] * 100
            st.success(f"✅ STATUS: SAFE")
            st.markdown(f"The object poses no immediate threat. (Confidence: {confidence:.2f}%)")
            st.image("Safe.png")
            
    except Exception as e:
        st.warning(f"Something went wrong during prediction: {e}")