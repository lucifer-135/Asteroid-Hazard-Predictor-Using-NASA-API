import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Set up the page layout
st.set_page_config(page_title="Asteroid Hazard Predictor", page_icon="☄️", layout="centered")

REPO_URL = "https://github.com/lucifer-135/Asteroid-Hazard-Predictor-Using-NASA-API"
AU_KM = 149_597_870.7          # one astronomical unit in km
LUNAR_DISTANCE_KM = 384_400    # average Earth-Moon distance

# NASA's definition of a Potentially Hazardous Asteroid
PHA_MAX_H = 22.0
PHA_MAX_MOID = 0.05

HAZARD_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzIzNzRqdzBpMnZnOWNtMGtyMm0xaml3OGRkOTJ1NzkweDNxc2VpbCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/dC9I6hjH33wiuNXG0D/giphy.gif"
SAFE_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaHR0bmM3NW80NzZ6bm53cDlwd3dlMW8yYXRuN2gyZ212aThreG8zNCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/n5iPVLeA1fvb0IYU5W/giphy.gif"

# Starting values for the inputs
DEFAULTS = {'abs_mag': 20.0, 'moid': 0.1, 'v_rel': 50000.0, 'miss_dist': 1000000.0}

# Real asteroids from the training data (NASA NeoWs + JPL SBDB, Jan-Jun 2023)
EXAMPLES = {
    "4486 Mithra": {'abs_mag': 15.62, 'moid': 0.0458, 'v_rel': 48432.0, 'miss_dist': 24334530.0},
    "2014 JO25": {'abs_mag': 18.02, 'moid': 0.00601, 'v_rel': 86963.0, 'miss_dist': 53456523.0},
    "2003 XF11": {'abs_mag': 17.08, 'moid': 0.126, 'v_rel': 122864.0, 'miss_dist': 27997679.0},
    "2018 CB": {'abs_mag': 25.9, 'moid': 0.0000438, 'v_rel': 20342.0, 'miss_dist': 8449670.0},
}


@st.cache_resource
def load_model():
    path = 'asteroid_hazard_model.pkl'
    if not os.path.exists(path):
        return None

    with open(path, 'rb') as f:
        return pickle.load(f)


def estimated_diameter_range(abs_mag):
    """Diameter range (km) from absolute magnitude, the same way NASA NeoWs computes it:
    D = 1329 / sqrt(albedo) * 10^(-H/5), for albedos from 0.25 (min size) to 0.05 (max size)."""
    scale = 1329 * 10 ** (-abs_mag / 5)
    return scale / np.sqrt(0.25), scale / np.sqrt(0.05)


def format_size(km):
    return f"{km:,.1f} km" if km >= 1 else f"{km * 1000:,.0f} m"


def describe_distance(km):
    size = f"{km / 1e6:,.2f} million km" if km >= 1e6 else f"{km:,.0f} km"
    lunar = km / LUNAR_DISTANCE_KM
    return f"{size}, {lunar:,.1f}× the Earth-Moon distance" if lunar >= 1 else f"{size}, closer than the Moon"


def load_example():
    if st.session_state.example:
        st.session_state.update(EXAMPLES[st.session_state.example])


def clear_example():
    # Once an input is edited, the values no longer belong to the selected asteroid
    st.session_state.example = None


# Load up the brain
clf = load_model()

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# Sidebar for context
with st.sidebar:
    st.header("About")
    st.markdown(
        f"""
        A **Potentially Hazardous Asteroid** is one large enough to cause serious damage
        (H ≤ {PHA_MAX_H:g}, about 140 m or larger) **and** whose orbit comes within
        {PHA_MAX_MOID:g} au (~7.5 million km) of Earth's orbit. It doesn't mean an impact
        is expected, just that the asteroid is worth watching.

        **How it works:** a Random Forest classifier, balanced with SMOTE, trained on
        1,055 asteroids from NASA NeoWs with orbit data (MOID) from JPL's Small-Body
        Database. In cross-validation it caught 98% of hazardous asteroids, and 99% of
        its alerts were correct.
        """
    )
    st.link_button("View source on GitHub", REPO_URL)
    st.divider()
    st.caption("Made By: SHIVANSH GAUTAM")

# Main Interface
st.title("☄️ Asteroid Hazard Predictor")
st.markdown("Check whether a near-Earth asteroid counts as **potentially hazardous**, "
            "using a machine-learning model trained on NASA NeoWs and JPL data.")

# Check if model loaded correctly before showing the form
if clf is None:
    st.error("⚠️ Critical Error: Model file missing. Please ensure 'asteroid_hazard_model.pkl' is in the directory.")
    st.stop()

st.pills("Try a real asteroid", list(EXAMPLES), key="example", on_change=load_example,
         help="Loads a real asteroid from the training data: two hazardous ones (Mithra, 2014 JO25), "
              "a large one whose orbit stays far from Earth's (2003 XF11), and one whose orbit crosses "
              "Earth's but is too small to count (2018 CB).")

# Input Section - what decides hazard status vs. details of one flyby
c1, c2 = st.columns(2)

with c1, st.container(border=True):
    st.markdown("#### 🪨 Size & orbit")
    abs_mag = st.number_input("Absolute Magnitude (H)", key="abs_mag", step=0.1, format="%.2f", on_change=clear_example,
                              help="How bright the asteroid is, which NASA uses to estimate its size. "
                                   "Lower values mean a larger asteroid. The size below is estimated from H "
                                   "the same way NASA does.")

    # Diameter estimates are derived from H (as NASA does), so they always match it
    d_min, d_max = estimated_diameter_range(abs_mag)
    st.caption(f"≈ {format_size(d_min)} – {format_size(d_max)} across")

    # Orbit geometry: how close the asteroid's orbit ever gets to Earth's orbit
    moid = st.number_input("MOID (au)", key="moid", step=0.001, format="%.5f", on_change=clear_example,
                           help="Minimum Orbit Intersection Distance: the closest the asteroid's orbit "
                                "comes to Earth's orbit. 1 au is the Earth-Sun distance.")
    st.caption(f"≈ {describe_distance(moid * AU_KM)}")

with c2, st.container(border=True):
    st.markdown("#### 🛰️ Close approach")
    v_rel = st.number_input("Velocity (km/h)", key="v_rel", step=1000.0, format="%.0f", on_change=clear_example,
                            help="Speed relative to Earth during this close approach.")
    st.caption(f"≈ {v_rel / 3600:,.1f} km/s")

    miss_dist = st.number_input("Miss Distance (km)", key="miss_dist", step=100000.0, format="%.0f", on_change=clear_example,
                                help="How close the asteroid gets to Earth during this flyby.")
    st.caption(f"≈ {describe_distance(miss_dist)}")

# Execution Logic
if st.button("Run Risk Assessment 🚀", type="primary", width="stretch"):

    # Input validation
    if v_rel < 0 or miss_dist < 0:
        st.warning("⚠️ Velocity and Miss Distance cannot be negative.")
        st.stop()
    if moid < 0:
        st.warning("⚠️ MOID cannot be negative.")
        st.stop()

    # Organizing features to match the training shape
    feats = pd.DataFrame([[abs_mag, d_min, d_max, v_rel, miss_dist, moid]],
                          columns=['absolute_magnitude', 'est_diameter_min',
                                   'est_diameter_max', 'relative_velocity',
                                   'miss_distance', 'moid'])

    try:
        prediction = clf.predict(feats)[0]
        hazard_prob = float(clf.predict_proba(feats)[0][1])  # [prob_safe, prob_danger]
    except Exception as e:
        st.warning(f"Something went wrong during prediction: {e}")
        st.stop()

    # The two conditions in NASA's definition, shown so the result can be checked
    large_enough = abs_mag <= PHA_MAX_H
    orbit_close = moid <= PHA_MAX_MOID

    st.subheader("Assessment Report")
    with st.container(border=True):
        info, reaction = st.columns([3, 2], vertical_alignment="center")

        with info:
            if prediction:
                st.error("🚨 ALERT: HAZARDOUS OBJECT DETECTED")
            else:
                st.success("✅ STATUS: SAFE")

            st.metric("Hazard probability", f"{hazard_prob:.0%}")
            st.progress(hazard_prob)

            st.markdown("**NASA's definition of a hazardous asteroid**")
            st.markdown(
                f"{'✅' if large_enough else '❌'} **Large enough:** H = {abs_mag:.2f} "
                f"(needs ≤ {PHA_MAX_H:g}, about 140 m or larger)\n\n"
                f"{'✅' if orbit_close else '❌'} **Orbit comes close:** MOID = {moid:{'.4f' if moid >= 0.001 else '.6f'}} au "
                f"(needs ≤ {PHA_MAX_MOID:g} au)"
            )
            if bool(prediction) != (large_enough and orbit_close):
                st.caption("The model and NASA's rule disagree for these values, "
                           "which usually means the asteroid sits close to one of the thresholds.")

        with reaction:
            st.image(HAZARD_GIF if prediction else SAFE_GIF, width=260)
