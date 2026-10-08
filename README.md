# ☄️ Asteroid Hazard Predictor Using NASA API and Machine Learning

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project is an end-to-end Machine Learning application that predicts whether a Near-Earth Object (asteroid) is **potentially hazardous** to Earth. 

It fetches real-time telemetry data from the **NASA NeoWs API**, processes it to handle class imbalance (only ~5% of asteroids are hazardous), and runs a **Random Forest Classifier** to assess impact risk. The model is deployed via a user-friendly **Streamlit** web interface.

### 📸 Project Demo
![Demo Screenshot](https://github.com/lucifer-135/Asteroid-Hazard-Predictor-Using-NASA-API/blob/main/demo.png?raw=true)

### Asteroid is Safe !
![Safe Screenshot](https://github.com/lucifer-135/Asteroid-Hazard-Predictor-Using-NASA-API/blob/main/Safe.png)

### Asteroid is Hazardous !!!
![Hazardous Screenshot](https://github.com/lucifer-135/Asteroid-Hazard-Predictor-Using-NASA-API/blob/main/hazardous.png)

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Source:** [NASA NeoWs REST API](https://api.nasa.gov/)
* **Machine Learning:** Scikit-Learn (Random Forest), Imbalanced-Learn (SMOTE)
* **Data Processing:** Pandas, NumPy
* **Frontend:** Streamlit
* **Serialization:** Pickle

---

## 📁 Project Structure
```
├── app.py                      # Streamlit web application
├── train_model.ipynb           # Data pipeline, model evaluation + training notebook
├── asteroid_hazard_model.pkl   # Trained Random Forest model
├── nasa_asteroids_big.csv      # Raw dataset collected from NASA API
├── requirements.txt            # Python dependencies
├── hazardous.png               # UI asset — hazardous result
├── Safe.png                    # UI asset — safe result
├── demo.png                    # README demo screenshot
└── README.md
```

---

## ⚙️ Key Features & Methodology

### 1. Automated Data Pipeline (`train_model.ipynb`)
* Connected to the NASA API to fetch 8 weeks of historical asteroid data.
* Parsed complex nested JSON responses into a structured Pandas DataFrame.
* Extracted key physical features: `Absolute Magnitude`, `Estimated Diameter`, `Relative Velocity`, and `Miss Distance`.

### 2. Handling Class Imbalance (The Core Challenge)
* **Problem:** Real-world data is imbalanced (roughly 19:1 Safe vs. Hazardous — only 55 of 1,095 asteroids in the dataset are hazardous). A standard model would bias towards "Safe" and miss actual threats.
* **Solution:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of hazardous asteroids during training, ensuring the model learns robust decision boundaries.

### 3. Model Training & Evaluation
* Trained a **Random Forest Classifier** (`n_estimators=100`) for its ability to handle non-linear relationships between velocity and diameter.
* Evaluated with **repeated stratified 5-fold cross-validation** (5 folds × 3 repeats) plus a held-out 20% test set. SMOTE is applied to the training folds only, so no synthetic data leaks into evaluation.

| Model | Recall (hazardous caught) | Precision (alerts that are real) | F1 |
|---|---|---|---|
| **RF + SMOTE** (deployed) | **0.64 ± 0.16** | 0.35 ± 0.08 | 0.45 |
| RF without SMOTE | 0.39 ± 0.22 | 0.58 ± 0.23 | 0.45 |
| Rule: H ≤ 22 (no-ML baseline) | 1.00 ± 0.00 | 0.30 ± 0.04 | 0.46 |

* **SMOTE raises recall from 39% to 64%**, at the cost of more false alarms — the right trade-off for a safety tool that should minimise missed threats.

### 4. Limitations & Next Steps
* **The model does not beat a simple rule** that flags every asteroid with absolute magnitude H ≤ 22. NASA defines a Potentially Hazardous Asteroid as **H ≤ 22.0 *and* Minimum Orbit Intersection Distance (MOID) ≤ 0.05 au**. Every hazardous asteroid in the dataset has H ≤ 22, but MOID is not one of the features — `Miss Distance` is the distance at one particular close approach, not the closest possible distance between the two orbits. Without it, the model cannot reliably tell which bright asteroids are hazardous.
* **Next step:** add MOID as a feature. It is available as `orbital_data.minimum_orbit_intersection` from the NeoWs per-asteroid lookup endpoint (`/neo/rest/v1/neo/{id}`).

---

## 🚀 How to Run Locally

### Prerequisites
* Python 3.11 or higher (required by the pinned `scikit-learn==1.8.0` that the saved model was trained with)
* *(Optional)* A free NASA API Key (Get one [here](https://api.nasa.gov/)) — only needed to re-download the dataset

### Step 1: Clone the Repository
```bash
git clone https://github.com/lucifer-135/Asteroid-Hazard-Predictor-Using-NASA-API.git
cd Asteroid-Hazard-Predictor-Using-NASA-API
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 (Optional): Retrain the model
The notebook runs end-to-end on the cached `nasa_asteroids_big.csv`, so no API key is needed to reproduce the evaluation and the saved model.

To download fresh data from NASA instead:
* Open the `train_model.ipynb` file.
* Look for the line: `API_KEY = 'YOUR_API_KEY_HERE'`
* Paste your actual NASA key inside the quotes and set `REFETCH_DATA = True`.

### Step 4: Launch the App
```bash
streamlit run app.py
```

> **⚠️ Security Note:** The `asteroid_hazard_model.pkl` file uses Python's pickle serialization. Only load `.pkl` files from sources you trust.
