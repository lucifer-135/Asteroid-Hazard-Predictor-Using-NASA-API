# ☄️ Asteroid Hazard Predictor Using NASA API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project is an end-to-end Machine Learning application that predicts whether a Near-Earth Object (asteroid) is **potentially hazardous** to Earth. 

It fetches real-time telemetry data from the **NASA NeoWs API**, processes it to handle severe class imbalance (less than 1% of asteroids are hazardous), and runs a **Random Forest Classifier** to assess impact risk. The model is deployed via a user-friendly **Streamlit** web interface.

### 📸 Project Demo
*(Place a screenshot of your Streamlit app here. Example: `![Demo Screenshot](assets/screenshot.png)`)*

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Source:** [NASA NeoWs REST API](https://api.nasa.gov/)
* **Machine Learning:** Scikit-Learn (Random Forest), Imbalanced-Learn (SMOTE)
* **Data Processing:** Pandas, NumPy
* **Frontend:** Streamlit
* **Serialization:** Pickle

---

## ⚙️ Key Features & Methodology

### 1. Automated Data Pipeline (`etl_script.py`)
* Connected to the NASA API to fetch 8 weeks of historical asteroid data.
* Parsed complex nested JSON responses into a structured Pandas DataFrame.
* Extracted key physical features: `Absolute Magnitude`, `Estimated Diameter`, `Relative Velocity`, and `Miss Distance`.

### 2. Handling Class Imbalance (The Core Challenge)
* **Problem:** Real-world data is highly imbalanced (Ratio 99:1 Safe vs. Hazardous). A standard model would bias towards "Safe" and miss actual threats.
* **Solution:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of hazardous asteroids during training, ensuring the model learns robust decision boundaries.

### 3. Model Training & Evaluation
* Trained a **Random Forest Classifier** (`n_estimators=100`) for its ability to handle non-linear relationships between velocity and diameter.
* Achieved **95%+ Recall** on the minority class (Hazardous), prioritizing safety (minimizing False Negatives).

---

## 🚀 How to Run Locally

### Prerequisites
* Python 3.8 or higher
* A free NASA API Key (Get one [here](https://api.nasa.gov/))

### Step 1: Clone the Repository
```bash
git clone [https://github.com/yourusername/nasa-asteroid-predictor.git](https://github.com/yourusername/nasa-asteroid-predictor.git)
cd nasa-asteroid-predictor
