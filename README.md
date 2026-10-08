# ☄️ Asteroid Hazard Predictor Using NASA API and Machine Learning

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project is an end-to-end Machine Learning application that predicts whether a Near-Earth Object (asteroid) is **potentially hazardous** to Earth. 

It fetches real-time telemetry data from the **NASA NeoWs API**, adds each asteroid's orbit geometry (MOID) from **JPL's Small-Body Database**, handles class imbalance (only ~10% of asteroids are hazardous), and runs a **Random Forest Classifier** to assess impact risk. The model is deployed via a user-friendly **Streamlit** web interface.

### 📸 Project Demo
![Demo Screenshot](https://github.com/lucifer-135/Asteroid-Hazard-Predictor-Using-NASA-API/blob/main/demo.png?raw=true)

### Asteroid is Safe !
![Safe Screenshot](https://github.com/lucifer-135/Asteroid-Hazard-Predictor-Using-NASA-API/blob/main/Safe.png)

### Asteroid is Hazardous !!!
![Hazardous Screenshot](https://github.com/lucifer-135/Asteroid-Hazard-Predictor-Using-NASA-API/blob/main/hazardous.png)

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Sources:** [NASA NeoWs REST API](https://api.nasa.gov/), [JPL Small-Body Database Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html)
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
├── nasa_asteroids_big.csv      # Dataset collected from the NASA NeoWs + JPL SBDB APIs
├── requirements.txt            # Python dependencies
├── hazardous.png               # UI asset — hazardous result
├── Safe.png                    # UI asset — safe result
├── demo.png                    # README demo screenshot
└── README.md
```

---

## ⚙️ Key Features & Methodology

### 1. Automated Data Pipeline (`train_model.ipynb`)
* Connected to the NASA API to fetch 26 weeks of historical asteroid data (January–June 2023), keeping one row per asteroid.
* Parsed complex nested JSON responses into a structured Pandas DataFrame.
* Extracted key physical features: `Absolute Magnitude`, `Estimated Diameter`, `Relative Velocity`, and `Miss Distance`.
* Added each asteroid's **MOID** (Minimum Orbit Intersection Distance — how close its orbit ever comes to Earth's orbit) from JPL's Small-Body Database in a single bulk request, matched by asteroid designation.

### 2. Handling Class Imbalance (The Core Challenge)
* **Problem:** Real-world data is imbalanced (roughly 9:1 Safe vs. Hazardous — only 103 of 1,055 asteroids in the dataset are hazardous). A standard model would bias towards "Safe" and miss actual threats.
* **Solution:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of hazardous asteroids during training, ensuring the model learns robust decision boundaries.

### 3. Model Training & Evaluation
* Trained a **Random Forest Classifier** (`n_estimators=100`) for its ability to handle non-linear relationships between velocity and diameter.
* Evaluated with **repeated stratified 5-fold cross-validation** (5 folds × 3 repeats) plus a held-out 20% test set. SMOTE is applied to the training folds only, so no synthetic data leaks into evaluation.

| Model | Recall (hazardous caught) | Precision (alerts that are real) | F1 |
|---|---|---|---|
| **RF + SMOTE** (deployed) | **0.980 ± 0.040** | **0.991 ± 0.019** | **0.985** |
| RF + SMOTE without MOID | 0.664 ± 0.099 | 0.378 ± 0.072 | 0.480 |
| RF without SMOTE | 0.974 ± 0.040 | 0.991 ± 0.019 | 0.982 |
| Rule: H ≤ 22 (no-ML baseline) | 1.000 ± 0.000 | 0.297 ± 0.022 | 0.457 |
| Rule: H ≤ 22 and MOID ≤ 0.05 au (NASA's definition) | 1.000 ± 0.000 | 0.991 ± 0.019 | 0.995 |

* **MOID is the key feature:** adding it raises recall from 66% to 98% and precision from 38% to 99%. It is also the model's most important feature.
* **SMOTE now makes little difference** (98.0% vs 97.4% recall), because with MOID available the two classes separate cleanly.

### 4. Limitations
* **The model is relearning NASA's definition.** NASA labels an asteroid a Potentially Hazardous Asteroid when **H ≤ 22.0 *and* MOID ≤ 0.05 au**, and that two-condition rule scores as well as the Random Forest. The model's rare misses are asteroids sitting right on one of the two thresholds, so the ML model is best seen as a demonstration of the workflow rather than an improvement over the rule.
* **Velocity and miss distance carry almost no signal** — being hazardous is a property of the orbit, not of one particular flyby. NeoWs also computes the diameter estimates directly from H, so they duplicate H — the app therefore derives the diameter range from H instead of asking for it, so the two always match.
* **NeoWs data changes over time.** Its feed now returns fewer close approaches for early 2023 than when this project's first dataset was collected, which is why the dataset was rebuilt over a 26-week window. Its hazard flag can also lag JPL's latest orbit solutions (e.g. 2012 KC6 meets the definition but isn't flagged).

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
* Create a `.env` file in the project folder containing your key (it is gitignored, so the key never gets committed):
  ```
  NASA_API_KEY=your_key_here
  ```
* Open `train_model.ipynb`, set `REFETCH_DATA = True`, and run all cells. Without a key the notebook falls back to NASA's `DEMO_KEY`, which is limited to a few requests per day.

### Step 4: Launch the App
```bash
streamlit run app.py
```

> **⚠️ Security Note:** The `asteroid_hazard_model.pkl` file uses Python's pickle serialization. Only load `.pkl` files from sources you trust.
