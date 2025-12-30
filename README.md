# 🔥 Fire Weather Index (FWI) Prediction

An end-to-end **Data Science and Machine Learning project** that predicts the **Fire Weather Index (FWI)** using the **Algerian Forest Fires dataset**.  
The project includes **data preprocessing, exploratory data analysis (EDA), feature engineering, model training**, and a **Flask web application** for real-time prediction.

---

## 📌 Project Overview

Forest fires cause severe environmental and economic damage.  
The **Fire Weather Index (FWI)** is an important indicator used to estimate fire risk based on weather conditions.

This project:
- Analyzes historical fire data
- Trains a machine learning model to predict FWI
- Deploys the model using a Flask web interface

---

## 📊 Dataset Information

- **Dataset:** Algerian Forest Fires Dataset
- **Region:** Algeria
- **Features include:**
  - Temperature
  - Relative Humidity (RH)
  - Wind Speed (Ws)
  - Rain
  - FFMC, DMC, ISI
  - Region & Classes

---

## 🛠️ Tech Stack & Tools

- **Programming Language:** Python
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib / Seaborn
  - Scikit-learn
- **Machine Learning:**
  - Linear Regression / Ridge Regression
  - StandardScaler
- **Web Framework:** Flask
- **Frontend:** HTML, Bootstrap
- **Version Control:** Git & GitHub

---

## 📁 Project Structur
FWI-Predication/
│
├── Algerian_forest_fires_dataset_UPDATE.csv
├── Algerian_forest_fires_cleaned_dataset.csv
│
├── Models/
│ ├── ridge.pkl
│ └── scaler.pkl
│
├── Notebooks/
│ ├── 2.0-EDA and FE Algerian Forest Fires.ipynb
│ └── 3.0-Model Training.ipynb
│
├── templates/
│ ├── home.html
│ └── index.html
│
├── application.py
├── practice.ipynb
├── requirement.txt
└── README.md


