# Climate AI POC - Technical Documentation

## 1. Project Overview
This project implements an AI-driven Climate Vulnerability Assessment system for Hyderabad/Telangana. It features a **multi-model experimental framework** where different machine learning algorithms compete to provide the most accurate predictions for:
1.  Monthly Rainfall
2.  Drought Severity
3.  Heatwave Risk
4.  Crop Yield Impact

## 2. Experimental Methodology
We adopted a **Champion/Challenger** approach. For each predictive task, we:
1.  Curated real-world datasets (IMD, Open-Meteo, Kaggle).
2.  Defined a consistent 80/20 Train/Test split.
3.  Trained 3 candidate models (e.g., Linear, Random Forest, Boosting).
4.  Selected the best performer based on metrics (R2, F1-Score).

## 3. Model Details

### Model 1: Rainfall Prediction
- **Task:** Predict monthly rainfall (mm).
- **Candidates:** Linear Regression, Random Forest, Gradient Boosting.
- **Winner:** See `models/rainfall/EXPERIMENT_REPORT.md`.
- **Key Features:** Lagged rainfall (t-1, t-2, t-3), Seasonal Lag (t-12), Cyclical Month Encoding.
- **Why it matters:** Enables flood risk early warning.

### Model 2: Drought Prediction
- **Task:** Predict drought severity score (0-100).
- **Candidates:** Ridge Regression, SVR, Random Forest.
- **Winner:** See `models/drought/EXPERIMENT_REPORT.md`.
- **Key Features:** Rainfall Deficit %, Rolling 3/6-mo Averages, Monsoon Strength.
- **Why it matters:** Critical for water resource planning.

### Model 3: Heatwave Prediction
- **Task:** Classify if a day is a Heatwave (Yes/No).
- **Candidates:** Logistic Regression, Random Forest, Gradient Boosting.
- **Winner:** See `models/heatwave/EXPERIMENT_REPORT.md`.
- **Key Features:** Max Temp Lags, 7-day Average, Humidity.
- **Why it matters:** Public health alerts.

### Model 4: Crop Yield Impact
- **Task:** Predict % yield deviation due to climate.
- **Candidates:** Lasso, Decision Tree, Random Forest.
- **Winner:** See `models/crop_impact/EXPERIMENT_REPORT.md`.
- **Key Features:** Crop Type, Rainfall Anomaly, Fertilizer Usage.
- **Why it matters:** Food security and insurance.

## 4. API & Deployment
The models are served via a FastAPI backend:
- **Endpoints:** `/predict_rainfall`, `/predict_drought`, `/predict_heatwave`, `/predict_crop_impact`.
- **Architecture:** Loads `best_model.joblib` dynamically.
- **Pipelines:** Preprocessing (Scaling, Encoding) is embedded in the model pipelines to ensure training/inference consistency.

## 5. Future Work
- Integrate real-time weather API for live inference.
- Expand to more districts beyond Hyderabad.
- Add Deep Learning (LSTM) models for sequence modeling.
