# Experiment Report: Heatwave Prediction (v2)

**Best Model:** XGBoost
**Test F1 Score:** 0.5714

## 1. Overview
This model classifies days as 'Heatwave' or 'Normal' based on previous days' temperatures and humidity.
**v2 upgrade:** Added SMOTE for class balancing (heatwave=~2% of data), XGBoost as candidate, and new features: diurnal range, min temp, precipitation, heat streak.

## 2. Methodology
- **Data Source:** Open-Meteo Historical Weather API (Hyderabad, 2015-2024).
- **Definition:** Max Temp > 40C or >4.5C above normal.
- **Features:** 3-day temperature lags, humidity, calendar month, diurnal range,
  min temp, 7-day precipitation sum, heat streak.
- **Class Balancing:** SMOTE (minority oversampled to 50% of majority in training set).
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost.

## 3. Visual Performance Benchmarks
### Confusion Matrix
![Confusion Matrix](/static/plots/heatwave_confusion_matrix.png)

### ROC Curve
![ROC Curve](/static/plots/heatwave_roc_curve.png)

### Feature Importance
![Feature Importance](/static/plots/heatwave_feature_importance.png)

## 4. Comparison Metrics
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 0.9290 | 0.2785 |
| Random Forest | 0.9826 | 0.5625 |
| Gradient Boosting | 0.9801 | 0.5294 |
| XGBoost | 0.9851 | 0.5714 |
