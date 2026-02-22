# Experiment Report: Heatwave Prediction

**Best Model:** Gradient Boosting
**Test F1 Score:** 0.4615

## 1. Overview
This model classifies days as 'Heatwave' or 'Normal' based on previous days' temperatures and humidity.

## 2. Methodology
- **Data Source:** Open-Meteo Historical Weather API (Hyderabad).
- **Definition:** Max Temp > 40 C or >4.5 C above normal.
- **Features:** 3-day temperature lags, humidity, and calendar month.

## 3. Visual Performance Benchmarks
### Confusion Matrix
![Confusion Matrix](/static/plots/heatwave_confusion_matrix.png)

### ROC Curve
![ROC Curve](/static/plots/heatwave_roc_curve.png)

## 4. Comparison Metrics
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 0.9014 | 0.2500 |
| Random Forest | 0.9795 | 0.4444 |
| Gradient Boosting | 0.9808 | 0.4615 |
