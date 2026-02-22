# Experiment Report: Rainfall Prediction

**Best Model:** Random Forest
**Test R2 Score:** 0.5676

## 1. Overview
This model predicts monthly rainfall in Hyderabad using historical lag features and rolling averages.

## 2. Methodology
- **Data Source:** IMD Historical Rainfall Data.
- **Features:** 3-month rolling average, lags (1, 2, 3 months), and seasonal sin/cos components.

## 3. Visual Performance Benchmarks
### Actual vs Predicted
![Actual vs Predicted](/static/plots/rainfall_actual_vs_pred.png)

### Error Distribution (Residuals)
![Residual Distribution](/static/plots/rainfall_residuals.png)

## 4. Comparison Metrics
| Model | R2 Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.5484 | 60.46 | 38.03 |
| Random Forest | 0.5676 | 59.16 | 35.85 |
| Gradient Boosting | 0.5338 | 61.43 | 38.69 |
