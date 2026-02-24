# Experiment Report: Rainfall Prediction (v2)

**Best Model:** Random Forest
**Test R2 Score:** 0.5912

## 1. Overview
This model predicts monthly rainfall in Hyderabad using historical lag features and rolling averages.
**v2 upgrade:** Added XGBoost with regularization as a candidate model.

## 2. Methodology
- **Data Source:** IMD Historical Rainfall Data (1901-2021, 121 years).
- **Features:** 3-month rolling average, lags (1, 2, 3, 12 months), and seasonal sin/cos components.
- **Models:** Linear Regression, Random Forest, Gradient Boosting, XGBoost.

## 3. Visual Performance Benchmarks
### Actual vs Predicted
![Actual vs Predicted](/static/plots/rainfall_actual_vs_pred.png)

### Error Distribution (Residuals)
![Residual Distribution](/static/plots/rainfall_residuals.png)

## 4. Comparison Metrics
| Model | R2 Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.5815 | 59.24 | 38.36 |
| Random Forest | 0.5912 | 58.55 | 36.28 |
| Gradient Boosting | 0.5424 | 61.94 | 39.44 |
| XGBoost | 0.5770 | 59.55 | 37.49 |
