# Experiment Report: Drought Prediction

**Best Model:** Ridge Regression
**Test R2 Score:** 1.0000

## 1. Overview
This model calculates a drought severity score (0-100) based on rainfall deficits and monsoon strength.

## 2. Methodology
- **Data Source:** IMD Rainfall Records (Historical).
- **Features:** 3/6 month rolling averages, rainfall deficit %, and previous year's drought status.

## 3. Visual Performance Benchmarks
### Actual vs Predicted
![Actual vs Predicted](/static/plots/drought_actual_vs_pred.png)

### Error Distribution (Residuals)
![Residual Distribution](/static/plots/drought_residuals.png)

## 4. Comparison Metrics
| Model | R2 Score | RMSE | MAE |
|-------|----------|------|-----|
| Ridge Regression | 1.0000 | 0.00 | 0.00 |
| SVR | 1.0000 | 0.08 | 0.08 |
| Random Forest | 1.0000 | 0.12 | 0.06 |
