# Experiment Report: Drought Prediction

**Best Model:** Random Forest
**Test R2 Score:** 0.1914

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
| Ridge Regression | 0.1892 | 37.07 | 32.05 |
| SVR | 0.1711 | 37.49 | 31.52 |
| Random Forest | 0.1914 | 37.02 | 31.84 |
