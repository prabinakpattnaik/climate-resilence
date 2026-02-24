# Experiment Report: Drought Prediction (v2)

**Best Model:** Ridge Regression
**Test R2 Score:** 0.1395

## 1. Overview
This model calculates a drought severity score (0-100) based on rainfall deficits and monsoon strength.
**v2 upgrade:** Added SPI-3/SPI-6 (Standardized Precipitation Index via gamma distribution), consecutive dry months feature, and XGBoost as a candidate model.

## 2. Methodology
- **Data Source:** IMD Rainfall Records (121 years, 1901-2021).
- **Features:** 3/6 month rolling averages, rainfall deficit %, previous year drought,
  monsoon strength, SPI-3, SPI-6, consecutive dry months.
- **Models:** Ridge, SVR, Random Forest, XGBoost.

## 3. Visual Performance Benchmarks
### Actual vs Predicted
![Actual vs Predicted](/static/plots/drought_actual_vs_pred.png)

### Error Distribution (Residuals)
![Residual Distribution](/static/plots/drought_residuals.png)

## 4. Comparison Metrics
| Model | R2 Score | RMSE | MAE |
|-------|----------|------|-----|
| Ridge Regression | 0.1395 | 38.13 | 32.82 |
| SVR | 0.0845 | 39.33 | 33.25 |
| Random Forest | 0.1218 | 38.52 | 33.58 |
| XGBoost | 0.1263 | 38.42 | 34.01 |
