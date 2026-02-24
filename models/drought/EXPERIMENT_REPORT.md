# Experiment Report: Drought Prediction (v2)

**Best Model:** Ridge Regression
**Test R2 Score:** 0.2139

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
| Ridge Regression | 0.2139 | 36.51 | 31.62 |
| SVR | 0.1842 | 37.19 | 31.36 |
| Random Forest | 0.2043 | 36.73 | 31.62 |
| XGBoost | 0.1948 | 36.95 | 31.87 |
