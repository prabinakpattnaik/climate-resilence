# Experiment Report: Crop Yield Impact

**Best Model:** Random Forest
**Test R2 Score:** 0.4396

## 1. Overview
This model predicts how climate anomalies (rainfall/temperature) impact crop yields in India.

## 2. Methodology
- **Data Source:** Kaggle India Crop Yield Dataset (1997-2020).
- **Features:** Rainfall, Rainfall Anomaly, Fertilizer usage, Pesticide usage, Crop Type, and State.

## 3. Visual Performance Benchmarks
### Actual vs Predicted
![Actual vs Predicted](/static/plots/crop_actual_vs_pred.png)

### Error Distribution (Residuals)
![Residual Distribution](/static/plots/crop_residuals.png)

## 4. Comparison Metrics
| Model | R2 Score | RMSE | MAE |
|-------|----------|------|-----|
| Lasso | 0.1534 | 15.88 | 12.70 |
| Decision Tree | 0.2787 | 14.65 | 10.55 |
| Random Forest | 0.4396 | 12.92 | 9.59 |
