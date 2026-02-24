# Experiment Report: Crop Yield Impact (v2)

**Best Model:** Random Forest
**Test R2 Score:** 0.4433

## 1. Overview
This model predicts how climate anomalies (rainfall/temperature) impact crop yields in India.
**v2 upgrade:** Added XGBoost with regularization as a candidate model.

## 2. Methodology
- **Data Source:** Kaggle India Crop Yield Dataset (1997-2020).
- **Features:** Rainfall, Rainfall Anomaly, Fertilizer usage, Pesticide usage, Crop Type, State, Season.
- **Models:** Lasso, Decision Tree, Random Forest, XGBoost.

## 3. Visual Performance Benchmarks
### Actual vs Predicted
![Actual vs Predicted](/static/plots/crop_actual_vs_pred.png)

### Error Distribution (Residuals)
![Residual Distribution](/static/plots/crop_residuals.png)

### Feature Importance
![Feature Importance](/static/plots/crop_feature_importance.png)

## 4. Comparison Metrics
| Model | R2 Score | RMSE | MAE |
|-------|----------|------|-----|
| Lasso | 0.1534 | 15.88 | 12.70 |
| Decision Tree | 0.2803 | 14.64 | 10.52 |
| Random Forest | 0.4433 | 12.87 | 9.57 |
| XGBoost | 0.3472 | 13.94 | 10.62 |
