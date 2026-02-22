# Technical Report: Crop Yield Impact (Data Science Notebook)

- **Dataset details**: Aggregated dataset combining Kaggle India Crop Yield (1997-2020) with localized Rainfall/Temperature anomalies. Focused on primary regional crops: Rice, Cotton, and Maize.
- **EDA and basic stats**: Mean yield deviation of -4.5% observed in deficit rainfall years. High variance between states suggests that irrigation infrastructure in Telangana mitigates but does not eliminate climate risk.
- **Feature Engineering Details**: 
  - **Anomalies**: 'Rainfall Anomaly %' used as the primary climate stressor.
  - **Agricultural Intensity**: Normalized Fertilizer and Pesticide application rates (kg/ha).
  - **Categorical Context**: Crop Type and State features encoded to capture regional soil and crop-specific resilience.
- **Model selection and training**: Random Forest Regressor selected to capture non-linear agricultural dependencies (e.g., the threshold where fertilizer efficacy drops due to lack of moisture).
- **Testing and metrics**: R²: 0.4396; RMSE: 12.92. The model shows moderate precision in predicting yield reduction percentages and strong directional accuracy for risk assessment.
- **Possible hyperparameter tuning/improvement steps**: Integration of NDVI (Normalized Difference Vegetation Index) from satellite imagery; inclusion of pest-incidence forecasts and market pricing volatility.
- **API details**: Endpoint: `POST /predict_crop_impact`. Input Schema: `crop_type`, `state`, `season`, `rainfall`, `rainfall_anomaly`, `fertilizer_per_area`, `pesticide_per_area`.
- **Real testing examples with expected result**: 
  - *Scenario (Drought Stressed Rice)*: `Rice, Telangana, Kharif, Rainfall=850mm, Anomaly=-22.5%` -> Expected Result: ~12.5% yield reduction.
- **How it integrates into the overall dashboard**: Drives the "Crop Impact Analysis" card, allowing stakeholders to simulate food security outcomes under varying climate projections.
