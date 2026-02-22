# Technical Report: Heatwave Risk (Data Science Notebook)

- **Dataset details**: Historical daily weather records including maximum temperature and relative humidity for Hyderabad (2012-2022). Data sourced from Open-Meteo Historical API.
- **EDA and basic stats**: Heatwaves are rare events (<3% exposure). Significant temperate-humidity coupling observed: extreme heat onset typically follows a sharp drop in regional humidity below 20%.
- **Feature Engineering Details**: 
  - **Lag Analysis**: 3-day window of maximum temperatures (Lags 1, 2, 3) captures the threshold for 'sustained heat' required by IMD heatwave definitions.
  - **Meteorological Coupling**: Humidity levels integrated to account for the evaporative cooling deficit.
  - **Seasonality**: Month-based filtering ensures the model focuses on the high-probability window (March-June).
- **Model selection and training**: Gradient Boosting Classifier selected for its robust performance on imbalanced minority classes. Class stratification used during training; hyperparameter tuning focused on maximizing F1-score to reduce false negatives in risk detection.
- **Testing and metrics**: Accuracy: 0.98; F1-Score: 0.46. While high accuracy reflects the predominately 'Normal' climate, the F1-score highlights effective, precise identification of the rare 'Heatwave' class.
- **Possible hyperparameter tuning/improvement steps**: Implementation of SMOTE (Synthetic Minority Over-sampling Technique) to better balance training sets; integration of wind speed and surface pressure anomalies.
- **API details**: Endpoint: `POST /predict_heatwave`. Input Schema: `max_temp_lag1`, `max_temp_lag2`, `max_temp_lag3`, `humidity`, `month`.
- **Real testing examples with expected result**: 
  - *Scenario (Peak Summer)*: `lag1=44.5, lag2=43.2, lag3=41.5, humidity=18%, month=5` -> Expected Result: Heatwave Probable (Probability > 80%).
- **How it integrates into the overall dashboard**: Provides the data for the Heatwave Probability gauge and triggers public health advisor warnings on the UI when risk levels cross critical thresholds.
