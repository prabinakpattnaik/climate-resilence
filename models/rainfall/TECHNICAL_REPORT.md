# Technical Report: Rainfall Prediction (Data Science Notebook)

- **Dataset details**: 120+ years of historical monthly rainfall data (mm) for Hyderabad/Telangana. Includes seasonal precipitation, monsoon onset dates, and cumulative annual totals.
- **EDA and basic stats**: Mean monthly rainfall is 65.4mm with a standard deviation of 84.8mm. The distribution is highly right-skewed, characterized by extreme events during the Southwest Monsoon (June-September), which accounts for 80% of total annual precipitation.
- **Feature Engineering Details**: 
  - **Temporal Lags**: 1, 2, 3, and 12-month lags capture autocorrelation and yearly periodicity.
  - **Rolling Averages**: 3-month window used to smooth short-term variance and identify seasonal shifts.
  - **Cyclic Encoding**: Sine and cosine transforms applied to month indices to ensure mathematical continuity between December and January.
- **Model selection and training**: A Random Forest Regressor was selected for its superior ability to model non-linear seasonal interactions. Evaluation involved an 80/20 temporal split and GridSearchCV for hyperparameter optimization (n_estimators, max_depth).
- **Testing and metrics**: R²: 0.5676; RMSE: 59.16. The model effectively captures monsoon peaks but exhibits higher variance during uncharacteristic dry spells or localized cloudbursts.
- **Possible hyperparameter tuning/improvement steps**: Integration of ENSO (El Niño Southern Oscillation) indices and Indian Ocean Dipole (IOD) data; experimentation with XGBoost or LSTM networks for better multi-step forecasting.
- **API details**: Endpoint: `POST /predict_rainfall`. Input Schema: `month` (Int), `lag_1` (Float), `lag_2` (Float), `lag_3` (Float), `lag_12` (Float).
- **Real testing examples with expected result**: 
  - *Scenario (Monsoon Peak)*: August simulation with `lag_1=150, lag_2=80, lag_3=35, month=8` -> Expected Result: ~180-220mm.
- **How it integrates into the overall dashboard**: Powers the primary Rainfall Prediction module, providing immediate monthly forecasts that feed the geospatial risk overlays and the high-level dashboard summary.
