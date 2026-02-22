import sys
import os
import pandas as pd
import importlib.util

# Load app.py directly from path
app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
spec = importlib.util.spec_from_file_location("app_module", app_path)
app_module = importlib.util.module_from_spec(spec)
sys.modules["app_module"] = app_module
spec.loader.exec_module(app_module)

# Import functions from the loaded module
predict_rainfall = app_module.predict_rainfall
predict_drought = app_module.predict_drought
predict_heatwave = app_module.predict_heatwave
predict_crop_impact = app_module.predict_crop_impact
RainfallRequest = app_module.RainfallRequest
DroughtRequest = app_module.DroughtRequest
HeatwaveRequest = app_module.HeatwaveRequest
CropRequest = app_module.CropRequest

print("Verifying Rainfall...")
req = RainfallRequest(lag_1=100, lag_2=120, lag_3=110, lag_12=90, month=7)
res = predict_rainfall(req)
print(f"Rainfall Result: {res['predicted_rainfall_mm']} mm, Risk: {res['risk_category']}")

print("\nVerifying Drought...")
req = DroughtRequest(rolling_3mo_avg=50, rolling_6mo_avg=60, deficit_pct=40, prev_year_drought=1, monsoon_strength=0.5)
res = predict_drought(req)
print(f"Drought Score: {res['drought_score']}, Category: {res['category']}")

print("\nVerifying Heatwave...")
req = HeatwaveRequest(max_temp_lag1=42, max_temp_lag2=41, max_temp_lag3=40, humidity=30, month=5)
res = predict_heatwave(req)
print(f"Heatwave Probability: {res['heatwave_probability']}")

print("\nVerifying Crop...")
req = CropRequest(rainfall=600, rainfall_anomaly=-10, fertilizer_per_area=150, pesticide_per_area=1, 
                 state="Andhra Pradesh", crop_type="Rice", season="Kharif")
res = predict_crop_impact(req)
print(f"Yield Deviation: {res['yield_deviation_pct']}%")

print("\nALL VERIFICATIONS PASSED")
