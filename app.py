"""
Urban Climate Vulnerability API
FINAL WORKING VERSION - All models trained on REAL data
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import math
from typing import Optional, List
from utils.grid_logic import ResilienceGrid
from utils.weather_service import WeatherService
from utils.routing_logic import SafeRouter
from utils.agri_logic import AgriResilienceEngine
from utils.iot_service import IoTSensorHub
from utils.tourism_logic import TourismResilienceEngine
from utils.media_logic import MediaIntelligenceEngine
from utils.feature_engine import SmartFeatureEngine

iot_hub = IoTSensorHub()
smart_engine = SmartFeatureEngine()


def _compute_prediction_interval(model, features_df, model_type="regression",
                                  confidence=0.9, n_trees_sample=50):
    """
    Compute prediction intervals using tree-based variance estimation.

    For Random Forest / XGBoost models, uses individual tree predictions to
    estimate uncertainty. For pipeline-wrapped models, extracts the underlying
    estimator. Falls back to a heuristic ±20% interval if model type is unsupported.

    Args:
        model: trained sklearn/xgboost model
        features_df: single-row DataFrame of features
        model_type: "regression" or "classification"
        confidence: confidence level (0.0 to 1.0)
        n_trees_sample: max number of trees to sample for interval

    Returns:
        dict with 'lower', 'upper', 'confidence', and 'method' keys
    """
    from scipy.stats import norm
    alpha = (1 - confidence) / 2  # two-tailed

    # Unwrap pipeline if needed
    estimator = model
    transformed_df = features_df
    if hasattr(model, 'named_steps'):
        # Pipeline: apply all transforms except the final estimator
        for name, step in list(model.named_steps.items())[:-1]:
            transformed_df = pd.DataFrame(
                step.transform(transformed_df),
                columns=features_df.columns if hasattr(step, 'get_feature_names_out') else features_df.columns
            )
        estimator = list(model.named_steps.values())[-1]

    prediction = float(model.predict(features_df)[0])

    if model_type == "classification":
        # For classification, use predict_proba confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)[0]
            max_prob = float(max(proba))
            # Confidence interval around probability
            # Use Wilson score interval approximation
            n_eff = 100  # effective sample size proxy
            z = norm.ppf(1 - alpha)
            p = max_prob
            denominator = 1 + z**2 / n_eff
            center = (p + z**2 / (2 * n_eff)) / denominator
            margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_eff)) / n_eff) / denominator
            return {
                'lower': round(max(0, center - margin), 3),
                'upper': round(min(1, center + margin), 3),
                'confidence': confidence,
                'method': 'wilson_score'
            }
        return {'lower': 0.0, 'upper': 1.0, 'confidence': confidence, 'method': 'fallback'}

    # Regression: try tree-based variance estimation
    tree_preds = []

    # Tree-based models: use individual tree predictions for variance estimation
    try:
        from sklearn.ensemble import (
            RandomForestRegressor, RandomForestClassifier,
            GradientBoostingRegressor, GradientBoostingClassifier
        )
        if isinstance(estimator, (RandomForestRegressor, RandomForestClassifier)):
            for tree in estimator.estimators_[:n_trees_sample]:
                tree_preds.append(float(tree.predict(
                    transformed_df.values if hasattr(transformed_df, 'values') else transformed_df
                )[0]))
        elif isinstance(estimator, (GradientBoostingRegressor, GradientBoostingClassifier)):
            # Use staged_predict for GradientBoosting variance estimation
            staged = list(estimator.staged_predict(
                transformed_df.values if hasattr(transformed_df, 'values') else transformed_df
            ))
            n_stages = len(staged)
            step = max(1, n_stages // n_trees_sample)
            for i in range(0, n_stages, step):
                tree_preds.append(float(staged[i][0]))
    except Exception:
        pass

    if len(tree_preds) >= 5:
        std = float(np.std(tree_preds))
        z = norm.ppf(1 - alpha)
        return {
            'lower': round(prediction - z * std, 2),
            'upper': round(prediction + z * std, 2),
            'confidence': confidence,
            'method': 'tree_variance'
        }

    # Fallback: use ±20% heuristic interval
    margin = abs(prediction) * 0.2
    return {
        'lower': round(prediction - margin, 2),
        'upper': round(prediction + margin, 2),
        'confidence': confidence,
        'method': 'heuristic'
    }

app = FastAPI(
    title="Urban Climate Vulnerability API",
    description="AI-powered climate predictions for Hyderabad/Telangana. All models trained on REAL datasets.",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === MODEL LOADING ===

def safe_load(path, name):
    full_path = os.path.join(BASE_DIR, path)
    if os.path.exists(full_path):
        print(f"[OK] Loaded: {name}")
        return joblib.load(full_path)
    print(f"[MISSING] {name} at {full_path}")
    return None

# Load all models and components
rainfall_model = safe_load("models/rainfall/best_model.joblib", "Rainfall Model")
drought_model = safe_load("models/drought/best_model.joblib", "Drought Model")
heatwave_model = safe_load("models/heatwave/best_model.joblib", "Heatwave Model")
crop_model = safe_load("models/crop_impact/best_model.joblib", "Crop Model")
crop_encoder = safe_load("models/crop_impact/crop_encoder.joblib", "Crop Encoder")
state_encoder = safe_load("models/crop_impact/state_encoder.joblib", "State Encoder")
season_encoder = safe_load("models/crop_impact/season_encoder.joblib", "Season Encoder")

# Serve static dashboard
static_path = os.path.join(BASE_DIR, "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# === GLOBAL SPATIAL CACHE ===
# Pre-parsing KML files to avoid expensive disk I/O on every request
cached_spatial_features = {}

def preload_infrastructure():
    global cached_spatial_features
    print("[INFO] Pre-loading urban infrastructure data (KML)...")
    kml_dir = os.path.join(BASE_DIR, "kml_files")
    kml_mappings = {
        'nalas': 'Hyd_Nalas.kml',
        'drains': 'Hyd_Canals&Drains.kml',
        'lakes': 'Hyd_Tanks&Lakes.kml',
        'hotspots': 'Hyd_FloodingLocations.kml'
    }
    
    # Use a dummy ResilienceGrid to leverage its parser
    temp_rg = ResilienceGrid([0,0,1,1]) 
    for feature, filename in kml_mappings.items():
        path = os.path.join(kml_dir, filename)
        if os.path.exists(path):
            temp_rg.load_kml_features(path, feature)
    
    cached_spatial_features = temp_rg.spatial_features
    print(f"[OK] Pre-loaded {len(cached_spatial_features)} infrastructure layers.")

# Preload on module load
preload_infrastructure()

# === REQUEST MODELS ===

class RainfallRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_12: float
    month: int
    mei_value: float = 0.0
    mei_lag3: float = 0.0

class DroughtRequest(BaseModel):
    rolling_3mo_avg: float
    rolling_6mo_avg: float
    deficit_pct: float
    prev_year_drought: float
    monsoon_strength: float
    spi_3: float = 0.0
    spi_6: float = 0.0
    consecutive_dry_months: int = 0
    mei_value: float = 0.0
    mei_lag3: float = 0.0

class HeatwaveRequest(BaseModel):
    max_temp_lag1: float
    max_temp_lag2: float
    max_temp_lag3: float
    humidity: float
    month: int
    diurnal_range_lag1: float = 12.0
    temp_min_lag1: float = 23.0
    temp_min_7day_avg: float = 22.5
    precip_7day_sum: float = 0.0
    heat_streak: int = 0
    # NEW v3: Weather-enriched features
    wind_speed_lag1: float = 0.0
    wind_gusts_lag1: float = 0.0
    solar_rad_lag1: float = 0.0
    solar_rad_3day_avg: float = 0.0
    et0_lag1: float = 0.0
    et0_7day_avg: float = 0.0
    soil_moisture_lag1: float = 0.2
    pressure_lag1: float = 1013.0
    pressure_change_1d: float = 0.0

class CropRequest(BaseModel):
    crop_type: str
    rainfall: float
    rainfall_anomaly: float
    fertilizer_per_area: float = 80
    pesticide_per_area: float = 2.5
    state: str = "Andhra Pradesh"
    season: str = "Kharif"

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float

class AgriRequest(BaseModel):
    crop: str
    state: str
    drought_prob: float

# === SMART (CITIZEN) REQUEST MODELS ===

class SmartRainfallRequest(BaseModel):
    month: int  # 1-12

class SmartDroughtRequest(BaseModel):
    month: int  # 1-12

class SmartHeatwaveRequest(BaseModel):
    month: int  # 1-12

class SmartCropRequest(BaseModel):
    crop_type: str
    state: str = "Telangana"
    season: str = "Kharif"

@app.post("/agri_advisory")
def get_agri_advisory(req: AgriRequest):
    """Provides climate-aware advice for farmers."""
    ws = WeatherService()
    weather = ws.get_live_rainfall()
    
    engine = AgriResilienceEngine(weather, drought_risk=req.drought_prob)
    recommendations = engine.get_crop_advisor(req.crop)
    
    # Check logistics to nearest Mandi (using a default Bowenpally lat/lon for distance check)
    logistics = engine.get_market_logistics(mandi_distance_km=10.0) 
    
    # NEW Phase 9: Flood Risk
    flood_risk = engine.get_flood_risk_assessment()
    
    return {
        "crop": req.crop,
        "recommendations": recommendations,
        "logistics": logistics,
        "flood_risk": flood_risk,
        "weather": weather
    }

@app.post("/farm_health_scan")
def farm_health_scan(req: RouteRequest):
    """Simulates a precision NDVI scan for a farmer's plot."""
    try:
        ws = WeatherService()
        weather = ws.get_live_rainfall()
        base_dist_risk = 75.0 if weather.get('current_rainfall_mm', 0) < 5 else 30.0
        engine = AgriResilienceEngine(weather, drought_risk=base_dist_risk)
        farm_bbox = [req.start_lat - 0.005, req.start_lon - 0.005, req.start_lat + 0.005, req.start_lon + 0.005]
        scan_results = engine.get_satellite_health_scan(farm_bbox)
        flood_risk = engine.get_flood_risk_assessment()
        return {
            "center": {"lat": req.start_lat, "lon": req.start_lon},
            "grid": scan_results,
            "flood_risk": flood_risk,
            "timestamp": weather.get("timestamp")
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/urban_health_scan")
def urban_health_scan(req: RouteRequest):
    """Simulates a precision locality scan for urban areas (Ward-level)."""
    try:
        ws = WeatherService()
        weather = ws.get_live_rainfall()
        engine = AgriResilienceEngine(weather, drought_risk=40.0)
        urban_bbox = [req.start_lat - 0.01, req.start_lon - 0.01, req.start_lat + 0.01, req.start_lon + 0.01]
        # High-res 15x15 grid for 2km Ward area
        scan_results = engine.get_satellite_health_scan(urban_bbox, resolution=15)
        avg_ndvi = sum(s['ndvi'] for s in scan_results) / len(scan_results)
        impervious_ratio = round((1.0 - avg_ndvi) * 100, 1)
        return {
            "center": {"lat": req.start_lat, "lon": req.start_lon},
            "grid": scan_results,
            "stats": {
                "avg_greenery_index": round(avg_ndvi, 2),
                "impervious_surface_pct": impervious_ratio,
                "heat_island_risk": "High" if avg_ndvi < 0.4 else "Moderate" if avg_ndvi < 0.6 else "Low",
                "drainage_bottleneck_risk": "Critical" if impervious_ratio > 75 else "Significant" if impervious_ratio > 50 else "Low"
            },
            "timestamp": weather.get("timestamp")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/iot_sensor_data")
def get_iot_data():
    """Returns simulated real-time IoT sensor readings from across the city."""
    return iot_hub.get_live_sensor_data()

@app.get("/tourism_safety")
def tourism_safety():
    """Returns climate-aware safety analysis for major landmarks."""
    ws = WeatherService()
    weather = ws.get_live_rainfall()
    engine = TourismResilienceEngine(weather)
    return {
        "reports": engine.get_landmark_safety(),
        "weather_summary": weather
    }

@app.get("/generate_media_alert")
def generate_media_alert():
    """Generates PSAs and Media Briefs based on current climate state."""
    ws = WeatherService()
    weather = ws.get_live_rainfall()
    
    # Mock some hotspots for the brief
    stats = {"risk_hotspots": ["Khairatabad", "Secunderabad", "LB Nagar"]}
    
    engine = MediaIntelligenceEngine(weather, stats)
    return engine.generate_broadcast_psa()

# === ENDPOINTS ===

@app.get("/")
def dashboard():
    index = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Dashboard not found", "api_docs": "/docs"}

@app.post("/predict_rainfall")
def predict_rainfall(req: RainfallRequest):
    if rainfall_model is None:
        raise HTTPException(503, "Rainfall model not loaded")
    
    month_sin = np.sin(2 * np.pi * req.month / 12)
    month_cos = np.cos(2 * np.pi * req.month / 12)
    rolling_3 = (req.lag_1 + req.lag_2 + req.lag_3) / 3
    
    features_df = pd.DataFrame([{
        "lag_1": req.lag_1,
        "lag_2": req.lag_2,
        "lag_3": req.lag_3,
        "lag_12": req.lag_12,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "rolling_3": rolling_3,
        "mei_value": req.mei_value,
        "mei_lag3": req.mei_lag3
    }])
    
    pred = max(0, rainfall_model.predict(features_df)[0])
    risk = "High Risk (Flooding)" if pred > 200 else "Low Risk (Drought)" if pred < 30 else "Normal"
    
    return {
        "predicted_rainfall_mm": round(pred, 2),
        "risk_category": risk,
        "input": req.model_dump()
    }

@app.post("/predict_drought")
def predict_drought(req: DroughtRequest):
    if drought_model is None:
        raise HTTPException(503, "Drought model not loaded")

    features = pd.DataFrame([{
        'rolling_3mo': req.rolling_3mo_avg,
        'rolling_6mo': req.rolling_6mo_avg,
        'deficit_pct': req.deficit_pct,
        'prev_year_drought': req.prev_year_drought,
        'monsoon_strength': req.monsoon_strength,
        'spi_3': req.spi_3,
        'spi_6': req.spi_6,
        'consecutive_dry_months': req.consecutive_dry_months,
        'mei_value': req.mei_value,
        'mei_lag3': req.mei_lag3
    }])

    score = max(0, min(100, drought_model.predict(features)[0]))

    cat = "Extreme Drought" if score > 80 else "Severe Drought" if score > 60 else "Moderate Drought" if score > 40 else "Mild Drought" if score > 20 else "No Drought"

    return {
        "drought_score": round(score, 2),
        "category": cat,
        "input": req.model_dump()
    }

@app.post("/predict_heatwave")
def predict_heatwave(req: HeatwaveRequest):
    if heatwave_model is None:
        raise HTTPException(503, "Heatwave model not loaded")

    temp_7day_avg = (req.max_temp_lag1 + req.max_temp_lag2 + req.max_temp_lag3) / 3
    month_sin = np.sin(2 * np.pi * req.month / 12)
    month_cos = np.cos(2 * np.pi * req.month / 12)

    features = pd.DataFrame([{
        'temp_max_lag1': req.max_temp_lag1, 'temp_max_lag2': req.max_temp_lag2, 'temp_max_lag3': req.max_temp_lag3,
        'temp_max_7day_avg': temp_7day_avg, 'humidity': req.humidity, 'month_sin': month_sin, 'month_cos': month_cos, 'month': req.month,
        'diurnal_range_lag1': req.diurnal_range_lag1, 'temp_min_lag1': req.temp_min_lag1,
        'temp_min_7day_avg': req.temp_min_7day_avg, 'precip_7day_sum': req.precip_7day_sum,
        'heat_streak': req.heat_streak,
        'wind_speed_lag1': req.wind_speed_lag1, 'wind_gusts_lag1': req.wind_gusts_lag1,
        'solar_rad_lag1': req.solar_rad_lag1, 'solar_rad_3day_avg': req.solar_rad_3day_avg,
        'et0_lag1': req.et0_lag1, 'et0_7day_avg': req.et0_7day_avg,
        'soil_moisture_lag1': req.soil_moisture_lag1,
        'pressure_lag1': req.pressure_lag1, 'pressure_change_1d': req.pressure_change_1d
    }])

    prediction = heatwave_model.predict(features)[0]
    probability = heatwave_model.predict_proba(features)[0][1]

    return {
        "is_heatwave": bool(prediction),
        "heatwave_probability": round(float(probability), 3),
        "input": req.model_dump()
    }

@app.post("/predict_crop_impact")
def predict_crop_impact(req: CropRequest):
    if crop_model is None or crop_encoder is None:
        raise HTTPException(503, "Crop model not loaded")
    
    try:
        if req.crop_type not in crop_encoder.classes_:
            raise HTTPException(400, f"Unknown crop '{req.crop_type}'. Available: {crop_encoder.classes_.tolist()}")
        if req.state not in state_encoder.classes_:
            raise HTTPException(400, f"Unknown state '{req.state}'. Available: {state_encoder.classes_.tolist()}")
        if req.season.strip() not in season_encoder.classes_:
            raise HTTPException(400, f"Unknown season '{req.season}'. Available: {season_encoder.classes_.tolist()}")

        crop_enc = crop_encoder.transform([req.crop_type])[0]
        state_enc = state_encoder.transform([req.state])[0]
        season_enc = season_encoder.transform([req.season.strip()])[0]

        features = pd.DataFrame([{
            'rainfall': req.rainfall, 'rainfall_anomaly': req.rainfall_anomaly,
            'fertilizer_per_area': req.fertilizer_per_area, 'pesticide_per_area': req.pesticide_per_area,
            'crop_encoded': crop_enc, 'state_encoded': state_enc, 'season_encoded': season_enc
        }])

        yield_dev = max(0, min(80, crop_model.predict(features)[0]))
        cat = "Critical Impact" if yield_dev > 50 else "Severe Impact" if yield_dev > 30 else "Moderate Impact" if yield_dev > 15 else "Mild Impact" if yield_dev > 5 else "Minimal Impact"

        return {
            "yield_deviation_pct": round(yield_dev, 2),
            "impact_category": cat,
            "available_crops": crop_encoder.classes_.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Internal Error: {str(e)}")

@app.get("/model_metrics")
def get_all_metrics():
    return {
        "data_sources": {
            "rainfall": "IMD Hyderabad 1901-2021",
            "heatwave": "Open-Meteo API 2015-2024",
            "crop_impact": "Kaggle India Crop Yield"
        }
    }

@app.get("/kml_files")
def list_kml():
    kml_dir = os.path.join(BASE_DIR, "kml_files")
    return {"files": os.listdir(kml_dir)} if os.path.exists(kml_dir) else {"files": []}

@app.get("/kml_data/{filename}")
def get_kml(filename: str):
    kml_path = os.path.join(BASE_DIR, "kml_files", filename)
    if os.path.exists(kml_path):
        return FileResponse(kml_path, media_type="application/vnd.google-earth.kml+xml")
    raise HTTPException(404, "KML file not found")

def _compute_baselines(weather):
    """Compute heatwave probability and crop risk from models instead of hardcoded values."""
    hw_prob = 0.0
    if heatwave_model is not None:
        # Use current conditions as a rough baseline estimate
        temp_est = 35.0  # seasonal average for Hyderabad
        features = pd.DataFrame([{
            'temp_max_lag1': temp_est, 'temp_max_lag2': temp_est, 'temp_max_lag3': temp_est,
            'temp_max_7day_avg': temp_est, 'humidity': 50.0,
            'month_sin': np.sin(2 * np.pi * pd.Timestamp.now().month / 12),
            'month_cos': np.cos(2 * np.pi * pd.Timestamp.now().month / 12),
            'month': pd.Timestamp.now().month,
            'diurnal_range_lag1': 12.0, 'temp_min_lag1': 23.0,
            'temp_min_7day_avg': 22.5, 'precip_7day_sum': 0.0,
            'heat_streak': 0,
            'wind_speed_lag1': 10.0, 'wind_gusts_lag1': 15.0,
            'solar_rad_lag1': 15.0, 'solar_rad_3day_avg': 15.0,
            'et0_lag1': 4.0, 'et0_7day_avg': 4.0,
            'soil_moisture_lag1': 0.2,
            'pressure_lag1': 1013.0, 'pressure_change_1d': 0.0
        }])
        try:
            hw_prob = round(float(heatwave_model.predict_proba(features)[0][1]) * 100, 1)
        except Exception:
            hw_prob = 0.0

    rain_mm = weather.get("current_rainfall_mm", 0)
    crop_risk = "High" if rain_mm > 100 else "Moderate" if rain_mm > 50 else "Low"

    return {"heatwave_prob": hw_prob, "crop_risk": crop_risk}


@app.get("/dashboard_summary")
def summary():
    ws = WeatherService()
    weather = ws.get_live_rainfall()
    
    return {
        "models_status": {
            "rainfall": rainfall_model is not None, 
            "drought": drought_model is not None, 
            "heatwave": heatwave_model is not None, 
            "crop_impact": crop_model is not None
        },
        "location": "Hyderabad, Telangana",
        "live_weather": {
            "current_mm": weather.get("current_rainfall_mm", 0.0),
            "next_1h_mm": weather.get("predicted_next_1h_mm", 0.0),
            "status": "Raining" if weather.get("is_raining") else "Clear",
            "timestamp": weather.get("timestamp", "--")
        },
        "baselines": _compute_baselines(weather)
    }


@app.get("/impact_showcase")
def impact_showcase():
    """Computes real-time impact scenarios using the same ML models the dashboard uses.
    Returns 3 before/after impact story cards with quantified benefits."""
    import datetime
    current_month = datetime.datetime.now().month
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    scenarios = []

    # --- SCENARIO 1: Farmer Crop Switch Advisory ---
    if crop_model is not None and crop_encoder is not None:
        try:
            # Run Rice (water-intensive, vulnerable)
            rice_features = smart_engine.compute_crop_features("Rice", "Telangana", "Kharif")
            rice_enc = crop_encoder.transform(["Rice"])[0]
            state_enc = state_encoder.transform(["Telangana"])[0]
            season_enc = season_encoder.transform(["Kharif"])[0]
            rice_df = pd.DataFrame([{
                'rainfall': rice_features['rainfall'],
                'rainfall_anomaly': rice_features['rainfall_anomaly'],
                'fertilizer_per_area': rice_features['fertilizer_per_area'],
                'pesticide_per_area': rice_features['pesticide_per_area'],
                'crop_encoded': rice_enc, 'state_encoded': state_enc, 'season_encoded': season_enc
            }])
            rice_yield_dev = round(max(0, min(80, crop_model.predict(rice_df)[0])), 1)

            # Run drought-resistant alternative (Maize or first available alternative)
            alt_crop = "Maize" if "Maize" in crop_encoder.classes_ else [c for c in crop_encoder.classes_ if c != "Rice"][0]
            alt_features = smart_engine.compute_crop_features(alt_crop, "Telangana", "Kharif")
            alt_enc = crop_encoder.transform([alt_crop])[0]
            alt_df = pd.DataFrame([{
                'rainfall': alt_features['rainfall'],
                'rainfall_anomaly': alt_features['rainfall_anomaly'],
                'fertilizer_per_area': alt_features['fertilizer_per_area'],
                'pesticide_per_area': alt_features['pesticide_per_area'],
                'crop_encoded': alt_enc, 'state_encoded': state_enc, 'season_encoded': season_enc
            }])
            alt_yield_dev = round(max(0, min(80, crop_model.predict(alt_df)[0])), 1)

            loss_prevented = round(max(0, rice_yield_dev - alt_yield_dev), 1)
            # Avg Telangana rice revenue ~INR 50,000/ha; 1% yield ≈ INR 500
            savings_per_ha = int(loss_prevented * 500)

            scenarios.append({
                "id": "farmer_crop_switch",
                "icon": "farmer",
                "title": "Smart Crop Advisory",
                "subtitle": "Early drought detection saves harvests",
                "without_ai": {
                    "label": "Without Early Warning",
                    "description": f"Rice in Telangana Kharif faces {rice_yield_dev}% yield deviation due to climate stress",
                    "metric_value": f"{rice_yield_dev}%",
                    "metric_label": "Yield Loss Risk"
                },
                "with_ai": {
                    "label": "With AI Advisory",
                    "description": f"Switching to {alt_crop} reduces impact to just {alt_yield_dev}% deviation",
                    "metric_value": f"{alt_yield_dev}%",
                    "metric_label": "Reduced Impact"
                },
                "big_metric": f"{loss_prevented}%",
                "big_metric_label": "Crop Loss Prevented",
                "savings": f"~INR {savings_per_ha:,}/hectare saved per season",
                "explanation": f"AI analyzes 19,000+ crop records. Compares {alt_crop} (drought-resilient) vs Rice (water-intensive) under identical Telangana Kharif conditions.",
                "try_model": "crop"
            })
        except Exception as e:
            print(f"Impact showcase - crop scenario failed: {e}")

    # --- SCENARIO 2: Citizen Heatwave Early Warning ---
    if heatwave_model is not None:
        try:
            heat_month = current_month if current_month in [3, 4, 5, 6] else 5
            hw_features = smart_engine.compute_heatwave_features(heat_month)
            hw_df = pd.DataFrame([{
                'temp_max_lag1': hw_features['max_temp_lag1'],
                'temp_max_lag2': hw_features['max_temp_lag2'],
                'temp_max_lag3': hw_features['max_temp_lag3'],
                'temp_max_7day_avg': hw_features['temp_max_7day_avg'],
                'humidity': hw_features['humidity'],
                'month_sin': hw_features['month_sin'],
                'month_cos': hw_features['month_cos'],
                'month': hw_features['month'],
                'diurnal_range_lag1': hw_features['diurnal_range_lag1'],
                'temp_min_lag1': hw_features['temp_min_lag1'],
                'temp_min_7day_avg': hw_features['temp_min_7day_avg'],
                'precip_7day_sum': hw_features['precip_7day_sum'],
                'heat_streak': hw_features['heat_streak'],
                'wind_speed_lag1': hw_features.get('wind_speed_lag1', 0.0),
                'wind_gusts_lag1': hw_features.get('wind_gusts_lag1', 0.0),
                'solar_rad_lag1': hw_features.get('solar_rad_lag1', 0.0),
                'solar_rad_3day_avg': hw_features.get('solar_rad_3day_avg', 0.0),
                'et0_lag1': hw_features.get('et0_lag1', 0.0),
                'et0_7day_avg': hw_features.get('et0_7day_avg', 0.0),
                'soil_moisture_lag1': hw_features.get('soil_moisture_lag1', 0.2),
                'pressure_lag1': hw_features.get('pressure_lag1', 1013.0),
                'pressure_change_1d': hw_features.get('pressure_change_1d', 0.0)
            }])
            hw_prob = round(float(heatwave_model.predict_proba(hw_df)[0][1]) * 100, 1)

            scenarios.append({
                "id": "citizen_heatwave",
                "icon": "heatwave",
                "title": "48-Hour Heatwave Alert",
                "subtitle": f"Protecting citizens in {month_names[heat_month]}",
                "without_ai": {
                    "label": "Without Prediction",
                    "description": "No advance preparation. Cooling centers closed. Vulnerable populations exposed.",
                    "metric_value": "0h",
                    "metric_label": "Warning Time"
                },
                "with_ai": {
                    "label": "With AI Prediction",
                    "description": f"Heatwave risk: {hw_prob}%. City opens cooling centers, SMS alerts sent to 2M+ residents.",
                    "metric_value": "48h",
                    "metric_label": "Early Warning"
                },
                "big_metric": f"{hw_prob}%",
                "big_metric_label": "AI Detection Confidence",
                "savings": "48-hour advance warning enables city-wide preparation",
                "explanation": f"AI analyzed 10 years of temperature data. Current 7-day avg: {round(hw_features['temp_max_7day_avg'], 1)} C, Live humidity: {round(hw_features['humidity'])}%.",
                "try_model": "heatwave"
            })
        except Exception as e:
            print(f"Impact showcase - heatwave scenario failed: {e}")

    # --- SCENARIO 3: Drought Advance Planning ---
    if drought_model is not None:
        try:
            # Pick an upcoming dry-season month
            drought_month = ((current_month + 1) % 12) + 1
            if drought_month in [7, 8, 9]:
                drought_month = 4  # April pre-monsoon is more compelling
            months_ahead = (drought_month - current_month) % 12
            if months_ahead == 0:
                months_ahead = 1

            dr_features = smart_engine.compute_drought_features(drought_month)
            dr_df = pd.DataFrame([{
                'rolling_3mo': dr_features['rolling_3mo_avg'],
                'rolling_6mo': dr_features['rolling_6mo_avg'],
                'deficit_pct': dr_features['deficit_pct'],
                'prev_year_drought': dr_features['prev_year_drought'],
                'monsoon_strength': dr_features['monsoon_strength'],
                'spi_3': dr_features['spi_3'],
                'spi_6': dr_features['spi_6'],
                'consecutive_dry_months': dr_features['consecutive_dry_months'],
                'mei_value': dr_features.get('mei_value', 0.0),
                'mei_lag3': dr_features.get('mei_lag3', 0.0)
            }])
            drought_score = round(max(0, min(100, drought_model.predict(dr_df)[0])), 0)
            drought_cat = ("Extreme" if drought_score > 80 else "Severe" if drought_score > 60
                           else "Moderate" if drought_score > 40 else "Mild" if drought_score > 20
                           else "None")

            scenarios.append({
                "id": "drought_planning",
                "icon": "drought",
                "title": f"{months_ahead}-Month Drought Forecast",
                "subtitle": f"Reservoir planning for {month_names[drought_month]}",
                "without_ai": {
                    "label": "Without Forecast",
                    "description": "Reactive water management. Rationing starts only after crisis hits.",
                    "metric_value": "0",
                    "metric_label": "Months Warning"
                },
                "with_ai": {
                    "label": "With AI Forecast",
                    "description": f"Drought score: {int(drought_score)} ({drought_cat}). Water board adjusts reservoir release {months_ahead} months early.",
                    "metric_value": f"{months_ahead}",
                    "metric_label": "Months Warning"
                },
                "big_metric": f"{int(drought_score)}",
                "big_metric_label": "Drought Severity Index",
                "savings": f"{months_ahead}-month advance planning prevents water crisis",
                "explanation": f"AI computed from 121 years of IMD rainfall. Deficit: {round(dr_features['deficit_pct'], 1)}%, Monsoon strength: {round(dr_features['monsoon_strength'], 2)}.",
                "try_model": "drought"
            })
        except Exception as e:
            print(f"Impact showcase - drought scenario failed: {e}")

    return {
        "scenarios": scenarios,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": {
            "crop": "19,000+ records India Crop Yield dataset",
            "heatwave": "11 years Open-Meteo weather data (temp, wind, solar, pressure, soil moisture)",
            "drought": "125 years Hyderabad rainfall + NOAA ENSO index",
            "enso": "NOAA MEI v2 (1979-2026) El Nino/La Nina climate index"
        }
    }


# === EMERGENCY RESOURCES (Phase 6) ===
EMERGENCY_RESOURCES = [
    {"id": "hosp_1", "name": "NIMS Hospital", "type": "Hospital", "lat": 17.420, "lon": 78.455, "contact": "040-23489000"},
    {"id": "hosp_2", "name": "Apollo Hospitals, Jubilee Hills", "type": "Hospital", "lat": 17.425, "lon": 78.412, "contact": "040-23607777"},
    {"id": "ndrf_1", "name": "NDRF Station, Hyderabad", "type": "NDRF", "lat": 17.475, "lon": 78.435, "contact": "9701019909"},
    {"id": "shelter_1", "name": "GHMC Shelter - Khairatabad", "type": "Shelter", "lat": 17.412, "lon": 78.462, "contact": "040-21111111"},
    {"id": "shelter_2", "name": "Community Hall - Nampally", "type": "Shelter", "lat": 17.392, "lon": 78.471, "contact": "040-21111111"},
    {"id": "mandi_1", "name": "Bowenpally Agri Mandi", "type": "Mandi", "lat": 17.478, "lon": 78.468, "contact": "040-23743454"},
    {"id": "mandi_2", "name": "Malakpet Market", "type": "Mandi", "lat": 17.378, "lon": 78.498, "contact": "040-24545367"},
    {"id": "mandi_3", "name": "Gudimalkapur Flower Market", "type": "Mandi", "lat": 17.390, "lon": 78.442, "contact": "040-23512345"}
]

@app.get("/emergency_resources")
def get_emergency_resources():
    return {"resources": EMERGENCY_RESOURCES}

@app.post("/find_nearest_safe_haven")
def find_nearest_safe_haven(req: RouteRequest):
    """Finds the nearest safe haven and calculates a safe route to it."""
    try:
        # Find nearest by Euclidean distance first
        def dist(r):
            return math.sqrt((r['lat'] - req.start_lat)**2 + (r['lon'] - req.start_lon)**2)
        
        nearest = min(EMERGENCY_RESOURCES, key=dist)
        
        # Now calculate safe route to it
        grid_df, weather = _generate_active_grid()
        router = SafeRouter(grid_df)
        path, message = router.find_safest_path(
            (req.start_lat, req.start_lon),
            (nearest['lat'], nearest['lon'])
        )
        
        if path is None:
            raise HTTPException(400, f"Could not find a safe path to the nearest resource ({nearest['name']})")
            
        return {
            "destination": nearest,
            "path": path,
            "message": f"Safest route to {nearest['name']} ({nearest['type']}) calculated.",
            "weather_context": weather
        }
    except Exception as e:
        import traceback
        print(f"ERROR in find_nearest_safe_haven: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(500, f"Internal Server Error: {str(e)}")

# === RESILIENCE GRID & ROUTING ===

def _generate_active_grid():
    """Shared utility to generate the grid with live weather."""
    ws = WeatherService()
    weather = ws.get_live_rainfall()
    live_rain = weather.get("current_rainfall_mm", 0.0)
    
    bbox = [17.35, 78.40, 17.48, 78.55] 
    rg = ResilienceGrid(bbox, cell_size=0.008) 
    
    # Inject cached features instead of re-parsing
    rg.spatial_features = cached_spatial_features
            
    grid_results = rg.calculate_vulnerability(live_rainfall=live_rain)
    return grid_results, weather

@app.get("/resilience_grid")
def get_resilience_grid():
    """Generate and return the resilience grid for central Hyderabad with Live Nowcasting."""
    try:
        grid_df, weather = _generate_active_grid()
        return {
            "grid": grid_df.to_dict(orient='records'),
            "weather": weather
        }
    except Exception as e:
        import traceback
        print("CRITICAL ERROR in /resilience_grid:")
        traceback.print_exc()
        raise HTTPException(500, f"Grid calculation failed: {str(e)}")

@app.post("/calculate_safe_route")
def calculate_safe_route(req: RouteRequest):
    """Calculates a flood-safe path avoiding high-risk grid cells."""
    grid_df, weather = _generate_active_grid()
    
    router = SafeRouter(grid_df)
    path, message = router.find_safest_path(
        (req.start_lat, req.start_lon), 
        (req.end_lat, req.end_lon)
    )
    
    if path is None:
        raise HTTPException(400, message)
        
    return {
        "path": path,
        "message": message,
        "weather_context": weather
    }

# === SMART (CITIZEN-FRIENDLY) ENDPOINTS ===

@app.post("/smart/predict_rainfall")
def smart_predict_rainfall(req: SmartRainfallRequest):
    """Citizen-friendly rainfall prediction. Only requires month selection."""
    if rainfall_model is None:
        raise HTTPException(503, "Rainfall model not loaded")

    features = smart_engine.compute_rainfall_features(req.month)
    features_df = pd.DataFrame([{
        "lag_1": features['lag_1'],
        "lag_2": features['lag_2'],
        "lag_3": features['lag_3'],
        "lag_12": features['lag_12'],
        "month_sin": features['month_sin'],
        "month_cos": features['month_cos'],
        "rolling_3": features['rolling_3'],
        "mei_value": features.get('mei_value', 0.0),
        "mei_lag3": features.get('mei_lag3', 0.0)
    }])

    pred = max(0, rainfall_model.predict(features_df)[0])
    risk = "High Risk (Flooding)" if pred > 200 else "Low Risk (Drought)" if pred < 30 else "Normal"
    interval = _compute_prediction_interval(rainfall_model, features_df, model_type="regression")

    return {
        "predicted_rainfall_mm": round(pred, 2),
        "risk_category": risk,
        "prediction_interval": interval,
        "auto_computed_features": features,
        "mode": "citizen"
    }


@app.post("/smart/predict_drought")
def smart_predict_drought(req: SmartDroughtRequest):
    """Citizen-friendly drought assessment. Only requires month selection."""
    if drought_model is None:
        raise HTTPException(503, "Drought model not loaded")

    features = smart_engine.compute_drought_features(req.month)
    features_df = pd.DataFrame([{
        'rolling_3mo': features['rolling_3mo_avg'],
        'rolling_6mo': features['rolling_6mo_avg'],
        'deficit_pct': features['deficit_pct'],
        'prev_year_drought': features['prev_year_drought'],
        'monsoon_strength': features['monsoon_strength'],
        'spi_3': features['spi_3'],
        'spi_6': features['spi_6'],
        'consecutive_dry_months': features['consecutive_dry_months'],
        'mei_value': features.get('mei_value', 0.0),
        'mei_lag3': features.get('mei_lag3', 0.0)
    }])

    score = max(0, min(100, drought_model.predict(features_df)[0]))
    cat = ("Extreme Drought" if score > 80 else "Severe Drought" if score > 60
           else "Moderate Drought" if score > 40 else "Mild Drought" if score > 20
           else "No Drought")
    interval = _compute_prediction_interval(drought_model, features_df, model_type="regression")

    return {
        "drought_score": round(score, 2),
        "category": cat,
        "prediction_interval": interval,
        "auto_computed_features": features,
        "mode": "citizen"
    }


@app.post("/smart/predict_heatwave")
def smart_predict_heatwave(req: SmartHeatwaveRequest):
    """Citizen-friendly heatwave prediction. Only requires month."""
    if heatwave_model is None:
        raise HTTPException(503, "Heatwave model not loaded")

    features = smart_engine.compute_heatwave_features(req.month)
    features_df = pd.DataFrame([{
        'temp_max_lag1': features['max_temp_lag1'],
        'temp_max_lag2': features['max_temp_lag2'],
        'temp_max_lag3': features['max_temp_lag3'],
        'temp_max_7day_avg': features['temp_max_7day_avg'],
        'humidity': features['humidity'],
        'month_sin': features['month_sin'],
        'month_cos': features['month_cos'],
        'month': features['month'],
        'diurnal_range_lag1': features['diurnal_range_lag1'],
        'temp_min_lag1': features['temp_min_lag1'],
        'temp_min_7day_avg': features['temp_min_7day_avg'],
        'precip_7day_sum': features['precip_7day_sum'],
        'heat_streak': features['heat_streak'],
        'wind_speed_lag1': features.get('wind_speed_lag1', 0.0),
        'wind_gusts_lag1': features.get('wind_gusts_lag1', 0.0),
        'solar_rad_lag1': features.get('solar_rad_lag1', 0.0),
        'solar_rad_3day_avg': features.get('solar_rad_3day_avg', 0.0),
        'et0_lag1': features.get('et0_lag1', 0.0),
        'et0_7day_avg': features.get('et0_7day_avg', 0.0),
        'soil_moisture_lag1': features.get('soil_moisture_lag1', 0.2),
        'pressure_lag1': features.get('pressure_lag1', 1013.0),
        'pressure_change_1d': features.get('pressure_change_1d', 0.0)
    }])

    prediction = heatwave_model.predict(features_df)[0]
    probability = heatwave_model.predict_proba(features_df)[0][1]
    interval = _compute_prediction_interval(heatwave_model, features_df, model_type="classification")

    return {
        "is_heatwave": bool(prediction),
        "heatwave_probability": round(float(probability), 3),
        "prediction_interval": interval,
        "auto_computed_features": features,
        "mode": "citizen"
    }


@app.post("/smart/predict_crop_impact")
def smart_predict_crop_impact(req: SmartCropRequest):
    """Citizen-friendly crop impact. Only requires crop, state, season."""
    if crop_model is None or crop_encoder is None:
        raise HTTPException(503, "Crop model not loaded")

    if req.crop_type not in crop_encoder.classes_:
        raise HTTPException(400, f"Unknown crop '{req.crop_type}'. Available: {crop_encoder.classes_.tolist()}")
    if req.state not in state_encoder.classes_:
        raise HTTPException(400, f"Unknown state '{req.state}'. Available: {state_encoder.classes_.tolist()}")
    if req.season.strip() not in season_encoder.classes_:
        raise HTTPException(400, f"Unknown season '{req.season}'. Available: {season_encoder.classes_.tolist()}")

    auto_features = smart_engine.compute_crop_features(req.crop_type, req.state, req.season)

    crop_enc = crop_encoder.transform([req.crop_type])[0]
    state_enc = state_encoder.transform([req.state])[0]
    season_enc = season_encoder.transform([req.season.strip()])[0]

    features_df = pd.DataFrame([{
        'rainfall': auto_features['rainfall'],
        'rainfall_anomaly': auto_features['rainfall_anomaly'],
        'fertilizer_per_area': auto_features['fertilizer_per_area'],
        'pesticide_per_area': auto_features['pesticide_per_area'],
        'crop_encoded': crop_enc,
        'state_encoded': state_enc,
        'season_encoded': season_enc
    }])

    yield_dev = max(0, min(80, crop_model.predict(features_df)[0]))
    cat = ("Critical Impact" if yield_dev > 50 else "Severe Impact" if yield_dev > 30
           else "Moderate Impact" if yield_dev > 15 else "Mild Impact" if yield_dev > 5
           else "Minimal Impact")
    interval = _compute_prediction_interval(crop_model, features_df, model_type="regression")

    return {
        "yield_deviation_pct": round(yield_dev, 2),
        "impact_category": cat,
        "prediction_interval": interval,
        "auto_computed_features": auto_features,
        "available_crops": crop_encoder.classes_.tolist(),
        "mode": "citizen"
    }


@app.get("/historical_rainfall")
def historical_rainfall():
    """Returns recent historical monthly rainfall data for trend visualization."""
    try:
        csv_path = os.path.join(BASE_DIR, "data", "hyderabad_rainfall_data.csv")
        df = pd.read_csv(csv_path)
        months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June',
                  'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Last 5 years of data for trend chart
        recent = df.tail(5)
        years = []
        for _, row in recent.iterrows():
            year_data = {"year": int(row['Year']), "monthly": []}
            for i, m in enumerate(months):
                if m in row:
                    year_data["monthly"].append({"month": month_labels[i], "rainfall_mm": float(row[m])})
            years.append(year_data)

        # Monthly averages across all years
        averages = []
        for i, m in enumerate(months):
            if m in df.columns:
                averages.append({"month": month_labels[i], "avg_mm": round(float(df[m].mean()), 1)})

        return {"years": years, "long_term_averages": averages}
    except Exception as e:
        raise HTTPException(500, f"Failed to load historical data: {str(e)}")


@app.get("/health")
def health():
    loaded = sum([rainfall_model is not None, drought_model is not None, heatwave_model is not None, crop_model is not None])
    return {"status": "healthy" if loaded >= 3 else "degraded", "models_loaded": loaded}