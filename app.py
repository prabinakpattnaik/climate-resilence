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
        print(f"✓ Loaded: {name}")
        return joblib.load(full_path)
    print(f"✗ Missing: {name} at {full_path}")
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
    print("⏳ Pre-loading urban infrastructure data (KML)...")
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
    print(f"✅ Pre-loaded {len(cached_spatial_features)} infrastructure layers.")

# Preload on module load
preload_infrastructure()

# === REQUEST MODELS ===

class RainfallRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_12: float
    month: int

class DroughtRequest(BaseModel):
    rolling_3mo_avg: float
    rolling_6mo_avg: float
    deficit_pct: float
    prev_year_drought: float
    monsoon_strength: float

class HeatwaveRequest(BaseModel):
    max_temp_lag1: float
    max_temp_lag2: float
    max_temp_lag3: float
    humidity: float
    month: int

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
        "rolling_3": rolling_3
    }])
    
    pred = max(0, rainfall_model.predict(features_df)[0])
    risk = "High Risk (Flooding)" if pred > 200 else "Low Risk (Drought)" if pred < 30 else "Normal"
    
    return {
        "predicted_rainfall_mm": round(pred, 2),
        "risk_category": risk,
        "input": req.dict()
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
        'monsoon_strength': req.monsoon_strength
    }])
    
    score = max(0, min(100, drought_model.predict(features)[0]))
    
    cat = "Extreme Drought" if score > 80 else "Severe Drought" if score > 60 else "Moderate Drought" if score > 40 else "Mild Drought" if score > 20 else "No Drought"
    
    return {
        "drought_score": round(score, 2),
        "category": cat,
        "input": req.dict()
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
        'temp_max_7day_avg': temp_7day_avg, 'humidity': req.humidity, 'month_sin': month_sin, 'month_cos': month_cos, 'month': req.month
    }])
    
    prediction = heatwave_model.predict(features)[0]
    probability = heatwave_model.predict_proba(features)[0][1]
    
    return {
        "is_heatwave": bool(prediction),
        "heatwave_probability": round(float(probability), 3),
        "input": req.dict()
    }

@app.post("/predict_crop_impact")
def predict_crop_impact(req: CropRequest):
    if crop_model is None or crop_encoder is None:
        raise HTTPException(503, "Crop model not loaded")
    
    try:
        crop_enc = crop_encoder.transform([req.crop_type])[0]
        state_enc = state_encoder.transform([req.state])[0] if req.state in state_encoder.classes_ else 0
        season_enc = season_encoder.transform([req.season.strip()])[0] if req.season.strip() in season_encoder.classes_ else 0
        
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
        "baselines": {
            "heatwave_prob": 12.5,
            "crop_risk": "Low"
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

@app.get("/health")
def health():
    loaded = sum([rainfall_model is not None, drought_model is not None, heatwave_model is not None, crop_model is not None])
    return {"status": "healthy" if loaded >= 3 else "degraded", "models_loaded": loaded}