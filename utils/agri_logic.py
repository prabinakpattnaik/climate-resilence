import math

class AgriResilienceEngine:
    def __init__(self, weather_data, drought_risk=0.0):
        self.weather = weather_data
        self.drought_risk = drought_risk

    def get_crop_advisor(self, current_crop):
        """Recommends better crops if climate risk is high."""
        recommendations = []
        
        # 1. Millet Switching Advisor (National Mission on Millets)
        if self.drought_risk > 60:
            if current_crop.lower() in ['paddy', 'sugarcane', 'rice']:
                recommendations.append({
                    "type": "SWITCH",
                    "title": "🌾 Switch to Millets (Shree Anna)",
                    "reason": "High drought risk predicted. Paddy/Rice will likely face water stress.",
                    "suggestion": "Consider Ragi (Finger Millet) or Jowar (Sorghum) which require 70% less water."
                })

        # 2. Irrigation Skip Alert (Power & Water Saving)
        rain_now = self.weather.get('current_rainfall_mm', 0)
        rain_pred = self.weather.get('predicted_next_1h_mm', 0)
        
        if rain_pred > 20 or rain_now > 30:
            recommendations.append({
                "type": "ACTION",
                "title": "💧 Save Power: Skip Irrigation",
                "reason": f"Heavy rain ({rain_pred}mm) predicted in next hour.",
                "suggestion": "Turn off water pumps. Soil moisture will be replenished naturally."
            })

        # 3. Pest Risk Nowcasting (Heat + Humidity)
        # Simplified logic: High Humidity + Sudden Heat correlates with specific pest booms
        # Assuming humidity is passed in or using rainfall as proxy for soil moisture/humidity
        if rain_now > 10 and self.weather.get('is_raining'):
             recommendations.append({
                "type": "WARNING",
                "title": "🪲 Pest Alert: High Risk Window",
                "reason": "Combination of high humidity and standing water creates breeding grounds.",
                "suggestion": "Inspect crops for fungal growth or aphids. Apply organic bio-pesticides if needed."
            })

        return recommendations

    def get_satellite_health_scan(self, farm_bbox):
        """
        Simulates an NDVI (Satellite Crop Health) scan for a specific farm area.
        Returns a high-resolution grid (0.001) for the farm.
        """
        min_lat, min_lon, max_lat, max_lon = farm_bbox
        # Create a dense precision grid
        lats = [min_lat + i*0.001 for i in range(10)]
        lons = [min_lon + i*0.001 for i in range(10)]
        
        scan_results = []
        for lat in lats:
            for lon in lons:
                # NDVI Simulation Logic:
                # - Base health is high (0.8)
                # - High local drought risk lowers it
                # - High heatwaves lower it
                # - We use a random seed based on lat/lon to keep it consistent for "this farm"
                seed = int((lat + lon) * 1000000) % 100
                health_variance = (seed - 50) / 250.0 # small random noise
                
                ndvi = 0.85 - (self.drought_risk / 200.0) + health_variance
                ndvi = max(0.1, min(1.0, ndvi))
                
                if ndvi > 0.75:
                    status = "✅ Healthy: Optimal Growth"
                    advice = "Vigor is high. Maintain current irrigation."
                elif ndvi > 0.55:
                    status = "🟡 Vulnerable: Early Moisture Stress"
                    advice = "Slight drooping detected. Increase water frequency."
                elif ndvi > 0.35:
                    status = "🟠 Stressed: Immediate Water Needed"
                    advice = "Critical soil dryness. Action required to save yield."
                else:
                    status = "🔴 Critical: Severe Crop Damage"
                    advice = "High mortality risk. Consult agriculture officer."
                
                scan_results.append({
                    "lat": lat,
                    "lon": lon,
                    "ndvi": round(ndvi, 2),
                    "status": status,
                    "advice": advice
                })
        return scan_results

    def get_market_logistics(self, mandi_distance_km):
        """Advises on Mandi transport safety."""
        rain_now = self.weather.get('current_rainfall_mm', 0)
        if rain_now > 50:
            return {
                "safe_to_travel": False,
                "warning": "Critical Flood Risk: Heavy rain may have blocked village access roads to the Mandi.",
                "advice": "Use the 'Safe Route' map to check transport corridors before starting tractors."
            }
        return {"safe_to_travel": True, "advice": "Weather looks stable for harvest transport."}
