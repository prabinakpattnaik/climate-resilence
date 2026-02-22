class TourismResilienceEngine:
    """AI engine for climate-aware tourism safety in Hyderabad."""
    
    def __init__(self, weather_data):
        self.weather = weather_data
        # Major Landmarks with coordinates
        self.landmarks = [
            {"id": "charminar", "name": "Charminar", "lat": 17.3616, "lon": 78.4747, "type": "Heritage"},
            {"id": "golconda", "name": "Golconda Fort", "lat": 17.3833, "lon": 78.4011, "type": "Fort"},
            {"id": "hussain_sagar", "name": "Hussain Sagar Lake", "lat": 17.4239, "lon": 78.4738, "type": "WaterBody"},
            {"id": "birla_mandir", "name": "Birla Mandir", "lat": 17.4062, "lon": 78.4691, "type": "Temple"},
            {"id": "salargunj", "name": "Salar Jung Museum", "lat": 17.3713, "lon": 78.4803, "type": "Museum"},
            {"id": "ramoji", "name": "Ramoji Film City", "lat": 17.2543, "lon": 78.6808, "type": "Themed-Park"}
        ]

    def get_landmark_safety(self):
        """Analyzes safety for each landmark based on current climate state."""
        rain_now = self.weather.get('current_rainfall_mm', 0)
        temp_c = 35.0 # Placeholder for live temp if not in weather_data
        
        safety_reports = []
        for l in self.landmarks:
            risk_score = 0
            advice = "Safe to visit today."
            status = "Low Risk"
            
            # Logic for Heritage Sites (Open to sky)
            if l['type'] in ['Heritage', 'Fort', 'Temple']:
                if rain_now > 10:
                    risk_score = 60
                    status = "Moderate Risk"
                    advice = "Slippery surfaces. Avoid steep climbs (like Golconda stairs)."
                if rain_now > 40:
                    risk_score = 90
                    status = "Critical"
                    advice = "Active Flooding Risk. Heritage sites poorly drained; avoid visiting."
            
            # Logic for Water Bodies
            if l['type'] == 'WaterBody':
                if rain_now > 20:
                    risk_score = 75
                    status = "High Risk"
                    advice = "Hussain Sagar levels rising. Boating likely suspended."
            
            # Logic for Indoor (Museums)
            if l['type'] == 'Museum':
                if rain_now > 50:
                    risk_score = 40
                    status = "Low Risk"
                    advice = "Good indoor alternative, but check access roads for local pooling."
            
            safety_reports.append({
                **l,
                "risk_score": risk_score,
                "status": status,
                "advice": advice
            })
            
        return safety_reports

    def get_visitor_guide(self, landmark_id):
        """Provides a detailed AI visitor guide for a specific landmark."""
        landmark = next((l for l in self.landmarks if l['id'] == landmark_id), None)
        if not landmark:
            return {"error": "Landmark not found"}
            
        rain_now = self.weather.get('current_rainfall_mm', 0)
        
        # Simulated 'Safety Window'
        return {
            "best_window": "08:00 AM - 11:00 AM" if rain_now < 5 else "Limited (Heavy Rain)",
            "environmental_psa": "Heritage structures are sensitive to moisture. Please stay on designated paved paths.",
            "emergency_nearest": "Osmania General Hospital (2.1km)" if landmark['id'] == 'charminar' else "Apollo Hospitals (3.8km)"
        }
