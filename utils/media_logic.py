class MediaIntelligenceEngine:
    """Engine for generating mass-media formatted alert briefs."""
    
    def __init__(self, weather_data, resilience_stats):
        self.weather = weather_data
        self.stats = resilience_stats

    def generate_broadcast_psa(self):
        """Generates a formatted PSA for news tickers and radio."""
        rain_now = self.weather.get('current_rainfall_mm', 0)
        risk_level = "CRITICAL" if rain_now > 50 else "WARNING" if rain_now > 20 else "ADVISORY"
        
        # Format for Ticker / SMS
        ticker = f"[{risk_level}] CLIMATE ALERT HYDERABAD: {rain_now}mm rain recorded. "
        if risk_level == "CRITICAL":
            ticker += "Major Nalas overflow risk. Avoid travel. Emergency centers active."
        else:
            ticker += "Localized water-logging expected in low-lying areas. Check Safe Route map."
            
        # Social Media Format (Hash-tags, etc.)
        social = f"🚨 #HyderabadFloodAlert: {risk_level} issued. "
        social += f"Live rainfall: {rain_now}mm. "
        social += f"Top Risk Wards: {', '.join(self.stats.get('risk_hotspots', ['Central Zone']))}. "
        social += "\n👉 Check live grid: resilience-grid.gov.in #ClimateResilience #GHMC"
        
        return {
            "risk_level": risk_level,
            "ticker_tape": ticker.upper(),
            "social_media_brief": social,
            "community_action": "Stay indoors if rainfall exceeds 30mm/hr. Use GHMC app to report pooling."
        }
